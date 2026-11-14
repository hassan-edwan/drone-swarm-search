"""Geometry-aware camera (M6): what a real search drone can see.

Five effects, all optional through `Scenario.camera`:

1. Altitude. Each drone flies low, mid or high. Higher sees a wider square
   footprint (1x1, 3x3, 5x5 cells) but each person is smaller in the image,
   so the chance of spotting them drops; cells off-centre drop a little more.
2. Line of sight. Buildings have heights. A footprint cell only counts if the
   line from the camera to the ground there clears every building it crosses.
3. Cover. Ground cells are open, rubble or tree canopy; rubble and canopy
   lower the chance of spotting someone (known from satellite imagery).
4. False alarms. From mid or high altitude a detector reports candidates:
   real sightings and false alarms alike. Only a low pass confirms a find.
5. Light and sensor. Day or dusk with an ordinary camera, night or midday
   with thermal: each scales detection and false alarms, and thermal sees
   through canopy better.

Everything that does not depend on the belief is precomputed once per
episode as P[alt, y, x, oy, ox]: the chance a drone at (x, y) and altitude
alt spots a survivor in cell (x + ox - R, y + oy - R) (0 if outside the
footprint, blocked or out of the map), and FA, the false-alarm chance there.
"""

from __future__ import annotations

import numpy as np

N_ALT = 3
ALT_NAMES = ("low", "mid", "high")
ALT_M = (15.0, 35.0, 60.0)  # flight height in metres (cells are 10 m)
FOOT_R = (0, 1, 2)  # footprint radius in cells
ALT_DETECT = (1.0, 0.7, 0.45)  # person shrinks in the image with height
OFF_NADIR = 0.15  # detection drop per ring away from the footprint centre
FA_BASE = (0.0, 0.001, 0.002)  # false alarms per visible cell per look; low passes confirm
START_ALT = 1
R = max(FOOT_R)
SIDE = 2 * R + 1
PERSON_M = 1.5

OPEN, RUBBLE, CANOPY = 0, 1, 2
COVER_NAMES = ("open", "rubble", "canopy")
COVER_FA = (1.0, 2.0, 1.5)  # debris and foliage cause more false alarms
RUBBLE_DETECT = 0.6

# name: (detection multiplier, false-alarm multiplier, canopy detection factor)
LIGHTING = {
    "day_rgb": (1.0, 1.0, 0.3),
    "dusk_rgb": (0.6, 1.5, 0.25),
    "night_thermal": (0.9, 0.5, 0.5),
    "midday_thermal": (0.6, 2.5, 0.5),
}
LIGHT_NAMES = tuple(LIGHTING)
BUILDING_M = (10.0, 50.0)  # building heights


def make_cover(rng: np.random.Generator, free: np.ndarray, frac: float) -> np.ndarray:
    """(g, g) cover type per cell: canopy as smooth blobs, rubble next to buildings."""
    g = free.shape[0]
    cover = np.zeros((g, g), dtype=np.int8)
    if frac <= 0:
        return cover
    coarse = rng.random((g // 4 + 2, g // 4 + 2))
    yy, xx = np.mgrid[0:g, 0:g] / 4.0
    x0, y0 = xx.astype(int), yy.astype(int)
    fx, fy = xx - x0, yy - y0
    noise = (coarse[y0, x0] * (1 - fx) * (1 - fy) + coarse[y0, x0 + 1] * fx * (1 - fy)
             + coarse[y0 + 1, x0] * (1 - fx) * fy + coarse[y0 + 1, x0 + 1] * fx * fy)
    canopy = free & (noise >= np.quantile(noise[free], 1 - 0.7 * frac))
    cover[canopy] = CANOPY
    near = np.zeros((g, g), dtype=bool)
    obs = ~free
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(obs, dy, 0), dx, 1)
    rubble = free & ~canopy & near & (rng.random((g, g)) < min(1.0, 2.0 * frac))
    cover[rubble] = RUBBLE
    return cover


def make_heights(rng: np.random.Generator, obstacles: np.ndarray) -> np.ndarray:
    """(g, g) building height in metres (0 on free cells); one height per building block."""
    g = obstacles.shape[0]
    h = np.zeros((g, g))
    seen = np.zeros((g, g), dtype=bool)
    for y, x in zip(*np.nonzero(obstacles)):
        if seen[y, x]:
            continue
        height = rng.uniform(*BUILDING_M)
        stack = [(y, x)]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            h[cy, cx] = height
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < g and 0 <= nx < g and obstacles[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return h


def line_of_sight(heights: np.ndarray, n_samples: int = 16) -> np.ndarray:
    """(N_ALT, g, g, SIDE, SIDE) bool: from each cell and altitude, is the
    ground at each footprint offset visible (no building above the sight line)?"""
    g = heights.shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    vis = np.ones((N_ALT, g, g, SIDE, SIDE), dtype=bool)
    ts = (np.arange(n_samples) + 0.5) / n_samples
    for oy in range(SIDE):
        for ox in range(SIDE):
            dy, dx = oy - R, ox - R
            if dx == 0 and dy == 0:
                continue
            for alt in range(N_ALT):
                blocked = np.zeros((g, g), dtype=bool)
                for t in ts:
                    sx = np.floor(xx + 0.5 + dx * t).astype(int)
                    sy = np.floor(yy + 0.5 + dy * t).astype(int)
                    inside = (sx >= 0) & (sx < g) & (sy >= 0) & (sy < g)
                    sxc, syc = np.clip(sx, 0, g - 1), np.clip(sy, 0, g - 1)
                    is_target = (sx == xx + dx) & (sy == yy + dy)
                    line_h = ALT_M[alt] * (1 - t) + PERSON_M * t
                    blocked |= inside & ~is_target & (heights[syc, sxc] > line_h)
                vis[alt, :, :, oy, ox] = ~blocked
    return vis


def tables(free: np.ndarray, heights: np.ndarray, cover: np.ndarray, base_p: float,
           lighting: str) -> tuple[np.ndarray, np.ndarray]:
    """P and FA, both (N_ALT, g, g, SIDE, SIDE)."""
    g = free.shape[0]
    det, fa_mult, canopy = LIGHTING[lighting]
    cover_det = np.choose(cover, [1.0, RUBBLE_DETECT, canopy])
    cover_fa = np.choose(cover, COVER_FA)
    vis = line_of_sight(heights)
    yy, xx = np.mgrid[0:g, 0:g]
    P = np.zeros((N_ALT, g, g, SIDE, SIDE))
    FA = np.zeros_like(P)
    for oy in range(SIDE):
        for ox in range(SIDE):
            dy, dx = oy - R, ox - R
            ring = max(abs(dx), abs(dy))
            tx, ty = xx + dx, yy + dy
            ok = (tx >= 0) & (tx < g) & (ty >= 0) & (ty < g)
            txc, tyc = np.clip(tx, 0, g - 1), np.clip(ty, 0, g - 1)
            ok &= free[tyc, txc]
            for alt in range(N_ALT):
                if ring > FOOT_R[alt]:
                    continue
                m = ok & vis[alt, :, :, oy, ox]
                p = base_p * det * ALT_DETECT[alt] * (1 - OFF_NADIR * ring) * cover_det[tyc, txc]
                P[alt, :, :, oy, ox] = np.where(m, np.minimum(p, 0.98), 0.0)
                FA[alt, :, :, oy, ox] = np.where(m, FA_BASE[alt] * fa_mult * cover_fa[tyc, txc], 0.0)
    return P, FA


def footprint_sum(grid: np.ndarray, P_alt: np.ndarray) -> np.ndarray:
    """(g, g): for a drone at each cell, sum over its footprint of grid[cell] * P."""
    g = grid.shape[0]
    pad = np.pad(grid, R)
    out = np.zeros((g, g))
    for oy in range(SIDE):
        for ox in range(SIDE):
            out += pad[oy:oy + g, ox:ox + g] * P_alt[:, :, oy, ox]
    return out


def light_vector(lighting: str) -> np.ndarray:
    det, fa, canopy = LIGHTING[lighting]
    return np.array([det, fa / 2.5, canopy], dtype=np.float32)
