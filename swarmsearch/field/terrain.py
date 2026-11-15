"""World generation for the field environment (register E1, E2, K5).

A World holds the true layers and the drones' map of them (which can be
wrong: missing trails, a stale canopy map, buildings that have since
collapsed). All layers are (g, g) arrays indexed [y, x].

- wilderness: fractal elevation with ridges; drainages from D8 flow routing
  on the pit-filled surface; trails routed from the trailhead to peaks and
  water along gentle slopes; forest density with a treeline; cliffs (slope
  over 45 degrees) people cannot cross.
- urban: building blocks with heights, canopy and rubble, a share of blocks
  collapsed into rubble that can be flown over and holds trapped people.
- flood: a valley with a river; the water level rises during the mission
  (see FieldEnv), low buildings where people take refuge.

`world_from_arrays` builds a wilderness World from real data (elevation,
canopy, trails, water), e.g. for a field-test site (scripts/import_site.py).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..obstacles import make_obstacles
from .. import camera as cam
from .. import realism as rl

N8 = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
CLIFF_SLOPE = 1.0  # rise over run: 45 degrees
TREE_M = 20.0  # tree height (low flights stay above it)


@dataclass
class World:
    family: str
    g: int
    cell_m: float
    elev: np.ndarray  # ground elevation, m
    struct_h: np.ndarray  # building height above ground, m (true)
    walk: np.ndarray  # people can be here (true)
    fly_block: np.ndarray  # drones cannot enter (true)
    canopy: np.ndarray  # tree cover density 0-1 (true)
    rubble: np.ndarray  # rubble cover (true)
    water: np.ndarray  # open water now
    trail: np.ndarray  # trails / streets (true)
    stream: np.ndarray  # drainages (true)
    flow: np.ndarray  # (2, g, g) downhill direction (dx, dy) per cell
    base: tuple[int, int]  # drone base and command post (x, y)
    ipp: tuple[int, int]  # true last-seen point
    shelters: list  # places people head for
    # The drones' map (prior knowledge)
    map_struct_h: np.ndarray
    map_walk: np.ndarray
    map_fly_block: np.ndarray
    map_canopy: np.ndarray
    map_trail: np.ndarray
    collapsed: np.ndarray = None  # urban: true collapses, unknown to the map
    river_level: float = 0.0  # flood: water surface height at launch, m
    extra: dict = field(default_factory=dict)

    @property
    def surface(self) -> np.ndarray:
        return self.elev + self.struct_h

    @property
    def map_surface(self) -> np.ndarray:
        return self.elev + self.map_struct_h


# ------------------------------------------------------------------ noise
def smooth_noise(rng, g: int, scale: int) -> np.ndarray:
    """Bilinear-interpolated random field with features about `scale` cells wide, in [0, 1]."""
    n = g // scale + 2
    c = rng.random((n, n))
    yy, xx = np.mgrid[0:g, 0:g] / float(scale)
    x0, y0 = xx.astype(int), yy.astype(int)
    fx, fy = xx - x0, yy - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)  # smoothstep: no grid artefacts
    return (c[y0, x0] * (1 - fx) * (1 - fy) + c[y0, x0 + 1] * fx * (1 - fy)
            + c[y0 + 1, x0] * (1 - fx) * fy + c[y0 + 1, x0 + 1] * fx * fy)


def fbm(rng, g: int, scales=(16, 8, 4), weights=(1.0, 0.5, 0.25)) -> np.ndarray:
    out = sum(w * smooth_noise(rng, g, s) for s, w in zip(scales, weights))
    return (out - out.min()) / max(out.max() - out.min(), 1e-9)


# ------------------------------------------------------------- hydrology
def fill_pits(elev: np.ndarray) -> np.ndarray:
    """Priority-flood: raise every pit to its spill height so water drains off the map edge."""
    g = elev.shape[0]
    out = elev.copy()
    done = np.zeros((g, g), dtype=bool)
    heap = []
    for y in range(g):
        for x in range(g):
            if x in (0, g - 1) or y in (0, g - 1):
                heapq.heappush(heap, (out[y, x], y, x))
                done[y, x] = True
    while heap:
        h, y, x = heapq.heappop(heap)
        for dx, dy in N8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and not done[ny, nx]:
                done[ny, nx] = True
                out[ny, nx] = max(out[ny, nx], h + 1e-3)
                heapq.heappush(heap, (out[ny, nx], ny, nx))
    return out


def d8_flow(elev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downhill direction (2, g, g) and flow accumulation (cells draining through each cell)."""
    g = elev.shape[0]
    filled = fill_pits(elev)
    flow = np.zeros((2, g, g), dtype=int)
    P = np.pad(filled, 1, constant_values=np.inf)
    best = np.full((g, g), np.inf)
    for dx, dy in N8:
        nb = (P[1 + dy:1 + dy + g, 1 + dx:1 + dx + g] - filled) / np.hypot(dx, dy)
        better = nb < best
        best = np.where(better, nb, best)
        flow[0][better], flow[1][better] = dx, dy
    edge = ~np.isfinite(best) | (best >= 0)  # map edge or flat: drains out
    flow[:, edge] = 0
    acc = np.ones((g, g))
    for i in np.argsort(-filled, axis=None):
        y, x = divmod(int(i), g)
        dx, dy = flow[0, y, x], flow[1, y, x]
        if dx or dy:
            acc[y + dy, x + dx] += acc[y, x]
    return flow, acc


def slope(elev: np.ndarray, cell_m: float) -> np.ndarray:
    gy, gx = np.gradient(elev, cell_m)
    return np.hypot(gx, gy)


def cheapest_path(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Dijkstra over 8-neighbours; cost per cell entered (inf = impassable). Returns (x, y) cells."""
    g = cost.shape[0]
    dist = np.full((g, g), np.inf)
    prev = {}
    sx, sy = start
    dist[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if (x, y) == goal:
            break
        if d > dist[y, x]:
            continue
        for dx, dy in N8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and np.isfinite(cost[ny, nx]):
                nd = d + cost[ny, nx] * np.hypot(dx, dy)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, nx, ny))
    path, c = [], goal
    while c in prev or c == start:
        path.append(c)
        if c == start:
            break
        c = prev[c]
    return path[::-1] if path and path[-1] == start else []


# ------------------------------------------------------------- families
def wilderness(rng, g: int, cell_m: float, map_error: float) -> World:
    relief = rng.uniform(150, 500)
    base_n = fbm(rng, g, (16, 8, 4), (1.0, 0.5, 0.25))
    ridge = 1.0 - np.abs(2 * fbm(rng, g, (12, 6), (1.0, 0.4)) - 1.0)
    elev = relief * (0.7 * base_n + 0.3 * ridge ** 2)
    sl = slope(elev, cell_m)
    cliff = sl > CLIFF_SLOPE
    flow, acc = d8_flow(elev)
    stream = acc >= max(12, g * g // 60)
    river = acc >= g * g // 6  # deep water: impassable on foot
    water = river.copy()
    walk = ~cliff & ~river
    # Trailhead and command post near the middle (the search area is drawn
    # around the last-seen point), on a low walkable cell.
    lo, hi = int(0.35 * g), int(0.65 * g)
    cand = [(x, y) for y in range(lo, hi) for x in range(lo, hi) if walk[y, x]]
    base = min(cand, key=lambda c: elev[c[1], c[0]] + rng.uniform(0, relief * 0.5))
    trail_cost = np.where(walk, 1.0 + 25.0 * sl ** 2, np.inf)
    trail = np.zeros((g, g), dtype=bool)
    peak = np.unravel_index(int(np.argmax(np.where(walk, elev, -1))), elev.shape)
    junction = np.unravel_index(int(np.argmax(np.where(walk & stream, acc, -1))), elev.shape)
    far = [(int(rng.integers(0, g)), int(rng.integers(0, g))) for _ in range(2)]
    dests = [(int(peak[1]), int(peak[0])), (int(junction[1]), int(junction[0])), *far]
    for d in dests:
        if walk[d[1], d[0]]:
            for x, y in cheapest_path(trail_cost, base, d):
                trail[y, x] = True
    trail[base[1], base[0]] = True
    canopy = np.clip(1.3 * fbm(rng, g, (10, 5), (1.0, 0.5)) - 0.15, 0, 1)
    treeline = np.quantile(elev, 0.85)
    canopy = np.where(elev > treeline, canopy * 0.2, canopy)
    canopy = np.where(trail, canopy * 0.6, canopy)
    canopy = np.where(water, 0.0, canopy)
    # Last seen: on a trail some way from the trailhead.
    tcells = np.argwhere(trail)
    d = np.hypot(tcells[:, 1] - base[0], tcells[:, 0] - base[1])
    pick = tcells[(d >= 2) & (d <= g // 6)] if ((d >= 2) & (d <= g // 6)).any() else tcells
    iy, ix = pick[rng.integers(len(pick))]
    # The drones' map: some trail segments missing, canopy map smoothed and noisy (K5).
    map_trail = trail.copy()
    if map_error > 0:  # whole stretches missing, about map_error of the trail in all
        segs = np.argwhere(trail)
        for y, x in segs[rng.random(len(segs)) < map_error / 3]:
            map_trail[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = False
        map_trail[base[1], base[0]] = True
    map_canopy = np.clip(canopy + map_error * (smooth_noise(rng, g, 4) - 0.5), 0, 1)
    zeros = np.zeros((g, g))
    return World("wilderness", g, cell_m, elev, zeros, walk, np.zeros((g, g), bool), canopy, np.zeros((g, g), bool),
                 water, trail, stream, flow, base, (int(ix), int(iy)), [base],
                 map_struct_h=zeros.copy(), map_walk=walk.copy(), map_fly_block=np.zeros((g, g), bool),
                 map_canopy=map_canopy, map_trail=map_trail)


def urban(rng, g: int, cell_m: float, collapse_frac: float, map_error: float) -> World:
    base = (0, g // 2)
    obst = make_obstacles(rng, g, float(rng.uniform(0.15, 0.3)), [base], "blocks")
    heights = cam.make_heights(rng, obst)
    cover = cam.make_cover(rng, ~obst, float(rng.uniform(0.0, 0.3)))
    collapsed = rl.collapse(rng, obst, collapse_frac)
    struct_h = np.where(collapsed, 3.0, heights)  # rubble piles
    walk = ~obst | collapsed
    canopy = np.where(cover == cam.CANOPY, 0.7, 0.0)
    rubble = (cover == cam.RUBBLE) | collapsed
    streets = ~obst & (rng.random((g, g)) < 1.0)
    shelters = []
    free_idx = np.argwhere(~obst)
    for _ in range(int(rng.integers(1, 4))):
        y, x = free_idx[rng.integers(len(free_idx))]
        shelters.append((int(x), int(y)))
    zeros = np.zeros((g, g))
    flow = np.zeros((2, g, g), dtype=int)
    w = World("urban", g, cell_m, zeros, struct_h, walk, obst & ~collapsed, canopy, rubble,
              np.zeros((g, g), bool), streets, np.zeros((g, g), bool), flow, base, base, shelters,
              map_struct_h=heights.copy(), map_walk=~obst, map_fly_block=obst.copy(),
              map_canopy=np.clip(canopy + map_error * (smooth_noise(rng, g, 4) - 0.5), 0, 1),
              map_trail=streets.copy(), collapsed=collapsed)
    w.extra["cover"] = cover
    return w


def flood(rng, g: int, cell_m: float, map_error: float) -> World:
    yy, xx = np.mgrid[0:g, 0:g]
    centre = g / 2 + (g / 6) * np.sin(xx / g * np.pi * rng.uniform(1, 2.5) + rng.uniform(0, 6))
    valley = rng.uniform(8, 20) * (np.abs(yy - centre) / (g / 2)) ** 1.3
    elev = valley + 4.0 * (1 - xx / g) + 3.0 * fbm(rng, g, (8, 4), (1.0, 0.5))
    river = np.abs(yy - centre) < 1.2
    level = float(np.quantile(elev[river], 0.9)) + 0.3
    water = elev < level
    flow = np.zeros((2, g, g), dtype=int)
    flow[0] = 1  # the river runs towards +x
    # Village: low buildings on the valley sides (people shelter on roofs).
    bld = np.zeros((g, g), dtype=bool)
    for _ in range(int(rng.integers(6, 16))):
        x, y = int(rng.integers(2, g - 2)), int(rng.integers(0, g))
        if not water[y, x]:
            bld[y:y + int(rng.integers(1, 3)), x:x + int(rng.integers(1, 3))] = True
    bld &= ~water
    struct_h = np.where(bld, rng.uniform(4, 10, (g, g)), 0.0)
    canopy = np.clip(fbm(rng, g, (6, 3), (1.0, 0.5)) - 0.4, 0, 0.6) * ~water
    high = np.argwhere(elev > np.quantile(elev, 0.9))
    shelters = [(int(x), int(y)) for y, x in high[rng.choice(len(high), size=min(3, len(high)), replace=False)]]
    side = 0 if elev[0].mean() > elev[-1].mean() else g - 1
    base = (int(np.argmax(elev[side])), side)
    zeros = np.zeros((g, g))
    walk = np.ones((g, g), dtype=bool)  # people can be anywhere: in water they drift
    w = World("flood", g, cell_m, elev, struct_h, walk, np.zeros((g, g), bool), canopy, np.zeros((g, g), bool),
              water, np.zeros((g, g), bool), river, flow, base, base, shelters,
              map_struct_h=struct_h.copy(), map_walk=walk.copy(), map_fly_block=np.zeros((g, g), bool),
              map_canopy=np.clip(canopy + map_error * (smooth_noise(rng, g, 4) - 0.5), 0, 1),
              map_trail=np.zeros((g, g), bool), river_level=level)
    return w


def make_world(rng, sc) -> World:
    fam = sc.fam
    if sc.family == "wilderness" and sc.site:
        d = np.load(sc.site)
        if abs(float(d["cell_m"]) - fam.cell_m) > 1e-6:
            raise ValueError(f"site cell size {float(d['cell_m'])} m: set FieldScenario(cell_m={float(d['cell_m'])})")
        return world_from_arrays(d["elev"], d["canopy"], d["trail"], d["water"], float(d["cell_m"]),
                                 tuple(int(v) for v in d["base"]), tuple(int(v) for v in d["ipp"]))
    if sc.family == "wilderness":
        return wilderness(rng, fam.grid, fam.cell_m, sc.map_error)
    if sc.family == "urban":
        return urban(rng, fam.grid, fam.cell_m, sc.collapse_frac, sc.map_error)
    if sc.family == "flood":
        return flood(rng, fam.grid, fam.cell_m, sc.map_error)
    raise ValueError(sc.family)


def world_from_arrays(elev: np.ndarray, canopy: np.ndarray, trail: np.ndarray, water: np.ndarray,
                      cell_m: float, base: tuple[int, int], ipp: tuple[int, int]) -> World:
    """A wilderness World from real map layers (all (g, g), already resampled
    to the grid): elevation in metres, canopy density 0-1, trail and water masks."""
    g = elev.shape[0]
    sl = slope(elev, cell_m)
    flow, acc = d8_flow(elev)
    walk = (sl <= CLIFF_SLOPE) & ~water
    zeros = np.zeros((g, g))
    return World("wilderness", g, cell_m, elev.astype(float), zeros, walk, np.zeros((g, g), bool),
                 canopy.astype(float), np.zeros((g, g), bool), water.astype(bool), trail.astype(bool),
                 acc >= max(12, g * g // 60), flow, base, ipp, [base],
                 map_struct_h=zeros.copy(), map_walk=walk.copy(), map_fly_block=np.zeros((g, g), bool),
                 map_canopy=canopy.astype(float).copy(), map_trail=trail.astype(bool).copy())
