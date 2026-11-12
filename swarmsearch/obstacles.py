"""Obstacle layouts for M2 and shortest-path distance fields.

"blocks": rectangles of 1-4 cells a side (collapsed buildings, rubble,
water), placed until the target fraction is covered. "scatter": single
random cells (the old layout). Either way every free cell is reachable from
the drone base: enclosed pockets are filled in, since nobody could search
them (or be found in them) anyway.
"""

from __future__ import annotations

import numpy as np

NEIGHBOURS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
LAYOUTS = ("blocks", "scatter")


def make_obstacles(rng: np.random.Generator, g: int, frac: float, keep_free: list[tuple[int, int]],
                   layout: str = "blocks", tries: int = 20) -> np.ndarray:
    """(g, g) bool obstacle map with about `frac` of cells blocked. Cells in
    `keep_free` stay free and reachable from keep_free[0]."""
    n_target = int(frac * g * g)
    keep = np.zeros((g, g), dtype=bool)
    for x, y in keep_free:
        keep[y, x] = True
    obs = np.zeros((g, g), dtype=bool)
    for _ in range(tries):
        obs = np.zeros((g, g), dtype=bool)
        if layout == "scatter":
            idx = rng.choice(np.flatnonzero(~keep), size=n_target, replace=False)
            obs.flat[idx] = True
        else:
            for _ in range(50 * g * g):
                if obs.sum() >= n_target:
                    break
                w, h = rng.integers(1, 5, size=2)
                x0, y0 = rng.integers(0, g - w + 1), rng.integers(0, g - h + 1)
                block = np.zeros((g, g), dtype=bool)
                block[y0:y0 + h, x0:x0 + w] = True
                if (block & keep).any() or (obs | block).sum() > n_target + 3:
                    continue
                obs |= block
        reach = reachable(~obs, keep_free[0])
        if reach[keep].all():
            return ~reach  # fill enclosed pockets
    return np.zeros((g, g), dtype=bool)  # give up: open map


def reachable(free: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Cells reachable from `start` with 8-neighbour moves over free cells."""
    return np.isfinite(distance_fields(free, [start])[0])


def distance_fields(free: np.ndarray, targets: list[tuple[int, int]]) -> np.ndarray:
    """(len(targets), g, g) 8-neighbour shortest-path distance from every cell
    to each target over free cells (inf where unreachable), all at once."""
    g = free.shape[0]
    D = np.full((len(targets), g, g), np.inf)
    for k, (tx, ty) in enumerate(targets):
        if free[ty, tx]:
            D[k, ty, tx] = 0.0
    blocked = ~free
    while True:
        P = np.pad(D, ((0, 0), (1, 1), (1, 1)), constant_values=np.inf)
        best = D.copy()
        for dx, dy in NEIGHBOURS:
            np.minimum(best, P[:, 1 + dy:1 + dy + g, 1 + dx:1 + dx + g] + 1.0, out=best)
        best[:, blocked] = np.inf
        if np.array_equal(best, D):
            return D
        D = best
