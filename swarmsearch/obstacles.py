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
                   layout: str = "blocks", tries: int = 20, corner_cut: bool = True) -> np.ndarray:
    """(g, g) bool obstacle map with about `frac` of cells blocked. Cells in
    `keep_free` stay free and reachable from keep_free[0] (without cutting
    a blocked cell's corner diagonally if not corner_cut)."""
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
        reach = reachable(~obs, keep_free[0], corner_cut)
        if reach[keep].all():
            return ~reach  # fill enclosed pockets
    return np.zeros((g, g), dtype=bool)  # give up: open map


def reachable(free: np.ndarray, start: tuple[int, int], corner_cut: bool = True) -> np.ndarray:
    """Cells reachable from `start` with 8-neighbour moves over free cells."""
    return np.isfinite(distance_fields(free, [start], corner_cut)[0])


def diagonal_ok(free: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """(g, g): the move (dx, dy) from each cell does not cut a blocked cell's
    corner (both orthogonal neighbours it passes are free); a full-cell
    building's corner lies exactly on the diagonal. Always true for straight moves."""
    g = free.shape[0]
    if dx == 0 or dy == 0:
        return np.ones((g, g), dtype=bool)
    P = np.pad(free, 1, constant_values=False)
    side_x = P[1:1 + g, 1 + dx:1 + dx + g]  # cell (x + dx, y)
    side_y = P[1 + dy:1 + dy + g, 1:1 + g]  # cell (x, y + dy)
    return side_x & side_y


def distance_fields(free: np.ndarray, targets: list[tuple[int, int]], corner_cut: bool = True) -> np.ndarray:
    """(len(targets), g, g) 8-neighbour shortest-path distance from every cell
    to each target over free cells (inf where unreachable), all at once.
    Without corner_cut, diagonal moves past a blocked cell's corner are not allowed."""
    g = free.shape[0]
    D = np.full((len(targets), g, g), np.inf)
    for k, (tx, ty) in enumerate(targets):
        if free[ty, tx]:
            D[k, ty, tx] = 0.0
    blocked = ~free
    allowed = [None if corner_cut else diagonal_ok(free, dx, dy) for dx, dy in NEIGHBOURS]
    while True:
        P = np.pad(D, ((0, 0), (1, 1), (1, 1)), constant_values=np.inf)
        best = D.copy()
        for (dx, dy), ok in zip(NEIGHBOURS, allowed):
            nb = P[:, 1 + dy:1 + dy + g, 1 + dx:1 + dx + g] + 1.0
            if ok is not None:
                nb = np.where(ok, nb, np.inf)
            np.minimum(best, nb, out=best)
        best[:, blocked] = np.inf
        if np.array_equal(best, D):
            return D
        D = best
