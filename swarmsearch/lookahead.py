"""Short lookahead on the team's belief map (no learning).

`path_value` scores a cell by the most expected survivor detections a drone
could collect on an H-cell path that starts there, searching each cell it
passes over. A cell searched twice on the path pays only what the first
search left behind (x (1 - p)). Paths are enumerated exactly (9^(H-1) per
start, so keep H small). Survivor movement over those few steps (at most
~0.15 cells per step) is ignored. Used at run time to re-rank the learned
policy's top moves.
"""

from __future__ import annotations

import numpy as np

from .belief import MOVES


def path_value(gain: np.ndarray, free: np.ndarray, start: tuple[int, int], horizon: int,
               p: float, corner_cut: bool = True) -> float:
    g = gain.shape[0]
    visits: dict[tuple[int, int], int] = {}

    def take(c) -> float:
        k = visits.get(c, 0)
        visits[c] = k + 1
        return float(gain[c[1], c[0]]) * (1.0 - p) ** k

    def ok(x, y, dx, dy) -> bool:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < g and 0 <= ny < g and free[ny, nx]):
            return False
        return corner_cut or not (dx and dy) or (free[y, nx] and free[ny, x])

    def best(c, h) -> float:
        if h == 0:
            return 0.0
        x, y = c
        out = 0.0
        for dx, dy in MOVES:
            dx, dy = int(dx), int(dy)
            if (dx or dy) and not ok(x, y, dx, dy):
                continue
            n = (x + dx, y + dy)
            v = take(n)
            v += best(n, h - 1)
            visits[n] -= 1
            out = max(out, v)
        return out

    s = (int(start[0]), int(start[1]))
    return take(s) + best(s, horizon - 1)
