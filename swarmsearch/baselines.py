"""Fixed search strategies used as baselines.

Both act through the same env interface as the learned policy:
`reset(env)` once per episode, then `act(env) -> {agent: action}`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .env import HOVER, MOVES, GridSearchEnv


def distance_field(free: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """8-connected BFS distance from every cell to `target` (inf if blocked)."""
    g = free.shape[0]
    dist = np.full((g, g), np.inf)
    tx, ty = target
    dist[ty, tx] = 0
    q = deque([(tx, ty)])
    while q:
        x, y = q.popleft()
        for dx, dy in MOVES[:8]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and free[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist


class RandomWalk:
    """Uniformly random valid move. Lower bound."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, env: GridSearchEnv):
        pass

    def act(self, env: GridSearchEnv) -> dict:
        out = {}
        for a in env.agents:
            valid = np.flatnonzero(env.action_mask(a))
            out[a] = int(self.rng.choice(valid))
        return out


class Lawnmower:
    """Split the rows evenly between drones; each sweeps its strip back and
    forth, then starts over. Ignores the heat map, the movement profile and
    lost teammates, which is the point: it is the same plan in every scenario."""

    def reset(self, env: GridSearchEnv):
        g = env.grid_size
        strips = np.array_split(np.arange(g), len(env.agents))
        self.routes = {}
        for a, rows in zip(env.agents, strips):
            route = []
            for i, y in enumerate(rows):
                xs = range(g) if i % 2 == 0 else range(g - 1, -1, -1)
                route += [(x, int(y)) for x in xs if env.free[y, x]]
            self.routes[a] = route
        self.idx = {a: 0 for a in env.agents}
        self._fields: dict[tuple[int, int], np.ndarray] = {}

    def _field(self, env, target):
        if target not in self._fields:
            self._fields[target] = distance_field(env.free, target)
        return self._fields[target]

    def act(self, env: GridSearchEnv) -> dict:
        out = {}
        for a in env.agents:
            route = self.routes[a]
            if not route:
                out[a] = HOVER
                continue
            pos = tuple(int(v) for v in env.pos[a])
            mask = env.action_mask(a)
            target = route[self.idx[a] % len(route)]
            if target == pos:
                self.idx[a] += 1
                target = route[self.idx[a] % len(route)]
            step = (target[0] - pos[0], target[1] - pos[1])
            if max(abs(step[0]), abs(step[1])) == 1:
                # Next waypoint is a neighbour: step straight to it.
                act = int(np.flatnonzero((MOVES == step).all(axis=1))[0])
                out[a] = act if mask[act] else HOVER
                continue
            # Otherwise path-find, skipping waypoints that are unreachable.
            for _ in range(len(route)):
                target = route[self.idx[a] % len(route)]
                field = self._field(env, target)
                if target != pos and np.isfinite(field[pos[1], pos[0]]):
                    break
                self.idx[a] += 1
            best, best_d = HOVER, field[pos[1], pos[0]]
            for act in np.flatnonzero(mask[:8]):
                nx, ny = pos[0] + MOVES[act][0], pos[1] + MOVES[act][1]
                if field[ny, nx] < best_d:
                    best, best_d = int(act), field[ny, nx]
            out[a] = best
        return out
