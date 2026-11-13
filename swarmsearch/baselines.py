"""Fixed search strategies used as baselines.

Both act through the same env interface as the learned policy:
`reset(env)` once per episode, then `act(env) -> {agent: action}`.
"""

from __future__ import annotations

import numpy as np

from .env import HOVER, MOVES, GridSearchEnv
from .obstacles import distance_fields


def distance_field(free: np.ndarray, target: tuple[int, int], corner_cut: bool = True) -> np.ndarray:
    """8-neighbour shortest-path distance from every cell to `target` (inf if blocked)."""
    return distance_fields(free, [target], corner_cut)[0]


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
            self._fields[target] = distance_field(env.free, target, env.scenario.corner_cutting)
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
                if mask[act]:
                    out[a] = act
                    continue
                if env.free[target[1], target[0]] and not env.scenario.corner_cutting:
                    pass  # blocked corner: path-find round it below
                else:
                    out[a] = HOVER
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


class BeliefGreedy:
    """Hand-written use of the same belief map the learned policy sees: each
    drone heads for the cell with the most expected detections (plus some
    credit for its neighbours) per step of travel, skipping cells near teammates'
    targets, and keeps that target until it gets there or most of its value
    is gone (re-picking every step makes drones dither between two spots).
    Tells apart what the belief map is worth from what the learning adds."""

    def __init__(self, claim_radius: int = 2, drop: float = 0.5, neighbour_weight: float = 0.0):
        self.r = claim_radius
        self.drop = drop
        self.nb = neighbour_weight

    def reset(self, env: GridSearchEnv):
        self.targets: dict[str, tuple[tuple[int, int], float]] = {}

    def act(self, env: GridSearchEnv) -> dict:
        live = [a for a in env.agents if env.alive.get(a)]
        if not live:
            return {}
        g = env.grid_size
        cc = env.scenario.corner_cutting
        gain = env.belief * env.scenario.detection_prob
        P = np.pad(gain, 1)
        box = sum(P[1 + dy:1 + dy + g, 1 + dx:1 + dx + g] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        val = gain + self.nb * (box - gain)  # the cell itself, plus some credit for a rich neighbourhood
        for a in list(self.targets):
            (tx, ty), v0 = self.targets[a]
            here = tuple(int(v) for v in env.pos[a]) == (tx, ty)
            if a not in live or here or val[ty, tx] < self.drop * v0:
                del self.targets[a]
        need = [a for a in live if a not in self.targets]
        if need:
            D = distance_fields(env.free, [tuple(int(v) for v in env.pos[a]) for a in need], cc)
            taken = ~env.free.copy()
            for (tx, ty), _ in self.targets.values():
                taken[max(0, ty - self.r):ty + self.r + 1, max(0, tx - self.r):tx + self.r + 1] = True
            for k, a in enumerate(need):
                x, y = (int(v) for v in env.pos[a])
                score = np.where(taken, -1.0, val / (1.0 + D[k]))
                score[y, x] = -1.0  # just searched it
                ty, tx = np.unravel_index(int(np.argmax(score)), score.shape)
                self.targets[a] = ((int(tx), int(ty)), float(val[ty, tx]))
                taken[max(0, ty - self.r):ty + self.r + 1, max(0, tx - self.r):tx + self.r + 1] = True
        Dt = distance_fields(env.free, [self.targets[a][0] for a in live], cc)
        out = {}
        for k, a in enumerate(live):
            mask = env.action_mask(a)
            nxt = env.pos[a] + MOVES
            best, best_key = HOVER, None
            for m in np.flatnonzero(mask):
                nx, ny = int(nxt[m, 0]), int(nxt[m, 1])
                key = (Dt[k, ny, nx], -gain[ny, nx])  # closer to target, then more to find on the way
                if best_key is None or key < best_key:
                    best, best_key = int(m), key
            out[a] = best
        return out
