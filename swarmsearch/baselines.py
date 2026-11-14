"""Fixed search strategies used as baselines.

Both act through the same env interface as the learned policy:
`reset(env)` once per episode, then `act(env) -> {agent: action}`.
"""

from __future__ import annotations

import numpy as np

from . import camera as cam
from .env import CLIMB, DESCEND, HOVER, MOVES, GridSearchEnv
from .obstacles import distance_fields

CONFIRM_RANGE = 6  # camera mode: go check candidates at most this many steps away


def _altitude_step(alt: int, want: int) -> int | None:
    """Climb or descend toward `want`; None when already there."""
    if alt == want:
        return None
    return CLIMB if want > alt else DESCEND


def _toward(env, a, field, tiebreak=None) -> int:
    """Move that most reduces `field` (a distance field), ties by higher tiebreak."""
    mask = env.action_mask(a)
    nxt = env.pos[a] + MOVES
    best, best_key = HOVER, None
    for m in np.flatnonzero(mask[:9]):
        nx, ny = int(nxt[m, 0]), int(nxt[m, 1])
        key = (field[ny, nx], 0.0 if tiebreak is None else -tiebreak[ny, nx])
        if best_key is None or key < best_key:
            best, best_key = int(m), key
    return best


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
    lost teammates, which is the point: it is the same plan in every scenario.

    Camera mode: sweep at mid altitude, whose 3-cell-wide footprint means one
    lane every third row, and break off to check any candidate within
    CONFIRM_RANGE (fly there, descend to low, climb back) before carrying on:
    the standard "sweep, then investigate" procedure."""

    SWEEP_ALT = 1

    def reset(self, env: GridSearchEnv):
        g = env.grid_size
        strips = np.array_split(np.arange(g), len(env.agents))
        self.routes = {}
        self.tasks: dict[str, tuple[int, int]] = {}
        for a, rows in zip(env.agents, strips):
            route = []
            if env.camera:  # lanes 3 rows apart, centred so every row is within one of a lane
                rows = rows[1::3] if len(rows) > 1 else rows
                if len(rows) and rows[-1] + 1 < (strips[env.agents.index(a)][-1]):
                    rows = np.append(rows, strips[env.agents.index(a)][-1])
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
        if env.camera:
            out = self._confirm(env)
        for a in env.agents:
            if a in out:
                continue
            if env.camera and (m := _altitude_step(env.alt[a], self.SWEEP_ALT)) is not None:
                out[a] = m
                continue
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

    def _confirm(self, env) -> dict:
        """Actions for drones busy checking a candidate (camera mode)."""
        cand = env.candidates()
        for a in list(self.tasks):  # done when the cell was checked from low, or the drone is gone
            cx, cy = self.tasks[a]
            if a not in env.agents or cand[cy, cx] == 0:
                del self.tasks[a]
        claimed = set(self.tasks.values())
        open_c = [(int(x), int(y)) for y, x in zip(*np.nonzero(cand)) if (int(x), int(y)) not in claimed]
        for a in env.agents:
            if a in self.tasks or not open_c:
                continue
            x, y = (int(v) for v in env.pos[a])
            near = [c for c in open_c if max(abs(c[0] - x), abs(c[1] - y)) <= CONFIRM_RANGE]
            if near:
                c = min(near, key=lambda c: max(abs(c[0] - x), abs(c[1] - y)))
                self.tasks[a] = c
                open_c.remove(c)
        return _confirm_actions(env, self.tasks, self._field)


def _confirm_actions(env, tasks, field_of) -> dict:
    out = {}
    for a, (cx, cy) in tasks.items():
        if tuple(int(v) for v in env.pos[a]) == (cx, cy):
            out[a] = _altitude_step(env.alt[a], 0) or HOVER
        else:
            out[a] = _toward(env, a, field_of(env, (cx, cy)))
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
        self.cam_targets: dict[str, tuple[tuple[int, int], int, float]] = {}
        self.tasks: dict[str, tuple[int, int]] = {}
        self._fields: dict[tuple[int, int], np.ndarray] = {}

    def _field(self, env, target):
        if target not in self._fields:
            self._fields[target] = distance_fields(env.free, [target], env.scenario.corner_cutting)[0]
        return self._fields[target]

    def act(self, env: GridSearchEnv) -> dict:
        live = [a for a in env.agents if env.alive.get(a)]
        if not live:
            return {}
        if env.camera:
            return self._act_camera(env, live)
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

    def _act_camera(self, env, live) -> dict:
        """Camera mode: check candidates within CONFIRM_RANGE first (nearest,
        one drone each). Otherwise pick an (altitude, cell) pair by expected
        detections over the footprint there, per step of travel and altitude
        change; set the altitude first, then fly. Same commitment rule as above."""
        g, cc = env.grid_size, env.scenario.corner_cutting
        cand = env.candidates()
        for a in list(self.tasks):
            cx, cy = self.tasks[a]
            if a not in live or cand[cy, cx] == 0:
                del self.tasks[a]
        claimed = set(self.tasks.values())
        open_c = [(int(x), int(y)) for y, x in zip(*np.nonzero(cand)) if (int(x), int(y)) not in claimed]
        for a in live:
            if a in self.tasks or not open_c:
                continue
            D = self._field(env, tuple(int(v) for v in env.pos[a]))
            dist = [D[c[1], c[0]] for c in open_c]
            k = int(np.argmin(dist))
            if dist[k] <= CONFIRM_RANGE:
                self.tasks[a] = open_c.pop(k)
                self.cam_targets.pop(a, None)
        out = _confirm_actions(env, {a: c for a, c in self.tasks.items() if a in live}, self._field)

        V = [cam.footprint_sum(env.belief, env.P[k]) for k in range(cam.N_ALT)]
        for a in list(self.cam_targets):
            (tx, ty), k, v0 = self.cam_targets[a]
            arrived = tuple(int(v) for v in env.pos[a]) == (tx, ty) and env.alt[a] == k
            if a not in live or a in self.tasks or arrived or V[k][ty, tx] < self.drop * v0:
                del self.cam_targets[a]
        need = [a for a in live if a not in self.tasks and a not in self.cam_targets]
        if need:
            taken = ~env.free.copy()
            for (tx, ty), k, _ in self.cam_targets.values():
                rr = self.r + cam.FOOT_R[k]
                taken[max(0, ty - rr):ty + rr + 1, max(0, tx - rr):tx + rr + 1] = True
            for a in need:
                x, y = (int(v) for v in env.pos[a])
                D = self._field(env, (x, y))
                best = None
                for k in range(cam.N_ALT):
                    score = np.where(taken, -1.0, V[k] / (1.0 + D + abs(k - env.alt[a])))
                    if k == env.alt[a]:
                        score[y, x] = -1.0  # just searched it from here
                    ty, tx = np.unravel_index(int(np.argmax(score)), score.shape)
                    if best is None or score[ty, tx] > best[0]:
                        best = (score[ty, tx], (int(tx), int(ty)), k)
                _, (tx, ty), k = best
                self.cam_targets[a] = ((tx, ty), k, float(V[k][ty, tx]))
                rr = self.r + cam.FOOT_R[k]
                taken[max(0, ty - rr):ty + rr + 1, max(0, tx - rr):tx + rr + 1] = True
        for a in live:
            if a in out:
                continue
            (tx, ty), k, _ = self.cam_targets[a]
            m = _altitude_step(env.alt[a], k)
            out[a] = m if m is not None else _toward(env, a, self._field(env, (tx, ty)), V[k])
        return out
