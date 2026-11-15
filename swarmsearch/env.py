"""2D grid search environment (milestones M1 and M2).

A team of drones moves on a grid of 10 m cells. Each step a drone moves to one
of its 8 neighbours or hovers, and its camera searches the cell under it.
Survivors may move; the drones only know the operator's movement profile (or
the default), the prior heat map, and what they have searched themselves.

The team keeps a shared belief map: the expected share of survivor mass in
each cell that has not been detected yet. Searching a cell removes the
detected share; each step the belief is moved with the same movement model
the survivors follow. With a known profile that model is the truth. Without
one, the belief is a mixture over movement hypotheses that the team reweights
from what it finds (see belief.py), so it learns how survivors move during
the search. A drone's reward is the survivor mass its search detects, so it
pays to be where survivors are likely to be now, not where they were.

Implements the PettingZoo ParallelEnv API. `possible_agents` is the maximum
swarm size; each episode activates `scenario.n_drones` of them.
"""

from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import belief as _belief
from . import camera as cam
from . import realism as rl
from .belief import DIRECTED, move_weights, propagate
from .obstacles import distance_fields, make_obstacles
from .scenario import PROFILE_DIM, Scenario, ScenarioSampler

# Action 8 is hover; 0-7 are the 8 neighbours.
MOVES = np.array(
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 0)],
    dtype=np.int64,
)
HOVER = 8
N_ACTIONS = len(MOVES)
# Camera mode (M6) adds two actions: change altitude by one level, staying in the cell.
CLIMB, DESCEND = 9, 10
N_ACTIONS_CAMERA = 11
CAND_TTL = 40  # steps an unconfirmed detection stays on the map
INFO_WEIGHT_HIGH = 0.25  # reward weight for ground cleared from mid/high (not confirmable)

REWARD_DETECT = 1.0  # x detected belief mass, scaled so a fresh uniform cell pays ~detection_prob
REWARD_SURVIVOR = 10.0
PENALTY_COLLISION = 10.0
PENALTY_STEP = 0.01
# reward="find_time": a find pays FIND_TIME_REWARD * (1 - t/T), so the summed
# find reward is exactly N * (1 - mean find time / T), the evaluation metric.
# Ground cleared pays CLEAR_WEIGHT at every altitude, a small dense signal
# that no longer favours slow low passes.
FIND_TIME_REWARD = 20.0
CLEAR_WEIGHT = 0.25
FRESH_REPEAT = 0.9  # searching a cell this fresh counts as a repeat (metric only)
WEIGHT_FLOOR = 1e-4  # keeps every movement hypothesis recoverable

MAX_DRONES = 8
PATCH_RADIUS = 2
COARSE = 5
N_TEAMMATES = 2
# Full shared-map view (optional): freshness, belief, obstacles, self, teammates.
MAP_CHANNELS = 5
# Team layers (optional, M5): every teammate drawn on maps, so the input size
# does not depend on team size. Positions (coarse count; the local patch
# already shows them), trails (each cell a teammate passed, fading with age)
# and intent (the next INTENT_STEPS cells along each teammate's recent heading).
TRAIL_TAU = 7.0  # trail weight exp(-age / tau): ~0.24 after 10 steps, ~0.06 after 20
HEADING_WINDOW = 5  # heading = mean move over the last few steps
INTENT_STEPS = 5
TEAM_DIM = 2 * (2 * PATCH_RADIUS + 1) ** 2 + 3 * COARSE * COARSE

# Camera mode observation extras: own altitude (one-hot), light/sensor,
# local patches of cover, building height and candidate detections, coarse
# candidates, expected detections here at each altitude, nearest teammates' altitude.
CAMERA_DIM = cam.N_ALT + 3 + 3 * (2 * PATCH_RADIUS + 1) ** 2 + COARSE * COARSE + cam.N_ALT + N_TEAMMATES

# M7 real world extras: link to base, age of the map since the last sync,
# group size, the evidence ratios (finds and reports seen vs what the map
# expected: how the sensor really performs), wind, GPS error level.
REAL_DIM = 8

# Guidance input (optional): direction to the guide's target cell + "has target".
GUIDE_DIM = 3
GUIDE_GAMMA = 0.99  # discount used in the potential-based shaping bonus


def obs_dim(patch_radius: int = PATCH_RADIUS, coarse: int = COARSE) -> int:
    side = 2 * patch_radius + 1
    return 4 * side * side + 2 * coarse * coarse + 4 + 3 * N_TEAMMATES + PROFILE_DIM


def map_dim(grid_size: int) -> int:
    return MAP_CHANNELS * grid_size * grid_size


def state_dim(grid_size: int, camera: bool = False, real: bool = False) -> int:
    g2 = grid_size * grid_size
    return 5 * g2 + 4 + PROFILE_DIM + ((3 * g2 + 3) if camera else 0) + ((g2 + 6) if real else 0)


def n_actions(camera: bool = False) -> int:
    return N_ACTIONS_CAMERA if camera else N_ACTIONS


_W_CACHE: dict = {}


class View:
    """One copy of the team map. Without radio limits a single view is
    shared by every drone; with them (M7) each connected group has its own,
    and groups merge theirs when they meet again."""

    __slots__ = ("B", "w", "belief", "decay", "cand_t", "last_searched", "finds", "mates",
                 "exp_low", "found_low", "exp_high", "reports", "synced",
                 "out", "N", "map", "s_est", "outbox")  # field environment (M8)

    def __init__(self):
        self.finds: dict[int, tuple[int, int]] = {}  # survivor -> cell where this map saw it found
        self.mates: dict[str, tuple[np.ndarray, int]] = {}  # last known (cell, time) of each drone
        self.exp_low = self.found_low = self.exp_high = self.reports = 0.0
        self.synced = 0
        self.out = self.N = self.map = None
        self.s_est = 1.0
        self.outbox: list = []

    def copy(self) -> "View":
        v = View()
        for k in self.__slots__:
            x = getattr(self, k, None)
            setattr(v, k, x.copy() if hasattr(x, "copy") else x)
        v.outbox = []  # undelivered reports stay with the drone that made them
        return v


class GridSearchEnv(ParallelEnv):
    metadata = {"name": "grid_search_v0", "render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenario: Scenario | None = None,
        sampler: ScenarioSampler | None = None,
        max_drones: int = MAX_DRONES,
        seed: int | None = None,
        full_map: bool = False,
        guided: bool = False,
        shield: bool = True,
        team: bool = False,
    ):
        if scenario is None and sampler is None:
            scenario = Scenario()
        self.fixed_scenario = scenario
        self.sampler = sampler
        grid = scenario.grid_size if scenario else sampler.grid_size
        self.grid_size = grid
        self.possible_agents = [f"drone_{i}" for i in range(max_drones)]
        self.rng = np.random.default_rng(seed)
        self.full_map = full_map
        self.guided = guided
        self.team = team
        self.camera = bool((scenario or sampler).camera)
        # Safety shield: drones never move into a teammate's cell, and
        # simultaneous claims on one cell (or head-on swaps) are resolved
        # before anyone moves. Active in training and deployment alike.
        self.shield = shield
        self.guide_coef = 0.0  # shaping weight, set (and annealed) by the trainer
        self.reward = "count"  # "count" (+REWARD_SURVIVOR per find) or "find_time"; set by the trainer
        self.goals: dict[str, tuple[int, int]] = {}
        self.real = bool((scenario or sampler).real)
        if self.real and not self.camera:
            raise ValueError("M7 real-world effects need camera mode")
        self._obs_dim = (obs_dim() + (GUIDE_DIM if guided else 0) + (TEAM_DIM if team else 0)
                         + (CAMERA_DIM if self.camera else 0) + (REAL_DIM if self.real else 0)
                         + (map_dim(grid) if full_map else 0))
        self._state_dim = state_dim(grid, self.camera, self.real)
        self.agents: list[str] = []
        self._v = View()
        self.views = None

    # The map lives in the current view (see View); these keep the old names.
    _B = property(lambda s: s._v.B, lambda s, x: setattr(s._v, "B", x))
    _w = property(lambda s: s._v.w, lambda s, x: setattr(s._v, "w", x))
    belief = property(lambda s: s._v.belief, lambda s, x: setattr(s._v, "belief", x))
    _decay = property(lambda s: s._v.decay, lambda s, x: setattr(s._v, "decay", x))
    cand_t = property(lambda s: s._v.cand_t, lambda s, x: setattr(s._v, "cand_t", x))
    last_searched = property(lambda s: s._v.last_searched, lambda s, x: setattr(s._v, "last_searched", x))

    def use_view(self, a) -> "GridSearchEnv":
        """Make drone a's copy of the map current (no-op with perfect radio)."""
        if self.views is not None:
            self._v = self.views[a]
        return self

    def view_of(self, a) -> View:
        return self.views[a] if self.views is not None else self._v

    def _distinct_views(self) -> list:
        if self.views is None:
            return [self._v]
        out = {}
        for v in self.views.values():
            out[id(v)] = v
        return list(out.values())

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-1.0, 5.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(n_actions(self.camera))

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        sc = options.get("scenario") or self.fixed_scenario or self.sampler.sample(self.rng)
        if sc.grid_size != self.grid_size:
            raise ValueError("scenario grid_size must match the env")
        if sc.camera != self.camera or sc.real != self.real:
            raise ValueError("scenario camera / real mode must match the env")
        self.scenario = sc
        g = sc.grid_size
        rng = self.rng

        self.t = 0
        self._v = View()
        self.views = None
        self.map_version = 0
        self.agents = self.possible_agents[: sc.n_drones]
        self.starts = self._start_cells(sc.n_drones)

        self.obstacles = np.zeros((g, g), dtype=bool)
        if sc.obstacle_frac > 0:
            keep = list(self.starts)
            if sc.movement.kind == "to_safety":
                keep.append(tuple(sc.movement.target))
            self.obstacles = make_obstacles(rng, g, sc.obstacle_frac, keep, sc.obstacle_layout,
                                            corner_cut=sc.corner_cutting)
        self.free = ~self.obstacles
        self.n_free = int(self.free.sum())
        if sc.camera:
            self.heights = cam.make_heights(rng, self.obstacles)
            self.cover = cam.make_cover(rng, self.free, sc.cover_frac)
            self.P, self.FA = cam.tables(self.free, self.heights, self.cover, sc.detection_prob, sc.lighting)
            self.cand_t = np.full((g, g), -1e9)  # time of the latest unconfirmed detection per cell
        # True world vs the drones' map (they differ only with M7 effects on).
        self.true_free = self.free
        self.collapsed = np.zeros((g, g), dtype=bool)
        if sc.camera:
            self.true_heights, self.P_true, self.FA_true = self.heights, self.P, self.FA
        if sc.real:
            self._reset_real_map(sc)

        self.prior, zone_mask = self._make_prior(sc)
        self.survivors = self._place_survivors(sc, zone_mask)
        self.found = np.zeros(len(self.survivors), dtype=bool)
        self.found_times: list[int] = []
        self.trapped = np.zeros(len(self.survivors), dtype=bool)
        self.vis = np.ones(len(self.survivors))
        self.people = None
        if sc.real:
            self._reset_real_people(sc)

        # Belief = expected remaining survivor mass per cell, starting from
        # the prior. Searching a cell scales it down; movement spreads it.
        self._init_belief(sc)
        self.last_searched = np.full((g, g), -1e9)
        self.visited = np.zeros((g, g), dtype=bool)

        self.pos = {a: np.array(self.starts[i]) for i, a in enumerate(self.agents)}
        # Where each drone really is; `pos` is where it thinks it is (the same object without M7).
        self.true_pos = {a: p.copy() for a, p in self.pos.items()} if sc.real else self.pos
        self.alive = {a: True for a in self.agents}
        self.trail_t = {a: np.full((g, g), -1e9) for a in self.agents}
        self.history = {a: [self.pos[a].copy()] for a in self.agents}
        for a in self.agents:
            self.trail_t[a][self.pos[a][1], self.pos[a][0]] = 0
        self.battery = {a: 1.0 for a in self.agents}
        self.alt = {a: cam.START_ALT for a in self.agents}
        self.fail_step = (
            int(rng.integers(sc.max_steps // 4, 3 * sc.max_steps // 4)) if sc.drone_failure else -1
        )
        self.fail_agent = self.agents[int(rng.integers(len(self.agents)))] if sc.drone_failure else None

        self.n_visits = 0
        self.n_repeats = 0
        self.n_collisions = 0
        self.n_shield = 0
        self.n_alarms = 0  # candidates reported (camera mode)
        self.n_false_alarms = 0
        self.n_bumps = 0  # M7: moves stopped because the drone was not where it thought
        self.n_slips = 0
        self.n_gps_off = 0
        self.linked_steps = 0.0  # M7: sum over steps of the share of drones linked to base
        # Survivors move with the true model; the belief with the hypotheses.
        self._W_true_cum = np.cumsum(self._move_weights([sc.movement], DIRECTED)[0], axis=0)
        self.goals = {}
        self.group_of = {a: list(self.agents) for a in self.agents}
        if sc.real:
            self.base = (0, g // 2)
            self.views = {n: self._v for n in [*self.agents, "base"]}
            self._v.mates = {a: (self.pos[a].copy(), 0) for a in self.agents}
            self._v.synced = 0
            self.group_of["base"] = list(self.agents) + ["base"]
            for a in self.agents:
                self.group_of[a] = list(self.agents) + ["base"]

        for a in self.agents:
            self._search(a)
        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a)} for a in self.agents}
        return obs, infos

    def _start_cells(self, n: int) -> list[tuple[int, int]]:
        # Drones launch from a base along the left edge, spaced out.
        g = self.grid_size
        ys = np.linspace(0, g - 1, n + 2)[1:-1].round().astype(int)
        return [(0, int(y)) for y in ys]

    def _make_prior(self, sc: Scenario):
        g = sc.grid_size
        yy, xx = np.mgrid[0:g, 0:g]
        zone_mask = np.zeros((g, g), dtype=bool)
        if sc.n_zones == 0:
            heat = np.ones((g, g))
        else:
            heat = np.full((g, g), 0.05)
            for _ in range(sc.n_zones):
                cx, cy = self.rng.integers(2, g - 2, size=2)
                r = self.rng.uniform(1.5, 3.0)
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                heat += np.exp(-d2 / (2 * r * r))
                zone_mask |= d2 <= (1.5 * r) ** 2
        heat = heat * self.free
        heat /= heat.sum()
        return heat, zone_mask & self.free

    def _place_survivors(self, sc: Scenario, zone_mask) -> np.ndarray:
        free_idx = np.flatnonzero(self.free)
        zone_idx = np.flatnonzero(zone_mask)
        out = []
        for _ in range(sc.n_survivors):
            if len(zone_idx) and self.rng.random() < sc.zone_accuracy:
                c = self.rng.choice(zone_idx)
            else:
                c = self.rng.choice(free_idx)
            out.append((c % self.grid_size, c // self.grid_size))
        return np.array(out, dtype=np.int64).reshape(-1, 2)

    # ------------------------------------------------------------------- step
    def targets(self, actions, count: bool = False) -> dict:
        """Cell each live drone will move to this step: invalid moves (wall,
        obstacle, occupied cell) become hover, then the shield resolves
        conflicts. Pure unless `count` (then shield stops are counted)."""
        live = [a for a in self.agents if self.alive[a]]
        target = {}
        for a in live:
            act = int(actions.get(a, HOVER))
            if act >= N_ACTIONS or not self.action_mask(a)[act]:
                act = HOVER  # altitude changes stay in the cell
            target[a] = self.pos[a] + MOVES[act]
        if self.shield:
            n = self.n_shield
            self._resolve_conflicts(live, target)
            if not count:
                self.n_shield = n
        return target

    def step(self, actions, searching: dict | None = None):
        """`searching` (3D, M3): drones mapped to False did not physically
        reach their cell in time and do not search this step."""
        sc = self.scenario
        g = sc.grid_size
        self.t += 1
        rewards = {a: -PENALTY_STEP for a in self.agents}
        phi0 = {a: self._potential(a) for a in self.agents} if self.guide_coef else {}

        live = [a for a in self.agents if self.alive[a]]
        if self.camera:
            for a in live:
                act = int(actions.get(a, HOVER))
                if act in (CLIMB, DESCEND) and self.action_mask(a)[act]:
                    self.alt[a] += 1 if act == CLIMB else -1
        target = self.targets(actions, count=True)
        for a in live:
            self.battery[a] = max(0.0, 1.0 - self.t / sc.max_steps)
            if self.real:
                self._move_real(a, target[a])
            else:
                self.pos[a] = target[a]
            self.trail_t[a][self.pos[a][1], self.pos[a][0]] = self.t
            self.history[a] = (self.history[a] + [self.pos[a].copy()])[-(HEADING_WINDOW + 1):]

        # Potential-based shaping toward the guide's target (Ng et al. 1999):
        # it speeds learning without changing which policy is optimal.
        for a, p0 in phi0.items():
            if self.alive[a]:
                rewards[a] += self.guide_coef * (GUIDE_GAMMA * self._potential(a) - p0)

        # Drone-drone collisions: two live drones in one cell.
        cells: dict[tuple, list[str]] = {}
        for a in self.agents:
            if self.alive[a]:  # M7: where drones really are; different altitudes pass safely
                key = (*self.true_pos[a], self.alt[a]) if self.real else tuple(self.pos[a])
                cells.setdefault(key, []).append(a)
        for group in cells.values():
            if len(group) > 1:
                self.n_collisions += len(group)
                for a in group:
                    rewards[a] -= PENALTY_COLLISION

        if self.real:
            self._sync()
        # Search, then survivors move, then belief spreads the same way.
        found_now = 0
        for a in self.agents:
            if self.alive[a] and (searching is None or searching.get(a, True)):
                r, f = self._search(a)
                if self.reward == "find_time":
                    r *= CLEAR_WEIGHT / (1.0 if not self.camera or self.alt[a] == 0 else INFO_WEIGHT_HIGH)
                rewards[a] += r
                found_now += f
        per_find = (FIND_TIME_REWARD * (1.0 - self.t / sc.max_steps) if self.reward == "find_time"
                    else REWARD_SURVIVOR)
        for a in self.agents:
            rewards[a] += per_find * found_now
        self._move_survivors()
        self._spread_belief()

        terminations = {a: False for a in self.agents}
        if self.t == self.fail_step and self.fail_agent in self.alive:
            self.alive[self.fail_agent] = False
            terminations[self.fail_agent] = True
        all_found = bool(self.found.all())
        truncated = self.t >= sc.max_steps
        for a in self.agents:
            if all_found:
                terminations[a] = True
        truncations = {a: truncated and not terminations[a] for a in self.agents}

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a)} for a in self.agents}
        done_agents = [a for a in self.agents if terminations[a] or truncations[a]]
        if all_found or truncated:
            self.agents = []
        else:
            self.agents = [a for a in self.agents if a not in done_agents]
        return obs, rewards, terminations, truncations, infos

    def _resolve_conflicts(self, live: list[str], target: dict):
        """Shield: cancel moves (to hover) until no two drones share a cell
        and no two drones swap cells. A drone already in a cell keeps it;
        otherwise the lowest-numbered claimant does."""
        changed = True
        while changed:
            changed = False
            here = {tuple(self.pos[a]): a for a in live}
            for a in live:  # head-on swaps
                b = here.get(tuple(target[a]))
                if b is not None and b != a and tuple(target[b]) == tuple(self.pos[a]):
                    for d in (a, b):
                        if tuple(target[d]) != tuple(self.pos[d]):
                            target[d] = self.pos[d].copy()
                            self.n_shield += 1
                            changed = True
            claims: dict[tuple[int, int], list[str]] = {}
            for a in live:
                claims.setdefault(tuple(target[a]), []).append(a)
            for cell, group in claims.items():
                if len(group) < 2:
                    continue
                stay = [a for a in group if tuple(self.pos[a]) == cell]
                keep = stay[0] if stay else group[0]
                for a in group:
                    # Already hovering here is final: with GPS error (M7) two
                    # drones can believe they share a cell they do not.
                    if a != keep and tuple(target[a]) != tuple(self.pos[a]):
                        target[a] = self.pos[a].copy()
                        self.n_shield += 1
                        changed = True

    def set_goals(self, goals: dict[str, tuple[int, int]]):
        """Target cell (x, y) per drone from a guide; drones not listed keep theirs."""
        self.goals.update(goals)

    def _potential(self, a) -> float:
        if a not in self.goals:
            return 0.0
        gx, gy = self.goals[a]
        x, y = self.pos[a]
        return -max(abs(gx - x), abs(gy - y)) / self.grid_size

    def freshness(self) -> np.ndarray:
        """1 = just searched, decaying toward 0 as survivors could have moved in."""
        age = self.t - self.last_searched
        if self._decay == 0.0:
            return (age < 1e8).astype(np.float64)
        return np.exp(-age * self._decay)

    def _search(self, a) -> tuple[float, int]:
        if self.real:
            return self._search_real(a)
        if self.camera:
            return self._search_camera(a)
        x, y = self.pos[a]
        self.n_visits += 1
        if self.freshness()[y, x] >= FRESH_REPEAT:
            self.n_repeats += 1
        self.last_searched[y, x] = self.t
        self.visited[y, x] = True
        p = self.scenario.detection_prob

        found = 0
        here = (~self.found) & (self.survivors[:, 0] == x) & (self.survivors[:, 1] == y)
        for i in np.flatnonzero(here):
            if self.rng.random() < p:
                self.found[i] = True
                self.found_times.append(self.t)
                found += 1
        if found and len(self._w) > 1:
            # Each find reweights the hypotheses by the mass they had here.
            self._w = self._w * (self._B[:, y, x] + 1e-12) ** found
            self._w = (1 - WEIGHT_FLOOR) * self._w / self._w.sum() + WEIGHT_FLOOR * self._w0
            self._mix()

        detected = self.belief[y, x] * p
        self._B[:, y, x] *= 1.0 - p
        self.belief[y, x] -= detected
        reward = REWARD_DETECT * detected * self.n_free
        return reward, found

    def _find(self, i: int, x: int, y: int):
        self.found[i] = True
        self.found_times.append(self.t)
        if len(self._w) > 1:  # a find reweights the movement hypotheses by the mass they had here
            self._w = self._w * (self._B[:, y, x] + 1e-12)
            self._w = (1 - WEIGHT_FLOOR) * self._w / self._w.sum() + WEIGHT_FLOOR * self._w0
            self._mix()

    def _search_camera(self, a) -> tuple[float, int]:
        """Camera search (M6). Every footprint cell in sight is looked at with
        its own detection chance. From low altitude a detection is a confirmed
        find. From higher up it is only a candidate on the map, and so is a
        false alarm; the team cannot tell them apart until someone checks low.
        A cell with no report counts as an empty search (belief scaled down);
        a cell with a report is left as it was."""
        x, y = (int(v) for v in self.pos[a])
        alt = self.alt[a]
        P, FA = self.P[alt, y, x], self.FA[alt, y, x]
        g, r = self.grid_size, cam.R
        self.n_visits += 1
        if self.freshness()[y, x] >= FRESH_REPEAT and alt == 0:
            self.n_repeats += 1
        reward, found = 0.0, 0
        for oy, ox in zip(*np.nonzero(P)):
            cx, cy = x + ox - r, y + oy - r
            p = P[oy, ox]
            self.last_searched[cy, cx] = self.t
            self.visited[cy, cx] = True
            here = np.flatnonzero((~self.found) & (self.survivors[:, 0] == cx) & (self.survivors[:, 1] == cy))
            seen = [i for i in here if self.rng.random() < p]
            if alt == 0:
                self.cand_t[cy, cx] = -1e9  # checked from low: resolved either way
                for i in seen:
                    self._find(i, cx, cy)
                found += len(seen)
                report = False
            else:
                false = self.rng.random() < FA[oy, ox]
                report = bool(seen) or false
                if report:
                    self.cand_t[cy, cx] = self.t
                    self.n_alarms += 1
                    self.n_false_alarms += int(false and not seen)
            if not report:
                detected = self.belief[cy, cx] * p
                self._B[:, cy, cx] *= 1.0 - p
                self.belief[cy, cx] -= detected
                # Only a low pass can turn this into a find; clearing ground
                # from higher up is worth less, or drones learn to farm it
                # from high altitude and never come down to confirm.
                reward += REWARD_DETECT * detected * self.n_free * (1.0 if alt == 0 else INFO_WEIGHT_HIGH)
        return reward, found

    # ------------------------------------------------------- M7 real world
    def _reset_real_map(self, sc: Scenario):
        """Collapse some buildings in the true world; the drones' map
        (free, obstacles, heights, cover, P, FA) stays the pre-disaster one."""
        rng = self.rng
        self.collapsed = rl.collapse(rng, self.obstacles, sc.collapse_frac)
        self.true_free = self.free | self.collapsed
        self.true_heights = np.where(self.collapsed, 0.0, self.heights)
        true_cover = np.where(self.collapsed, cam.RUBBLE, self.cover).astype(np.int8)
        # True sensing: the real detector (sensor_scale) on the real ground.
        self.P_true, FA = cam.tables(self.true_free, self.true_heights, true_cover,
                                     sc.detection_prob * sc.sensor_scale, sc.lighting)
        self.FA_true = FA * sc.fa_scale
        self.unseen = self.collapsed.copy()
        self.block_id = np.full(self.collapsed.shape, -1)
        for k, cells in enumerate(rl.building_blocks(self.collapsed)):
            for y, x in cells:
                self.block_id[y, x] = k
        self.free = self.free.copy()  # the map now changes as collapses are found
        self.obstacles = self.obstacles.copy()
        self.heights = self.heights.copy()
        self.cover = self.cover.copy()
        self.P, self.FA = self.P.copy(), self.FA.copy()
        self.n_free0 = self.n_free
        d = rl.MOVES8[rng.integers(8)]
        self.wind_dir = np.array(d)

    def _reset_real_people(self, sc: Scenario):
        rng = self.rng
        g = self.grid_size
        if self.collapsed.any():  # some people are trapped in the collapsed buildings
            cells = np.flatnonzero(self.collapsed)
            for i in range(len(self.survivors)):
                if rng.random() < rl.TRAPPED_SHARE:
                    c = int(rng.choice(cells))
                    self.survivors[i] = (c % g, c // g)
                    self.trapped[i] = True
        if sc.vis_sd > 0:  # clothing, posture, partly under debris
            self.vis = np.clip(np.exp(rng.normal(0.0, sc.vis_sd, len(self.survivors))), 0.3, 2.0)
        if sc.behaviour == "mixed":
            self.people = rl.People(rng, sc, self.true_free, self.survivors, self.trapped, DIRECTED)
        # P6: the team has only an estimate of how many people are missing (log-sd 0.4).
        n = len(self.survivors)
        self.n_est = max(1, int(round(n * np.exp(rng.normal(0.0, 0.4)))))

    def _move_real(self, a, target):
        """The drone flies the move it chose from where it thinks it is; wind
        may push it a cell further; then it gets a new (maybe wrong) GPS fix."""
        sc, g, rng = self.scenario, self.grid_size, self.rng
        # Onboard detect-and-avoid: a drone senses a teammate really in the
        # next cell at its own altitude and holds (the shield only knows
        # where drones think they are).
        near = {tuple(self.true_pos[b]) for b in self.agents
                if b != a and self.alive[b] and self.alt[b] == self.alt[a]}

        def ok(c):
            return 0 <= c[0] < g and 0 <= c[1] < g and self.true_free[c[1], c[0]] and (c[0], c[1]) not in near

        tp = self.true_pos[a]
        nxt = tp + (target - self.pos[a])
        if not ok(nxt):
            self.n_bumps += 1  # obstacle or drone avoidance stops it
            nxt = tp.copy()
        if sc.wind_mps > 0 and rng.random() < rl.SLIP_PER_MPS * sc.wind_mps:
            d = self.wind_dir if rng.random() >= rl.GUST else np.array(rl.MOVES8[rng.integers(8)])
            if ok(nxt + d):
                nxt = nxt + d
                self.n_slips += 1
        self.true_pos[a] = nxt
        est = nxt.copy()
        if sc.gps_err > 0:
            x, y = nxt
            near = self.true_heights[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
            p = sc.gps_err * (rl.URBAN_GPS if near > cam.ALT_M[self.alt[a]] else 1.0)
            if rng.random() < p:
                e = nxt + np.array(rl.MOVES8[rng.integers(8)])
                if 0 <= e[0] < g and 0 <= e[1] < g and self.free[e[1], e[0]]:
                    est = e
                    self.n_gps_off += 1
        self.pos[a] = est

    def _sync(self):
        """Radio: connected drones (and the base) share one map; a group that
        was split merges its copies; each drone learns its group's positions."""
        sc = self.scenario
        live = [a for a in self.agents if self.alive[a]]
        nodes = {a: (tuple(int(v) for v in self.true_pos[a]), cam.ALT_M[self.alt[a]]) for a in live}
        nodes["base"] = (self.base, rl.BASE_ALT_M)
        used, new = set(), {}
        for grp in rl.groups(nodes, self.true_heights, self.rng, sc.comm_range, sc.comm_loss):
            objs = list({id(self.views[n]): self.views[n] for n in grp}.values())
            if len(objs) == 1 and id(objs[0]) not in used:
                v = objs[0]
            else:
                v = self._merge(objs)
            used.add(id(v))
            for n in grp:
                new[n] = v
                self.group_of[n] = grp
                if n != "base":
                    v.mates[n] = (self.pos[n].copy(), self.t)
            v.synced = self.t
            if "base" in grp:
                self.linked_steps += (len(grp) - 1) / max(1, len(live))
        self.views.update(new)

    def _merge(self, objs: list) -> View:
        """Combine copies of the map. A search only ever lowers the map, so
        the lower value per cell is the better-informed one; finds one copy
        has not seen reweight its movement hypotheses."""
        v = objs[0].copy()
        for o in objs[1:]:
            np.minimum(v.B, o.B, out=v.B)
            np.maximum(v.cand_t, o.cand_t, out=v.cand_t)
            np.maximum(v.last_searched, o.last_searched, out=v.last_searched)
            for i, (x, y) in o.finds.items():
                if i not in v.finds:
                    v.finds[i] = (x, y)
                    if len(v.w) > 1:
                        v.w = v.w * (v.B[:, y, x] + 1e-12)
                        v.w = (1 - WEIGHT_FLOOR) * v.w / v.w.sum() + WEIGHT_FLOOR * self._w0
            for b, (p, t) in o.mates.items():
                if b not in v.mates or v.mates[b][1] < t:
                    v.mates[b] = (p, t)
            for k in ("exp_low", "found_low", "exp_high", "reports"):
                setattr(v, k, max(getattr(v, k), getattr(o, k)))
        self._v = v
        self._mix()
        return v

    def _reveal(self, cells: np.ndarray):
        """A drone saw collapsed rubble: the whole building's footprint becomes
        searchable ground on everyone's map, with extra prior weight (SAR
        teams search collapses first)."""
        sc = self.scenario
        mask = np.isin(self.block_id, np.unique(self.block_id[cells]))
        mask &= self.unseen
        if not mask.any():
            return
        self.unseen[mask] = False
        self.free[mask] = True
        self.obstacles[mask] = False
        self.heights[mask] = 0.0
        self.cover[mask] = cam.RUBBLE
        self.n_free = int(self.free.sum())
        m0 = rl.COLLAPSE_PRIOR / self.n_free0
        cur = self._v
        for v in self._distinct_views():
            v.B[:, mask] = m0
            self._v = v
            self._mix()
        self._v = cur
        cam.update_tables(self.P, self.FA, self.free, self.heights, self.cover, sc.detection_prob,
                          sc.lighting, mask)
        self.map_version += 1

    def _search_real(self, a) -> tuple[float, int]:
        """Camera search with M7 effects. Sensing happens where the drone
        really is, with the real detector and ground; the map is updated where
        the drone thinks it is, with the modelled detector."""
        self.use_view(a)
        v = self._v
        g, r = self.grid_size, cam.R
        tx, ty = (int(c) for c in self.true_pos[a])
        ex, ey = (int(c) for c in self.pos[a])
        alt = self.alt[a]
        Pt, FAt = self.P_true[alt, ty, tx], self.FA_true[alt, ty, tx]
        self.n_visits += 1
        if self.freshness()[ey, ex] >= FRESH_REPEAT and alt == 0:
            self.n_repeats += 1
        # 1. What the camera really sees.
        seen_cells = np.argwhere(Pt > 0)
        cy_t, cx_t = seen_cells[:, 0] + ty - r, seen_cells[:, 1] + tx - r
        if self.unseen[cy_t, cx_t].any():
            hit = self.unseen[cy_t, cx_t]
            self._reveal((cy_t[hit], cx_t[hit]))
        found, reports = 0, set()
        for (oy, ox), cx, cy in zip(seen_cells, cx_t, cy_t):
            self.visited[cy, cx] = True
            here = np.flatnonzero((~self.found) & (self.survivors[:, 0] == cx) & (self.survivors[:, 1] == cy))
            seen = [i for i in here if self.rng.random() < min(0.98, Pt[oy, ox] * self.vis[i])]
            if alt == 0:
                for i in seen:
                    self._find_real(i, ex, ey)
                found += len(seen)
            else:
                false = self.rng.random() < FAt[oy, ox]
                if seen or false:
                    # reported where the drone thinks the camera was looking
                    rx, ry = min(g - 1, max(0, cx + ex - tx)), min(g - 1, max(0, cy + ey - ty))
                    reports.add((rx, ry))
                    self.n_alarms += 1
                    self.n_false_alarms += int(false and not seen)
        v.reports += len(reports)
        # 2. What the map does with it (modelled detector, believed position).
        Pm = self.P[alt, ey, ex]
        reward = 0.0
        n_est = self.n_est  # the team's estimate, not the true count (P6)
        for oy, ox in zip(*np.nonzero(Pm)):
            mx, my = ex + ox - r, ey + oy - r
            p = Pm[oy, ox]
            self.last_searched[my, mx] = self.t
            if alt == 0:
                self.cand_t[my, mx] = -1e9
            if (mx, my) in reports:
                self.cand_t[my, mx] = self.t
                continue
            detected = self.belief[my, mx] * p
            self._B[:, my, mx] *= 1.0 - p
            self.belief[my, mx] -= detected
            if alt == 0:
                v.exp_low += detected * n_est
            else:
                v.exp_high += detected * n_est
            reward += REWARD_DETECT * detected * self.n_free * (1.0 if alt == 0 else INFO_WEIGHT_HIGH)
        if alt > 0:  # expected false alarms count as expected reports
            v.exp_high += float(self.FA[alt, ey, ex].sum())
        return reward, found

    def _find_real(self, i: int, ex: int, ey: int):
        self.found[i] = True
        self.found_times.append(self.t)
        v = self._v
        v.finds[i] = (ex, ey)
        v.found_low += 1
        if len(self._w) > 1:
            self._w = self._w * (self._B[:, ey, ex] + 1e-12)
            self._w = (1 - WEIGHT_FLOOR) * self._w / self._w.sum() + WEIGHT_FLOOR * self._w0
            self._mix()

    def _real_obs(self, a) -> np.ndarray:
        sc = self.scenario
        v = self._v
        grp = self.group_of.get(a, [])
        return np.array([
            float("base" in grp),
            min(1.0, (self.t - v.synced) / 20.0),
            (len(grp) - ("base" in grp)) / MAX_DRONES,
            np.clip((v.found_low + 1.0) / (v.exp_low + 1.0), 0.0, 3.0) / 3.0,  # finds vs expected
            np.clip((v.reports + 1.0) / (v.exp_high + 1.0), 0.0, 5.0) / 5.0,  # reports vs expected
            v.exp_low / max(1, self.n_est),  # how much evidence there is
            sc.wind_mps / 10.0,  # weather is known
            sc.gps_err * 10.0,  # so is the GPS fix quality
        ], dtype=np.float32)

    def candidates(self) -> np.ndarray:
        """(g, g) unconfirmed detections, 1 when new, fading to 0 after CAND_TTL steps."""
        return np.clip(1.0 - (self.t - self.cand_t) / CAND_TTL, 0.0, 1.0)

    def _init_belief(self, sc: Scenario):
        if sc.profile_known:
            profiles, prior = [sc.movement], np.ones(1)
        else:
            profiles, prior = _belief.hypothesis_bank(self.grid_size, self.free)
        key = ((self.grid_size, tuple(profiles), sc.profile_known, _belief.HYP_DIRECTED)
               if not self.obstacles.any() else None)
        W = _W_CACHE.get(key) if key else None
        if W is None:
            directed = DIRECTED if sc.profile_known else _belief.HYP_DIRECTED
            W = self._move_weights(profiles, directed)
            if key:
                _W_CACHE[key] = W
        self._W = W
        self._w0 = prior
        self._w = prior.copy()
        self._B = np.repeat(self.prior[None], len(profiles), axis=0)
        self._V = np.stack([pr.as_vector(False, self.grid_size) for pr in profiles])
        self._V_moveprob = np.array([pr.move_prob for pr in profiles])
        self._mix()

    def _move_weights(self, profiles, directed) -> np.ndarray:
        """(K, 9, g, g) move probabilities; with obstacles, survivors heading
        to safety follow shortest paths around them."""
        dists = {}
        if self.obstacles.any():
            targets = sorted({pr.target for pr in profiles if pr.kind == "to_safety"})
            if targets:
                dists = dict(zip(targets, distance_fields(self.free, targets)))
        return np.stack([move_weights(pr, self.free, directed, dists.get(pr.target) if pr.kind == "to_safety" else None)
                         for pr in profiles])

    def _mix(self):
        """Shared belief = hypothesis maps weighted by the find-based weights."""
        self.belief = np.tensordot(self._w, self._B, axes=1)
        post = self.profile_posterior()
        self._decay = float(post @ self._V_moveprob)

    def profile_posterior(self) -> np.ndarray:
        """P(hypothesis | finds and empty searches so far)."""
        q = self._w * self._B.sum(axis=(1, 2))
        return q / max(q.sum(), 1e-300)

    def _move_survivors(self):
        if self.people is not None:
            self.people.step(self.rng, self.survivors, self.found, MOVES)
            return
        if self.scenario.movement.move_prob == 0.0:
            return
        idx = np.flatnonzero(~self.found & ~self.trapped)
        if not len(idx):
            return
        xs, ys = self.survivors[idx, 0], self.survivors[idx, 1]
        cum = self._W_true_cum[:, ys, xs]  # (9, n)
        m = np.minimum((cum < self.rng.random(len(idx))).sum(axis=0), 8)
        self.survivors[idx] = self.survivors[idx] + MOVES[m]

    def _spread_belief(self):
        for v in self._distinct_views():
            self._v = v
            self._B = propagate(self._B, self._W)
            self._mix()

    def action_mask(self, a) -> np.ndarray:
        g = self.grid_size
        nxt = self.pos[a] + MOVES
        ok = (nxt[:, 0] >= 0) & (nxt[:, 0] < g) & (nxt[:, 1] >= 0) & (nxt[:, 1] < g)
        ok[ok] &= self.free[nxt[ok, 1], nxt[ok, 0]]
        if not self.scenario.corner_cutting:  # M4: don't cut a building's corner diagonally
            x, y = self.pos[a]
            for m in range(8):
                dx, dy = MOVES[m]
                if ok[m] and dx and dy:
                    ok[m] = self.free[y, x + dx] and self.free[y + dy, x]
        if self.shield:
            for b in self.agents:
                if b != a and self.alive.get(b):
                    hit = (nxt[:, 0] == self.pos[b][0]) & (nxt[:, 1] == self.pos[b][1])
                    ok &= ~hit
        ok[HOVER] = True
        if self.camera:
            ok = np.concatenate([ok, [self.alt[a] < cam.N_ALT - 1, self.alt[a] > 0]])
        return ok.astype(np.int8)

    def _patch(self, grid: np.ndarray, x: int, y: int, fill: float) -> np.ndarray:
        r = PATCH_RADIUS
        padded = np.pad(grid, r, constant_values=fill)
        return padded[y : y + 2 * r + 1, x : x + 2 * r + 1].ravel()

    def _coarse(self, grid: np.ndarray, reduce) -> np.ndarray:
        out = []
        for rows in np.array_split(grid, COARSE, axis=0):
            for block in np.array_split(rows, COARSE, axis=1):
                out.append(reduce(block))
        return np.array(out)

    def _belief_density(self) -> np.ndarray:
        # Scale so that a uniform prior reads 1.0 per free cell.
        return np.clip(self.belief * self.n_free, 0.0, 5.0)

    def team_maps(self, a) -> tuple[np.ndarray, np.ndarray]:
        """Trail and intent maps of a's live teammates (values in [0, 1])."""
        g = self.grid_size
        trail = np.zeros((g, g))
        intent = np.zeros((g, g))
        for b in self.possible_agents[: self.scenario.n_drones]:
            if b == a or not self.alive.get(b):
                continue
            np.maximum(trail, np.exp(-(self.t - self.trail_t[b]) / TRAIL_TAU), out=trail)
            h = self.history[b]
            if len(h) < 2:
                continue
            v = (h[-1] - h[0]) / (len(h) - 1)
            if np.hypot(*v) < 0.2:  # hovering: no heading
                continue
            for s in range(1, INTENT_STEPS + 1):
                cx, cy = (int(round(c)) for c in h[-1] + v * s)
                if not (0 <= cx < g and 0 <= cy < g) or self.obstacles[cy, cx]:
                    break
                intent[cy, cx] = max(intent[cy, cx], 1.0 - (s - 1) / INTENT_STEPS)
        return trail, intent

    def _obs(self, a, full_map: bool | None = None, guided: bool | None = None,
             team: bool | None = None) -> np.ndarray:
        """Local observation. With full_map, the whole shared team map is
        appended as MAP_CHANNELS flattened g x g channels."""
        sc = self.scenario
        g = self.grid_size
        self.use_view(a)
        x, y = (int(v) for v in self.pos[a])
        fresh = self.freshness()
        dens = self._belief_density()
        mates = np.zeros((g, g))
        others = [b for b in self.possible_agents[: sc.n_drones] if b != a and self.alive.get(b)]
        mpos = self._mate_positions(a, others)
        for b in others:
            bx, by = mpos[b]
            mates[by, bx] = 1.0

        parts = [
            self._patch(fresh, x, y, 1.0),
            self._patch(dens, x, y, 0.0),
            self._patch(self.obstacles.astype(float), x, y, 1.0),
            self._patch(mates, x, y, 0.0),
            self._coarse(fresh, np.mean),
            self._coarse(dens, np.mean),
            np.array([x / (g - 1), y / (g - 1), self.battery[a], self.t / sc.max_steps]),
        ]
        rel = sorted(
            (mpos[b] - self.pos[a] for b in others), key=lambda d: abs(d[0]) + abs(d[1])
        )[:N_TEAMMATES]
        mate_feats = np.zeros(3 * N_TEAMMATES)
        for i, d in enumerate(rel):
            mate_feats[3 * i : 3 * i + 3] = (d[0] / g, d[1] / g, 1.0)
        parts.append(mate_feats)
        if sc.profile_known:
            parts.append(sc.movement.as_vector(True, g))
        else:  # expected profile under the current posterior over hypotheses
            parts.append(self.profile_posterior() @ self._V)
        if self.guided if guided is None else guided:
            if a in self.goals:
                gx, gy = self.goals[a]
                parts.append(np.array([(gx - x) / g, (gy - y) / g, 1.0]))
            else:
                parts.append(np.zeros(GUIDE_DIM))
        if self.camera:
            cand = self.candidates()
            onehot = np.zeros(cam.N_ALT)
            onehot[self.alt[a]] = 1.0
            here = [cam.footprint_sum(self.belief, self.P[k])[y, x] * self.n_free for k in range(cam.N_ALT)]
            mate_alt = np.zeros(N_TEAMMATES)
            near = sorted(others, key=lambda b: abs(mpos[b][0] - x) + abs(mpos[b][1] - y))[:N_TEAMMATES]
            for i, b in enumerate(near):
                mate_alt[i] = (self.alt[b] + 1) / cam.N_ALT
            cover_det = np.choose(self.cover, [1.0, cam.RUBBLE_DETECT, cam.LIGHTING[sc.lighting][2]])
            parts += [onehot, cam.light_vector(sc.lighting),
                      self._patch(cover_det, x, y, 1.0), self._patch(self.heights / cam.BUILDING_M[1], x, y, 0.0),
                      self._patch(cand, x, y, 0.0), np.clip(self._coarse(cand, np.sum), 0.0, 5.0),
                      np.clip(here, 0.0, 5.0), mate_alt]
        if self.real:
            parts.append(self._real_obs(a))
        if self.team if team is None else team:
            trail, intent = self.team_maps(a)
            parts += [self._patch(trail, x, y, 0.0), self._patch(intent, x, y, 0.0),
                      np.clip(self._coarse(mates, np.sum), 0.0, 5.0),
                      self._coarse(trail, np.mean), self._coarse(intent, np.mean)]
        if self.full_map if full_map is None else full_map:
            me = np.zeros((g, g))
            me[y, x] = 1.0
            parts += [fresh.ravel(), (dens / 5.0).ravel(), self.obstacles.ravel().astype(float),
                      me.ravel(), mates.ravel()]
        return np.concatenate(parts).astype(np.float32)

    def _mate_positions(self, a, others) -> dict:
        """Where drone a thinks its teammates are: their reported cells, or
        with limited radio the last cell its group heard (M7)."""
        if self.views is None:
            return {b: self.pos[b] for b in others}
        known = self._v.mates
        return {b: known[b][0] if b in known else self.pos[b] for b in others}

    def state(self) -> np.ndarray:
        """Global state for the centralized critic (training only)."""
        sc = self.scenario
        if self.views is not None:  # the team map as the base station holds it
            self._v = self.views["base"]
        g = self.grid_size
        drones = np.zeros((g, g))
        for a, alive in self.alive.items():
            if alive:
                x, y = self.true_pos[a]
                drones[y, x] += 1.0
        unfound = np.zeros((g, g))
        for (x, y), f in zip(self.survivors, self.found):
            if not f:
                unfound[y, x] += 1.0
        found_frac = self.found.mean() if len(self.found) else 1.0
        extra = []
        if self.camera:
            alt = np.zeros((g, g))
            for a, alive in self.alive.items():
                if alive:
                    alt[self.pos[a][1], self.pos[a][0]] = (self.alt[a] + 1) / cam.N_ALT
            cover_det = np.choose(self.cover, [1.0, cam.RUBBLE_DETECT, cam.LIGHTING[sc.lighting][2]])
            extra = [cover_det.ravel(), self.candidates().ravel(), alt.ravel(), cam.light_vector(sc.lighting)]
        if self.real:  # the critic may know the truth (training only)
            extra += [self.unseen.ravel().astype(float),
                      [sc.sensor_scale, sc.fa_scale / 3.0, sc.wind_mps / 10.0, sc.gps_err * 10.0,
                       sc.comm_range / 30.0, sc.comm_loss * 5.0]]
        return np.concatenate(
            [
                self.freshness().ravel(),
                self._belief_density().ravel(),
                self.obstacles.ravel().astype(float),
                drones.ravel(),
                unfound.ravel(),
                [self.t / sc.max_steps, found_frac, sc.n_drones / MAX_DRONES, len(self.found) / 20],
                sc.movement.as_vector(True, g),
                *extra,
            ]
        ).astype(np.float32)

    # ---------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        n = len(self.found)
        need = int(np.ceil(0.9 * n)) if n else 0
        t90 = self.found_times[need - 1] if need and len(self.found_times) >= need else None
        # Mean time to find a survivor; one never found counts as the whole budget.
        budget = self.scenario.max_steps
        find_time = (sum(self.found_times) + (n - len(self.found_times)) * budget) / n if n else 0.0
        return {
            "survivors": n,
            "found_frac": float(self.found.mean()) if n else 1.0,
            "time_to_90": t90,
            "find_time": float(find_time),
            "find_time_frac": float(find_time / budget),
            "coverage": float(self.visited[self.free].mean()),
            "repeat_frac": self.n_repeats / max(1, self.n_visits),
            "collisions": self.n_collisions,
            "shield": self.n_shield,
            "steps": self.t,
            "alarms": self.n_alarms,
            "false_alarms": self.n_false_alarms,
            "bumps": self.n_bumps,
            "linked": self.linked_steps / max(1, self.t),
            "revealed": float(1.0 - self.unseen.sum() / max(1, self.collapsed.sum())) if self.real else 1.0,
        }
