"""2D grid search environment (milestones M1 and M2).

A team of drones moves on a grid of 10 m cells. Each step a drone moves to one
of its 8 neighbours or hovers, and its camera searches the cell under it.
Survivors may move; the drones only know the operator's movement profile (or
the default), the prior heat map, and what they have searched themselves.

Implements the PettingZoo ParallelEnv API. `possible_agents` is the maximum
swarm size; each episode activates `scenario.n_drones` of them.
"""

from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .scenario import PROFILE_DIM, Scenario, ScenarioSampler

# Action 8 is hover; 0-7 are the 8 neighbours.
MOVES = np.array(
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 0)],
    dtype=np.int64,
)
HOVER = 8
N_ACTIONS = len(MOVES)

REWARD_NEW_CELL = 1.0
REWARD_SURVIVOR = 10.0
PENALTY_REPEAT = 0.5
PENALTY_COLLISION = 10.0
PENALTY_STEP = 0.01
FRESH_REPEAT = 0.9  # searching a cell this fresh counts as a repeat

MAX_DRONES = 8
PATCH_RADIUS = 2
COARSE = 5
N_TEAMMATES = 2
# Full shared-map view (optional): freshness, belief, obstacles, self, teammates.
MAP_CHANNELS = 5
# Guidance input (optional): direction to the guide's target cell + "has target".
GUIDE_DIM = 3
GUIDE_GAMMA = 0.99  # discount used in the potential-based shaping bonus


def obs_dim(patch_radius: int = PATCH_RADIUS, coarse: int = COARSE) -> int:
    side = 2 * patch_radius + 1
    return 4 * side * side + 2 * coarse * coarse + 4 + 3 * N_TEAMMATES + PROFILE_DIM


def map_dim(grid_size: int) -> int:
    return MAP_CHANNELS * grid_size * grid_size


def state_dim(grid_size: int) -> int:
    return 5 * grid_size * grid_size + 4 + PROFILE_DIM


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
        self.guide_coef = 0.0  # shaping weight, set (and annealed) by the trainer
        self.goals: dict[str, tuple[int, int]] = {}
        self._obs_dim = obs_dim() + (GUIDE_DIM if guided else 0) + (map_dim(grid) if full_map else 0)
        self._state_dim = state_dim(grid)
        self.agents: list[str] = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-1.0, 5.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        sc = options.get("scenario") or self.fixed_scenario or self.sampler.sample(self.rng)
        if sc.grid_size != self.grid_size:
            raise ValueError("scenario grid_size must match the env")
        self.scenario = sc
        g = sc.grid_size
        rng = self.rng

        self.t = 0
        self.agents = self.possible_agents[: sc.n_drones]
        self.starts = self._start_cells(sc.n_drones)

        self.obstacles = np.zeros((g, g), dtype=bool)
        if sc.obstacle_frac > 0:
            n_obs = int(sc.obstacle_frac * g * g)
            free = np.ones(g * g, dtype=bool)
            for x, y in self.starts:
                free[y * g + x] = False
            idx = rng.choice(np.flatnonzero(free), size=n_obs, replace=False)
            self.obstacles.flat[idx] = True
        self.free = ~self.obstacles
        self.n_free = int(self.free.sum())

        self.prior, zone_mask = self._make_prior(sc)
        self.survivors = self._place_survivors(sc, zone_mask)
        self.found = np.zeros(len(self.survivors), dtype=bool)
        self.found_times: list[int] = []

        # Belief = expected remaining survivor mass per cell, starting from
        # the prior. Searching a cell scales it down; movement spreads it.
        self.belief = self.prior.copy()
        self.last_searched = np.full((g, g), -1e9)
        self.visited = np.zeros((g, g), dtype=bool)

        self.pos = {a: np.array(self.starts[i]) for i, a in enumerate(self.agents)}
        self.alive = {a: True for a in self.agents}
        self.battery = {a: 1.0 for a in self.agents}
        self.fail_step = (
            int(rng.integers(sc.max_steps // 4, 3 * sc.max_steps // 4)) if sc.drone_failure else -1
        )
        self.fail_agent = self.agents[int(rng.integers(len(self.agents)))] if sc.drone_failure else None

        self.n_visits = 0
        self.n_repeats = 0
        self.n_collisions = 0
        self._decay = sc.observed_profile().move_prob
        self.goals = {}

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
    def step(self, actions):
        sc = self.scenario
        g = sc.grid_size
        self.t += 1
        rewards = {a: -PENALTY_STEP for a in self.agents}
        phi0 = {a: self._potential(a) for a in self.agents} if self.guide_coef else {}

        # Move drones; invalid moves (wall, obstacle) become hover.
        for a in self.agents:
            if not self.alive[a]:
                continue
            act = int(actions.get(a, HOVER))
            if not self.action_mask(a)[act]:
                act = HOVER
            self.pos[a] = self.pos[a] + MOVES[act]
            self.battery[a] = max(0.0, 1.0 - self.t / sc.max_steps)

        # Potential-based shaping toward the guide's target (Ng et al. 1999):
        # it speeds learning without changing which policy is optimal.
        for a, p0 in phi0.items():
            if self.alive[a]:
                rewards[a] += self.guide_coef * (GUIDE_GAMMA * self._potential(a) - p0)

        # Drone-drone collisions: two live drones in one cell.
        cells: dict[tuple[int, int], list[str]] = {}
        for a in self.agents:
            if self.alive[a]:
                cells.setdefault(tuple(self.pos[a]), []).append(a)
        for group in cells.values():
            if len(group) > 1:
                self.n_collisions += len(group)
                for a in group:
                    rewards[a] -= PENALTY_COLLISION

        # Search, then survivors move, then belief spreads the same way.
        found_now = 0
        for a in self.agents:
            if self.alive[a]:
                r, f = self._search(a)
                rewards[a] += r
                found_now += f
        for a in self.agents:
            rewards[a] += REWARD_SURVIVOR * found_now
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
        x, y = self.pos[a]
        fresh = self.freshness()[y, x]
        reward = REWARD_NEW_CELL * (1.0 - fresh)
        self.n_visits += 1
        if fresh >= FRESH_REPEAT:
            reward -= PENALTY_REPEAT
            self.n_repeats += 1
        self.last_searched[y, x] = self.t
        self.visited[y, x] = True
        self.belief[y, x] *= 1.0 - self.scenario.detection_prob

        found = 0
        here = (~self.found) & (self.survivors[:, 0] == x) & (self.survivors[:, 1] == y)
        for i in np.flatnonzero(here):
            if self.rng.random() < self.scenario.detection_prob:
                self.found[i] = True
                self.found_times.append(self.t)
                found += 1
        return reward, found

    def _move_survivors(self):
        mv = self.scenario.movement
        p = mv.move_prob
        if p == 0.0:
            return
        g = self.grid_size
        for i in np.flatnonzero(~self.found):
            if self.rng.random() >= p:
                continue
            x, y = self.survivors[i]
            if mv.kind == "drift" and self.rng.random() < 0.8:
                dx, dy = mv.drift_dir
            else:
                dx, dy = MOVES[self.rng.integers(8)]
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and self.free[ny, nx]:
                self.survivors[i] = (nx, ny)

    def _spread_belief(self):
        prof = self.scenario.observed_profile()
        p = prof.move_prob
        if p == 0.0:
            return
        b = self.belief
        if prof.kind == "drift":
            dx, dy = prof.drift_dir
            moved = np.roll(b, shift=(dy, dx), axis=(0, 1))
            # Mass that drifts off the map is lost, not wrapped around.
            if dx > 0:
                moved[:, :dx] = 0
            elif dx < 0:
                moved[:, dx:] = 0
            if dy > 0:
                moved[:dy, :] = 0
            elif dy < 0:
                moved[dy:, :] = 0
        else:
            padded = np.pad(b, 1, mode="edge")
            g = self.grid_size
            moved = sum(
                padded[1 + dy : 1 + dy + g, 1 + dx : 1 + dx + g] for dx, dy in MOVES[:8]
            ) / 8.0
        self.belief = ((1 - p) * b + p * moved) * self.free

    # ------------------------------------------------------------ observation
    def action_mask(self, a) -> np.ndarray:
        g = self.grid_size
        nxt = self.pos[a] + MOVES
        ok = (nxt[:, 0] >= 0) & (nxt[:, 0] < g) & (nxt[:, 1] >= 0) & (nxt[:, 1] < g)
        ok[ok] &= self.free[nxt[ok, 1], nxt[ok, 0]]
        ok[HOVER] = True
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

    def _obs(self, a, full_map: bool | None = None, guided: bool | None = None) -> np.ndarray:
        """Local observation. With full_map, the whole shared team map is
        appended as MAP_CHANNELS flattened g x g channels."""
        sc = self.scenario
        g = self.grid_size
        x, y = (int(v) for v in self.pos[a])
        fresh = self.freshness()
        dens = self._belief_density()
        mates = np.zeros((g, g))
        others = [b for b in self.possible_agents[: sc.n_drones] if b != a and self.alive.get(b)]
        for b in others:
            bx, by = self.pos[b]
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
            (self.pos[b] - self.pos[a] for b in others), key=lambda d: abs(d[0]) + abs(d[1])
        )[:N_TEAMMATES]
        mate_feats = np.zeros(3 * N_TEAMMATES)
        for i, d in enumerate(rel):
            mate_feats[3 * i : 3 * i + 3] = (d[0] / g, d[1] / g, 1.0)
        parts.append(mate_feats)
        parts.append(sc.observed_profile().as_vector(sc.profile_known))
        if self.guided if guided is None else guided:
            if a in self.goals:
                gx, gy = self.goals[a]
                parts.append(np.array([(gx - x) / g, (gy - y) / g, 1.0]))
            else:
                parts.append(np.zeros(GUIDE_DIM))
        if self.full_map if full_map is None else full_map:
            me = np.zeros((g, g))
            me[y, x] = 1.0
            parts += [fresh.ravel(), (dens / 5.0).ravel(), self.obstacles.ravel().astype(float),
                      me.ravel(), mates.ravel()]
        return np.concatenate(parts).astype(np.float32)

    def state(self) -> np.ndarray:
        """Global state for the centralized critic (training only)."""
        sc = self.scenario
        g = self.grid_size
        drones = np.zeros((g, g))
        for a, alive in self.alive.items():
            if alive:
                x, y = self.pos[a]
                drones[y, x] += 1.0
        unfound = np.zeros((g, g))
        for (x, y), f in zip(self.survivors, self.found):
            if not f:
                unfound[y, x] += 1.0
        found_frac = self.found.mean() if len(self.found) else 1.0
        return np.concatenate(
            [
                self.freshness().ravel(),
                self._belief_density().ravel(),
                self.obstacles.ravel().astype(float),
                drones.ravel(),
                unfound.ravel(),
                [self.t / sc.max_steps, found_frac, sc.n_drones / MAX_DRONES, len(self.found) / 20],
                sc.movement.as_vector(True),
            ]
        ).astype(np.float32)

    # ---------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        n = len(self.found)
        need = int(np.ceil(0.9 * n)) if n else 0
        t90 = self.found_times[need - 1] if need and len(self.found_times) >= need else None
        return {
            "survivors": n,
            "found_frac": float(self.found.mean()) if n else 1.0,
            "time_to_90": t90,
            "coverage": float(self.visited[self.free].mean()),
            "repeat_frac": self.n_repeats / max(1, self.n_visits),
            "collisions": self.n_collisions,
            "steps": self.t,
        }
