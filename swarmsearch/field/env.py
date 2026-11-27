"""FieldEnv (M8): one mission in a scenario family, built to the
assumptions register (docs/ASSUMPTIONS.md). Same PettingZoo interface as
GridSearchEnv, so the trainer, the rule, the sweep and the evaluation run on
it unchanged.

Per step:
1. Each drone acts (cruise move, hover, climb, descend, or sprint two cells)
   unless the autopilot has it: returning on low battery, out of contact or
   for weather, landed for a battery swap, holding over a find, or leaving a
   newly closed no-fly zone.
2. Drones fly from where they think they are; wind and GPS error decide
   where they really end up; onboard detect-and-avoid stops real conflicts.
3. Radio groups form over the terrain; each group shares one map copy.
4. Cameras look from where the drones really are (true world, true
   detector, correlated per viewing direction); detections go to the drone's
   outbox and on to the operator queue when the drone is in contact. The
   map is updated where the drones think they are, with the modelled
   detector and look counts per direction.
5. Operators review; confirmed finds send a drone to hold over the person
   and a ground team to them. Ground teams also search trails.
6. People move (and react to drones), water rises, smoke drifts, aftershocks
   collapse buildings, the clock runs (light, thermal contrast, rain).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from gymnasium import spaces

from .. import realism as rl
from ..belief import propagate
from ..env import (CLEAR_WEIGHT, FIND_TIME_REWARD, MAX_DRONES, MOVES, N_TEAMMATES, PENALTY_COLLISION,
                   PENALTY_STEP, REWARD_DETECT, WEIGHT_FLOOR, GridSearchEnv, View)
from ..obstacles import distance_fields
from . import lpb
from . import evidence as ev
from . import people as pp
from . import sensing as sn
from .params import FAMILY, FieldSampler, FieldScenario
from .terrain import make_world

HOVER, CLIMB, DESCEND = 8, 9, 10
SPRINT0 = 11
N_ACTIONS_FIELD = 19
CLIMB_MPS, DESCEND_MPS = 4.0, 3.0
LAUNCH_FAIL = 0.02
OVERWATCH_MAX_S = 900.0
# Under-canopy check of an alert in forest: from the lowest search height the
# drone descends through a gap to ~3 m above the ground, flies under the
# crowns to the alerted spot (alerts are located to a few metres in the
# image), looks with nothing between camera and person, and climbs back out.
UNDER_MIN_COVER = 0.5  # tree cover where looking from above stops being enough
UNDER_RUN_M = 100.0  # horizontal run under the crowns, there and back to a gap
UNDER_MPS = 2.0  # slow, among trunks
UNDER_DETECT = 0.9  # thermal at a few metres, no canopy in between; trunks and undergrowth can still hide
UNDER_FA_DISMISS = 0.97  # an animal or a warm stump is easy to tell apart up close
LINK_TIMEOUT_S = 30.0
WEATHER_LIMIT = 12.0  # m/s
TEAM_MPS = 1.0
TEAM_DETECT = 0.6
POOL = 16  # critic map size
N_PATCH = 7
N_COARSE = 4
COARSE = 5
PATCH = 5
FIELD_SCALARS = 50
FIELD_OBS_DIM = N_PATCH * PATCH * PATCH + N_COARSE * COARSE * COARSE + FIELD_SCALARS
# Optional parts of the observation (Phase 2):
# - planning features: for each of 8 directions, the best expected detections per step
#   of travel there (any height) and how far it is: what the rule ranks by.
# - the full map at a fixed size for a CNN, whatever the grid (20, 32, 64 or 128 cells).
PLAN_DIM = 16
FIELD_MAP_RES = 32
FIELD_MAP_CHANNELS = 6  # belief, freshness, blocked, alerts waiting, teammates, self
FIELD_MAP_DIM = FIELD_MAP_CHANNELS * FIELD_MAP_RES * FIELD_MAP_RES


def field_obs_dim(plan: bool = False, full_map: bool = False) -> int:
    return FIELD_OBS_DIM + (PLAN_DIM if plan else 0) + (FIELD_MAP_DIM if full_map else 0)


def to_res(a: np.ndarray, how: str = "mean", res: int = FIELD_MAP_RES) -> np.ndarray:
    """A (g, g) layer at res x res: block mean/max when g is a multiple, else nearest cell."""
    g = a.shape[0]
    if g == res:
        return a
    if g > res and g % res == 0:
        b = a.reshape(res, g // res, res, g // res)
        return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))
    idx = (np.arange(res) * g / res).astype(int)
    return a[np.ix_(idx, idx)]
FIELD_STATE_DIM = 9 * POOL * POOL + 18
CRASH_PENALTY = 10.0


class MapState:
    """One group's map of the ground (changes when a collapse is seen)."""

    def __init__(self, walk, fly, struct_h, elev, G, rubble, revealed):
        self.walk, self.fly, self.struct_h, self.elev, self.G = walk, fly, struct_h, elev, G
        self.rubble, self.revealed = rubble, revealed

    def copy(self):
        return MapState(self.walk.copy(), self.fly.copy(), self.struct_h.copy(), self.elev, self.G.copy(),
                        self.rubble.copy(), self.revealed.copy())


def _pool_matrix(g: int, n: int = POOL) -> np.ndarray:
    M = np.zeros((n, g))
    for i in range(n):
        a, b = i * g / n, (i + 1) * g / n
        for j in range(g):
            M[i, j] = max(0.0, min(b, j + 1) - max(a, j))
        M[i] /= M[i].sum()
    return M


class FieldEnv(GridSearchEnv):
    metadata = {"name": "field_search_v0", "render_modes": []}

    def __init__(self, scenario: FieldScenario | None = None, sampler: FieldSampler | None = None,
                 max_drones: int = MAX_DRONES, seed: int | None = None, shield: bool = True, **_):
        self.fixed_scenario = scenario
        self.sampler = sampler or (None if scenario else FieldSampler())
        self.possible_agents = [f"drone_{i}" for i in range(max_drones)]
        self.rng = np.random.default_rng(seed)
        self.shield = shield
        self.camera = self.real = self.field = True
        self.full_map = self.guided = self.team = False
        self.plan_feats = False  # Phase 2 options (field_obs_dim)
        self.guide_coef = 0.0
        self.reward = "find_time"
        # training only: share of the find reward paid as expected finds, i.e. each look earns the chance it
        # gives of detecting a true person not yet found (privileged: the simulator knows where they are),
        # times a find's value; the rest is paid on actual finds. 0: finds only (evaluation is unchanged)
        self.reward_pod = 0.0
        self._pod = {}
        # the value of a look (footprint_values, the rule and the policy's inputs) counts earlier looks per
        # viewing direction, as the map does; False: only looks from straight above (before 2026-11-27)
        self.view_value = False
        self.goals = {}
        self._v = View()
        self.views = None
        self.agents = []
        self.grid_size = (scenario.fam.grid if scenario else max(FAMILY[f].grid for f in self.sampler.families))
        self._obs_dim = FIELD_OBS_DIM
        self._state_dim = FIELD_STATE_DIM

    def observation_space(self, agent):
        return spaces.Box(-1.0, 5.0, shape=(field_obs_dim(self.plan_feats, self.full_map),), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(N_ACTIONS_FIELD)

    # Map layers of the current drone's group.
    free = property(lambda s: s._v.map.fly & ~s.nofly)
    obstacles = property(lambda s: ~(s._v.map.fly & ~s.nofly))
    n_free = property(lambda s: int(s._v.map.walk.sum()))

    def footprint_values(self, k: int) -> np.ndarray:
        """Expected detections over the footprint at height k from every cell
        (hovering), with the same diminishing returns the map uses (S2): a
        cell already looked at from straight above at this height is worth
        less to look at that way again."""
        from ..camera import footprint_sum
        o = self.occ_model
        pl = np.minimum(0.98, np.clip(0.5 * self.C_model, 1e-3, 0.98) / np.maximum(1e-6, 1 - o))

        def f(m):
            return o + (1 - o) * (1 - pl) ** m

        def disc(n):
            return (1 - f(n + 1) / f(n)) / np.maximum(1 - f(1), 1e-9)
        base = self.belief * self.C_model
        if not self.view_value:  # every cell discounted by its looks from straight above at this height
            return footprint_sum(base * disc(self._v.N[k * sn.N_CLASS + 4]), self._v.map.G[k])
        # each cell of the footprint discounted by its looks from the direction it would be seen from
        # (sensing.view_class): a cell searched from one side is worth looking at again from another (S2)
        G = self._v.map.G[k]
        g, r = self.grid_size, sn.R
        out = np.zeros((g, g))
        for oy in range(2 * r + 1):
            for ox in range(2 * r + 1):
                w = G[:, :, oy, ox]
                if not w.any():
                    continue
                X = np.pad(base * disc(self._v.N[sn.view_class(k, ox - r, oy - r)]), r)
                out += X[oy:oy + g, ox:ox + g] * w
        return out

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        sc = options.get("scenario") or self.fixed_scenario or self.sampler.sample(self.rng)
        self.scenario = sc
        fam = sc.fam
        self.fam = fam
        rng = self.rng
        w = make_world(np.random.default_rng(sc.seed_world), sc)
        self.w = w
        g = self.grid_size = w.g
        # lowest search height allowed per cell (crowns, roofs, wires, towers); 0 everywhere by default
        self.min_level = np.asarray(w.extra.get("min_level", np.zeros((w.g, w.g), int)))
        self.under_p = w.extra.get("under_p")
        self.under_s = w.extra.get("under_s")
        self.under_tried = set()
        self.land_dist = w.extra.get("land_dist")
        self.exec_stats = w.extra.get("exec")
        wm = w.extra.get("wind_mult")
        self._wind_mult_all = wm
        # low cloud: no search height within 10 m of the cloud base; cells needing more are no-fly
        ceil = getattr(sc, "ceiling_m", 1e9)
        lv_ok = [k for k, h in enumerate(fam.alt_m) if h <= ceil - 10.0]
        self.max_level = max(lv_ok) if lv_ok else 0
        if self.max_level < 2:
            over = self.min_level > self.max_level
            w.fly_block |= over
            w.map_fly_block |= over
            self.min_level = np.where(over, 3, self.min_level)
        self.n_emergency = self.n_bird = self.n_under_crash = 0
        self._M = _pool_matrix(g)
        self.alt_m_levels = fam.alt_m
        self.foot_r = fam.foot_r
        self.sweep_alt = int(np.argmax([(2 * r + 1) ** 2 * d for r, d in zip(fam.foot_r, fam.alt_detect)]))
        self.has_sprint = True
        self.t = 0
        self.agents = self.possible_agents[: sc.n_drones]
        self.map_version = 0

        # True geometry and conditions.
        self.G_true = sn.geo_table(w.surface, w.elev, w.walk, fam.alt_m, fam.foot_r, fam.alt_detect)
        self.water = w.water.copy()
        self.smoke = np.zeros((g, g))
        self._smoke_shape = self._make_smoke(sc, w, rng)
        self.rain_on = False
        self._rain_spell = self._spell(sc.rain, rng)
        self._gust_spell = self._spell(sc.gust, rng)
        self._aftershock_t = int(rng.integers(fam.max_steps // 6, fam.max_steps)) if rng.random() < sc.aftershock else -1
        self.wind_dir = np.array(rl.MOVES8[rng.integers(8)])
        # No-fly zones (D8)
        yy, xx = np.mgrid[0:g, 0:g]
        self._nofly_static = np.zeros((g, g), dtype=bool)
        for _ in range(sc.n_nofly):
            cx, cy, r = rng.integers(g // 4, g), rng.integers(0, g), rng.uniform(1.5, 3.5)
            self._nofly_static |= np.hypot(xx - cx, yy - cy) <= r
        self._nofly_static[max(0, w.base[1] - 2):w.base[1] + 3, max(0, w.base[0] - 2):w.base[0] + 3] = False
        self._tfr = None
        if rng.random() < sc.p_tfr:
            t0 = int(rng.integers(0, fam.max_steps * 2 // 3))
            cx, cy, r = rng.integers(g // 4, g), rng.integers(0, g), rng.uniform(2, 4)
            mask = np.hypot(xx - cx, yy - cy) <= r
            mask[max(0, w.base[1] - 2):w.base[1] + 3, max(0, w.base[0] - 2):w.base[0] + 3] = False
            self._tfr = (t0, t0 + int(rng.uniform(600, 1200) / fam.step_s), mask)
        self.nofly = self._nofly_static.copy()

        # People and what the drones know.
        self.people = pp.People(w, sc, rng)
        n = sc.n_people
        self.survivors = self.people.pos  # the evaluation and renderers read these names
        self.found = np.zeros(n, dtype=bool)
        self.found_times: list[int] = []
        self.rescued_t = np.full(n, -1)
        self.n_est = n if sc.n_est_err == 0 else max(1, int(round(n * np.exp(rng.normal(0, sc.n_est_err)))))
        ping = None
        if sc.phone_ping and not self.people.outside[0]:  # P2
            px, py = self.people.pos[0]
            rad = rng.uniform(300, 1000) / fam.cell_m
            ping = ((int(np.clip(px + rng.normal(0, rad / 2), 0, g - 1)), int(np.clip(py + rng.normal(0, rad / 2), 0, g - 1))), rad)
        dens, out, self.rep_ipp = pp.prior(w, sc, rng, ping)
        names, W, pw, mps = pp.hypothesis_bank(w, sc, self.rep_ipp, rng)
        self._W, self._w0, self._V_moveprob, self.hyp_names = W, pw, mps, names

        G_map = sn.geo_table(w.map_surface, w.elev, w.map_walk, fam.alt_m, fam.foot_r, fam.alt_detect)
        rubble_map = w.extra["cover"] == 1 if "cover" in w.extra else np.zeros((g, g), bool)
        self._v = View()
        v = self._v
        v.map = MapState(w.map_walk.copy(), ~w.map_fly_block, w.map_struct_h.copy(), w.elev, G_map,
                         rubble_map.copy(), np.zeros((g, g), dtype=bool))
        v.B = np.repeat(dens[None], len(names), axis=0)
        v.out = np.full(len(names), out)
        v.w = pw.copy()
        v.N = np.zeros((3 * sn.N_CLASS, g, g), dtype=np.float32)
        v.cand_t = np.full((g, g), -1e9)
        v.last_searched = np.full((g, g), -1e9)
        v.s_est = 1.0
        self._mix()
        self.visited = np.zeros((g, g), dtype=bool)
        self.unseen = (w.collapsed.copy() if w.collapsed is not None else np.zeros((g, g), bool))
        self.collapsed = self.unseen.copy()

        # Drones
        self.pos = {a: np.array(w.base) for a in self.agents}
        self.true_pos = {a: np.array(w.base) for a in self.agents}
        self.alive = {a: True for a in self.agents}
        self.alt = {a: 0 for a in self.agents}
        self.alt_tgt = {a: 0 for a in self.agents}
        self.alt_now = {a: fam.alt_m[0] for a in self.agents}
        self.battery = {a: 1.0 for a in self.agents}
        self.mode = {a: None for a in self.agents}  # autopilot mode
        self.mode_t = {a: 0 for a in self.agents}
        self.mode_target = {a: None for a in self.agents}
        self.heading = {a: np.array([1, 0]) for a in self.agents}
        self.last_act = {a: HOVER for a in self.agents}
        self.unlinked_s = {a: 0.0 for a in self.agents}
        self.history = {a: [self.pos[a].copy()] for a in self.agents}
        self.trail_t = {a: np.full((g, g), -1e9) for a in self.agents}
        self.views = {n_: v for n_ in [*self.agents, "base"]}
        v.mates = {a: (self.pos[a].copy(), 0) for a in self.agents}
        self.group_of = {n_: list(self.agents) + ["base"] for n_ in [*self.agents, "base"]}
        self.queue = sn.OperatorQueue(sc.n_operators, int(np.ceil(sc.review_s / fam.step_s)), sc.op_miss, rng)
        self.teams = [dict(pos=np.array(w.base, float), task=None, prog=0.0, patrol=0) for _ in range(sc.n_teams)]
        self._patrol = self._patrol_route(w)
        self.link_mask = self._link_mask(sc, w)
        self._fields: dict = {}
        self._fields_ver = -1

        self.n_visits = self.n_repeats = self.n_collisions = self.n_shield = 0
        self.n_alarms = self.n_false_alarms = self.n_bumps = self.n_slips = self.n_gps_off = 0
        self.n_near = self.n_crash = self.n_swaps = self.n_launch_fail = 0
        self.n_under = 0  # alerts checked from under the trees
        self.n_under_finds = 0  # finds confirmed from such a check
        self.n_under_blocked = 0  # checks that could not get close (undergrowth, wires)
        self.linked_steps = 0.0
        # Evidence and others in the area (K7, S8), on their own random stream so that
        # missions without them run exactly as before.
        self._ev_rng = np.random.default_rng((sc.seed_world, 7))
        self.clues = ev.Clues(w, sc, self.people, self._ev_rng)
        self.others = ev.Others(w, sc, self._ev_rng)
        self._report_t = (int(self._ev_rng.uniform(0.1, 0.8) * fam.max_steps)
                          if self._ev_rng.random() < sc.p_report else -1)
        self.n_clues_seen = self.n_clues_used = self.n_other_alarms = self.n_reports = 0
        self.expanded_t = -1  # step at which the search area grew (K2), -1: not yet
        self.grid_epoch = 0  # bumped when the grid changes (planners re-plan)
        self._update_conditions()
        for a in self.agents:
            self._search(a, HOVER)
        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a)} for a in self.agents}
        return obs, infos

    # ---------------------------------------------------------- world bits
    @staticmethod
    def _spell(p, rng):
        if rng.random() >= p:
            return None
        return (float(rng.uniform(0, 0.8)), float(rng.uniform(0.1, 0.3)))  # start, length (fraction of mission)

    def _in_spell(self, spell) -> bool:
        if spell is None:
            return False
        f = self.t / self.fam.max_steps
        return spell[0] <= f < spell[0] + spell[1]

    def _make_smoke(self, sc, w, rng):
        if sc.family != "urban" or sc.smoke <= 0 or w.collapsed is None or not w.collapsed.any():
            return None
        g = w.g
        yy, xx = np.mgrid[0:g, 0:g]
        d = np.array(rl.MOVES8[rng.integers(8)], float)
        d /= np.hypot(*d)
        shape = np.zeros((g, g))
        cc = np.argwhere(w.collapsed)
        for y, x in cc[rng.choice(len(cc), size=min(2, len(cc)), replace=False)]:
            along = (xx - x) * d[0] + (yy - y) * d[1]
            perp = -(xx - x) * d[1] + (yy - y) * d[0]
            shape = np.maximum(shape, np.where(along >= -0.5, np.exp(-perp ** 2 / (2 * (1 + 0.3 * np.maximum(along, 0)) ** 2))
                                               * np.exp(-np.maximum(along, 0) / 8.0), 0.0))
        return sc.smoke * shape

    def _link_mask(self, sc, w) -> np.ndarray:
        """Cells where the radio plan says a drone at its top height keeps a link to
        the base (range, and terrain/buildings for the shorter blocked range).
        Planners stay inside it when drones return on link loss (D6)."""
        g, fam = w.g, sc.fam
        if sc.always_linked or not sc.rtb_on_link_loss:
            return np.ones((g, g), dtype=bool)
        rng_c = sc.comm_range_m / fam.cell_m
        base_h = w.elev[w.base[1], w.base[0]] + (100.0 if sc.tether_relay else rl.BASE_ALT_M)
        out = np.zeros((g, g), dtype=bool)
        for y in range(g):
            for x in range(g):
                d = np.hypot(x - w.base[0], y - w.base[1])
                if d <= rng_c * rl.NLOS_FACTOR:
                    out[y, x] = True
                elif d <= 0.9 * rng_c:
                    out[y, x] = rl.line_clear(w.map_surface, w.base, (x, y), base_h, w.elev[y, x] + fam.alt_m[2])
        return out

    def _patrol_route(self, w):
        """Ground teams' hasty search: walk the mapped trails / streets out from the base."""
        trail = np.argwhere(w.map_trail & w.walk)
        if not len(trail):
            return [w.base]
        d = np.hypot(trail[:, 1] - w.base[0], trail[:, 0] - w.base[1])
        order = trail[np.argsort(d)]
        return [(int(x), int(y)) for y, x in order[: 4 * w.g]]

    def hour(self) -> float:
        return self.scenario.hour + self.t * self.fam.step_s / 3600.0

    def _update_conditions(self):
        sc, w = self.scenario, self.w
        self.rain_on = self._in_spell(self._rain_spell)
        if self._smoke_shape is not None:
            self.smoke = self._smoke_shape * min(1.0, self.t * self.fam.step_s / 600.0 + 0.2)
        if sc.family == "flood":
            level = w.river_level + sc.water_rise * self.t / self.fam.max_steps
            self.water = w.elev < level
        rgb, th = sn.conditions(w.canopy, w.rubble | self.collapsed, self.smoke, self.water, self.hour(), sc.air_c,
                                self.rain_on, sc.canopy_model)
        self.C_rgb, self.C_th = rgb, th
        map_rubble = (self.views["base"] if self.views else self._v).map.rubble
        rgb_m, th_m = sn.conditions(w.map_canopy, map_rubble, self.smoke, self.water, self.hour(), sc.air_c,
                                    self.rain_on, sc.canopy_model)
        self.C_model = np.maximum(rgb_m, th_m)
        self.occ_true = sn.occlusion(w.canopy, w.rubble | self.collapsed, sc.canopy_model)
        self.occ_model = sn.occlusion(w.map_canopy, map_rubble, sc.canopy_model)
        self.FA_true = sn.false_alarm_rate(sc.family, w.canopy, w.rubble | self.collapsed, self.hour(),
                                           0.001) * sc.fa_scale
        if self._tfr is not None:
            t0, t1, mask = self._tfr
            self.nofly = self._nofly_static | (mask if t0 <= self.t < t1 else False)

    # ------------------------------------------------------------- belief
    def _mix(self):
        self.belief = np.tensordot(self._w, self._B, axes=1)
        post = self.profile_posterior()
        self._decay = float(post @ self._V_moveprob)

    def profile_posterior(self) -> np.ndarray:
        q = self._w * (self._B.sum(axis=(1, 2)) + self._v.out)
        return q / max(q.sum(), 1e-300)

    def p_outside(self) -> float:
        tot = self._w @ (self._B.sum(axis=(1, 2)) + self._v.out)
        return float(self._w @ self._v.out / max(tot, 1e-300))

    def _spread_belief(self):
        for v in self._distinct_views():
            self._v = v
            s0 = v.B.sum(axis=(1, 2))
            v.B = propagate(v.B, self._W)
            v.out = v.out + np.maximum(s0 - v.B.sum(axis=(1, 2)), 0.0)  # walked off the map (K2)
            v.N *= np.exp(-self._decay)  # looks lose their meaning as people may have moved (S2)
            self._mix()

    def _apply_find(self, v: View, i: int, cell):
        if i in v.finds:
            return
        v.finds[i] = cell
        x, y = cell
        if len(v.w) > 1:
            v.w = v.w * (v.B[:, y, x] + 1e-12)
            v.w = (1 - WEIGHT_FLOOR) * v.w / v.w.sum() + WEIGHT_FLOOR * self._w0
        cur, self._v = self._v, v
        self._mix()
        self._v = cur

    # -------------------------------------------------------------- masks
    def _flyable(self, x, y, use_true=False) -> bool:
        g = self.grid_size
        if not (0 <= x < g and 0 <= y < g):
            return False
        ok = (~self.w.fly_block[y, x]) if use_true else self._v.map.fly[y, x]
        return bool(ok and not self.nofly[y, x])

    def action_mask(self, a) -> np.ndarray:
        self.use_view(a)
        m = np.zeros(N_ACTIONS_FIELD, dtype=np.int8)
        m[HOVER] = 1
        if self.mode[a] is not None or not self.alive[a]:
            return m  # the autopilot has it
        x, y = self.pos[a]
        occupied = {tuple(self.pos[b]) for b in self.agents if b != a and self.alive[b] and self.mode[b] != "landed"}
        lv = self.alt[a]
        for k, (dx, dy) in enumerate(MOVES[:8]):
            nx, ny = x + dx, y + dy
            if self._flyable(nx, ny) and (nx, ny) not in occupied and self._high_enough(nx, ny, lv):
                m[k] = 1
                if (self._flyable(nx + dx, ny + dy) and (nx + dx, ny + dy) not in occupied
                        and self._high_enough(nx + dx, ny + dy, lv)):
                    m[SPRINT0 + k] = 1
        m[CLIMB] = self.alt_tgt[a] < min(2, getattr(self, "max_level", 2))
        m[DESCEND] = self.alt_tgt[a] > max(0, int(self.min_level[y, x]))
        return m

    def _wind_here(self, a) -> float:
        """Mean wind at the drone's search height in its cell: the scenario's wind shaped by the terrain."""
        sc = self.scenario
        if self._wind_mult_all is None:
            return sc.wind_mps
        dx, dy = (float(v) for v in self.wind_dir)  # the way the wind blows (grid: +y is south)
        frm = (math.degrees(math.atan2(dx, -dy)) + 180.0) % 360.0
        x, y = self.pos[a]
        return sc.wind_mps * float(self._wind_mult_all[int(round(frm / 45.0)) % 8, y, x])

    def _high_enough(self, x, y, level) -> bool:
        g = self.grid_size
        return not (0 <= x < g and 0 <= y < g) or level >= self.min_level[y, x]

    def _needs_climb(self, a, act) -> bool:
        """A move that is blocked only because the next cell needs a higher search height."""
        if act >= 8 and act < SPRINT0:
            return False
        k = act if act < 8 else act - SPRINT0
        steps = 1 if act < 8 else 2
        x, y = self.pos[a]
        for s_ in range(1, steps + 1):
            nx, ny = x + s_ * MOVES[k][0], y + s_ * MOVES[k][1]
            if self._flyable(nx, ny) and not self._high_enough(nx, ny, self.alt[a]):
                return True
        return False

    def targets(self, actions, count: bool = False) -> dict:
        live = [a for a in self.agents if self.alive[a] and self.mode[a] != "landed"]
        target = {}
        for a in live:
            act = self._auto_action(a) if self.mode[a] is not None else int(actions.get(a, HOVER))
            if (act < 8 or act >= SPRINT0) and self._needs_climb(a, act):
                # trees, wires or a tower ahead need a higher search height: climb first, then go
                self.alt_tgt[a] = min(2, self.max_level, max(self.alt_tgt[a], self.alt[a] + 1))
                act = HOVER
            elif not self.action_mask(a)[act] and self.mode[a] is None:
                act = HOVER
            self.last_act[a] = act
            if act < 8:
                target[a] = self.pos[a] + MOVES[act]
            elif act >= SPRINT0:
                target[a] = self.pos[a] + 2 * MOVES[act - SPRINT0]
            else:
                target[a] = self.pos[a].copy()
        if self.shield:
            n = self.n_shield
            self._resolve_conflicts(live, target)
            if not count:
                self.n_shield = n
        return target

    # ----------------------------------------------------------- autopilot
    def _field_to(self, cell):
        if self._fields_ver != self.map_version:
            self._fields, self._fields_ver = {}, self.map_version
        key = (tuple(cell), id(self._v.map))
        if key not in self._fields:
            fly = self._v.map.fly & ~self._nofly_static
            fly[cell[1], cell[0]] = True
            self._fields[key] = distance_fields(fly, [tuple(cell)])[0]
        return self._fields[key]

    def _auto_action(self, a) -> int:
        """Next move for a drone the autopilot controls."""
        mode = self.mode[a]
        if mode == "under":
            return HOVER
        if mode == "overwatch":
            i = self.mode_target[a]
            goal = tuple(int(v) for v in self.people.pos[i]) if not self.people.outside[i] else self.w.base
        elif mode == "nofly_exit":
            free = np.argwhere(~self.nofly & self._v.map.fly)
            x, y = self.pos[a]
            d = np.abs(free[:, 1] - x) + np.abs(free[:, 0] - y)
            yy, xx = free[np.argmin(d)]
            goal = (int(xx), int(yy))
        else:
            goal = self.w.base
        if tuple(self.pos[a]) == tuple(goal):
            return HOVER
        D = self._field_to(goal)
        x, y = self.pos[a]
        best, bd = HOVER, D[y, x]
        g = self.grid_size
        for k, (dx, dy) in enumerate(MOVES[:8]):
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and D[ny, nx] < bd and (mode == "nofly_exit" or not self.nofly[ny, nx]):
                best, bd = k, D[ny, nx]
        return best

    def _energy_home(self, a) -> float:
        sc, fam = self.scenario, self.fam
        D = self._field_to(self.w.base)
        x, y = self.pos[a]
        steps = D[y, x] if np.isfinite(D[y, x]) else 2 * self.grid_size
        # the return reserve uses the wind the drone measures where it is (terrain-shaped), with a margin
        wind = max(sc.wind_mps, 1.25 * self._wind_here(a)) if self._wind_mult_all is not None else sc.wind_mps
        per = fam.step_s / (sc.endurance_min * 60.0) * (1 + 0.02 * wind)
        return float(steps * per * 1.5 + 0.12)  # 12% kept for climbing to safe heights, landing and gusts

    def _update_modes(self, a):
        """Battery, link loss, weather, no-fly and overwatch decide who the autopilot flies (D1, D6, D8, O1)."""
        sc, fam = self.scenario, self.fam
        if not self.alive[a]:
            return
        mode = self.mode[a]
        at_base = tuple(self.true_pos[a]) == tuple(self.w.base)
        if mode == "landed":
            self.mode_t[a] -= 1
            if self.mode_t[a] <= 0 and not self._in_spell(self._gust_spell):
                if self.rng.random() < LAUNCH_FAIL:
                    self.alive[a] = False
                    self.n_launch_fail += 1
                    return
                self.mode[a] = None
                self.battery[a] = 1.0
                self.alt[a] = self.alt_tgt[a] = 0
                self.alt_now[a] = fam.alt_m[0]
            return
        if mode in ("rtb", "link", "weather") and at_base:
            if mode == "rtb" or self.battery[a] < 0.5:
                self.mode[a] = "landed"
                self.mode_t[a] = int(np.ceil(sc.swap_s / fam.step_s))
                self.n_swaps += 1
                return
            if mode == "weather" and self._in_spell(self._gust_spell):
                self.mode[a] = "landed"  # sits the gusts out on the ground
                self.mode_t[a] = 1
                return
            self.mode[a] = None
        if mode == "under":
            self.mode_t[a] -= 1
            if self.battery[a] <= self._energy_home(a):
                self.mode[a] = "rtb"  # battery first: the check is abandoned
            return
        if mode == "overwatch":
            self.mode_t[a] -= 1
            i = self.mode_target[a]
            if self.mode_t[a] <= 0 or self.rescued_t[i] >= 0:
                self.mode[a] = None
        if self.battery[a] <= self._energy_home(a):
            self.mode[a] = "rtb"
        elif self._in_spell(self._gust_spell) and self.mode[a] != "rtb":
            self.mode[a] = "weather"
        elif sc.rtb_on_link_loss and self.unlinked_s[a] >= LINK_TIMEOUT_S and self.mode[a] is None:
            self.mode[a] = "link"
        elif self.mode[a] == "link" and self.unlinked_s[a] == 0:
            self.mode[a] = None
        x, y = self.pos[a]
        if self.mode[a] is None and self.nofly[y, x]:
            self.mode[a] = "nofly_exit"
        elif self.mode[a] == "nofly_exit" and not self.nofly[y, x]:
            self.mode[a] = None

    # ------------------------------------------------------------------ step
    def step(self, actions, searching=None):
        sc, fam, rng = self.scenario, self.fam, self.rng
        g = self.grid_size
        self.t += 1
        self._update_conditions()
        rewards = {a: -PENALTY_STEP for a in self.agents}
        self._pod = {a: 0.0 for a in self.agents}
        live = [a for a in self.agents if self.alive[a]]
        for a in live:
            self.use_view(a)
            self._update_modes(a)
        # altitude (D2)
        for a in live:
            if self.mode[a] is None:
                act = int(actions.get(a, HOVER))
                if act == CLIMB and self.alt_tgt[a] < min(2, self.max_level):
                    self.alt_tgt[a] += 1
                elif act == DESCEND and self.alt_tgt[a] > 0:
                    self.alt_tgt[a] -= 1
            cx_, cy_ = self.pos[a]
            self.alt_tgt[a] = min(max(self.alt_tgt[a], min(2, int(self.min_level[cy_, cx_]))), self.max_level)
            goal = fam.alt_m[self.alt_tgt[a]]
            rate = (CLIMB_MPS if goal > self.alt_now[a] else DESCEND_MPS) * fam.step_s
            new_h = self.alt_now[a] + float(np.clip(goal - self.alt_now[a], -rate, rate))
            # detect-and-avoid also vertically: no climbing or descending into a drone in the same spot
            if any(b != a and self.alive[b] and self.mode[b] != "landed"
                   and tuple(self.true_pos[b]) == tuple(self.true_pos[a]) and abs(self.alt_now[b] - new_h) < 10.0
                   for b in live):
                new_h = self.alt_now[a]
            self.alt_now[a] = new_h
            self.alt[a] = int(np.argmin([abs(self.alt_now[a] - h) for h in fam.alt_m]))
        target = self.targets(actions, count=True)
        for a in live:
            if self.mode[a] == "landed" or not self.alive[a]:
                continue
            self._fly(a, target[a])
            self.history[a] = (self.history[a] + [self.pos[a].copy()])[-6:]
            self.trail_t[a][self.pos[a][1], self.pos[a][0]] = self.t
            # battery (D1)
            act = self.last_act[a]
            power = 1.5 if act >= SPRINT0 else 1.0
            power += 0.3 if abs(self.alt_now[a] - fam.alt_m[self.alt_tgt[a]]) > 1 else 0.0
            power *= (1 + 0.02 * self._wind_here(a)) * (1.2 if sc.air_c < 0 else 1.0) * (1.1 if self.rain_on else 1.0)
            self.battery[a] -= power * fam.step_s / (sc.endurance_min * 60.0)
            fail = rng.random() < sc.fail_per_h * fam.step_s / 3600.0
            bird = rng.random() < getattr(sc, "bird_per_h", 0.0) * fam.step_s / 3600.0
            if self.battery[a] <= 0 or fail or bird:
                self.alive[a] = False  # battery exhausted, in-flight failure (D6) or a bird of prey
                self.n_bird += int(bird and not fail)
                cx_, cy_ = self.pos[a]
                # with a failure (not an empty battery) it can still glide down under control for ~150 m:
                # an open spot within reach means an emergency landing, else it comes down in the trees
                if (self.battery[a] > 0 and self.land_dist is not None
                        and float(self.land_dist[cy_, cx_]) <= 150.0):
                    self.n_emergency += 1
                else:
                    self.n_crash += 1
                rewards[a] -= CRASH_PENALTY
        # collisions where drones really are, at the same height
        up = [a for a in self.agents if self.alive[a] and self.mode[a] != "landed"]
        for i, a in enumerate(up):  # same spot and within 10 m of height, away from the pad
            for b in up[i + 1:]:
                if (tuple(self.true_pos[a]) == tuple(self.true_pos[b]) != tuple(self.w.base)
                        and abs(self.alt_now[a] - self.alt_now[b]) < 10.0):
                    self.n_collisions += 2
                    rewards[a] -= PENALTY_COLLISION
                    rewards[b] -= PENALTY_COLLISION
        self._sync()
        # sensing (S) and the map
        for a in self.agents:
            if self.alive[a] and self.mode[a] != "landed":
                r = self._search(a, self.last_act[a])
                rewards[a] += r * CLEAR_WEIGHT
        # operators (S5), finds, overwatch and ground teams (O1, O2)
        found_now = self._review()
        found_now += self._teams_step()
        per_find = FIND_TIME_REWARD * (1.0 - self.t / fam.max_steps)
        w_pod = float(self.reward_pod)
        for a in self.agents:  # finds are a team reward; expected detections credit the drone that looked
            rewards[a] += (1.0 - w_pod) * per_find * found_now + w_pod * per_find * self._pod.get(a, 0.0)
        # people, water, aftershocks
        self._move_people()
        if self.t == self._report_t:
            self._new_report()
        if self.t == self._aftershock_t:
            self._aftershock()
        self._spread_belief()
        if sc.expand and self.expanded_t < 0 and "big" in self.w.extra and not self.found.all():
            self._v = self.views["base"]
            due = sc.expand_min > 0 and self.t * fam.step_s >= sc.expand_min * 60.0
            if due or self.p_outside() > sc.expand_at:
                self._expand()

        terminations = {a: not self.alive[a] for a in self.agents}
        truncated = self.t >= fam.max_steps
        # Ends when everyone is found and rescued; otherwise at the time limit
        # (nobody knows when someone is outside the area).
        done_all = bool(self.found.all() and (self.rescued_t >= 0).all())
        truncations = {a: truncated and not terminations[a] for a in self.agents}
        if done_all:
            terminations = {a: True for a in self.agents}
        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a)} for a in self.agents}
        if done_all or truncated:
            self.agents = []
        else:
            self.agents = [a for a in self.agents if not terminations[a]]
        return obs, rewards, terminations, truncations, infos

    def _expand(self):
        """Grow the search area (K2): the base's map now says the person is more
        likely outside than inside, so the incident commander widens it to the
        world built EXPAND times wider at launch. Same cells, clock and drones;
        everything on the old grid is placed at its offset in the new one. The
        new ring gets the share of each hypothesis's outside mass that the
        distance data put there; the rest stays outside. People outside whose
        true spot is now on the map come back in."""
        big, (ox, oy) = self.w.extra["big"], self.w.extra["origin"]
        sc, fam, rng = self.scenario, self.fam, self.rng
        g0, G = self.grid_size, big.g
        off = np.array([ox, oy])
        inner = (slice(oy, oy + g0), slice(ox, ox + g0))
        ring = np.ones((G, G), dtype=bool)
        ring[inner] = False

        def emb(a, fill):
            out = np.full((*a.shape[:-2], G, G), fill, dtype=a.dtype)
            out[(..., *inner)] = a
            return out

        self.w = big
        self.grid_size = G
        # the wider world's per-cell flight limits and conditions (field/hazards.py; none: obstacle-free)
        self.min_level = np.asarray(big.extra.get("min_level", np.zeros((G, G), int)))
        self.under_p, self.under_s = big.extra.get("under_p"), big.extra.get("under_s")
        self.land_dist = big.extra.get("land_dist")
        self._wind_mult_all = big.extra.get("wind_mult")
        if self.max_level < 2 and self.min_level is not None:
            over = self.min_level > self.max_level
            big.fly_block |= over
            big.map_fly_block |= over
            self.min_level = np.where(over, 3, self.min_level)
        self._M = _pool_matrix(G)
        self.G_true = sn.geo_table(big.surface, big.elev, big.walk, fam.alt_m, fam.foot_r, fam.alt_detect)
        self.water = big.water.copy()
        self.smoke = emb(self.smoke, 0.0)
        self._smoke_shape = emb(self._smoke_shape, 0.0) if isinstance(self._smoke_shape, np.ndarray) else self._smoke_shape
        self._nofly_static = emb(self._nofly_static, False)
        if self._tfr is not None:
            self._tfr = (self._tfr[0], self._tfr[1], emb(self._tfr[2], False))
        self.nofly = emb(self.nofly, False)
        self.visited = emb(self.visited, False)
        self.unseen = emb(self.unseen, False)
        self.collapsed = emb(self.collapsed, False)
        # People
        ppl = self.people
        ppl.w, ppl.walk, ppl._fields = big, big.walk, {}
        ppl.pos += off
        ppl.far += off
        for i in np.flatnonzero(ppl.outside & np.isfinite(ppl.far[:, 0])):
            x, y = (int(round(c)) for c in ppl.far[i])
            if 0 <= x < G and 0 <= y < G:
                if not big.walk[y, x]:  # nearest walkable cell
                    cells = np.argwhere(big.walk)
                    y, x = cells[np.argmin(np.hypot(cells[:, 0] - y, cells[:, 1] - x))]
                ppl.pos[i] = (x, y)
                ppl.outside[i] = False
                ppl.far[i] = np.nan
                ppl.occ[i] = {}
        self.survivors = ppl.pos
        # The maps
        self.rep_ipp = (self.rep_ipp[0] + ox, self.rep_ipp[1] + oy)
        names, W, _pw, mps = pp.hypothesis_bank(big, sc, self.rep_ipp, rng)
        if names != self.hyp_names:
            raise RuntimeError("hypotheses changed when the area grew")
        self._W, self._V_moveprob = W, mps
        # How much of "outside the old area" the distance data put in the new ring (vs beyond it),
        # spread over the ring like the prior (terrain, direction of travel).
        raw, beyond = lpb.ring_probability(G, big.cell_m, self.rep_ipp, lpb.CATEGORIES[sc.category]["dist_km"])
        to_ring = float(raw[ring].sum() / max(raw[ring].sum() + beyond, 1e-300))
        dens, _ = pp.ring_density(big, sc, self.rep_ipp)
        ring_dens = np.where(ring, dens, 0.0) / max(dens[ring].sum(), 1e-300)
        G_map = sn.geo_table(big.map_surface, big.elev, big.map_walk, fam.alt_m, fam.foot_r, fam.alt_detect)
        for v in self._distinct_views():
            m = v.map
            walk, fly, sh = big.map_walk.copy(), ~big.map_fly_block, big.map_struct_h.copy()
            walk[inner], fly[inner], sh[inner] = m.walk, m.fly, m.struct_h
            v.map = MapState(walk, fly, sh, big.elev, G_map.copy(), emb(m.rubble, False), emb(m.revealed, False))
            v.B = emb(v.B, 0.0) + (v.out * to_ring)[:, None, None] * ring_dens[None]
            v.out = v.out * (1.0 - to_ring)
            v.N = emb(v.N, 0.0)
            v.cand_t = emb(v.cand_t, -1e9)
            v.last_searched = emb(v.last_searched, -1e9)
            v.finds = {i: (c[0] + ox, c[1] + oy) for i, c in v.finds.items()}
            v.mates = {a: (p + off, t) for a, (p, t) in v.mates.items()}
            for it in v.outbox:
                it["cell"] = (it["cell"][0] + ox, it["cell"][1] + oy)
            self._v = v
            self._mix()
        for it in self.queue.items + [c for c in self.queue.current if c is not None]:
            it["cell"] = (it["cell"][0] + ox, it["cell"][1] + oy)
        # Drones and teams
        for a in self.possible_agents[: sc.n_drones]:
            if a in self.pos:
                self.pos[a] = self.pos[a] + off
                self.true_pos[a] = self.true_pos[a] + off
                self.history[a] = [h + off for h in self.history[a]]
                self.trail_t[a] = emb(self.trail_t[a], -1e9)
        for tm in self.teams:
            tm["pos"] = tm["pos"] + off
            tm["patrol"] = 0
        self.clues.shift(ox, oy)
        self.others.shift(ox, oy, big)
        self._patrol = self._patrol_route(big)
        self.link_mask = self._link_mask(sc, big)
        self._fields, self._fields_ver = {}, -1
        self.map_version += 1
        self.grid_epoch += 1
        self.expanded_t = self.t
        self._update_conditions()

    def _fly(self, a, target):
        """Fly the chosen move from the believed cell; wind and GPS decide the truth (D4)."""
        sc, fam, rng = self.scenario, self.fam, self.rng
        g = self.grid_size
        near = {tuple(self.true_pos[b]) for b in self.agents
                if b != a and self.alive[b] and abs(self.alt_now[b] - self.alt_now[a]) < 10.0
                and self.mode[b] != "landed"}

        def ok(c):
            return (0 <= c[0] < g and 0 <= c[1] < g and not self.w.fly_block[c[1], c[0]]
                    and (c[0], c[1]) not in near)

        tp = self.true_pos[a]
        d = target - self.pos[a]
        if d.any():
            self.heading[a] = np.sign(d)
        nxt = tp + d
        mid = tp + np.sign(d)
        if not ok(nxt) or not ok(mid):
            self.n_bumps += 1
            if any((c[0], c[1]) in near for c in (nxt, mid)):
                self.n_near += 1
            nxt = tp.copy()
        slip = rl.SLIP_PER_MPS * self._wind_here(a) * (10.0 / fam.cell_m)
        if rng.random() < slip:
            dd = self.wind_dir if rng.random() >= rl.GUST else np.array(rl.MOVES8[rng.integers(8)])
            if ok(nxt + dd):
                nxt = nxt + dd
                self.n_slips += 1
        self.true_pos[a] = nxt
        est = nxt.copy()
        x, y = nxt
        tall = self.w.surface[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() - self.w.elev[y, x]
        p = sc.gps_err * (10.0 / fam.cell_m) * (rl.URBAN_GPS if tall > self.alt_now[a] else 1.0)
        if rng.random() < p:
            e = nxt + np.array(rl.MOVES8[rng.integers(8)])
            if 0 <= e[0] < g and 0 <= e[1] < g and self._v.map.fly[e[1], e[0]]:
                est = e
                self.n_gps_off += 1
        self.pos[a] = est

    # ------------------------------------------------------------- radio
    def _sync(self):
        """Groups over the terrain (C1); maps merge; reports reach the operator only through a link."""
        sc, fam = self.scenario, self.fam
        w = self.w
        up = [a for a in self.agents if self.alive[a] and self.mode[a] != "landed"]
        nodes = {a: (tuple(int(v) for v in self.true_pos[a]), w.elev[self.true_pos[a][1], self.true_pos[a][0]] + self.alt_now[a])
                 for a in up}
        base_h = w.elev[w.base[1], w.base[0]] + (100.0 if sc.tether_relay else rl.BASE_ALT_M)
        nodes["base"] = (w.base, base_h)
        landed = [a for a in self.agents if self.alive[a] and self.mode[a] == "landed"]
        # Undelivered reports belong to the drone that made them, wherever its group goes.
        pending = {}
        for v in self._distinct_views():
            for it in v.outbox:
                pending[id(it)] = it
            v.outbox = []
        used, new = set(), {}
        grps = ([list(nodes)] if sc.always_linked
                else rl.groups(nodes, w.surface, self.rng, sc.comm_range_m / fam.cell_m, sc.comm_loss))
        for grp in grps:
            if "base" in grp:
                grp = grp + landed
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
                self.linked_steps += (len(grp) - 1) / max(1, len(up) + len(landed))
        self.views.update(new)
        for it in sorted(pending.values(), key=lambda it: it["t"]):
            d = it["drone"]
            if d not in new:
                continue  # the drone is gone, and its unsent reports with it
            if "base" in self.group_of[d]:
                self.queue.add(it)
            else:
                new[d].outbox.append(it)
        for a in self.agents:
            if a in up:
                linked = "base" in self.group_of.get(a, [])
                self.unlinked_s[a] = 0.0 if linked else self.unlinked_s[a] + fam.step_s

    def _merge(self, objs) -> View:
        v = objs[0].copy()
        for o in objs[1:]:
            np.minimum(v.B, o.B, out=v.B)
            np.minimum(v.out, o.out, out=v.out)
            np.maximum(v.N, o.N, out=v.N)
            np.maximum(v.cand_t, o.cand_t, out=v.cand_t)
            np.maximum(v.last_searched, o.last_searched, out=v.last_searched)
            for b, (p, t) in o.mates.items():
                if b not in v.mates or v.mates[b][1] < t:
                    v.mates[b] = (p, t)
            for k in ("exp_low", "found_low", "exp_high", "reports"):
                setattr(v, k, max(getattr(v, k), getattr(o, k)))
            # Map discoveries travel with the radio groups (C2).
            new = o.map.revealed & ~v.map.revealed
            if new.any():
                self._reveal_in(v, new)
        for o in objs[1:]:
            for i, cell in o.finds.items():
                self._apply_find(v, i, cell)
        self._v = v
        self._mix()
        return v

    def _reveal_in(self, v: View, cells: np.ndarray):
        """Collapsed rubble seen: in this group's map it becomes flyable, searchable ground (urban)."""
        w, fam = self.w, self.fam
        mask = cells & ~v.map.revealed & self.collapsed
        if not mask.any():
            return
        v.map.revealed |= mask
        v.map.walk[mask] = True
        v.map.fly[mask] = True
        v.map.struct_h[mask] = 3.0
        v.map.rubble[mask] = True
        m0 = rl.COLLAPSE_PRIOR / max(1, int(w.map_walk.sum()))
        v.B[:, mask] = np.maximum(v.B[:, mask], m0)
        # local rebuild of the geometry table
        ys, xs = np.nonzero(mask)
        R2 = 2 * sn.R
        g = self.grid_size
        y0, y1 = max(0, ys.min() - R2), min(g, ys.max() + R2 + 1)
        x0, x1 = max(0, xs.min() - R2), min(g, xs.max() + R2 + 1)
        cy0, cy1, cx0, cx1 = max(0, y0 - sn.R), min(g, y1 + sn.R), max(0, x0 - sn.R), min(g, x1 + sn.R)
        sub = (slice(cy0, cy1), slice(cx0, cx1))
        G = _geo_rect(v.map.elev[sub] + v.map.struct_h[sub], v.map.elev[sub], v.map.walk[sub], fam)
        v.map.G[:, y0:y1, x0:x1] = G[:, y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
        self.map_version += 1
        cur, self._v = self._v, v
        self._mix()
        self._v = cur

    # ------------------------------------------------------------ sensing
    def _search(self, a, act) -> float:
        """Look from where the drone really is; update the map where it thinks it is."""
        self.use_view(a)
        v, sc, rng, w = self._v, self.scenario, self.rng, self.w
        g, r = self.grid_size, sn.R
        if self.mode[a] == "under":
            return self._under_look(a) if self.mode_t[a] <= 0 else 0.0
        k = self.alt[a]
        motion = 1.0 if act in (HOVER, CLIMB, DESCEND) else sn.SPRINT_BLUR if act >= SPRINT0 else sn.BLUR
        slant, shift = 1.0, np.zeros(2, dtype=int)
        if sc.tilt and k > 0:  # forward-tilted camera looks a cell ahead (D7)
            slant, shift = 0.85, self.heading[a]
        tx, ty = (int(c) for c in np.clip(self.true_pos[a] + shift, 0, g - 1))
        ex, ey = (int(c) for c in np.clip(self.pos[a] + shift, 0, g - 1))
        self.n_visits += 1
        if self.freshness()[ey, ex] >= 0.9 and k == 0:
            self.n_repeats += 1
        pending_here = k == 0 and v.cand_t[ey, ex] >= 0  # an unresolved alert where we now are, low
        Gt = self.G_true[k, ty, tx]
        ppl = self.people
        reports = set()
        hit = []
        for oy, ox in zip(*np.nonzero(Gt)):
            cx, cy = tx + ox - r, ty + oy - r
            self.visited[cy, cx] = True
            if self.unseen[cy, cx]:
                hit.append((cy, cx))
            here = np.flatnonzero(~self.found & ~ppl.outside & (ppl.pos[:, 0] == cx) & (ppl.pos[:, 1] == cy))
            cls = sn.view_class(k, ox - r, oy - r)
            seen_i = None
            for i in here:
                o = self.occ_true[cy, cx]
                if ppl.hiding[i]:
                    o = min(0.95, o * 1.5 + 0.2)
                elif ppl.signalling[i]:
                    o *= 0.3
                if cls not in ppl.occ[i]:
                    ppl.occ[i][cls] = bool(rng.random() < o)
                if ppl.occ[i][cls]:
                    continue
                post = sn_posture(ppl, i)
                rgb = self.C_rgb[cy, cx] * ppl.vis[i] * (2.0 if ppl.signalling[i] else 1.0)
                th = self.C_th[cy, cx] * ppl.warm[i]
                p = Gt[oy, ox] * max(rgb, th) * post * motion * slant * sc.sensor_scale
                if self.reward_pod:  # expected detection of this person by this look
                    self._pod[a] = self._pod.get(a, 0.0) + min(0.98, p / max(1e-6, 1 - o))
                if rng.random() < min(0.98, p / max(1e-6, 1 - o)):
                    seen_i = int(i)
                    break
            false = seen_i is None and rng.random() < self.FA_true[cy, cx]
            if seen_i is not None or false:
                rx, ry = int(np.clip(cx + ex - tx, 0, g - 1)), int(np.clip(cy + ey - ty, 0, g - 1))
                reports.add((rx, ry))
                if self._known_spot(v, rx, ry):
                    continue  # already waiting for review, or a known find: grouped with it
                self.n_alarms += 1
                self.n_false_alarms += int(seen_i is None)
                item = dict(cell=(rx, ry), alt=k, true=seen_i, t=self.t, p=0.0, drone=a)
                if "base" in self.group_of.get(a, []):
                    self.queue.add(item)  # streamed to the operator while in contact
                else:
                    v.outbox.append(item)
        self._search_extras(a, k, Gt, tx, ty, ex, ey, motion * slant, v, reports)
        if hit:
            ys, xs = zip(*hit)
            m = np.zeros((g, g), dtype=bool)
            m[list(ys), list(xs)] = True
            blocks = np.isin(self._block_ids(), np.unique(self._block_ids()[m]))
            self.unseen &= ~blocks
            self._reveal_in(v, blocks & self.collapsed)
        v.reports += len(reports)
        # The map: modelled detector at the believed position, looks counted per direction.
        Gm = v.map.G[k, ey, ex]
        reward = 0.0
        n_est = self.n_est
        s = v.s_est if sc.adaptive_map else 1.0
        for oy, ox in zip(*np.nonzero(Gm)):
            mx, my = ex + ox - r, ey + oy - r
            self.last_searched[my, mx] = self.t
            if (mx, my) in reports:
                self.cand_t[my, mx] = self.t
                continue
            if k == 0:
                self.cand_t[my, mx] = -1e9  # checked from low: resolved either way
            p_nom = Gm[oy, ox] * self.C_model[my, mx] * motion * slant
            p = min(0.98, p_nom * s)
            o = self.occ_model[my, mx]
            pl = min(0.98, p / max(1e-6, 1 - o))
            cls = sn.view_class(k, ox - r, oy - r)
            n = v.N[cls, my, mx]
            f0 = o + (1 - o) * (1 - pl) ** n
            f1 = o + (1 - o) * (1 - pl) ** (n + 1)
            ratio = f1 / max(f0, 1e-12)
            detected = self.belief[my, mx] * (1 - ratio)
            v.B[:, my, mx] *= ratio
            self.belief[my, mx] -= detected
            v.N[cls, my, mx] = n + 1
            v.exp_low += detected * n_est / max(s, 1e-6)  # expected finds with the nominal sensor
            reward += REWARD_DETECT * detected * max(1, self.n_free)
        self._maybe_go_under(a, k, ex, ey, pending_here)
        return reward

    def _maybe_go_under(self, a, k, ex, ey, pending=False):
        """At the lowest height over an unresolved alert in dense forest (a look from above
        through the leaves does not settle it), check it from under the trees, where the 3D
        planner says that can work (flight_limits.npz); once per alert spot."""
        sc, fam, v = self.scenario, self.fam, self._v
        if not sc.under_canopy or k != 0 or self.mode[a] is not None:
            return
        if not (pending or v.cand_t[ey, ex] >= 0) or self.w.map_canopy[ey, ex] < UNDER_MIN_COVER:
            return
        if (ex, ey) in self.under_tried:
            return
        p = self.under_p[ey, ex] if self.under_p is not None else -1.0
        if p == 0.0:  # no way in under the crowns here (undergrowth, no gap, wires)
            self.under_tried.add((ex, ey))
            return
        v.cand_t[ey, ex] = max(v.cand_t[ey, ex], self.t)  # still open until checked from under the trees
        self.under_tried.add((ex, ey))
        h = fam.alt_m[0]
        secs = h / DESCEND_MPS + UNDER_RUN_M / UNDER_MPS + h / CLIMB_MPS
        if self.under_s is not None and self.under_s[ey, ex] > 0:
            secs = float(self.under_s[ey, ex])  # the planned trip through the real obstacles
        if self.exec_stats:  # stopping, re-planning and holding in the real world take longer
            secs += max(0.0, float(self.exec_stats.get("extra_s", 0.0)))
        self.mode[a] = "under"
        self.mode_t[a] = max(1, int(np.ceil(secs / fam.step_s)))
        self.mode_target[a] = (int(ex), int(ey))
        self.n_under += 1

    def _under_look(self, a) -> float:
        """The look from under the crowns at the alerted spot; then back to the search."""
        v, sc, rng, g = self._v, self.scenario, self.rng, self.grid_size
        ex, ey = self.mode_target[a]
        tx, ty = (int(c) for c in self.true_pos[a])
        self.mode[a] = None
        p_close = self.under_p[ey, ex] if self.under_p is not None and self.under_p[ey, ex] >= 0 else 1.0
        if rng.random() >= p_close:  # could not get within sight of the spot under the trees
            self.n_under_blocked += 1
            return 0.0
        if self.exec_stats:  # what the map did not show: collided, or saw it and turned back
            u = rng.random()
            if u < self.exec_stats.get("p_collide", 0.0):
                self.alive[a] = False
                self.n_crash += 1
                self.n_under_crash += 1
                return 0.0
            if u < self.exec_stats.get("p_collide", 0.0) + self.exec_stats.get("p_abort", 0.0):
                self.n_under_blocked += 1
                return 0.0
        ppl = self.people
        here = np.flatnonzero(~self.found & ~ppl.outside & (ppl.pos[:, 0] == tx) & (ppl.pos[:, 1] == ty))
        seen_i = None
        for i in here:
            hide = 0.6 if ppl.hiding[i] else 0.0
            p = UNDER_DETECT * max(ppl.vis[i], ppl.warm[i]) * (1 - hide) * sc.sensor_scale / 0.75
            if self.reward_pod:
                self._pod[a] = self._pod.get(a, 0.0) + min(0.98, p)
            if rng.random() < min(0.98, p):
                seen_i = int(i)
                break
        false = seen_i is None and rng.random() < 0.3 * self.FA_true[ty, tx]
        if (seen_i is not None or false) and not self._known_spot(v, ex, ey):
            self.n_alarms += 1
            self.n_false_alarms += int(seen_i is None)
            self._send(a, v, dict(cell=(ex, ey), alt=0, true=seen_i, t=self.t, p=0.0, drone=a, under=True,
                                  dismiss=UNDER_FA_DISMISS))
        # the map: this spot was checked closely
        pm = min(0.98, UNDER_DETECT * (v.s_est if sc.adaptive_map else 1.0))
        detected = self.belief[ey, ex] * pm
        v.B[:, ey, ex] *= 1 - pm
        self.belief[ey, ex] -= detected
        v.cand_t[ey, ex] = -1e9
        self.last_searched[ey, ex] = self.t
        return REWARD_DETECT * detected * max(1, self.n_free)

    def _search_extras(self, a, k, Gt, tx, ty, ex, ey, factor, v, reports):
        """Clues and others in the footprint (K7, S8): what the camera sees
        besides the missing person and the background false-alarm rate."""
        g, r, sc, rng = self.grid_size, sn.R, self.scenario, self._ev_rng
        here: dict = {}
        for p, kind in zip(self.others.pos, self.others.kind):
            here.setdefault((int(p[0]), int(p[1])), []).append(kind)
        for tm in self.teams:  # the command post itself is masked out
            c = tuple(int(c) for c in tm["pos"])
            if c != tuple(self.w.base):
                here.setdefault(c, []).append("searcher")
        for oy, ox in zip(*np.nonzero(Gt)):
            cx, cy = tx + ox - r, ty + oy - r
            clues = self.clues.at(cx, cy) if self.clues.pos else []
            kinds = here.get((cx, cy), [])
            if not clues and not kinds:
                continue
            base = Gt[oy, ox] * factor * sc.sensor_scale * (1 - self.occ_true[cy, cx])
            rx, ry = int(np.clip(cx + ex - tx, 0, g - 1)), int(np.clip(cy + ey - ty, 0, g - 1))
            for j in clues:
                if rng.random() < min(0.98, base * self.C_rgb[cy, cx] * ev.CLUE_VIS):
                    self.clues.found[j] = True
                    self.n_clues_seen += 1
                    self._send(a, v, dict(cell=(rx, ry), alt=k, true=None, t=self.t, p=0.0, drone=a, clue=j))
            for kind in kinds:
                la_rgb, la_th, dismiss = ev.LOOKS[kind]
                p = base * max(self.C_rgb[cy, cx] * la_rgb, self.C_th[cy, cx] * la_th)
                if rng.random() < min(0.98, p):
                    reports.add((rx, ry))
                    if self._known_spot(v, rx, ry):
                        continue
                    self.n_alarms += 1
                    self.n_false_alarms += 1
                    self.n_other_alarms += 1
                    self._send(a, v, dict(cell=(rx, ry), alt=k, true=None, t=self.t, p=0.0, drone=a,
                                          other=kind, dismiss=dismiss))

    def _send(self, a, v, item):
        if "base" in self.group_of.get(a, []):
            self.queue.add(item)
        else:
            v.outbox.append(item)

    def _use_clue(self, v, cell):
        """The person passed this cell at some unknown time (K7)."""
        self.n_clues_used += 1
        L = ev.clue_likelihood(self.grid_size, self.fam.cell_m, self.scenario.family, cell)
        ev.reweight(self, v, L, ev.CLUE_FLOOR)

    def _new_report(self):
        """A late phone ping, a witness or a trail register (K7): a rough area
        around where the person really is, or word that they are far away."""
        ppl, fam, rng, g = self.people, self.fam, self._ev_rng, self.grid_size
        todo = np.flatnonzero(~self.found)
        if not len(todo):
            return
        i = int(todo[0])
        self.n_reports += 1
        if not ppl.outside[i]:
            rad = rng.uniform(300, 1500) / fam.cell_m
            c = ppl.pos[i] + rng.normal(0, rad / 2, 2)
            yy, xx = np.mgrid[0:g, 0:g]
            L = np.where(np.hypot(xx - c[0], yy - c[1]) <= rad, 4.0, 1.0)
            ev.reweight(self, self.views["base"], L, 1.0)
        elif np.isfinite(ppl.far[i, 0]):
            ev.reweight(self, self.views["base"], np.ones((g, g)), 4.0)

    def _known_spot(self, v, x, y) -> bool:
        """A detection next to one already queued or found is the same sighting."""
        def near(c):
            return abs(c[0] - x) <= 1 and abs(c[1] - y) <= 1
        if any(near(c) for c in v.finds.values()):
            return True
        items = list(self.queue.items) + [c for c in self.queue.current if c is not None] + list(v.outbox)
        return any(near(it["cell"]) for it in items)

    def _block_ids(self):
        if not hasattr(self, "_bid") or self._bid_ver != id(self.w):
            bid = np.full((self.grid_size, self.grid_size), -1)
            for k, cells in enumerate(rl.building_blocks(self.collapsed | self.w.fly_block)):
                for y, x in cells:
                    bid[y, x] = k
            self._bid, self._bid_ver = bid, id(self.w)
        return self._bid

    def _review(self) -> int:
        """Operators review detections; confirmed finds start the rescue (S5, O1)."""
        sc, fam = self.scenario, self.fam
        found_now = 0
        base_v = self.views["base"]
        for item, verdict in self.queue.step(self.t * fam.step_s / 3600.0):
            x, y = item["cell"]
            i = item["true"]
            if verdict == "confirm" and i is not None and not self.found[i]:
                self.n_under_finds += int(bool(item.get("under")))
                self.found[i] = True
                self.found_times.append(self.t)
                found_now += 1
                base_v.found_low += 1
                self._apply_find(base_v, i, (x, y))
                base_v.cand_t[y, x] = -1e9
                self._dispatch(i)
            elif verdict == "clue":
                self._use_clue(base_v, (x, y))
            elif verdict == "closer_look":
                base_v.cand_t[y, x] = self.t  # stays on the map for a low pass
            else:
                base_v.cand_t[y, x] = -1e9
                if i is not None:  # a missed person: the map wrongly counts the cell as searched
                    base_v.B[:, y, x] *= 0.5
            if sc.adaptive_map:  # M3: the team's estimate of its own sensor
                base_v.s_est = float(np.clip((base_v.found_low + 1.0) / (base_v.exp_low + 1.0), 0.3, 1.2))
        return found_now

    def _dispatch(self, i):
        """After a find: a drone may hold over the person (O1); the nearest free ground team goes (O2)."""
        sc, fam = self.scenario, self.fam
        px, py = self.people.pos[i]
        if self.rng.random() < sc.overwatch:
            free = [a for a in self.agents if self.alive[a] and self.mode[a] is None]
            if len(free) > 1:
                a = min(free, key=lambda b: abs(self.pos[b][0] - px) + abs(self.pos[b][1] - py))
                self.mode[a] = "overwatch"
                self.mode_target[a] = i
                self.mode_t[a] = int(OVERWATCH_MAX_S / fam.step_s)
        idle = [tm for tm in self.teams if tm["task"] is None]
        if idle:
            tm = min(idle, key=lambda t: abs(t["pos"][0] - px) + abs(t["pos"][1] - py))
            tm["task"] = i

    def _teams_step(self) -> int:
        """Ground teams walk (Tobler speed on the true ground) to finds, or search trails (O2)."""
        w, fam, ppl = self.w, self.fam, self.people
        found_now = 0
        for tm in self.teams:
            if tm["task"] is not None:
                i = tm["task"]
                if ppl.outside[i]:
                    tm["task"] = None
                    continue
                goal = tuple(int(v) for v in ppl.pos[i])
            elif not self.scenario.ground_search:
                continue  # responders only: they wait at the base until a find
            else:
                route = self._patrol
                goal = route[tm["patrol"] % len(route)]
                if tuple(int(v) for v in tm["pos"]) == goal:
                    tm["patrol"] += 1
            x, y = (int(v) for v in tm["pos"])
            walked = [(x, y)]
            key = ("team", goal)
            if key not in self._fields:
                self._fields[key] = distance_fields(w.walk, [goal])[0]
            D = self._fields[key]
            tm["prog"] += TEAM_MPS * fam.step_s / fam.cell_m
            while tm["prog"] >= 1.0 and D[y, x] > 0 and np.isfinite(D[y, x]):
                best = min(((x + dx, y + dy) for dx, dy in pp.N8
                            if 0 <= x + dx < w.g and 0 <= y + dy < w.g), key=lambda c: D[c[1], c[0]])
                x, y = best
                dz = abs(w.elev[y, x] - w.elev[int(tm["pos"][1]), int(tm["pos"][0])]) / fam.cell_m
                tm["prog"] -= 1.0 / max(0.2, float(pp.tobler(dz)))
                tm["pos"] = np.array([x, y], float)
                walked.append((x, y))
            tm["prog"] = min(tm["prog"], 1.0)
            self._team_looks(walked)
            here = np.flatnonzero(~ppl.outside & (ppl.pos[:, 0] == x) & (ppl.pos[:, 1] == y))
            for i in here:
                if not self.found[i] and self.rng.random() < TEAM_DETECT * (1 - 0.4 * w.canopy[y, x]) * (
                        2.0 if ppl.signalling[i] else 1.0):
                    self.found[i] = True
                    self.found_times.append(self.t)
                    found_now += 1
                    self._apply_find(self.views["base"], int(i), (x, y))
                if self.found[i] and self.rescued_t[i] < 0:
                    self.rescued_t[i] = self.t
                    if tm["task"] == i:
                        tm["task"] = None
        return found_now

    def _team_looks(self, cells):
        """What a ground team learns on the cells it walks through (O2, K7):
        clues it spots, and (by radio to the base) that nobody it could see is
        there, which lowers the base's map like a drone's look does."""
        base_v, w = self.views["base"], self.w
        for x, y in dict.fromkeys(cells):
            p = TEAM_DETECT * (1 - 0.4 * w.map_canopy[y, x])  # modelled, like the drones' map
            base_v.B[:, y, x] *= (1 - p)
            for j in self.clues.at(x, y):
                if self._ev_rng.random() < ev.TEAM_CLUE:
                    self.clues.found[j] = True
                    self.n_clues_seen += 1
                    self._use_clue(base_v, (x, y))
        cur, self._v = self._v, base_v
        self._mix()
        self._v = cur

    def _move_people(self):
        ppl, fam = self.people, self.fam
        hear = max(1, int(round(pp.HEAR_M / fam.cell_m)))
        near = np.zeros(len(ppl.pos), dtype=bool)
        for a in self.agents:
            if self.alive[a] and self.mode[a] != "landed" and self.alt[a] < 2:
                d = np.abs(ppl.pos - self.true_pos[a]).max(axis=1)
                near |= d <= hear
        hours = self.scenario.hours_lost + self.t * fam.step_s / 3600.0
        before = ppl.pos.copy()
        ppl.step(hours, near, self.water, self.rain_on, hold=self.found)  # found people wait for the team
        self.clues.drop(ppl, (ppl.pos != before).any(axis=1), fam.step_s / 3600.0)
        self.others.step(fam, sn.light_level(self.hour()) < 0.1)

    def _aftershock(self):
        """E3: more buildings collapse; the drones' maps do not know."""
        w, rng = self.w, self.rng
        if self.scenario.family != "urban":
            return
        intact = w.fly_block & ~self.collapsed
        blocks = rl.building_blocks(intact)
        if not blocks:
            return
        for cells in [blocks[j] for j in rng.choice(len(blocks), size=min(2, len(blocks)), replace=False)]:
            for y, x in cells:
                self.collapsed[y, x] = self.unseen[y, x] = True
                w.fly_block[y, x] = False
                w.walk[y, x] = True
                w.struct_h[y, x] = 3.0
        self.G_true = sn.geo_table(w.surface, w.elev, w.walk, self.fam.alt_m, self.fam.foot_r, self.fam.alt_detect)
        self._bid_ver = None

    # ---------------------------------------------------------- observation
    def _obs(self, a, full_map=None, guided=None, team=None) -> np.ndarray:
        self.use_view(a)
        sc, fam, v = self.scenario, self.fam, self._v
        g = self.grid_size
        x, y = (int(c) for c in self.pos[a])
        fresh = self.freshness()
        dens = np.clip(self.belief * max(1, self.n_free), 0.0, 5.0)
        others = [b for b in self.possible_agents[: sc.n_drones] if b != a and self.alive.get(b)]
        mpos = self._mate_positions(a, others)
        mates = np.zeros((g, g))
        for b in others:
            mates[mpos[b][1], mpos[b][0]] = 1.0
        cand = self.candidates()
        blocked = (~v.map.fly | self.nofly).astype(float)
        elev_rel = np.clip((self.w.elev - self.w.elev[y, x]) / 100.0, -1, 1)
        parts = [self._patch(fresh, x, y, 1.0), self._patch(dens, x, y, 0.0), self._patch(blocked, x, y, 1.0),
                 self._patch(mates, x, y, 0.0), self._patch(self.w.map_canopy, x, y, 0.0),
                 self._patch(elev_rel, x, y, 0.0), self._patch(cand, x, y, 0.0),
                 self._coarse(fresh, np.mean), self._coarse(dens, np.mean),
                 np.clip(self._coarse(cand, np.sum), 0, 5), self._coarse(self.nofly.astype(float), np.mean)]
        onehot = np.zeros(3)
        onehot[self.alt[a]] = 1.0
        grp = self.group_of.get(a, [])
        here = [float(np.sum(v.map.G[k, y, x] * self._patch(self.belief * self.C_model, x, y, 0.0).reshape(5, 5)))
                * max(1, self.n_free) for k in range(3)]
        rel = sorted((mpos[b] - self.pos[a] for b in others), key=lambda d: abs(d[0]) + abs(d[1]))[:N_TEAMMATES]
        mate_feats = np.zeros(3 * N_TEAMMATES)
        for i, d in enumerate(rel):
            mate_feats[3 * i:3 * i + 3] = (d[0] / g, d[1] / g, 1.0)
        near = sorted(others, key=lambda b: abs(mpos[b][0] - x) + abs(mpos[b][1] - y))[:N_TEAMMATES]
        mate_alt = np.zeros(N_TEAMMATES)
        for i, b in enumerate(near):
            mate_alt[i] = (self.alt[b] + 1) / 3
        fam_onehot = [float(sc.family == f) for f in ("urban", "wilderness", "flood")]
        bx, by = self.w.base
        scal = [
            x / (g - 1), y / (g - 1), self.battery[a], self.t / fam.max_steps, *onehot,
            self.alt_now[a] / 120.0, float(self.mode[a] is not None), float(self.mode[a] in ("rtb", "landed")),
            float(sc.tilt), float("base" in grp), min(1.0, (self.t - v.synced) / 20.0),
            (len(grp) - ("base" in grp)) / MAX_DRONES,
            np.clip((v.found_low + 1.0) / (v.exp_low + 1.0), 0, 3) / 3, np.clip(v.reports / max(1.0, self.t), 0, 1),
            min(1.0, v.exp_low / max(1, self.n_est)), v.s_est, sc.wind_mps / 10.0, sc.gps_err * 10.0,
            sn.light_level(self.hour()), sn.thermal_contrast(sc.air_c, self.hour(), self.rain_on),
            float(self.rain_on), self.p_outside(), min(1.0, len(self.queue) / 10), min(1.0, len(v.outbox) / 10),
            self._decay, self.n_est / 20.0, len(v.finds) / max(1, self.n_est), *fam_onehot,
            fam.cell_m / 50.0, fam.step_s / 5.0, *np.clip(here, 0, 5), *mate_feats, *mate_alt,
            (abs(x - bx) + abs(y - by)) / g, sc.hours_lost / 12.0, float(self._in_spell(self._gust_spell)),
            self._energy_home(a), float(self.nofly[y, x]),
        ]
        parts.append(np.array(scal, dtype=np.float64))
        if self.plan_feats:
            parts.append(self._plan_features(x, y, a))
        if full_map if full_map is not None else self.full_map:
            own = np.zeros((g, g))
            own[y, x] = 1.0
            parts += [to_res(dens / 5.0).ravel(), to_res(fresh).ravel(), to_res(blocked).ravel(),
                      to_res(np.clip(cand, 0, 1), "max").ravel(), to_res(mates, "max").ravel(),
                      to_res(own, "max").ravel()]
        out = np.concatenate(parts).astype(np.float32)
        return out

    def _plan_features(self, x, y, a=None) -> np.ndarray:
        """For each of 8 directions from the drone: the best expected detections
        per step of travel (and height change) at any cell that way, relative
        to the best anywhere, and how far that cell is.

        plan_feats 1: straight-line distance. plan_feats 2: the rule's own score,
        travel distance around obstacles plus the height change, with blocked
        cells and cells out of radio coverage left out."""
        key = (id(self._v), self.t, self.map_version, self.grid_size)
        if getattr(self, "_fv_key", None) != key:
            self._fv = np.stack([self.footprint_values(k) for k in range(3)])
            self._fv_key = key
        g = self.grid_size
        yy, xx = np.mgrid[0:g, 0:g]
        if self.plan_feats == 2:
            D = self.travel_distance(x, y)
            alt = self.alt[a] if a is not None else 0
            dk = np.abs(np.arange(3) - alt)[:, None, None]
            score = np.where(np.isfinite(D)[None], self._fv / (1.0 + np.where(np.isfinite(D), D, 0)[None] + dk), 0.0)
            blocked = ~self.free
            if getattr(self, "link_mask", None) is not None:
                blocked = blocked | ~self.link_mask
            score = np.where(blocked[None], 0.0, score)
            if a is not None:
                score[alt, y, x] = 0.0  # just searched it from here
            score = score.max(axis=0)
            d = np.where(np.isfinite(D), D, 2 * g)
        else:
            d = np.maximum(np.abs(xx - x), np.abs(yy - y))
            score = (self._fv / (1.0 + d[None])).max(axis=0)
        score[y, x] = 0.0
        top = max(float(score.max()), 1e-12)
        sector = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        out = np.zeros(PLAN_DIM)
        for k in range(8):
            m = sector == k
            m[y, x] = False
            if m.any():
                i = int(np.argmax(np.where(m, score, -1.0)))
                out[k] = score.flat[i] / top
                out[8 + k] = d.flat[i] / g
        return out

    def travel_distance(self, x, y) -> np.ndarray:
        """Shortest-path steps from (x, y) to every cell around obstacles (inf where
        unreachable), cached per map version."""
        key = (x, y, self.map_version, self.grid_size, getattr(self, "grid_epoch", 0))
        cache = self.__dict__.setdefault("_dcache", {})
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = distance_fields(self.free, [(int(x), int(y))], self.scenario.corner_cutting)[0]
        return cache[key]

    def state(self) -> np.ndarray:
        """Critic input (training only): pooled maps at a fixed size, plus the truth."""
        sc, fam, w = self.scenario, self.fam, self.w
        self._v = self.views["base"]
        g = self.grid_size
        M = self._M
        pool = lambda X: (M @ X @ M.T).ravel()  # noqa: E731
        drones = np.zeros((g, g))
        for a, alive in self.alive.items():
            if alive and self.mode[a] != "landed":
                drones[self.true_pos[a][1], self.true_pos[a][0]] += 1.0
        people = np.zeros((g, g))
        for (x, y), f, o in zip(self.people.pos, self.found, self.people.outside):
            if not f and not o:
                people[y, x] += 1.0
        maps = [self.freshness(), np.clip(self.belief * max(1, self.n_free), 0, 5), people * g * g / 16, drones * g * g / 16,
                w.canopy, (w.elev - w.elev.min()) / max(np.ptp(w.elev), 1.0), self.candidates(),
                self.nofly.astype(float), self.unseen.astype(float)]
        scal = [self.t / fam.max_steps, self.found.mean() if len(self.found) else 1.0, sc.n_drones / MAX_DRONES,
                sc.n_people / 20.0, sc.sensor_scale, sc.fa_scale / 3.0, sc.wind_mps / 10.0, sc.gps_err * 10.0,
                sc.comm_range_m / 2500.0, sc.comm_loss * 5.0, float(self.people.outside.mean()),
                sn.light_level(self.hour()), sn.thermal_contrast(sc.air_c, self.hour(), self.rain_on), float(self.rain_on),
                min(1.0, len(self.queue) / 10), *[float(sc.family == f) for f in ("urban", "wilderness", "flood")]]
        return np.concatenate([np.concatenate([pool(m) for m in maps]), scal]).astype(np.float32)

    # ------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        fam = self.fam
        n = len(self.found)
        T = fam.max_steps
        ft = sorted(self.found_times)
        need = int(np.ceil(0.9 * n)) if n else 0
        t90 = ft[need - 1] if need and len(ft) >= need else None
        find_time = (sum(ft) + (n - len(ft)) * T) / n if n else 0.0
        rescue = np.where(self.rescued_t >= 0, self.rescued_t, T)
        inside = ~self.people.outside
        return {
            "survivors": n,
            "found_frac": float(self.found.mean()) if n else 1.0,
            "time_to_90": t90,
            "find_time": float(find_time),
            "find_time_frac": float(find_time / T),
            "found_by_quarter": float(np.mean([t <= T / 4 for t in ft] + [False] * (n - len(ft)))) if n else 1.0,
            "found_by_half": float(np.mean([t <= T / 2 for t in ft] + [False] * (n - len(ft)))) if n else 1.0,
            "rescue_time": float(rescue.mean()) if n else 0.0,
            "rescued_frac": float((self.rescued_t >= 0).mean()) if n else 1.0,
            "found_inside_frac": float(self.found[inside].mean()) if inside.any() else 1.0,
            "outside_frac": float(self.people.outside.mean()) if n else 0.0,
            "clues_seen": self.n_clues_seen,
            "clues_used": self.n_clues_used,
            "other_alarms": self.n_other_alarms,
            "reports": self.n_reports,
            "expanded_min": self.expanded_t * self.fam.step_s / 60.0 if self.expanded_t >= 0 else None,
            "under_checks": self.n_under,
            "under_finds": self.n_under_finds,
            "under_blocked": self.n_under_blocked,
            "under_crashes": self.n_under_crash,
            "emergency_landings": self.n_emergency,
            "bird_losses": self.n_bird,
            "coverage": float(self.visited[self.w.walk].mean()),
            "repeat_frac": self.n_repeats / max(1, self.n_visits),
            "collisions": self.n_collisions,
            "near_misses": self.n_near,
            "shield": self.n_shield,
            "steps": self.t,
            "alarms": self.n_alarms,
            "false_alarms": self.n_false_alarms,
            "op_reviewed": self.queue.reviewed,
            "op_missed": self.queue.missed,
            "bumps": self.n_bumps,
            "crashes": self.n_crash,
            "launch_fails": self.n_launch_fail,
            "swaps": self.n_swaps,
            "linked": self.linked_steps / max(1, self.t),
            "revealed": float(1.0 - self.unseen.sum() / max(1, self.collapsed.sum())),
            "minutes": self.t * fam.step_s / 60.0,
        }


def sn_posture(ppl, i) -> float:
    """P7: injured or dead people lie down; walkers stand; resting people sit or lie."""
    if ppl.injured[i] or ppl.dead[i]:
        return pp.POSTURE["lying"]
    if ppl.signalling[i]:
        return pp.POSTURE["standing"]
    if ppl.resting[i] or ppl.kind[i] == "stay":
        return pp.POSTURE["sitting"]
    return pp.POSTURE["standing"]


def _geo_rect(surface, elev, walk, fam):
    """geo_table on a rectangular crop (pads to square and crops back)."""
    h, wd = surface.shape
    n = max(h, wd)
    S = np.full((n, n), -1e9)
    E = np.zeros((n, n))
    W = np.zeros((n, n), dtype=bool)
    S[:h, :wd], E[:h, :wd], W[:h, :wd] = surface, elev, walk
    G = sn.geo_table(S, E, W, fam.alt_m, fam.foot_r, fam.alt_detect)
    return G[:, :h, :wd]
