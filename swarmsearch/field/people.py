"""People and the drones' prior about them (register P1-P8, K1-K4, K6).

Truth: each person has a behaviour (lost-person research [1][4][13]):
stay, wander, trail (follow trails), downhill (follow drainages), uphill
(view enhancing), direction (straight line), backtrack (towards the
last-seen point), shelter (towards a known refuge), drift (carried by
water), or "none" (a persistent random walk that follows no modelled
pattern). Walking speed follows Tobler's hiking function on slopes; people
tire (mobility halves every `fatigue_h` hours), mostly stop at night, rest
and resume, sometimes change behaviour, and are sometimes injured and
immobile. Leaving the map is permanent (they are outside the search area).

Each person also has a responsiveness (P1): unresponsive (ignores drones),
signals (when a drone is close: stands, waves, moves into the open), or
evasive (hides under cover and moves away); a posture (standing, sitting,
lying), clothing visibility and warmth (hypothermic or deceased people show
little thermal contrast).

Prior (the drones' map): distance rings around the *reported* last-seen
point (which can be off or wrong), weighted by terrain features, with mass
left outside the map; a rough phone-ping area when there is one; a bank of
movement hypotheses built on the drones' (imperfect) map.
"""

from __future__ import annotations

import numpy as np

from ..obstacles import distance_fields
from . import lpb
from .sensing import light_level

N8 = np.array([(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)])
MOVES9 = np.vstack([N8, [(0, 0)]])
STAY = 8

BEHAVIOURS = ("stay", "wander", "trail", "downhill", "uphill", "direction", "backtrack", "shelter", "drift", "none")
# Behaviour mix by subject category (wilderness) or family. Illustrative
# shares in the spirit of Koester's categories [1][13]; replace with ISRID.
MIX = {
    "hiker": dict(trail=0.3, downhill=0.15, uphill=0.1, direction=0.15, backtrack=0.1, stay=0.1, wander=0.1),
    "hunter": dict(direction=0.25, downhill=0.2, trail=0.15, uphill=0.1, stay=0.15, wander=0.15),
    "child_7_12": dict(wander=0.35, trail=0.2, stay=0.25, downhill=0.1, direction=0.1),
    "dementia": dict(direction=0.45, wander=0.2, downhill=0.15, trail=0.1, stay=0.1),
    "despondent": dict(stay=0.35, uphill=0.25, trail=0.15, wander=0.15, direction=0.1),
    "urban": dict(stay=0.5, shelter=0.3, wander=0.2),
    "flood": dict(uphill=0.45, shelter=0.3, stay=0.25),
}
POSTURE = {"standing": 1.0, "sitting": 0.8, "lying": 0.65}
WALK_MPS = (0.4, 1.4)
REST_S, RESUME_S = 900.0, 600.0  # mean seconds walking before a rest / resting before walking on
SWITCH_S = 3600.0  # mean seconds before a person changes behaviour
HEAR_M = 150.0  # people notice a drone within this distance
TRAPPED_SHARE = 0.4  # urban: share of people in collapsed buildings (when there are any)


def tobler(slope_signed: np.ndarray | float) -> np.ndarray | float:
    """Walking speed relative to flat ground (Tobler's hiking function)."""
    return np.exp(-3.5 * np.abs(slope_signed + 0.05)) / np.exp(-3.5 * 0.05)


def _norm(mix: dict) -> dict:
    s = sum(mix.values())
    return {k: v / s for k, v in mix.items()}


def behaviour_mix(sc) -> dict:
    if sc.family == "wilderness":
        return _norm(MIX[sc.category])
    return _norm(MIX[sc.family])


# ----------------------------------------------------------------- prior
def prior(w, sc, rng, ping=None) -> tuple[np.ndarray, float, tuple[int, int]]:
    """The drones' prior over cells, the mass they put outside the map, and
    the last-seen point they were told."""
    g, cell_m = w.g, w.cell_m
    walk = w.map_walk
    if sc.family == "wilderness":
        err = rng.normal(0, sc.ipp_err_m / cell_m, 2)
        if sc.ipp_wrong:
            ang = rng.uniform(0, 2 * np.pi)
            err = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(600, 1500) / cell_m
        rep = (int(np.clip(round(w.ipp[0] + err[0]), 0, g - 1)), int(np.clip(round(w.ipp[1] + err[1]), 0, g - 1)))
        dens, out = ring_density(w, sc, rep)
    elif sc.family == "urban":
        rep = w.base
        dens = walk * (1.0 + 1.0 * (w.extra["cover"] == 1))  # rubble by buildings: people trapped nearby
        for sx, sy in w.shelters:
            yy, xx = np.mgrid[0:g, 0:g]
            dens = dens * (1.0 + 1.0 * (np.hypot(xx - sx, yy - sy) < 3))
        out = 0.03
    else:  # flood: along the river and on roofs and high ground
        rep = w.base
        dens = walk * (1.0 + 2.0 * w.water + 1.0 * (w.map_struct_h > 0))
        out = 0.05
    if ping is not None:  # P2: a rough area from a phone ping
        yy, xx = np.mgrid[0:g, 0:g]
        (px, py), rad = ping
        dens = dens * np.where(np.hypot(xx - px, yy - py) <= rad, 4.0, 1.0)
    dens = dens / max(dens.sum(), 1e-12) * (1.0 - out)
    return dens, out, rep


def ring_density(w, sc, rep) -> tuple[np.ndarray, float]:
    """Wilderness prior around the reported point: ISRID distance rings for the
    category [JSAR20][K20][NZSAR], terrain multipliers [J16] on the drones' map,
    and the dispersion angle around the direction of travel when it is known.
    Per-cell weights (unnormalised) and the chance of being beyond the map."""
    cat = lpb.CATEGORIES[sc.category]
    dens, out = lpb.ring_probability(w.g, w.cell_m, rep, cat["dist_km"])
    dens = dens * lpb.terrain_multiplier(w.map_trail, w.stream, w.elev) * w.map_walk
    if sc.dot_known:
        dens = dens * lpb.angle_weight(w.g, rep, sc.dot_deg, cat["disp_deg"])
    return dens, max(out, 0.05)


# -------------------------------------------------- movement hypotheses
def _main_moves(kind, w, walk, trail, target_field=None, heading=None) -> np.ndarray:
    """(8, g, g) share of the directed move per neighbour for one behaviour."""
    g = w.g
    out = np.zeros((8, g, g))
    P = lambda a, fill: np.pad(a, 1, constant_values=fill)  # noqa: E731
    if kind in ("downhill", "uphill", "backtrack", "shelter", "trail"):
        if kind == "downhill":
            pot = w.elev
        elif kind == "uphill":
            pot = -w.elev
        else:
            pot = target_field
        Pp = P(np.where(walk, pot, np.inf), np.inf)
        nb = np.stack([Pp[1 + dy:1 + dy + g, 1 + dx:1 + dx + g] for dx, dy in N8])
        best = nb.min(axis=0)
        better = np.isfinite(best) & (best < pot)
        pick = np.argmin(nb, axis=0)
        for m in range(8):
            out[m] = (pick == m) & better
        if kind == "trail":  # on a trail: walk along it (either way)
            Pt = P(trail, False)
            on = np.stack([Pt[1 + dy:1 + dy + g, 1 + dx:1 + dx + g] for dx, dy in N8]).astype(float)
            n_on = on.sum(axis=0)
            along = np.where(n_on > 0, on / np.maximum(n_on, 1), 0.0)
            out = np.where(trail[None] & (n_on > 0)[None], along, out)
    elif kind == "direction":
        m = int(np.flatnonzero((N8 == heading).all(axis=1))[0])
        out[m] = 1.0
    return out


def move_weights(kind, w, move_p: np.ndarray, directed: float, walk, trail, target_field=None,
                 heading=None, open_edges: bool = True) -> np.ndarray:
    """(9, g, g): chance of each move per cell. Moves into cells people
    cannot enter become staying put; moves off the map are kept (the mass
    leaves the search area) when open_edges."""
    g = w.g
    W = np.zeros((9, g, g))
    if kind == "stay":
        W[STAY] = 1.0
        return W
    main = _main_moves(kind, w, walk, trail, target_field, heading) if kind != "wander" else np.zeros((8, g, g))
    has_main = main.sum(axis=0) > 0
    rand = np.where(has_main, 1 - directed, 1.0) / 8.0
    W[:8] = move_p * (rand[None] + directed * main)
    yy, xx = np.mgrid[0:g, 0:g]
    for m, (dx, dy) in enumerate(N8):
        nx, ny = xx + dx, yy + dy
        inside = (nx >= 0) & (nx < g) & (ny >= 0) & (ny < g)
        ok = inside.copy()
        ok[inside] = walk[ny[inside], nx[inside]]
        keep = ok | (~inside & open_edges)
        W[m] = np.where(keep, W[m], 0.0)
    W[STAY] = 1.0 - W[:8].sum(axis=0)
    W[:, ~walk] = 0.0
    W[STAY, ~walk] = 1.0
    return W


def hypothesis_bank(w, sc, rep_ipp, rng) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Movement hypotheses on the drones' map: names, (K, 9, g, g) weights,
    prior weights, and each one's move probability per step."""
    fam = sc.fam
    walk, trail = w.map_walk, w.map_trail
    sl = _slope(w)
    base_p = 0.8 * fam.step_s / fam.cell_m  # a typical walker, before slope
    hours = sc.hours_lost
    fatigue = 0.5 ** (hours / 6.0)
    move_p = np.clip(base_p * tobler(sl) * fatigue, 0.0, 1.0)
    mix = behaviour_mix(sc)
    if sc.profile_told:  # K6: the operator names one behaviour (sometimes wrongly)
        told = max(mix, key=mix.get) if rng.random() < 0.7 else list(mix)[rng.integers(len(mix))]
        mix = _norm({k: v * (3.0 if k == told else 1.0) for k, v in mix.items()})
    names, Ws, prior_w, mps = [], [], [], []

    def add(name, kind, weight, **kw):
        names.append(name)
        Ws.append(move_weights(kind, w, move_p, 0.75, walk, trail, **kw))
        prior_w.append(weight)
        mps.append(0.0 if kind == "stay" else float(move_p[walk].mean()))

    for kind, p in mix.items():
        if kind == "direction":
            for d in N8:
                add(f"direction{tuple(d)}", "direction", p / 8, heading=d)
        elif kind == "backtrack":
            add(kind, kind, p, target_field=distance_fields(walk, [rep_ipp])[0])
        elif kind == "shelter":
            for s in w.shelters:
                if walk[s[1], s[0]]:
                    add(f"shelter{s}", kind, p / len(w.shelters), target_field=distance_fields(walk, [s])[0])
        elif kind == "trail":
            if trail.any():
                add(kind, kind, p, target_field=_dist_to_mask(walk, trail))
            else:
                add(kind, "wander", p)
        else:
            add(kind, kind, p)
    if sc.family == "flood":  # water carries people downstream
        Wd = _drift_weights(w, fam)
        names.append("drift")
        Ws.append(Wd)
        prior_w.append(0.2)
        mps.append(0.5)
    pw = np.array(prior_w, float)
    return names, np.stack(Ws), pw / pw.sum(), np.array(mps)


def _drift_weights(w, fam) -> np.ndarray:
    g = w.g
    W = np.zeros((9, g, g))
    p = np.clip(0.8 * fam.step_s / fam.cell_m, 0, 1)  # current ~0.8 m/s
    W[4] = np.where(w.water, p, 0.0)  # (+1, 0): downstream
    W[4, :, g - 1] = np.where(w.water[:, g - 1], p, 0.0)  # off the map downstream
    W[STAY] = 1.0 - W[4]
    return W


def _dist_to_mask(walk, mask) -> np.ndarray:
    """Walking distance from every cell to the nearest cell of `mask`."""
    g = walk.shape[0]
    D = np.where(mask & walk, 0.0, np.inf)
    N = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    while True:
        P = np.pad(D, 1, constant_values=np.inf)
        best = D.copy()
        for dx, dy in N:
            np.minimum(best, P[1 + dy:1 + dy + g, 1 + dx:1 + dx + g] + 1.0, out=best)
        best[~walk] = np.inf
        if np.array_equal(best, D):
            return D
        D = best


def _slope(w) -> np.ndarray:
    gy, gx = np.gradient(w.elev, w.cell_m)
    return np.hypot(gx, gy)


# ------------------------------------------------------------------- truth
class People:
    def __init__(self, w, sc, rng):
        self.w, self.sc, self.rng = w, sc, rng
        fam = sc.fam
        self.fam = fam
        g = w.g
        n = sc.n_people
        mix = behaviour_mix(sc)
        self.pos = np.zeros((n, 2), dtype=int)
        self.outside = np.zeros(n, dtype=bool)
        # Where someone outside the area really is, in this grid's cells (NaN: unknown,
        # e.g. beyond any area the search could grow to). A grown area can take them in (K2).
        self.far = np.full((n, 2), np.nan)
        self.kind = [""] * n
        self.heading = np.zeros((n, 2), dtype=int)
        self.speed = rng.uniform(*WALK_MPS, n)
        self.directed = rng.uniform(0.6, 0.9, n)
        self.injured = rng.random(n) < sc.p_injured
        self.resting = np.zeros(n, dtype=bool)
        self.trapped = np.zeros(n, dtype=bool)
        cat = lpb.CATEGORIES.get(sc.category, lpb.CATEGORIES["hiker"])
        p_ev = sc.p_evasive if sc.p_evasive is not None else cat["evasive"]
        p_un = sc.p_unresponsive if sc.p_unresponsive is not None else max(0.0, 1.0 - cat["respond"] - p_ev)
        r = rng.random(n)
        self.resp = np.where(r < p_un, "unresponsive", np.where(r < p_un + p_ev, "evasive", "signals")).astype(object)
        if sc.force == "unresponsive":
            self.resp[:] = "unresponsive"
        elif sc.force == "evasive":
            self.resp[:] = "evasive"
        self.vis = np.clip(np.exp(rng.normal(0.0, 0.4, n)), 0.3, 2.0)  # clothing contrast (P7)
        hours = sc.hours_lost
        if sc.family == "wilderness":  # P8: published survival curves by category [K20]
            p_dead = 1.0 - lpb.survival(hours / 24.0, cat["survive_d"])
        else:
            p_dead = min(0.3, 0.01 * hours)
        dead = rng.random(n) < p_dead
        # P4: how long each person keeps moving before stopping for good [K20][K06]
        self.mobile_h = np.array([lpb.sample_mobile_hours(rng, cat["mobile_h"]) for _ in range(n)])
        hypo = rng.random(n) < sc.hypothermic
        self.warm = np.where(dead, 0.15, np.where(hypo, 0.5, 1.0))
        self.dead = dead
        self.injured |= dead
        self.resp[dead] = "unresponsive"
        self.signalling = np.zeros(n, dtype=bool)
        self.hiding = np.zeros(n, dtype=bool)
        self.occ: list[dict] = [dict() for _ in range(n)]  # S2: hidden per viewing direction
        self.walk = w.walk
        self._fields = {}
        kinds = list(mix)
        # Place people (P5, K2, K4) and give them behaviours (P3).
        if sc.family == "wilderness":
            dens, p_off = None, 0.0  # placed by distance and direction below
        elif sc.family == "urban":
            dens = w.walk.astype(float)
            p_off = 0.02
        else:
            dens = w.walk * (1.0 + 2.0 * w.water + 2.0 * (w.struct_h > 0))
            p_off = 0.03
        p_off = min(0.95, p_off + sc.p_outside)
        if sc.force == "outside":
            p_off = 1.0
        if dens is not None:
            dens = dens / max(dens.sum(), 1e-12)
            cells = np.flatnonzero(dens > 0)
        i = 0
        while i < n:
            size = min(sc.group_size, n - i)
            if rng.random() < p_off:
                self.outside[i:i + size] = True
                c = int(w.ipp[1] * g + w.ipp[0])
            elif dens is None:
                c, out, far = self._place_wild(cat, mix, rng)
                self.outside[i:i + size] = out
                if out:
                    self.far[i:i + size] = far
            else:
                c = int(rng.choice(cells, p=dens[np.unravel_index(cells, dens.shape)] / dens.flat[cells].sum()))
            cy, cx = divmod(c, g)
            kind = kinds[rng.choice(len(kinds), p=[mix[k] for k in kinds])]
            for j in range(i, i + size):
                ox, oy = (0, 0) if j == i else N8[rng.integers(8)]
                x, y = int(np.clip(cx + ox, 0, g - 1)), int(np.clip(cy + oy, 0, g - 1))
                if not w.walk[y, x]:
                    x, y = cx, cy
                self.pos[j] = (x, y)
                self.kind[j] = kind
                self.heading[j] = N8[rng.integers(8)]
            i += size
        if sc.family == "urban" and w.collapsed is not None and w.collapsed.any():
            cc = np.argwhere(w.collapsed)
            for j in range(n):
                if not self.outside[j] and rng.random() < TRAPPED_SHARE:
                    y, x = cc[rng.integers(len(cc))]
                    self.pos[j] = (x, y)
                    self.trapped[j] = True
                    self.kind[j] = "stay"
        if sc.family == "flood":
            for j in range(n):
                x, y = self.pos[j]
                if w.water[y, x]:
                    self.kind[j] = "drift"
                elif w.struct_h[y, x] > 0:
                    self.kind[j] = "stay"  # stranded on a roof
        for j in range(n):  # already stopped for good before the drones launch
            if hours >= self.mobile_h[j] and self.kind[j] != "drift":
                self.kind[j] = "stay"
        p_none = 1.0 if sc.force == "off_model" else sc.p_off_model
        for j in range(n):
            if not self.trapped[j] and self.kind[j] not in ("drift",) and rng.random() < p_none:
                self.kind[j] = "none"
        self.mix = mix

    def _place_wild(self, cat, mix, rng) -> tuple[int, bool, tuple[float, float]]:
        """Where a lost person is at launch: an ISRID distance for the category
        (spread ×d50_scale, K4), scaled by how much of their moving time has
        passed, in a direction drawn from the dispersion angle around their
        direction of travel, favouring trails, streams and low and high points
        [J16]. Off the map means outside the search area (K2)."""
        w, sc = self.w, self.sc
        g = w.g
        q = tuple(d * sc.d50_scale for d in cat["dist_km"])
        tm = lpb.terrain_multiplier(w.trail, w.stream, w.elev)
        mh = lpb.sample_mobile_hours(rng, cat["mobile_h"])
        frac = min(1.0, sc.hours_lost / max(mh, 1e-3))
        last = None
        for _ in range(300):
            d = lpb.sample_distance(rng, q) * frac * 1000.0 / w.cell_m
            dev = min(180.0, lpb.sample_distance(rng, cat["disp_deg"])) * rng.choice([-1, 1])
            ang = np.radians(sc.dot_deg + dev)
            x, y = int(round(w.ipp[0] + d * np.cos(ang))), int(round(w.ipp[1] - d * np.sin(ang)))
            if not (0 <= x < g and 0 <= y < g):
                return int(w.ipp[1] * g + w.ipp[0]), True, (x, y)
            if not w.walk[y, x]:
                continue
            last = (x, y)
            if rng.random() < tm[y, x] / tm.max():
                break
        x, y = last if last is not None else w.ipp
        return int(y * g + x), False, (np.nan, np.nan)

    # -------------------------------------------------------------- fields
    def _field(self, key):
        if key not in self._fields:
            w = self.w
            if key == "ipp":
                f = distance_fields(w.walk, [w.ipp])[0]
            elif key == "trail":
                f = _dist_to_mask(w.walk, w.trail) if w.trail.any() else np.zeros((w.g, w.g))
            else:
                f = distance_fields(w.walk, [key])[0]
            self._fields[key] = f
        return self._fields[key]

    # ---------------------------------------------------------------- step
    def step(self, t_hours: float, drones_near: np.ndarray, water: np.ndarray, rain: bool, hold=None):
        """One step for everyone. drones_near[i]: a drone is within hearing range."""
        rng, w, sc, fam = self.rng, self.w, self.sc, self.fam
        g = w.g
        dt = fam.step_s
        dark = light_level(sc.hour + t_hours - sc.hours_lost) < 0.1
        self.signalling[:] = False
        self.hiding[:] = False
        for i in range(len(self.pos)):
            if self.outside[i] or self.injured[i] or self.trapped[i] or (hold is not None and hold[i]):
                if drones_near[i] and self.resp[i] == "signals" and not self.dead[i]:
                    self.signalling[i] = True  # waves where they are
                continue
            x, y = self.pos[i]
            if water[y, x]:
                self.kind[i] = "drift"
                if rng.random() < dt / 120.0:  # most people reach the bank or something to hold on to
                    dry = [(x + dx, y + dy) for dx, dy in N8 if 0 <= x + dx < g and 0 <= y + dy < g
                           and not water[y + dy, x + dx]]
                    if dry:
                        self.pos[i] = dry[rng.integers(len(dry))]
                        self.kind[i] = "uphill"
                        self.occ[i] = {}
                        continue
            elif self.kind[i] == "drift":
                self.kind[i] = "uphill"  # washed ashore: climbs out
            # stop-and-go, behaviour changes
            if self.resting[i]:
                if rng.random() < dt / RESUME_S:
                    self.resting[i] = False
            elif rng.random() < dt / REST_S:
                self.resting[i] = True
            if rng.random() < dt / SWITCH_S and self.kind[i] not in ("drift",):
                kinds = list(self.mix)
                self.kind[i] = kinds[rng.choice(len(kinds), p=[self.mix[k] for k in kinds])]
            # reacting to a drone (P1)
            react = None
            if drones_near[i]:
                if self.resp[i] == "signals":
                    self.signalling[i] = True
                    react = "open"
                elif self.resp[i] == "evasive":
                    self.hiding[i] = True
                    react = "cover"
            if t_hours >= self.mobile_h[i] and self.kind[i] != "drift":
                self.kind[i] = "stay"  # stopped for good (P4)
            mobility = 0.5 ** (t_hours / sc.fatigue_h) * (0.15 if dark else 1.0) * (0.5 if rain else 1.0)
            if self.kind[i] == "drift":
                mobility = 1.0
            if (self.resting[i] or self.kind[i] == "stay") and react is None:
                continue
            if self.kind[i] == "stay" and react is not None and rng.random() > 0.3:
                continue
            nxt = self._choose(i, x, y, react, water)
            if nxt is None:
                continue
            nx, ny = nxt
            if self.kind[i] == "drift":
                p_move = min(1.0, 0.5 * dt / fam.cell_m)  # carried at about 0.5 m/s
            else:
                dz = 0.0
                if 0 <= nx < g and 0 <= ny < g:
                    dz = (w.elev[ny, nx] - w.elev[y, x]) / (fam.cell_m * np.hypot(nx - x, ny - y))
                p_move = min(1.0, self.speed[i] * dt / fam.cell_m * float(tobler(dz)) * mobility)
            if rng.random() >= p_move:
                continue
            if not (0 <= nx < g and 0 <= ny < g):
                self.outside[i] = True  # walked off the map: outside the search area (K2)
                self.far[i] = (nx, ny)
                continue
            self.heading[i] = (nx - x, ny - y)
            self.pos[i] = (nx, ny)
            self.occ[i] = {}  # new spot, new hiding geometry (S2)

    def _choose(self, i, x, y, react, water):
        """Next cell (may be off the map), or None to stay."""
        rng, w = self.rng, self.w
        g = w.g
        kind = self.kind[i]
        cand = []
        for dx, dy in N8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < g and 0 <= ny < g):
                cand.append((nx, ny, np.inf if kind == "drift" else 0.0))
            elif self.walk[ny, nx]:
                cand.append((nx, ny, 0.0))
        if not cand:
            return None
        if react == "open":  # into the open, to be seen
            return min(cand, key=lambda c: w.canopy[c[1], c[0]] if 0 <= c[0] < g and 0 <= c[1] < g else 9)[:2]
        if react == "cover":
            return max(cand, key=lambda c: w.canopy[c[1], c[0]] if 0 <= c[0] < g and 0 <= c[1] < g else -9)[:2]
        if kind == "drift":
            nx, ny = x + 1, y + int(rng.integers(-1, 2)) * (rng.random() < 0.3)
            return (nx, ny)
        if rng.random() >= self.directed[i] or kind == "wander":
            return cand[rng.integers(len(cand))][:2]
        if kind == "none":  # persistent random walk
            if rng.random() < 0.3:
                self.heading[i] = N8[rng.integers(8)]
            nx, ny = x + self.heading[i][0], y + self.heading[i][1]
            return (nx, ny) if not (0 <= nx < g and 0 <= ny < g) or self.walk[ny, nx] else cand[rng.integers(len(cand))][:2]
        if kind == "direction":
            nx, ny = x + self.heading[i][0], y + self.heading[i][1]
            if not (0 <= nx < g and 0 <= ny < g) or self.walk[ny, nx]:
                return (nx, ny)
            return cand[rng.integers(len(cand))][:2]
        inside = [c for c in cand if 0 <= c[0] < g and 0 <= c[1] < g]
        if not inside:
            return None
        if kind == "downhill":
            pot = lambda c: w.elev[c[1], c[0]]  # noqa: E731
        elif kind == "uphill":
            pot = lambda c: -w.elev[c[1], c[0]]  # noqa: E731
        elif kind == "backtrack":
            f = self._field("ipp")
            pot = lambda c: f[c[1], c[0]]  # noqa: E731
        elif kind == "shelter":
            s = w.shelters[i % len(w.shelters)]
            f = self._field(tuple(s))
            pot = lambda c: f[c[1], c[0]]  # noqa: E731
        elif kind == "trail":
            on = [c for c in inside if w.trail[c[1], c[0]]]
            if w.trail[y, x] and on:
                ahead = [c for c in on if (c[0] - x, c[1] - y) == tuple(self.heading[i])]
                return (ahead or on)[0][:2] if ahead or rng.random() < 0.5 else on[rng.integers(len(on))][:2]
            f = self._field("trail")
            pot = lambda c: f[c[1], c[0]]  # noqa: E731
        else:
            return None
        here = pot((x, y))
        best = min(inside, key=pot)
        if pot(best) >= here:
            if kind == "uphill":
                self.kind[i] = "stay"  # reached a high point: waits there (view enhancing)
            return None
        return best[:2]
