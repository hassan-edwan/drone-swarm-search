"""M7 real-world effects: what the earlier environments assumed away.

The drones' map (belief, camera tables, building map) is their *model* of
the world; with these effects on, the world no longer matches it:

1. Sensor quality. True detection is the modelled chance times
   `sensor_scale` (smoke, haze, posture, detector quality), times a
   per-person visibility factor; false alarms are scaled by `fa_scale`.
   Field trials of person detectors report well below lab rates, so the
   scale is drawn from 0.4 to 1.1. The map keeps the nominal values.
2. Out-of-date map. The drones fly with the pre-disaster map. A share of
   buildings has collapsed into low rubble that can be flown over and is
   where trapped people are. A collapse is known only once a drone's camera
   has seen it (discoveries go out on the low-rate telemetry link, so the
   whole team learns them).
3. Radio. A link works within `comm_range` cells with clear line of sight,
   a quarter of that through buildings, and drops for a step with chance
   `comm_loss`. Drones (and the base station at the launch site) in one
   connected group share one map; groups merge maps when they meet again.
4. Position. Wind pushes a drone one cell downwind with chance growing with
   speed; a GPS fix is a cell off now and then, more often next to tall
   buildings (multipath). The camera senses where the drone really is, the
   map is updated where the drone thinks it is.
5. People. With behaviour="mixed" each person has their own behaviour, from
   lost-person research (Koester, Lost Person Behavior): trapped, wander,
   travel in a fixed direction, head to one of several shelters, and
   stop-and-go. One behaviour dominates; the operator's profile, when
   given, describes only that one.
"""

from __future__ import annotations

import numpy as np

from .belief import move_weights
from .obstacles import distance_fields
from .scenario import MOVEMENTS, MovementProfile

# Sampling ranges (per mission)
SENSOR_SCALE = (0.4, 1.1)
FA_SCALE = (0.5, 3.0)
VIS_SD = 0.4
COLLAPSE_FRAC = (0.0, 0.5)
COMM_RANGE = (8.0, 30.0)  # cells (10 m): 80-300 m line of sight, like small-drone mesh radios
COMM_LOSS = (0.0, 0.2)
WIND = (0.0, 10.0)  # m/s; small quadrotors work up to about 10-12 m/s
GPS_ERR = (0.02, 0.08)  # consumer GPS: 2-5 m open sky, 10-20 m in urban canyons

NLOS_FACTOR = 0.25  # radio range through buildings
BASE_ALT_M = 5.0  # base station antenna
SLIP_PER_MPS = 0.03  # chance per step of ending a cell downwind, per m/s of wind
GUST = 0.3  # share of slips in a random direction instead of downwind
URBAN_GPS = 3.0  # GPS error multiplier within one cell of a building taller than the drone
TRAPPED_SHARE = 0.4  # of survivors, placed in collapsed buildings when there are any
COLLAPSE_PRIOR = 3.0  # map mass of newly seen rubble, x mean prior density (collapses are searched first)
REST_P, RESUME_P = 1 / 40, 1 / 25  # stop-and-go walkers
DOMINANT = (0.5, 0.8)  # share of people following the mission's main behaviour


def sample_real(rng: np.random.Generator) -> dict:
    return dict(
        real=True,
        sensor_scale=float(rng.uniform(*SENSOR_SCALE)),
        fa_scale=float(rng.uniform(*FA_SCALE)),
        vis_sd=VIS_SD,
        collapse_frac=float(rng.uniform(*COLLAPSE_FRAC)),
        comm_range=float(rng.uniform(*COMM_RANGE)),
        comm_loss=float(rng.uniform(*COMM_LOSS)),
        wind_mps=float(rng.uniform(*WIND)),
        gps_err=float(rng.uniform(*GPS_ERR)),
        behaviour="mixed",
    )


# --------------------------------------------------------------- map
def building_blocks(obstacles: np.ndarray) -> list[list[tuple[int, int]]]:
    """Connected building blocks as lists of (y, x) cells."""
    g = obstacles.shape[0]
    seen = np.zeros_like(obstacles, dtype=bool)
    blocks = []
    for y, x in zip(*np.nonzero(obstacles)):
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < g and 0 <= nx < g and obstacles[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        blocks.append(cells)
    return blocks


def collapse(rng, obstacles: np.ndarray, frac: float) -> np.ndarray:
    """(g, g) bool: building cells that collapsed (whole blocks)."""
    out = np.zeros_like(obstacles, dtype=bool)
    if frac <= 0:
        return out
    for cells in building_blocks(obstacles):
        if rng.random() < frac:
            for y, x in cells:
                out[y, x] = True
    return out


# --------------------------------------------------------------- radio
def line_clear(heights: np.ndarray, a, b, h_a: float, h_b: float, n: int = 12) -> bool:
    """True if no building between cells a and b (x, y) rises above the radio line."""
    (ax, ay), (bx, by) = a, b
    for t in (np.arange(n) + 0.5) / n:
        x, y = int(np.floor(ax + 0.5 + (bx - ax) * t)), int(np.floor(ay + 0.5 + (by - ay) * t))
        if (x, y) in ((ax, ay), (bx, by)):
            continue
        if heights[y, x] > h_a * (1 - t) + h_b * t:
            return False
    return True


def groups(nodes: dict, heights: np.ndarray, rng, comm_range: float, loss: float) -> list[list]:
    """Connected groups of nodes. nodes: name -> ((x, y), antenna height in m)."""
    names = list(nodes)
    if comm_range <= 0:
        return [names]
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (pa, ha), (pb, hb) = nodes[a], nodes[b]
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if d > comm_range:
                continue
            if d > comm_range * NLOS_FACTOR and not line_clear(heights, pa, pb, ha, hb):
                continue
            if rng.random() < loss:
                continue
            parent[find(a)] = find(b)
    out: dict = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return list(out.values())


# --------------------------------------------------------------- people
class People:
    """Per-person movement on the true map (behaviour="mixed")."""

    def __init__(self, rng, sc, true_free: np.ndarray, survivors: np.ndarray, trapped: np.ndarray,
                 directed: float):
        g = true_free.shape[0]
        n = len(survivors)
        main = sc.movement
        shelters = [main.target] if main.kind == "to_safety" else []
        free_idx = np.flatnonzero(true_free)
        while len(shelters) < int(rng.integers(1, 4)):
            c = int(rng.choice(free_idx))
            shelters.append((c % g, c // g))
        share = rng.uniform(*DOMINANT)
        profiles, self.which = [], np.zeros(n, dtype=int)
        dirs8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        for i in range(n):
            if trapped[i]:
                pr = MovementProfile("still", 0.0)
            elif rng.random() < share:
                pr = MovementProfile(main.kind, main.speed_mps, main.drift_dir,
                                     main.target if main.kind != "to_safety" else shelters[0])
                if main.kind != "still":  # people differ in pace
                    pr = MovementProfile(pr.kind, float(np.clip(main.speed_mps * rng.uniform(0.6, 1.4), 0.2, 2.0)),
                                         pr.drift_dir, pr.target)
            else:
                kind = MOVEMENTS[rng.integers(len(MOVEMENTS))]
                v = 0.0 if kind == "still" else float(rng.uniform(0.3, 1.5))
                d = dirs8[rng.integers(8)] if kind == "drift" else (0, 0)
                t = shelters[rng.integers(len(shelters))] if kind == "to_safety" else (0, 0)
                pr = MovementProfile(kind, v, d, t)
            if pr not in profiles:
                profiles.append(pr)
            self.which[i] = profiles.index(pr)
        targets = sorted({pr.target for pr in profiles if pr.kind == "to_safety"})
        dist = dict(zip(targets, distance_fields(true_free, targets))) if targets else {}
        W = np.stack([move_weights(pr, true_free, directed, dist.get(pr.target) if pr.kind == "to_safety" else None)
                      for pr in profiles])
        self.cum = np.cumsum(W, axis=1)  # (P, 9, g, g)
        self.resting = np.zeros(n, dtype=bool)
        self.profiles = profiles
        self.shelters = shelters

    def step(self, rng, survivors: np.ndarray, found: np.ndarray, moves: np.ndarray):
        flip = rng.random(len(survivors))
        self.resting = np.where(self.resting, flip >= RESUME_P, flip < REST_P)
        idx = np.flatnonzero(~found & ~self.resting)
        if not len(idx):
            return
        xs, ys = survivors[idx, 0], survivors[idx, 1]
        cum = self.cum[self.which[idx], :, ys, xs]  # (n, 9)
        m = np.minimum((cum < rng.random(len(idx))[:, None]).sum(axis=1), 8)
        survivors[idx] = survivors[idx] + moves[m]


MOVES8 = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
