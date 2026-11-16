"""Evidence the map gets from outside the drones' own looks, and things in the
area that are not the missing person (register K7, S8, O2).

Clues (K7). Lost people leave things behind: a pack, clothing, footprints at a
stream crossing. Clues are left along the way they walked before launch and
while they keep moving. A drone or a ground team that comes across one reports
it, and an operator confirms it. The map then learns that the person passed
there at some unknown time: every cell is re-weighted by how reachable it is
from the clue, and so is each movement hypothesis.

Reports (K7). Partway through, a new report can arrive: a late phone ping, a
witness, a trail register. It gives a rough area around where the person
really is (or says they are beyond the map).

Others in the area (S8). Other walkers, campers, animals and the searchers
themselves are what drone cameras actually flag. They cluster on trails and
near the command post, which is also where lost people are likely to be, so
they are modelled as moving things in places, not only as a flat rate. An
operator dismisses most animals and the team's own searchers (whose
positions are tracked) at once, but often wants a closer look at a walker.

Rates marked "assumed" have no published source and are randomised per
mission.
"""

from __future__ import annotations

import numpy as np

from . import people as pp

CLUE_VIS = 0.5  # a clue against a person, for the colour camera (assumed); clues show no heat
CLUE_CONFIRM = 0.8  # an operator accepts a real clue (assumed)
CLUE_FLOOR = 0.1  # likelihood far from a clue (the clue may be old or not theirs)
TEAM_CLUE = 0.4  # a ground team spots a clue on a cell it walks through (assumed)
# How much each kind looks like a person to the colour and thermal detectors, and
# how often the operator dismisses it from the image alone (assumed).
LOOKS = {"walker": (0.9, 1.0, 0.3), "animal": (0.25, 0.6, 0.85), "searcher": (0.9, 1.0, 0.95)}


def reach_m(family: str) -> float:
    """Typical distance a person covers after leaving a clue (km scale in the wilderness)."""
    return 1500.0 if family == "wilderness" else 60.0 if family == "urban" else 150.0


class Clues:
    def __init__(self, w, sc, ppl, rng):
        self.rng, self.sc = rng, sc
        self.pos: list[tuple[int, int]] = []
        self.found: list[bool] = []
        rate = sc.clue_rate
        if rate <= 0:
            return
        for i in range(len(ppl.pos)):  # left along the way before launch
            moved_h = min(sc.hours_lost, ppl.mobile_h[i])
            end = ppl.far[i] if ppl.outside[i] and np.isfinite(ppl.far[i, 0]) else ppl.pos[i]
            if ppl.outside[i] and not np.isfinite(ppl.far[i, 0]):
                continue
            for _ in range(rng.poisson(rate * moved_h)):
                f = rng.uniform(0.05, 1.0)
                c = np.array(w.ipp) + f * (np.asarray(end, float) - np.array(w.ipp)) + rng.normal(0, 1.0, 2)
                x, y = int(round(c[0])), int(round(c[1]))
                if 0 <= x < w.g and 0 <= y < w.g and w.walk[y, x]:
                    self.add((x, y))

    def add(self, cell):
        self.pos.append((int(cell[0]), int(cell[1])))
        self.found.append(False)

    def drop(self, ppl, moved: np.ndarray, dt_h: float):
        """People who moved this step may leave something behind."""
        if self.sc.clue_rate <= 0:
            return
        for i in np.flatnonzero(moved & ~ppl.outside):
            if self.rng.random() < self.sc.clue_rate * dt_h:
                self.add(ppl.pos[i])

    def at(self, x, y) -> list[int]:
        return [j for j, c in enumerate(self.pos) if c == (x, y) and not self.found[j]]

    def shift(self, ox, oy):
        self.pos = [(x + ox, y + oy) for x, y in self.pos]


class Others:
    """Walkers and campers, animals; the env adds its ground teams as searchers."""

    def __init__(self, w, sc, rng):
        self.rng, self.sc = rng, sc
        n_w, n_a = sc.n_walkers, sc.n_animals
        kinds = ["walker"] * n_w + ["animal"] * n_a
        self.kind = np.array(kinds, dtype=object)
        self.pos = np.zeros((len(kinds), 2), dtype=int)
        self.heading = np.zeros((len(kinds), 2), dtype=int)
        self.camped = np.zeros(len(kinds), dtype=bool)
        trail = np.argwhere(w.trail & w.walk) if w.trail.any() else np.argwhere(w.walk)
        walk = np.argwhere(w.walk)
        bx, by = w.base
        for j, k in enumerate(kinds):
            if k == "walker":  # on trails, more of them near the trailhead
                d = np.hypot(trail[:, 1] - bx, trail[:, 0] - by)
                p = np.exp(-d / max(4.0, w.g / 6))
                y, x = trail[rng.choice(len(trail), p=p / p.sum())]
                self.camped[j] = sc.light == "night" or rng.random() < 0.2
            else:
                y, x = walk[rng.integers(len(walk))]
            self.pos[j] = (x, y)
            self.heading[j] = pp.N8[rng.integers(8)]
        self.w = w

    def step(self, fam, dark: bool):
        w, rng = self.w, self.rng
        for j in range(len(self.pos)):
            if self.camped[j] or (dark and self.kind[j] == "walker"):
                continue
            speed = 1.2 if self.kind[j] == "walker" else 0.4
            if rng.random() >= min(1.0, speed * fam.step_s / fam.cell_m):
                continue
            x, y = self.pos[j]
            opts = []
            for dx, dy in pp.N8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w.g and 0 <= ny < w.g and w.walk[ny, nx]:
                    score = 1.0 + 3.0 * (dx * self.heading[j][0] + dy * self.heading[j][1] > 0)
                    if self.kind[j] == "walker":
                        score *= 6.0 if w.trail[ny, nx] else 0.3
                    opts.append(((nx, ny), score))
            if not opts:
                continue
            p = np.array([s for _, s in opts])
            (nx, ny) = opts[rng.choice(len(opts), p=p / p.sum())][0]
            self.heading[j] = (nx - x, ny - y)
            self.pos[j] = (nx, ny)

    def shift(self, ox, oy, w):
        self.pos += np.array([ox, oy])
        self.w = w


def reweight(env, v, L: np.ndarray, L_out: float):
    """Bayes update of one map copy with a likelihood over cells (and for
    being outside): each hypothesis's map is re-weighted and renormalised to
    its old mass, and the hypothesis weights move by how well each explains
    the evidence. Unlike a look, evidence of this kind moves probability; it
    does not remove it."""
    m0 = v.B.sum(axis=(1, 2)) + v.out
    m1 = (v.B * L[None]).sum(axis=(1, 2)) + v.out * L_out
    z = m0 / np.maximum(m1, 1e-300)
    v.B = v.B * L[None] * z[:, None, None]
    v.out = v.out * L_out * z
    if len(v.w) > 1:
        from ..env import WEIGHT_FLOOR
        ev = m1 / np.maximum(m0, 1e-300)
        v.w = v.w * ev
        v.w = (1 - WEIGHT_FLOOR) * v.w / v.w.sum() + WEIGHT_FLOOR * env._w0
    cur, env._v = env._v, v
    env._mix()
    env._v = cur


def clue_likelihood(g: int, cell_m: float, family: str, cell) -> np.ndarray:
    yy, xx = np.mgrid[0:g, 0:g]
    d = np.hypot(xx - cell[0], yy - cell[1]) * cell_m
    return CLUE_FLOOR + (1 - CLUE_FLOOR) * np.exp(-d / reach_m(family))
