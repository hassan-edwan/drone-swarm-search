"""Scenario families and per-mission parameters for the field environment.

A family fixes the scale (cell size, step length, map size, flying heights)
and how its world and people are generated; a FieldScenario draws one
mission from it. Each number cites the register item it implements
(docs/ASSUMPTIONS.md); values marked "illustrative" are placeholders to be
replaced by field measurements or the cited data sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .lpb import CATEGORIES, CATEGORY_WEIGHTS

FAMILIES = ("urban", "wilderness", "flood")


@dataclass(frozen=True)
class Family:
    name: str
    grid: int  # cells per side
    cell_m: float  # metres per cell
    step_s: float  # seconds per step (cruise speed = cell_m / step_s)
    max_steps: int  # mission length
    alt_m: tuple[float, float, float]  # flying heights above ground (low, mid, high); <= 120 m legal ceiling (D8)
    foot_r: tuple[int, int, int]  # camera footprint radius in cells per height
    alt_detect: tuple[float, float, float]  # detection per height (person smaller in the image higher up)
    n_drones: tuple[int, int]
    n_teams: tuple[int, int]  # ground teams (O2)
    hfov_deg: float = 84.0  # wide-angle camera
    cruise_mps: float = 10.0

    def scaled(self, cell_m: float, mission_s: float) -> "Family":
        """This family at another cell size (the search area is sized per mission):
        the step is the time to cross a cell at cruise speed, and the camera
        footprint in cells follows from its height and field of view. When the
        footprint is narrower than a cell, a pass covers that share of it."""
        half = [h * np.tan(np.radians(self.hfov_deg / 2)) for h in self.alt_m]
        foot_r = tuple(int(min(2, max(0, round(hw / cell_m - 0.5)))) for hw in half)
        cover = [min(1.0, 2 * hw / cell_m) if r == 0 else 1.0 for hw, r in zip(half, foot_r)]
        base = FAMILY_BASE_DETECT[self.name]
        step_s = cell_m / self.cruise_mps
        return replace(self, cell_m=cell_m, step_s=step_s, max_steps=int(round(mission_s / step_s)), foot_r=foot_r,
                       alt_detect=tuple(float(d * c) for d, c in zip(base, cover)))


# E1: scale. Cruise speeds 5-10 m/s; missions of 30-60 min with battery swaps (D1).
URBAN = Family("urban", grid=20, cell_m=10.0, step_s=2.0, max_steps=900, alt_m=(15.0, 35.0, 60.0),
               foot_r=(0, 1, 2), alt_detect=(1.0, 0.7, 0.45), n_drones=(2, 6), n_teams=(1, 3))
# Wilderness: 64 x 64 cells around the trailhead. The cell size is set per
# mission so the map reaches the subject category's 75% distance from the
# last-seen point (ISRID, see lpb.py): planners size the search area to the
# category. 90-minute missions (about two batteries).
WILDERNESS = Family("wilderness", grid=64, cell_m=75.0, step_s=7.5, max_steps=720, alt_m=(40.0, 70.0, 120.0),
                    foot_r=(0, 0, 1), alt_detect=(0.9, 0.75, 0.35), n_drones=(2, 4), n_teams=(1, 3))
WILDERNESS_MISSION_S = 5400.0
CELL_RANGE_M = (50.0, 200.0)
FLOOD = Family("flood", grid=32, cell_m=25.0, step_s=3.0, max_steps=600, alt_m=(20.0, 45.0, 90.0),
               foot_r=(0, 1, 2), alt_detect=(1.0, 0.75, 0.45), n_drones=(2, 5), n_teams=(1, 2))
FAMILY = {f.name: f for f in (URBAN, WILDERNESS, FLOOD)}
FAMILY_BASE_DETECT = {f.name: f.alt_detect for f in (URBAN, WILDERNESS, FLOOD)}
_SCALED: dict = {}


def wilderness_cell_m(category: str) -> float:
    """Cell size that puts the category's 75% distance at the map edge."""
    q75_m = CATEGORIES[category]["dist_km"][2] * 1000.0
    return float(np.clip(2 * q75_m / WILDERNESS.grid, *CELL_RANGE_M))

# Subject categories and their data: lpb.py (ISRID and related studies).
LIGHTS = ("day", "dusk", "night")


@dataclass(frozen=True)
class FieldScenario:
    family: str = "wilderness"
    seed_world: int = 0  # terrain / city layout
    n_drones: int = 3
    # People (P)
    category: str = "hiker"
    n_people: int = 1
    group_size: int = 1  # people placed together (P6)
    n_est_err: float = 0.0  # log-sd of the reported head count (P6)
    p_unresponsive: float | None = None  # None: from the category (P1)
    p_evasive: float | None = None
    p_off_model: float = 0.15  # people following none of the modelled behaviours (P3)
    p_injured: float = 0.2  # immobile (P4)
    fatigue_h: float = 6.0  # mobility halves every this many hours (P4)
    hours_lost: float = 2.0  # time since lost when drones launch (K3)
    hypothermic: float = 0.15  # share with weak thermal contrast (P8)
    cell_m: float = 0.0  # wilderness: per-mission cell size (0 = from the category)
    dot_deg: float = 0.0  # the person's true direction of travel from the last-seen point (deg)
    dot_known: bool = False  # the team knows it (planned route, footprints, last sighting heading)
    # Knowledge (K)
    ipp_err_m: float = 150.0  # error of the reported last-seen point (K1)
    ipp_wrong: bool = False  # last-seen point plainly wrong (K1, hard case)
    d50_scale: float = 1.0  # true distance spread / the map's (K4)
    p_outside: float = 0.0  # chance the person is outside the area at launch, beyond the prior's (K2)
    phone_ping: bool = False  # a rough area from a phone ping (P2)
    map_error: float = 0.2  # share of trails missing from the map, canopy map noise (K5)
    profile_told: bool = False  # operator gives a behaviour (only the dominant one, sometimes wrong) (K6)
    # Sensing (S)
    sensor_scale: float = 1.0  # true detection / modelled (S1)
    fa_scale: float = 1.0
    light: str = "day"  # light at launch (E4); the clock runs on during the mission
    hour: float = 12.0  # local time at launch
    air_c: float = 15.0  # air temperature: thermal contrast (S3)
    rain: float = 0.0  # chance of a rain spell (E4)
    n_operators: int = 1  # people reviewing detections (S5)
    review_s: float = 20.0  # seconds per review (S5)
    op_miss: float = 0.1  # base chance a reviewer misses a true detection (S5)
    adaptive_map: bool = False  # the team software estimates sensor quality online (M3)
    # Flight (D)
    endurance_min: float = 45.0  # per battery (D1)
    swap_s: float = 150.0
    wind_mps: float = 4.0
    gust: float = 0.0  # chance of a gust spell above the limit (D6)
    gps_err: float = 0.04  # chance a fix is a cell off (D4)
    tilt: bool = False  # camera tilted forward (D7)
    n_nofly: int = 0  # static no-fly zones (D8)
    p_tfr: float = 0.0  # chance a no-fly zone appears mid-mission (D8)
    rtb_on_link_loss: bool = True  # return when out of contact (D6)
    fail_per_h: float = 0.01  # in-flight failure rate (D6)
    # Radio (C)
    comm_range_m: float = 800.0  # line of sight; a quarter of that when blocked (C1)
    comm_loss: float = 0.05
    tether_relay: bool = False  # relay drone on a tether above the base (C1)
    # Operations (O)
    overwatch: float = 0.7  # chance a drone stays over a find until a ground team arrives (O1)
    n_teams: int = 1  # ground teams (O2)
    # World (E)
    collapse_frac: float = 0.3  # urban: buildings collapsed (unknown until seen)
    aftershock: float = 0.0  # urban: chance of new collapses mid-mission (E3)
    smoke: float = 0.0  # urban: smoke from fires (E3)
    water_rise: float = 0.0  # flood: metres the water rises over the mission (E3)
    # Hard test cases (forced)
    force: str = ""  # "", "unresponsive", "evasive", "outside", "off_model"
    corner_cutting: bool = True  # interface field shared with the grid scenarios
    site: str = ""  # wilderness: real map layers from scripts/import_site.py instead of generated terrain
    # Growing the search (K2): wilderness only. The world is built EXPAND times wider;
    # the drones start on the centre and the area grows once their map puts the
    # person more likely outside than inside (or at a set time).
    # Evidence from outside the drones, and others in the area (K7, S8; field/evidence.py)
    clue_rate: float = 0.0  # clues a moving person leaves per hour
    p_report: float = 0.0  # chance of a new report (late ping, witness) during the mission
    ground_search: bool = True  # ground teams search; if not, they wait at the base and only go to finds
    n_walkers: int = 0  # other walkers and campers
    n_animals: int = 0  # deer, cattle, dogs: warm and person-sized to a detector
    expand: bool = False
    expand_at: float = 0.5  # the map's chance the person is outside the area that triggers growth
    expand_min: float = 0.0  # also grow at this mission minute (an operational-period boundary; 0: off)
    mission_s: float = 0.0  # mission length in seconds (0: the family's default)

    @property
    def fam(self) -> Family:
        if self.family != "wilderness":
            return FAMILY[self.family]
        cell = self.cell_m or wilderness_cell_m(self.category)
        mission = self.mission_s or WILDERNESS_MISSION_S
        key = (round(cell, 3), round(mission, 1))
        if key not in _SCALED:
            _SCALED[key] = WILDERNESS.scaled(cell, mission)
        return _SCALED[key]

    # Names shared with the grid scenarios (evaluation, baselines).
    camera = property(lambda self: True)
    real = property(lambda self: True)
    grid_size = property(lambda self: self.fam.grid)
    max_steps = property(lambda self: self.fam.max_steps)
    n_survivors = property(lambda self: self.n_people)


@dataclass
class FieldSampler:
    """Draws missions across families (weights) with every effect randomised."""

    families: dict = field(default_factory=lambda: {"wilderness": 0.5, "urban": 0.5})
    camera: bool = True  # interface flags shared with the grid environment
    real: bool = True
    field: bool = True
    grid_size: int = 64  # largest map; each mission uses its family's size
    max_steps: int = 1100
    expand_share: float = 0.0  # wilderness missions whose area can grow (K2)
    long_share: float = 0.0  # wilderness missions run as a long operation (long_s) instead of one flight period
    long_s: float = 4 * 3600.0

    def sample(self, rng: np.random.Generator) -> FieldScenario:
        names = list(self.families)
        p = np.array([self.families[n] for n in names], float)
        fam = FAMILY[names[rng.choice(len(names), p=p / p.sum())]]
        sc = sample_scenario(rng, fam.name)
        if fam.name == "wilderness" and (self.expand_share > 0 or self.long_share > 0):
            long = bool(rng.random() < self.long_share)
            sc = replace(sc, expand=bool(rng.random() < self.expand_share),
                         expand_at=float(rng.uniform(0.4, 0.7)), mission_s=self.long_s if long else 0.0)
        return sc


def sample_scenario(rng: np.random.Generator, family: str, **override) -> FieldScenario:
    fam = FAMILY[family]
    u = rng.uniform
    if family == "wilderness":
        cats = list(CATEGORY_WEIGHTS)
        cat = cats[rng.choice(len(cats), p=[CATEGORY_WEIGHTS[c] for c in cats])]
        n_people = 1 if rng.random() < 0.85 else int(rng.integers(2, 4))
        group = n_people
        n_est_err = 0.0  # the reporting party knows who is missing
    else:
        cat = "hiker"  # disaster survivors: mostly ordinary adults
        n_people = int(rng.integers(3, 16))
        group = int(rng.integers(1, 4))
        n_est_err = 0.4
    hour = float(u(0, 24))
    light = "day" if 7 <= hour < 18 else "dusk" if 18 <= hour < 20 or 6 <= hour < 7 else "night"
    dot = float(u(0, 360))
    sc = FieldScenario(
        family=family, seed_world=int(rng.integers(1 << 30)),
        dot_deg=dot, dot_known=bool(rng.random() < 0.5),
        n_drones=int(rng.integers(fam.n_drones[0], fam.n_drones[1] + 1)),
        category=cat, n_people=n_people, group_size=group, n_est_err=n_est_err,
        p_off_model=float(u(0.05, 0.3)), p_injured=float(u(0.05, 0.4)), fatigue_h=float(u(3, 12)),
        hours_lost=float(u(0.5, 12)) if family == "wilderness" else float(u(0.2, 6)),
        hypothermic=float(u(0.0, 0.3)),
        ipp_err_m=float(u(30, 500)), ipp_wrong=bool(rng.random() < 0.1), d50_scale=float(np.exp(rng.normal(0, 0.35))),
        p_outside=float(u(0.0, 0.05)),  # beyond what the distance data already put outside
        profile_told=bool(rng.random() < 0.3),
        sensor_scale=float(u(0.4, 1.1)), fa_scale=float(u(0.5, 3.0)), light=light, hour=hour,
        air_c=float(u(-5, 35)), rain=float(u(0, 0.4)), n_operators=int(rng.integers(1, 3)),
        review_s=float(u(8, 40)), op_miss=float(u(0.03, 0.2)), adaptive_map=False,
        endurance_min=float(u(35, 55)), swap_s=float(u(90, 240)), wind_mps=float(u(0, 10)),
        gust=float(u(0, 0.2)), gps_err=float(u(0.02, 0.08)), tilt=bool(rng.random() < 0.3),
        n_nofly=int(rng.integers(0, 3)), p_tfr=float(u(0, 0.4)), rtb_on_link_loss=bool(rng.random() < 0.5),
        fail_per_h=float(u(0.0, 0.03)),
        # Drone video links are rated for km in line of sight; terrain and buildings cut them (C1).
        comm_range_m=float(u(200, 1000)) if family == "urban" else float(u(2000, 10000)),
        comm_loss=float(u(0, 0.2)), tether_relay=bool(rng.random() < 0.5),
        overwatch=float(u(0.3, 1.0)), n_teams=int(rng.integers(fam.n_teams[0], fam.n_teams[1] + 1)),
        collapse_frac=float(u(0, 0.5)), aftershock=float(u(0, 0.3)), smoke=float(u(0, 0.6)),
        water_rise=float(u(0, 2.0)),
    )
    # Drawn last so the draws above (and so every earlier scenario) are unchanged: these two
    # were lost to a comment until 2026-11-15 and every mission used the defaults.
    sc = replace(sc, phone_ping=bool(rng.random() < 0.1), map_error=float(u(0.0, 0.4)))
    # Added 2026-11-16, also drawn last (K7, S8). Rates assumed: clues are found in many but not all
    # searches; other walkers mostly by day.
    n_walk = {"wilderness": (0, 6), "urban": (0, 12), "flood": (0, 6)}[family]
    n_anim = {"wilderness": (0, 8), "urban": (0, 3), "flood": (0, 3)}[family]
    # Often there is no ground search, the person leaves nothing to find, and nothing new is reported
    # (shares assumed): ground search in half the missions, clues in 40%, a new report possible in 40%.
    sc = replace(sc, ground_search=bool(rng.random() < 0.5),
                 clue_rate=float(u(0.05, 0.5)) if rng.random() < 0.4 else 0.0,
                 p_report=float(u(0.0, 0.2)) if rng.random() < 0.4 else 0.0,
                 n_walkers=int(rng.integers(n_walk[0], n_walk[1] + 1) * (0.3 if light == "night" else 1.0)),
                 n_animals=int(rng.integers(n_anim[0], n_anim[1] + 1)))
    return replace(sc, **override) if override else sc
