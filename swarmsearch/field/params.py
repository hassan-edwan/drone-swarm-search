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


# E1: scale. Cruise speeds 5-10 m/s; missions of 30-60 min with battery swaps (D1).
URBAN = Family("urban", grid=20, cell_m=10.0, step_s=2.0, max_steps=900, alt_m=(15.0, 35.0, 60.0),
               foot_r=(0, 1, 2), alt_detect=(1.0, 0.7, 0.45), n_drones=(2, 6), n_teams=(1, 3))
# Wilderness: 3.6 km square centred on the trailhead, so the area holds about
# the median lost-hiker distance in every direction (planners draw the search
# area around the last-seen point). 75 m cells; wide-angle camera at 120 m
# sees about three cells across.
WILDERNESS = Family("wilderness", grid=48, cell_m=75.0, step_s=7.5, max_steps=480, alt_m=(40.0, 70.0, 120.0),
                    foot_r=(0, 0, 1), alt_detect=(0.9, 0.75, 0.35), n_drones=(2, 4), n_teams=(1, 3))
FLOOD = Family("flood", grid=32, cell_m=25.0, step_s=3.0, max_steps=600, alt_m=(20.0, 45.0, 90.0),
               foot_r=(0, 1, 2), alt_detect=(1.0, 0.75, 0.45), n_drones=(2, 5), n_teams=(1, 2))
FAMILY = {f.name: f for f in (URBAN, WILDERNESS, FLOOD)}

# Subject categories (P1, P5): share unresponsive / evasive (rest signal when a
# drone is close), distance median from the last-seen point in km (wilderness,
# illustrative until replaced by ISRID values [1]). Dementia subjects usually do
# not respond [3][12]; children 7-12 often hide [13].
CATEGORIES = {
    "hiker": dict(unresponsive=0.5, evasive=0.02, d50_km=1.5),
    "hunter": dict(unresponsive=0.5, evasive=0.02, d50_km=1.8),
    "child_7_12": dict(unresponsive=0.5, evasive=0.35, d50_km=1.0),
    "dementia": dict(unresponsive=0.85, evasive=0.1, d50_km=0.8),
    "despondent": dict(unresponsive=0.7, evasive=0.25, d50_km=0.9),
}
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

    @property
    def fam(self) -> Family:
        return FAMILY[self.family]

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
    grid_size: int = 48  # largest map; each mission uses its family's size
    max_steps: int = 900

    def sample(self, rng: np.random.Generator) -> FieldScenario:
        names = list(self.families)
        p = np.array([self.families[n] for n in names], float)
        fam = FAMILY[names[rng.choice(len(names), p=p / p.sum())]]
        return sample_scenario(rng, fam.name)


def sample_scenario(rng: np.random.Generator, family: str, **override) -> FieldScenario:
    fam = FAMILY[family]
    u = rng.uniform
    if family == "wilderness":
        cat = list(CATEGORIES)[rng.choice(len(CATEGORIES), p=[0.45, 0.2, 0.1, 0.1, 0.15])]
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
    sc = FieldScenario(
        family=family, seed_world=int(rng.integers(1 << 30)),
        n_drones=int(rng.integers(fam.n_drones[0], fam.n_drones[1] + 1)),
        category=cat, n_people=n_people, group_size=group, n_est_err=n_est_err,
        p_off_model=float(u(0.05, 0.3)), p_injured=float(u(0.05, 0.4)), fatigue_h=float(u(3, 12)),
        hours_lost=float(u(0.5, 12)) if family == "wilderness" else float(u(0.2, 6)),
        hypothermic=float(u(0.0, 0.3)),
        ipp_err_m=float(u(30, 500)), ipp_wrong=bool(rng.random() < 0.1), d50_scale=float(np.exp(rng.normal(0, 0.35))),
        p_outside=float(u(0.0, 0.15)), phone_ping=bool(rng.random() < 0.1), map_error=float(u(0.0, 0.4)),
        profile_told=bool(rng.random() < 0.3),
        sensor_scale=float(u(0.4, 1.1)), fa_scale=float(u(0.5, 3.0)), light=light, hour=hour,
        air_c=float(u(-5, 35)), rain=float(u(0, 0.4)), n_operators=int(rng.integers(1, 3)),
        review_s=float(u(8, 40)), op_miss=float(u(0.03, 0.2)), adaptive_map=False,
        endurance_min=float(u(35, 55)), swap_s=float(u(90, 240)), wind_mps=float(u(0, 10)),
        gust=float(u(0, 0.2)), gps_err=float(u(0.02, 0.08)), tilt=bool(rng.random() < 0.3),
        n_nofly=int(rng.integers(0, 3)), p_tfr=float(u(0, 0.4)), rtb_on_link_loss=bool(rng.random() < 0.5),
        fail_per_h=float(u(0.0, 0.03)),
        comm_range_m=float(u(80, 300)) if family == "urban" else float(u(300, 2500)),  # buildings vs open ground
        comm_loss=float(u(0, 0.2)), tether_relay=bool(rng.random() < 0.2),
        overwatch=float(u(0.3, 1.0)), n_teams=int(rng.integers(fam.n_teams[0], fam.n_teams[1] + 1)),
        collapse_frac=float(u(0, 0.5)), aftershock=float(u(0, 0.3)), smoke=float(u(0, 0.6)),
        water_rise=float(u(0, 2.0)),
    )
    return replace(sc, **override) if override else sc
