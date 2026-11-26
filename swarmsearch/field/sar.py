"""The simulator's sensing in standard search-and-rescue terms.

Effective sweep width W (metres): the width of a strip in which a searcher
finds as many people as it misses outside it; for one pass at cruise speed,
W = the integral of the single-pass detection chance across the track.
For a downward camera of footprint width F that sees a person with chance
p anywhere in the footprint, W = F * p.

Coverage = W * track length / area. Probability of detection for an area
searched with coverage c, tracks spaced evenly but imperfectly (the
"random search" curve used in SAR planning, Koopman): POD = 1 - exp(-c).

Search rate (km2/h) = W * speed: how fast one drone sweeps area at that POD
per unit coverage. These let the simulator's detection model be checked
against field sweep-width numbers, and its results read like a SAR plan.
"""

from __future__ import annotations

import math

import numpy as np

from . import sensing as sn
from .params import FAMILY, FAMILY_BASE_DETECT

NOMINAL_SENSOR = 0.75  # mean of the per-mission sensor_scale draw (0.4-1.1)


def single_pass(family: str, k: int, canopy: float, sensor: str = "both", hour: float = 9.0, air_c: float = 15.0,
                rain: bool = False, model: str = "visibility", sensor_scale: float = NOMINAL_SENSOR,
                cruise: bool = True) -> float:
    """Chance one pass at height index k finds a person directly under the track."""
    rgb, th = sn.conditions(np.array([canopy]), np.array([False]), 0.0, np.array([False]), hour, air_c, rain, model)
    c = {"rgb": rgb[0], "thermal": th[0], "both": max(rgb[0], th[0])}[sensor]
    occ = float(sn.occlusion(np.array([canopy]), np.array([False]), model)[0])
    motion = sn.BLUR if cruise else 1.0
    return float(FAMILY_BASE_DETECT[family][k] * c * motion * sensor_scale * (1 - occ))


def footprint_m(family: str, k: int) -> float:
    fam = FAMILY[family]
    return 2 * fam.alt_m[k] * math.tan(math.radians(fam.hfov_deg / 2))


def sweep_width(family: str, k: int, canopy: float, **kw) -> float:
    return footprint_m(family, k) * single_pass(family, k, canopy, **kw)


def search_rate_km2_h(family: str, k: int, canopy: float, **kw) -> float:
    return sweep_width(family, k, canopy, **kw) * FAMILY[family].cruise_mps * 3.6 / 1000.0


def pod(coverage):
    return 1.0 - np.exp(-np.asarray(coverage, float))


def table(family: str = "wilderness", canopies=(0.0, 0.5, 0.8, 0.9), model: str = "visibility") -> str:
    fam = FAMILY[family]
    rows = ["| Height | Footprint | Tree cover | Sensor | Day: W (m) | Day: km²/h | Night: W (m) | Night: km²/h |",
            "|---|---|---|---|---|---|---|---|"]
    for k, h in enumerate(fam.alt_m):
        for cv in canopies:
            for s in ("rgb", "thermal", "both"):
                d = dict(sensor=s, model=model)
                wd, wn = sweep_width(family, k, cv, hour=10.0, air_c=15.0, **d), sweep_width(family, k, cv, hour=23.0, air_c=10.0, **d)
                rows.append(f"| {h:.0f} m | {footprint_m(family, k):.0f} m | {cv:.0%} | {s} | {wd:.0f} | "
                            f"{wd * fam.cruise_mps * 3.6 / 1000:.2f} | {wn:.0f} | {wn * fam.cruise_mps * 3.6 / 1000:.2f} |")
    return "\n".join(rows)
