"""Lost person behaviour data (ISRID and related studies) for the field environment.

Every number carries its source. "published" values are quoted from the
sources; "derived" values are fitted to the published points; "assumed"
values have no published source we could find and should be replaced when
the data are available (Koester, Lost Person Behavior, 2008, has the full
ISRID tables).

Sources
[JSAR20] Journal of Search & Rescue 4(1), March 2020, "A Pragmatic Approach to Applied Search
         Theory", Table 1: hiker, mountain temperate (ISRID). https://wasspu.org/wp-content/uploads/2021/09/A-Pragmatic-Approach-to-Applied-Search-Theory.pdf
[K20]    Koester et al., Journal of Search & Rescue 4(1), April 2020, ASD vs typically developing
         children (ISRID), Tables 3, 5-7, 9. https://journalofsar.com/wp-content/uploads/2020/04/v4-8-Koester-ASD-vTD.pdf.pdf
[NZSAR]  New Zealand SAR, Lost Person Behaviour course (ISRID dementia).
         https://www.nzsar.govt.nz/assets/Downloadable-Files/5.SAR-92-Lost-Person-Behaviour-v5.pdf
[K06]    Koester, Missing Person Behaviour (2006 handout, BSAR). https://www.bsar.org/wp-content/uploads/2008/04/Missing%20Person%20Behaviour%20v1.0.pdf
[IKAR]   IKAR, Predicting Lost Person Behavior. https://www.alpine-rescue.org/system/production/article/documents/file/004/113/90ecc0736c02b837fa37891a4ffd75669b3133ee41c347273bb4b0af950d5326/18-Predicting-Lost-Person-Behavior.pdf
[J16]    Jacobs, Terrain Based Probability Models for SAR (ISRID hikers, hunters, gatherers; OR, NY, AZ),
         Table 1. https://mra.org/wp-content/uploads/2016/05/TerrainProbabilityModelsReport.pdf
"""

from __future__ import annotations

import numpy as np

Q = (0.25, 0.50, 0.75, 0.95)

CATEGORIES = {
    "hiker": dict(
        dist_km=(1.1, 3.1, 5.8, 11.3), dist_src="published [JSAR20], mountain temperate",
        disp_deg=(2, 23, 64, 132), disp_src="published [JSAR20]: angle from the direction of travel",
        respond=0.7, respond_src="published [K06]: the majority are communicative (share assumed)",
        evasive=0.0,
        mobile_h=(1.5, 4.0, 9.0, 24.0), mobile_src="derived [K06]: 90% mobile for less than 24 h (shape assumed)",
        survive_d=(2, 5, 10, 24), survive_src="published [K20] Table 9: 75/50/25/5% found alive by day",
        follow_trail=0.75, trail_src="published [K06]: 75% followed a trail or drainage at some time",
    ),
    "hunter": dict(
        dist_km=(1.0, 2.1, 4.2, 11.7), dist_src="derived [K06]: log-normal through 66% < 3.2 km and 90% < 8 km",
        disp_deg=(2, 23, 64, 132), disp_src="assumed: hiker values",
        respond=0.8, respond_src="published [K06]: respond to calls and whistles, may fire shots (share assumed)",
        evasive=0.0,
        mobile_h=(1.5, 4.0, 9.0, 24.0), mobile_src="assumed: hiker values; [K06] some walk out at daybreak",
        survive_d=(2, 5, 10, 24), survive_src="assumed: hiker curve; [K06] high survivability",
        follow_trail=0.39, trail_src="published [K06]: 39% follow drainages",
    ),
    "child_7_12": dict(
        dist_km=(0.3, 0.9, 2.4, 11.3), dist_src="published [K20] Table 3: typically developing children 1-16, temperate",
        disp_deg=(5, 30, 75, 150), disp_src="assumed",
        respond=0.3, respond_src="[IKAR]: children 7-12 rarely respond (share assumed)",
        evasive=0.35, evasive_src="[IKAR]: often avoid or hide from searchers (share assumed)",
        mobile_h=(1.0, 3.0, 5.0, 11.0), mobile_src="published [K20] Table 3: hours mobile",
        survive_d=(2, 3, 5, 11.4), survive_src="published [K20] Table 9",
        follow_trail=0.5, trail_src="published [K20] Table 3: track offset median 14 m (share assumed)",
    ),
    "dementia": dict(
        dist_km=(0.3, 0.8, 1.9, 8.3), dist_src="published [NZSAR] (ISRID)",
        disp_deg=(5, 25, 66, 120), disp_src="partly published: 75% within 66 deg of the direction of travel (NEWSAR); rest assumed",
        respond=0.01, respond_src="published [K06]: only 1% respond to shouts",
        evasive=0.1, evasive_src="[BMC Geriatrics]: some hide (share assumed)",
        mobile_h=(1.0, 2.0, 4.0, 8.0), mobile_src="assumed: 'go until they get stuck' [K06]",
        survive_d=(1, 2, 4, 7), survive_src="assumed: shorter than children ([K06] succumb to the environment)",
        follow_trail=0.5, trail_src="published [K06]: half found within 33 m of a road",
    ),
    "despondent": dict(
        dist_km=(0.08, 0.32, 1.33, 10.2), dist_src="derived [K06]: 50% < 0.32 km, 96% < 12.8 km (two patterns: just out of sight, or a significant place)",
        disp_deg=(10, 45, 100, 170), disp_src="assumed",
        respond=0.1, respond_src="assumed: do not want to be found [K06]",
        evasive=0.2, evasive_src="assumed",
        mobile_h=(0.5, 1.0, 3.0, 8.0), mobile_src="assumed",
        survive_d=(0.5, 1, 2, 4), survive_src="derived [K06]: mortality over 55% (curve assumed)",
        follow_trail=0.5, trail_src="published via [J16]: 95% found within 500 m of a linear feature",
    ),
}
# Category mix of wilderness incidents (ISRID is mostly hikers; [J16] backcountry group: 74% hikers, 14% hunters).
CATEGORY_WEIGHTS = {"hiker": 0.55, "hunter": 0.15, "child_7_12": 0.1, "dementia": 0.1, "despondent": 0.1}

# [J16] Table 1: how much more likely a person is found on/near each feature (uninjured, injured).
TERRAIN_PDEN = {
    "road": (3.0, 1.5), "trail": (5.0, 7.0), "lake": (2.0, 2.0), "stream": (2.0, 3.5),
    "low_point": (2.0, 5.0), "high_point": (1.5, 2.0), "ridge": (2.0, 1.0), "drainage": (1.5, 3.0),
}
INJURED_SHARE = 0.3  # assumed mix when building a prior


def quantile_cdf(d_km: np.ndarray, q_km) -> np.ndarray:
    """P(distance <= d) from the four ISRID quantiles: piecewise linear in
    log distance between them, a short ramp from 0 below the 25% point
    (some are found at the IPP itself [K20] Table 5), and a log-linear tail
    to 99% at twice the 95% distance."""
    d = np.maximum(np.asarray(d_km, float), 1e-6)
    xs = np.log(np.array([q_km[0] / 20, *q_km, 2 * q_km[3]]))
    ps = np.array([0.02, *Q, 0.99])
    return np.interp(np.log(d), xs, ps, left=0.0, right=0.99)


def sample_distance(rng, q_km) -> float:
    """Distance in km drawn from the quantile model (inverse of quantile_cdf)."""
    u = rng.uniform(0.0, 0.99)
    xs = np.log(np.array([q_km[0] / 20, *q_km, 2 * q_km[3]]))
    ps = np.array([0.02, *Q, 0.99])
    if u < 0.02:
        return float(q_km[0] / 20 * u / 0.02)
    return float(np.exp(np.interp(u, ps, xs)))


def ring_probability(g: int, cell_m: float, centre, q_km) -> tuple[np.ndarray, float]:
    """Probability per cell (spread evenly around each ring) and the
    probability of being beyond the map, from the quantile model."""
    yy, xx = np.mgrid[0:g, 0:g]
    r_in = np.maximum(np.hypot(xx - centre[0], yy - centre[1]) - 0.5, 0.0) * cell_m / 1000
    r_out = (np.hypot(xx - centre[0], yy - centre[1]) + 0.5) * cell_m / 1000
    band = quantile_cdf(r_out, q_km) - quantile_cdf(r_in, q_km)
    # share of the ring's area this cell takes
    ring_area = np.pi * (r_out ** 2 - r_in ** 2) / (cell_m / 1000) ** 2
    dens = band / np.maximum(ring_area, 1.0)
    inside = float(dens.sum())
    return dens, float(np.clip(1.0 - inside, 0.0, 0.99))


def angle_weight(g: int, centre, dot_deg: float, disp_deg) -> np.ndarray:
    """Relative weight per cell from the dispersion-angle quartiles around a
    direction of travel (deg, 0 = +x, counter-clockwise with y down)."""
    yy, xx = np.mgrid[0:g, 0:g]
    ang = np.degrees(np.arctan2(-(yy - centre[1]), xx - centre[0]))
    dev = np.abs((ang - dot_deg + 180) % 360 - 180)
    # density of |deviation|: 25% of finds in each band up to the 75% value, 20% to the 95% value, 5% beyond
    edges = [0.0, *disp_deg, 180.0]
    mass = [0.25, 0.25, 0.25, 0.20, 0.05]
    w = np.zeros_like(dev)
    for lo, hi, m in zip(edges[:-1], edges[1:], mass):
        sel = (dev >= lo) & (dev < hi) if hi < 180 else (dev >= lo)
        w[sel] = m / max(hi - lo, 1e-6)
    w[(xx == centre[0]) & (yy == centre[1])] = np.mean(w)
    return w / w.mean()


def terrain_multiplier(trail, stream, elev, injured_share: float = INJURED_SHARE) -> np.ndarray:
    """[J16] feature multipliers on a grid, mixed over injured and uninjured subjects."""
    def m(feat):
        u, i = TERRAIN_PDEN[feat]
        return (1 - injured_share) * u + injured_share * i
    g = elev.shape[0]
    lo = elev <= np.quantile(elev, 0.15)
    hi = elev >= np.quantile(elev, 0.85)
    out = np.ones((g, g))
    out = np.where(trail, m("trail"), out)
    out = np.where(~trail & stream, m("stream"), out)
    out = np.where(~trail & ~stream & lo, m("low_point"), out)
    out = np.where(~trail & ~stream & hi, m("high_point"), out)
    near_trail = np.zeros_like(trail)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_trail |= np.roll(np.roll(trail, dy, 0), dx, 1)
    out = np.where(near_trail & ~trail, np.maximum(out, 0.5 * (1 + m("trail"))), out)
    return out


def survival(days: float, survive_d) -> float:
    """Chance of being found alive after `days`, from the 75/50/25/5% points."""
    return float(np.interp(days, [0.0, *survive_d], [1.0, 0.75, 0.5, 0.25, 0.05], right=0.02))


def sample_mobile_hours(rng, q_h) -> float:
    """How long this person keeps moving before stopping for good."""
    return sample_distance(rng, q_h)
