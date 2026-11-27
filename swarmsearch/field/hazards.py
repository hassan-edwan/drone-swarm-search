"""Obstacles and local conditions for GENERATED worlds (wilderness, urban, flood).

The real sites get these from lidar, footprints and mapped power lines
(scripts/real_site/flight_limits.py -> flight_limits.npz, exec_stats.json).
Generated worlds have no such data, so this module draws plausible
infrastructure from each map's own layers and turns it into the same per-cell
layers the field environment reads:

- min_level: the lowest search height that clears every crown, roof, wire,
  mast and crane in the cell by flight3d.CLEAR_M (3: no-fly cell);
- land_dist: metres to the nearest cell with an open, gentle, dry spot
  (emergency landings after a failure);
- wind_mult: wind speed at search height per cell for 8 directions, relative
  to open flat ground (swarmsearch.windfield, the terrain and the trees);
- under_p, under_s, exec: under-canopy checks, from the 3D executor run on
  synthetic 3D patches of each family (scripts/family_exec.py ->
  exec_family.json next to this module); left out when that file is missing.

What is drawn (ASSUMPTIONS: rates and sizes are typical values, not data):

- Trees: crown height per cell from the canopy density and a smooth random
  field; wilderness 15-30 m, urban street and park trees 8-18 m, flood
  (riparian) 10-25 m.
- Wilderness: an access road from the trailhead to the nearest map edge with
  a distribution line on 10 m poles (80% of maps); a high-voltage line across
  the map on 30 m pylons in a cleared corridor (35%); a guyed mast of 40-110 m
  on high ground (30%; guy wires out to 0.6 of its height).
- Urban: distribution lines along half of the long street runs (wires up to
  9.5 m); a construction crane of 40-70 m with a 30-60 m jib (30%); masts on
  20% of tall roofs (+3-8 m).
- Flood: a road along each valley side above the flood level with a
  distribution line (wires 9.5 m; in flood some sag to the water: that is in
  the executor's real-world draw); a high-voltage line across the valley
  (40%).
- Winter (urban, flood): street and riparian trees leaf-off, half the cover
  seen from above (as the generated wilderness).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

CLEAR_M = 5.0  # = flight3d.CLEAR_M: clearance over the highest obstacle in a cell
DRONE_H = 0.4  # = flight3d.DRONES["sar"]["height"]
WIRE_TOP = 9.5  # top conductor of a distribution line
HV_TOP = 30.0  # high-voltage pylons
TREE_H = {"wilderness": (15.0, 30.0), "urban": (8.0, 18.0), "flood": (10.0, 25.0)}
EXEC_FILE = Path(__file__).with_name("exec_family.json")
_EXEC: dict | None = None


def _smooth(rng, g, scale):
    from .terrain import smooth_noise
    return smooth_noise(rng, g, scale)


def _line_cells(p0, p1, g):
    """Grid cells (x, y) on the straight segment p0 -> p1 (cell coordinates)."""
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) * 2 + 1
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    out = []
    for x, y in zip(np.round(xs).astype(int), np.round(ys).astype(int)):
        if 0 <= x < g and 0 <= y < g and (not out or out[-1] != (x, y)):
            out.append((int(x), int(y)))
    return out


def _crossing(rng, g):
    """A straight line across the whole map at a random angle and offset (cell coordinates)."""
    ang = rng.uniform(0, np.pi)
    c = np.array([g / 2, g / 2]) + rng.uniform(-g / 4, g / 4, 2)
    d = np.array([math.cos(ang), math.sin(ang)]) * g
    return _line_cells(c - d, c + d, g)


def tree_heights(rng, w) -> np.ndarray:
    lo, hi = TREE_H[w.family]
    f = np.clip(0.5 + 0.8 * (_smooth(rng, w.g, 6) - 0.5), 0, 1)
    h = lo + (hi - lo) * f
    return np.where(w.canopy >= 0.15, h, 0.0)


# ------------------------------------------------------------------ layouts (cell coordinates)
def layout(rng, w) -> dict:
    """Infrastructure drawn for one generated world: lines as lists of cells, masts and cranes."""
    g, cell = w.g, w.cell_m
    H = dict(distribution=[], transmission=[], towers=[], cranes=[])
    if w.family == "wilderness":
        from .terrain import cheapest_path, slope
        if rng.random() < 0.8:  # access road to the trailhead, with power along it
            bx, by = w.base
            edges = [(0, by), (g - 1, by), (bx, 0), (bx, g - 1)]
            end = min(edges, key=lambda e: abs(e[0] - bx) + abs(e[1] - by))
            sl = slope(w.elev, cell)
            cost = np.where(w.water, 50.0, 1.0 + 25.0 * sl ** 2)
            H["distribution"].append(cheapest_path(cost, w.base, end))
        if rng.random() < 0.35:
            H["transmission"].append(_crossing(rng, g))
        if rng.random() < 0.3:
            hi = np.argwhere(w.elev >= np.quantile(w.elev, 0.9))
            y, x = hi[rng.integers(len(hi))]
            h = float(rng.uniform(40, 110))
            H["towers"].append(dict(x=int(x), y=int(y), h=h, guy_m=0.6 * h))
    elif w.family == "urban":
        street = ~w.fly_block & ~w.collapsed if w.collapsed is not None else ~w.fly_block
        for axis in (0, 1):
            for k in range(g):
                run = street[k, :] if axis == 0 else street[:, k]
                if run.mean() > 0.6 and rng.random() < 0.5:
                    H["distribution"].append([(j, k) if axis == 0 else (k, j) for j in np.flatnonzero(run)])
        if rng.random() < 0.3:
            free = np.argwhere(street)
            y, x = free[rng.integers(len(free))]
            H["cranes"].append(dict(x=int(x), y=int(y), h=float(rng.uniform(40, 70)), jib_m=float(rng.uniform(30, 60))))
    elif w.family == "flood":
        lvl = w.river_level
        for side in (0, 1):  # a road along each valley side, a few metres above the water at launch
            pts = []
            for x in range(g):
                col = w.elev[:, x]
                ys = np.arange(g)
                ok = (col > lvl + 1.5) & ((ys < g / 2) if side == 0 else (ys >= g / 2))
                if ok.any():
                    cand = ys[ok]
                    pts.append((x, int(cand[np.argmin(np.abs(col[ok] - (lvl + 3.0)))])))
            if len(pts) > g // 2:
                H["distribution"].append(pts)
        if rng.random() < 0.4:
            H["transmission"].append(_crossing(rng, g))
    return H


# ------------------------------------------------------------------ per-cell layers
def obstacle_top(w, H, tree_h, rng) -> np.ndarray:
    """Highest obstacle per cell, metres above the ground."""
    g, cell = w.g, w.cell_m
    top = np.maximum(tree_h, w.struct_h)
    if w.family == "urban":  # masts on some tall roofs
        tall = w.struct_h > np.quantile(w.struct_h[w.struct_h > 0], 0.7) if (w.struct_h > 0).any() else w.struct_h > 0
        mast = tall & (rng.random((g, g)) < 0.2)
        top = np.where(mast, w.struct_h + rng.uniform(3, 8, (g, g)), top)
    for line in H["distribution"]:
        for x, y in line:
            top[y, x] = max(top[y, x], WIRE_TOP)
    for line in H["transmission"]:
        for x, y in line:
            top[y, x] = max(top[y, x], HV_TOP)
    yy, xx = np.mgrid[0:g, 0:g]
    for t in H["towers"]:  # mast, and guy wires falling from 95% of its height to the ground at guy_m
        d = np.hypot(xx - t["x"], yy - t["y"]) * cell
        guy = np.where(d <= t["guy_m"], 0.95 * t["h"] * (1 - np.maximum(d - cell / 2, 0) / t["guy_m"]), 0.0)
        top = np.maximum(top, guy)
        top[t["y"], t["x"]] = max(top[t["y"], t["x"]], t["h"])
    for c in H["cranes"]:  # tower and the circle the jib sweeps
        d = np.hypot(xx - c["x"], yy - c["y"]) * cell
        top = np.maximum(top, np.where(d <= c["jib_m"], c["h"] - 5.0, 0.0))
        top[c["y"], c["x"]] = max(top[c["y"], c["x"]], c["h"])
    return top


def min_levels(top, alt_m) -> np.ndarray:
    need = top + CLEAR_M + DRONE_H / 2
    lv = np.full(top.shape, 3, int)
    for k in reversed(range(len(alt_m))):
        lv = np.where(alt_m[k] >= need, k, lv)
    return lv


def landing_distance(w, top, H) -> np.ndarray:
    """Metres to the nearest cell with an open, gentle (< 15 deg), dry spot, free of trees, roofs and wires."""
    from .terrain import slope
    sl = np.degrees(np.arctan(slope(w.elev, w.cell_m)))
    wires = np.zeros((w.g, w.g), bool)
    for line in H["distribution"] + H["transmission"]:
        for x, y in line:
            wires[y, x] = True
    ok = (w.canopy < 0.3) & (w.struct_h <= 0) & ~w.water & (sl < 15) & ~w.fly_block & ~wires
    if w.family == "urban":  # streets and squares: about one cell in three has room to set down
        ok &= ~w.rubble
    return ndimage.distance_transform_edt(~ok) * w.cell_m


def wind_multipliers(w, tree_h, alt_m) -> np.ndarray:
    """(8, g, g): wind at the middle search height over each cell, per direction it comes from (N, NE, ...),
    relative to open flat ground at the same height."""
    from ..windfield import WindField
    g = w.g
    z = float(alt_m[1])
    u_open = math.log(z / 0.03) / math.log(10.0 / 0.03)
    out = np.zeros((8, g, g))
    hc = tree_h
    for d in range(8):
        wf = WindField(w.elev, hc, w.cell_m, 1.0, 45.0 * d)
        u100 = wf.mult * math.log(100.0 / 0.03) / math.log(10.0 / 0.03)
        zz = z + hc  # search heights are above the ground; in forest, above the crowns' surface
        forest = hc > 2.0
        z0 = np.where(forest, 0.1 * hc, 0.03)
        dd = np.where(forest, 0.67 * hc, 0.0)
        prof = np.log(np.maximum(np.minimum(zz, 100.0) - dd, z0 * 1.01) / z0) / np.log((100.0 - dd) / z0)
        out[d] = u100 * prof / u_open
    return out


def exec_family() -> dict:
    global _EXEC
    if _EXEC is None:
        _EXEC = json.load(open(EXEC_FILE)) if EXEC_FILE.exists() else {}
    return _EXEC


def add(w, sc, rng) -> None:
    """Draw obstacles for a generated world and set its per-cell layers (w.extra), in place."""
    fam = sc.fam
    H = layout(rng, w)
    tree_h = tree_heights(rng, w)
    if w.family == "wilderness":  # high-voltage corridors are kept clear of trees (about 40 m wide)
        keep = max(0.0, 1.0 - 40.0 / w.cell_m)
        for line in H["transmission"]:
            for x, y in line:
                for a in (w.canopy, w.map_canopy):
                    a[y, x] *= keep
                tree_h[y, x] *= keep > 0.3
    if w.family in ("urban", "flood") and getattr(sc, "season", "summer") == "winter":
        w.canopy = w.canopy * 0.5  # leaf-off street, park and riparian trees: half the cover from above
        w.map_canopy = w.map_canopy * 0.5
    top = obstacle_top(w, H, tree_h, rng)
    lv = min_levels(top, fam.alt_m)
    if w.family == "urban":  # buildings were already no-fly cells; keep their level out of the way
        lv = np.where(w.fly_block, 0, lv)
    nofly = lv >= 3
    w.fly_block |= nofly
    w.map_fly_block |= nofly
    w.extra.update(hazards=H, tree_h=tree_h, top_m=top, min_level=lv,
                   land_dist=landing_distance(w, top, H), wind_mult=wind_multipliers(w, tree_h, fam.alt_m))
    st = exec_family().get(w.family)
    if st:
        dense = w.map_canopy >= 0.5
        w.extra["under_p"] = np.where(dense, st["under_p"], -1.0)
        w.extra["under_s"] = np.where(dense, st["under_s"], -1.0)
        w.extra["exec"] = st["exec"]
