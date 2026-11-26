"""Two 3D worlds at 1 m: the MAP a drone plans with, and the REAL world it meets.

Voxels: 1 m squares on the ground, 1 m slices of height ABOVE THE GROUND
(level k covers k..k+1 m), flagged by kind. The map is built from public data:

- trees: the lidar points themselves (leaf-off), layered: returns at a height
  in a 3 x 3 m neighbourhood make that slice occupied, and gaps shorter than
  4 m inside a column are closed (branches between returns); in summer
  (leaf-on) the foliage is grown out by 1 m around every return.
  Individual trees by watershed segmentation of the crown surface, each with a
  trunk (diameter from height: DBH (cm) ~ 2 x (height - 1.3 m), a generic
  eastern-hardwood fit) up to its crown bottom.
- buildings: footprints, with the roof surface from the lidar points inside
  them (pitched roofs, not flat boxes).
- power lines: poles every 45 m (10 m high) along one side of every road,
  three conductors (9.5, 8 and 6 m; the lowest is telephone/cable), sagging
  1 m mid-span; a service wire from the nearest pole to each house (down to
  4 m at the house). High-voltage lines: 30 m pylons every 300 m, conductors
  at 24 m sagging 6 m, an earth wire on top. Registered towers: a mast and
  guy wires to three anchors at 0.8 of the height.
  (Local lines and wire heights are assumed; see fetch_hazards.py.)

The REAL world is the map plus what the map does not know, drawn at random
(rates below, per area or per house, are assumptions where no data exists):

- growth: five years since the 2021 lidar, about 0.3 m a year at crown tops
  and edges;
- misregistration: each building's footprint off by about 1 m, each trunk
  0.7 m from the crown peak it was placed under;
- local power lines on the other side of the road than the map assumes
  (half of all roads), unmapped spans across fields, wires to sheds;
- sheds, carports and flagpoles not in the footprints;
- dead branches and saplings at flying height in the forest;
- storm-felled trees leaning on others;
- new buildings since the footprints were made;
- in winter with snow: branches weighed down by 1 m.

Also: open spots where a drone with a failure could land.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

VEG, TRUNK, BLDG, WIRE, POLE, STRUCT = 1, 2, 4, 8, 16, 32
KIND_NAMES = {VEG: "foliage", TRUNK: "trunk", BLDG: "building", WIRE: "wire", POLE: "pole", STRUCT: "structure"}
NZ = 50  # metres above the ground


@dataclass
class World3D:
    occ: np.ndarray  # (n, n, NZ) uint8 kind flags; [i, j, k]: row i (north at 0), column j, k m above ground
    dtm: np.ndarray  # (n, n) ground height, metres above sea level
    east0: float
    north0: float
    trees: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))  # x, y, height, crown r, crown base, trunk r
    surprises: list = field(default_factory=list)  # what the real world adds (kind, where)
    season: str = "leaf-on"

    @property
    def n(self):
        return self.occ.shape[0]

    def ij(self, x, y):
        return (int(np.clip(math.floor((self.north0 + self.n / 2) - y), 0, self.n - 1)),
                int(np.clip(math.floor(x - (self.east0 - self.n / 2)), 0, self.n - 1)))

    def xy(self, i, j):
        return self.east0 - self.n / 2 + j + 0.5, self.north0 + self.n / 2 - i - 0.5

    def inside(self, x, y, margin=0.0):
        return abs(x - self.east0) < self.n / 2 - margin and abs(y - self.north0) < self.n / 2 - margin

    def ground(self, x, y):
        return float(self.dtm[self.ij(x, y)])

    def copy(self):
        return World3D(self.occ.copy(), self.dtm, self.east0, self.north0, self.trees.copy(), list(self.surprises),
                       self.season)

    # ------------------------------------------------------------------ drawing
    def put_segment(self, p0, p1, kind, radius=0.0):
        """Mark voxels along a 3D segment (x, y, height above ground at each end)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1[:2] - p0[:2])) + abs(p1[2] - p0[2])
        m = max(2, int(L / 0.35) + 1)
        u = np.linspace(0, 1, m)[:, None]
        P = p0 + u * (p1 - p0)
        r = int(math.ceil(radius))
        for x, y, h in P:
            if not self.inside(x, y):
                continue
            i, j = self.ij(x, y)
            k = int(math.floor(h))
            if 0 <= k < NZ:
                self.occ[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1, k] |= kind

    def put_wire(self, a, b, h_a, h_b, sag, kind=WIRE):
        """A wire between two points sagging by `sag` in the middle (parabola for the catenary)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        m = max(2, int(L / 0.5) + 1)
        prev = None
        for u in np.linspace(0, 1, m):
            x, y = a + u * (b - a)
            h = h_a + u * (h_b - h_a) - 4 * sag * u * (1 - u)
            if prev is not None:
                self.put_segment(prev, (x, y, h), kind)
            prev = (x, y, h)

    def put_box(self, x, y, w, l, h, kind, ang=0.0):
        c, s_ = math.cos(ang), math.sin(ang)
        R = int(math.ceil(math.hypot(w, l) / 2)) + 1
        ci, cj = self.ij(x, y)
        for i in range(max(0, ci - R), min(self.n, ci + R + 1)):
            for j in range(max(0, cj - R), min(self.n, cj + R + 1)):
                px, py = self.xy(i, j)
                dx, dy = px - x, py - y
                if abs(dx * c + dy * s_) <= w / 2 and abs(-dx * s_ + dy * c) <= l / 2:
                    self.occ[i, j, :max(1, min(NZ, int(math.ceil(h))))] |= kind

    def put_pole(self, x, y, h, kind=POLE):
        if self.inside(x, y):
            i, j = self.ij(x, y)
            self.occ[i, j, :min(NZ, int(math.ceil(h)))] |= kind

    # ------------------------------------------------------------------ queries
    def top(self):
        """Highest occupied height above the ground per column (0 where empty)."""
        any_ = self.occ.any(axis=2)
        last = NZ - np.argmax(self.occ[:, :, ::-1] != 0, axis=2)
        return np.where(any_, last, 0).astype(float)

    def kinds_near(self, x, y, h, r):
        """Kinds of the occupied voxels within r (m) of a point (h above the ground), with distances."""
        if not self.inside(x, y):
            return []
        i, j = self.ij(x, y)
        R = int(math.ceil(r)) + 1
        k0, k1 = max(0, int(h - R)), min(NZ, int(h + R) + 1)
        i0, i1, j0, j1 = max(0, i - R), min(self.n, i + R + 1), max(0, j - R), min(self.n, j + R + 1)
        blk = self.occ[i0:i1, j0:j1, k0:k1]
        ii, jj, kk = np.nonzero(blk)
        if not len(ii):
            return []
        cx = self.east0 - self.n / 2 + (j0 + jj) + 0.5
        cy = self.north0 + self.n / 2 - (i0 + ii) - 0.5
        cz = k0 + kk + 0.5
        d = np.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - h) ** 2) - 0.5  # to the voxel's near side
        m = d <= r
        return list(zip(blk[ii[m], jj[m], kk[m]], d[m]))

    def landing_spots(self, radius=4.0, max_slope_deg=15.0):
        """Columns with nothing above them within `radius` and gentle ground: where a failing drone can land."""
        clear = ~self.occ.any(axis=2)
        gy, gx = np.gradient(self.dtm)
        gentle = np.degrees(np.arctan(np.hypot(gx, gy))) <= max_slope_deg
        return ndimage.distance_transform_edt(clear & gentle) > radius


# ---------------------------------------------------------------------- building the map
def segment_trees(top, bldg):
    """Watershed segmentation of the crown surface: markers at crown peaks (window grows with height)."""
    sm = ndimage.gaussian_filter(top, 1.0)
    w5 = ndimage.maximum_filter(sm, size=5)
    w9 = ndimage.maximum_filter(sm, size=9)
    peak = (sm >= 5.0) & ~bldg & (((sm == w5) & (sm < 15)) | ((sm == w9) & (sm >= 15)))
    markers, nm = ndimage.label(peak)
    img = np.clip((sm.max() - sm) * 10, 0, 65535).astype(np.uint16)
    markers = markers.astype(np.int32)
    markers[sm < 2.0] = -1  # background
    lab = ndimage.watershed_ift(img, markers)
    return lab, nm, peak


def build_map(fine_npz, points_npz, buildings_json, hazards_json, shift=(0.0, 0.0), season="leaf-on") -> World3D:
    d = np.load(fine_npz)
    n = d["dtm"].shape[0]
    e0, n0 = float(d["east0"]) + shift[0], float(d["north0"]) + shift[1]
    W = World3D(np.zeros((n, n, NZ), np.uint8), d["dtm"].astype(float), e0, n0, season=season)
    # --- vegetation from the points (layered)
    P = np.load(points_npz)
    xyz, hag, cls = P["xyz"].astype(float), P["hag"].astype(float), P["cls"]
    xyz[:, 0] += shift[0]
    xyz[:, 1] += shift[1]
    veg = (cls == 1) & (hag > 1.5) & (hag < NZ - 1)
    i = np.floor((n0 + n / 2) - xyz[veg, 1]).astype(int)
    j = np.floor(xyz[veg, 0] - (e0 - n / 2)).astype(int)
    k = np.floor(hag[veg]).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    cnt = np.zeros((n, n, NZ), np.uint16)
    np.add.at(cnt, (i[ok], j[ok], k[ok]), 1)
    hit = ndimage.uniform_filter(cnt.astype(np.float32), size=(3, 3, 1)) * 9 >= 1.0
    # close gaps shorter than 4 m inside a column (branches between the returns)
    hit = ndimage.binary_closing(hit, structure=np.ones((1, 1, 4), bool))
    if season == "leaf-on":  # foliage grows out about 1 m around the branches the laser hit
        hit = ndimage.binary_dilation(hit, structure=ndimage.generate_binary_structure(3, 1))
        hit[:, :, :1] = False
    bmask = d["bldg"] > 0.3
    hit[bmask] = False
    W.occ[hit] |= VEG
    # --- buildings with the lidar roof surface
    roof = np.zeros((n, n))
    b = cls == 6
    if b.any():
        bi = np.floor((n0 + n / 2) - xyz[b, 1]).astype(int)
        bj = np.floor(xyz[b, 0] - (e0 - n / 2)).astype(int)
        ok = (bi >= 0) & (bi < n) & (bj >= 0) & (bj < n)
        np.maximum.at(roof, (bi[ok], bj[ok]), hag[b][ok])
    roof = np.where(bmask, roof, 0.0)
    if bmask.any():  # fill footprint cells the laser missed with the building's typical roof height
        lab, nl = ndimage.label(bmask)
        med = ndimage.median(np.where(roof > 2, roof, np.nan), lab, index=np.arange(1, nl + 1))
        med = np.nan_to_num(np.asarray(med, float), nan=6.0)
        fill = med[np.maximum(lab - 1, 0)]
        roof = np.where(bmask & (roof < 2), fill, roof)
        roof = np.where(bmask, np.clip(roof, 3.0, 30.0), 0.0)
    ks = np.arange(NZ)[None, None, :]
    W.occ[(ks < np.ceil(roof)[:, :, None]) & bmask[:, :, None]] |= BLDG
    # --- trees: segmentation, trunks
    top = W.top() * (W.occ.any(axis=2) & ~bmask)
    lab, nl, peak = segment_trees(top, bmask)
    trees = []
    for (pi, pj) in np.argwhere(peak):
        h = float(top[pi, pj])
        if h < 5:
            continue
        seg = lab == lab[pi, pj]
        area = float(seg[max(0, pi - 10):pi + 11, max(0, pj - 10):pj + 11].sum())
        cr = float(np.clip(math.sqrt(area / math.pi), 1.5, 9.0))
        col = W.occ[max(0, pi - 1):pi + 2, max(0, pj - 1):pj + 2, :] & VEG
        zs = np.nonzero(col.any(axis=(0, 1)))[0]
        zs = zs[zs >= 3]
        cb = float(zs.min()) if len(zs) else 0.5 * h
        dbh_cm = max(8.0, 2.0 * (h - 1.3))
        tr = dbh_cm / 200.0
        x, y = W.xy(pi, pj)
        trees.append((x, y, h, cr, min(cb, h - 1.0), tr))
    W.trees = np.array(trees) if trees else np.zeros((0, 6))
    for x, y, h, cr, cb, tr in W.trees:
        W.put_segment((x, y, 0.0), (x, y, max(1.0, cb)), TRUNK, radius=0 if tr < 0.5 else 1)
    # --- power lines, poles, towers (map: assumed geometry)
    H = json.load(open(hazards_json)) if hazards_json and Path(hazards_json).exists() else {}
    B = json.load(open(buildings_json)) if buildings_json and Path(buildings_json).exists() else []
    W.lines = put_power(W, H, B, side=+1, rng=None)
    return W


def _poles_along(line, spacing, side, offset):
    """Pole positions along a polyline, `offset` m to one side."""
    P = np.asarray(line, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    out = []
    for s_ in np.arange(0, seg[-1] + 1e-6, spacing):
        k = min(int(np.searchsorted(seg, s_, side="right")) - 1, len(P) - 2)
        u = (s_ - seg[k]) / max(seg[k + 1] - seg[k], 1e-9)
        p = P[k] + u * (P[k + 1] - P[k])
        t = P[k + 1] - P[k]
        t = t / max(np.linalg.norm(t), 1e-9)
        nrm = np.array([-t[1], t[0]])
        out.append(p + side * offset * nrm)
    return out


def put_power(W: World3D, H: dict, B: list, side=+1, rng=None, side_of=None):
    """Distribution lines on poles along roads, service wires to houses, high-voltage lines, towers."""
    poles_all = []
    for r_i, road in enumerate(H.get("distribution", [])):
        sd = side_of(r_i) if side_of else side
        poles = _poles_along(road, 45.0 if rng is None else float(rng.uniform(35, 55)), sd, 5.0)
        for p in poles:
            W.put_pole(p[0], p[1], 10.0)
        for a, b in zip(poles[:-1], poles[1:]):
            for hh in (9.5, 8.0, 6.0):
                W.put_wire(a, b, hh, hh, 1.0)
        poles_all += poles
    if poles_all:
        PA = np.array(poles_all)
        for pg in B:
            c = np.mean(np.asarray(pg, float), axis=0)
            d = np.hypot(*(PA - c).T)
            k = int(np.argmin(d))
            if d[k] <= 80:
                # to the nearest wall point facing the pole
                pgv = np.asarray(pg, float)
                e = pgv[int(np.argmin(np.hypot(*(pgv - PA[k]).T)))]
                W.put_wire(PA[k], e, 6.0, 4.0, 0.6)
    for ln in H.get("transmission", []):
        towers = _poles_along(ln["pts"], 300.0, 0, 0.0)
        for t in towers:
            W.put_pole(t[0], t[1], 30.0, POLE)
        for a, b in zip(towers[:-1], towers[1:]):
            for off in (-4.0, 0.0, 4.0):
                dvec = b - a
                nrm = np.array([-dvec[1], dvec[0]]) / max(np.linalg.norm(dvec), 1e-9)
                W.put_wire(a + off * nrm, b + off * nrm, 24.0, 24.0, 6.0)
            W.put_wire(a, b, 30.0, 30.0, 4.0)
    for t in H.get("towers", []):
        x, y, h = t["x"], t["y"], min(t["h"], NZ - 1)
        W.put_pole(x, y, h, POLE)
        if t.get("guy_r", 0) > 3:
            for ang in (0.3, 0.3 + 2.094, 0.3 + 4.189):
                ax, ay = x + t["guy_r"] * math.cos(ang), y + t["guy_r"] * math.sin(ang)
                for frac in (0.35, 0.65, 0.95):
                    W.put_segment((x, y, h * frac), (ax, ay, 0.0), WIRE)
    return poles_all


# ---------------------------------------------------------------------- the real world
RATES = dict(  # assumptions where no data exists; per hectare of forest / of open land, or per house
    snag_per_ha_forest=25.0, sapling_per_ha_forest=40.0, felled_per_ha_forest=0.3,
    shed_per_house=0.35, flagpole_per_house=0.05, shed_wire_frac=0.5,
    field_span_per_ha_open=0.03, new_building_per_ha_open=0.01, road_side_flip=0.5,
    growth_m=1.5, building_shift_sd=1.0, trunk_shift_sd=0.7,
)


def make_truth(M: World3D, H: dict, B: list, seed=0, season=None, rates=None) -> World3D:
    """The real world: the map plus what the map does not know (see the module notes)."""
    R = dict(RATES, **(rates or {}))
    rng = np.random.default_rng(seed)
    T = M.copy()
    T.surprises = []
    n = T.n
    veg = (T.occ & VEG) > 0
    forest = veg.any(axis=2) & (T.top() > 8)
    open_ = ~T.occ.any(axis=2)
    area_forest = forest.sum() / 1e4
    area_open = open_.sum() / 1e4
    # growth: tops up by growth_m, crown edges out by 1 m (half of them)
    g = int(round(R["growth_m"]))
    if g > 0:
        grown = np.zeros_like(veg)
        grown[:, :, g:] = veg[:, :, :-g]
        edge = ndimage.binary_dilation(veg, structure=np.ones((3, 3, 1), bool)) & ~veg
        edge &= rng.random(edge.shape) < 0.5
        edge[:, :, :3] = False
        add = (grown | edge) & ~((T.occ & (BLDG | WIRE | POLE)) > 0)
        T.occ[add] |= VEG
    # winter with snow: branches weighed down by about 1 m
    if (season or M.season) == "snow":
        low = np.zeros_like(veg)
        low[:, :, :-1] = veg[:, :, 1:]
        T.occ[low & (np.arange(NZ)[None, None, :] >= 2)] |= VEG
    # trunks off the crown peak they were placed under
    T.occ &= ~np.uint8(TRUNK)
    for x, y, h, cr, cb, tr in T.trees:
        dx, dy = rng.normal(0, R["trunk_shift_sd"], 2)
        T.put_segment((x + dx, y + dy, 0.0), (x + dx, y + dy, max(1.0, cb)), TRUNK, radius=0 if tr < 0.5 else 1)
    # buildings off by about 1 m: shift each footprint's voxels
    bm = (T.occ & BLDG) > 0
    lab, nl = ndimage.label(bm.any(axis=2))
    newb = np.zeros_like(bm)
    for l_ in range(1, nl + 1):
        sl = ndimage.find_objects((lab == l_).astype(int))[0]
        di, dj = (int(round(v)) for v in rng.normal(0, R["building_shift_sd"], 2))
        src = bm[sl] & (lab[sl] == l_)[:, :, None]
        i0, j0 = sl[0].start + di, sl[1].start + dj
        if 0 <= i0 and 0 <= j0 and i0 + src.shape[0] <= n and j0 + src.shape[1] <= n:
            newb[i0:i0 + src.shape[0], j0:j0 + src.shape[1]] |= src
        else:
            newb[sl] |= src
    T.occ &= ~np.uint8(BLDG)
    T.occ[newb] |= BLDG
    T.occ[newb & ((T.occ & VEG) > 0)] &= ~np.uint8(VEG)
    # power lines: on the other side of the road for some roads; spans across fields; wires to sheds
    T.occ &= ~np.uint8(WIRE | POLE)
    flips = rng.random(len(H.get("distribution", []))) < R["road_side_flip"]
    poles = put_power(T, H, B, rng=rng, side_of=lambda r: -1 if flips[r] else +1)
    T.surprises.append(dict(kind="power lines on the other side of the road", n=int(flips.sum())))
    oi, oj = np.nonzero(open_)
    for _ in range(rng.poisson(R["field_span_per_ha_open"] * area_open)):
        k = rng.integers(len(oi))
        a = np.array(T.xy(oi[k], oj[k]))
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(80, 250)
        b = a + L * np.array([math.cos(ang), math.sin(ang)])
        for p in (a, b):
            T.put_pole(p[0], p[1], 10.0)
        T.put_wire(a, b, rng.uniform(7, 10), rng.uniform(7, 10), 1.2)
        T.surprises.append(dict(kind="unmapped span", a=a.tolist(), b=b.tolist()))
    # sheds, carports, flagpoles near houses
    for pg in B:
        c = np.mean(np.asarray(pg, float), axis=0)
        if not T.inside(*c, 30):
            continue
        if rng.random() < R["shed_per_house"]:
            ang = rng.uniform(0, 2 * np.pi)
            p = c + rng.uniform(12, 30) * np.array([math.cos(ang), math.sin(ang)])
            T.put_box(p[0], p[1], rng.uniform(2.5, 5), rng.uniform(3, 7), rng.uniform(2.5, 4.0), STRUCT,
                      rng.uniform(0, np.pi))
            T.surprises.append(dict(kind="shed", x=float(p[0]), y=float(p[1])))
            if rng.random() < R["shed_wire_frac"]:
                T.put_wire(c, p, 3.5, 2.8, 0.3)
                T.surprises.append(dict(kind="wire to a shed", a=c.tolist(), b=p.tolist()))
        if rng.random() < R["flagpole_per_house"]:
            p = c + rng.uniform(6, 15) * np.array([math.cos(rng.uniform(0, 6.3)), math.sin(rng.uniform(0, 6.3))])
            T.put_pole(p[0], p[1], rng.uniform(6, 9), STRUCT)
            T.surprises.append(dict(kind="flagpole", x=float(p[0]), y=float(p[1])))
    # forest clutter at flying height: dead branches, saplings, storm-felled trees
    fi, fj = np.nonzero(forest)
    for _ in range(rng.poisson(R["snag_per_ha_forest"] * area_forest)):
        k = rng.integers(len(fi))
        x, y = T.xy(fi[k], fj[k])
        h = rng.uniform(1.5, 6.0)
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(1.5, 4.0)
        T.put_segment((x, y, h), (x + L * math.cos(ang), y + L * math.sin(ang), h + rng.uniform(-1, 1)), VEG)
    T.surprises.append(dict(kind="dead branches at flying height", n=int(R["snag_per_ha_forest"] * area_forest)))
    for _ in range(rng.poisson(R["sapling_per_ha_forest"] * area_forest)):
        k = rng.integers(len(fi))
        x, y = T.xy(fi[k], fj[k])
        h = rng.uniform(2.0, 6.0)
        T.put_segment((x, y, 0.0), (x, y, h), VEG)
        T.put_box(x, y, 1.5, 1.5, 0, VEG)  # no-op height guard
        i, j = T.ij(x, y)
        T.occ[max(0, i - 1):i + 2, max(0, j - 1):j + 2, int(h * 0.6):int(h) + 1] |= VEG
    T.surprises.append(dict(kind="saplings", n=int(R["sapling_per_ha_forest"] * area_forest)))
    for _ in range(rng.poisson(R["felled_per_ha_forest"] * area_forest)):
        k = rng.integers(len(fi))
        x, y = T.xy(fi[k], fj[k])
        L = rng.uniform(12, 25)
        ang = rng.uniform(0, 2 * np.pi)
        lean = math.radians(rng.uniform(25, 60))  # caught by the crowns of its neighbours
        tip = (x + L * math.cos(lean) * math.cos(ang), y + L * math.cos(lean) * math.sin(ang), L * math.sin(lean))
        T.put_segment((x, y, 0.5), tip, TRUNK, radius=0)
        T.surprises.append(dict(kind="storm-felled tree", a=[x, y], b=[tip[0], tip[1]]))
    for _ in range(rng.poisson(R["new_building_per_ha_open"] * area_open)):
        k = rng.integers(len(oi))
        x, y = T.xy(oi[k], oj[k])
        T.put_box(x, y, rng.uniform(8, 14), rng.uniform(9, 16), rng.uniform(5, 9), BLDG, rng.uniform(0, np.pi))
        T.surprises.append(dict(kind="new building", x=float(x), y=float(y)))
    return T


def save(W: World3D, path):
    np.savez_compressed(path, occ=W.occ, dtm=W.dtm, east0=W.east0, north0=W.north0, trees=W.trees,
                        season=W.season, surprises=json.dumps(W.surprises))


def load(path) -> World3D:
    d = np.load(path, allow_pickle=False)
    return World3D(d["occ"], d["dtm"], float(d["east0"]), float(d["north0"]), d["trees"],
                   json.loads(str(d["surprises"])), str(d["season"]))
