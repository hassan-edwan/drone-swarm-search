"""Read USGS 3DEP lidar (Entwine Point Tiles on AWS) for a site, and make
raster layers from it.

    python scripts/real_site/lidar.py coarse --site sites/catoctin --depth 7
    python scripts/real_site/lidar.py fine --site sites/catoctin --east 0 --north -700 --size-m 1000

Points come from s3://usgs-lidar-public/<project> (public, no key), stored as
an octree: each level down adds detail, so reading levels 0..d gives an
evenly thinned cloud. Classes (ASPRS): 2 ground, 3-5 vegetation, 6 building,
9 water, 7/18 noise.

coarse (whole site, 10 m cells): ground height (dtm), canopy height (chm,
95th percentile of vegetation above ground), canopy cover (share of first
returns above 2 m), building cover and height. Written to lidar10.npz.

fine (a smaller area, full density): the points themselves (local metres,
height above ground, class) in fine_points.npz, and 1 m rasters (dtm,
surface, canopy top, crown base, buildings) in fine1.npz. The crown base is
the bottom of the lowest dense layer of vegetation over each 1 m cell: the
clearance a drone could fly under.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET = "https://usgs-lidar-public.s3.amazonaws.com"
R_EARTH = 6378137.0
# Vegetation classes. Many 3DEP projects leave trees "unclassified" (1); buildings are
# then also in class 1, so building footprints come from another source (buildings.py).
VEG = (1, 3, 4, 5)


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def merc(lat, lon):
    return lon * math.pi / 180 * R_EARTH, math.log(math.tan((90 + lat) * math.pi / 360)) * R_EARTH


class EPT:
    def __init__(self, project: str):
        self.url = f"{BUCKET}/{project}"
        self.meta = json.loads(get(f"{self.url}/ept.json"))
        self.b = self.meta["bounds"]
        self._h: dict = {}

    def _hier(self, key="0-0-0-0"):
        h = json.loads(get(f"{self.url}/ept-hierarchy/{key}.json"))
        for k, n in h.items():
            if n == -1:
                self._hier(k)
            else:
                self._h[k] = n

    def node_box(self, key):
        d, x, y, z = (int(v) for v in key.split("-"))
        s = (self.b[3] - self.b[0]) / 2 ** d
        return self.b[0] + x * s, self.b[1] + y * s, self.b[0] + (x + 1) * s, self.b[1] + (y + 1) * s

    def nodes(self, box, max_depth):
        """Keys of nodes down to max_depth that overlap box (xmin, ymin, xmax, ymax), in web mercator."""
        if not self._h:
            self._hier()
        out = []
        for k, n in self._h.items():
            if n <= 0 or int(k.split("-")[0]) > max_depth:
                continue
            x0, y0, x1, y1 = self.node_box(k)
            if x0 < box[2] and x1 > box[0] and y0 < box[3] and y1 > box[1]:
                out.append(k)
        return sorted(out, key=lambda k: int(k.split("-")[0]))

    def read(self, key):
        import laspy
        las = laspy.read(io.BytesIO(get(f"{self.url}/ept-data/{key}.laz")))
        return np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], np.asarray(las.classification), \
            np.asarray(las.return_number), np.asarray(las.number_of_returns)

    def points(self, box, max_depth, workers=8, log=print):
        keys = self.nodes(box, max_depth)
        log(f"{len(keys)} nodes, {sum(self._h[k] for k in keys):,} points before cropping")
        P, C, RN = [], [], []
        with ThreadPoolExecutor(workers) as ex:
            for i, (xyz, c, rn, nr) in enumerate(ex.map(self.read, keys)):
                m = (xyz[:, 0] >= box[0]) & (xyz[:, 0] < box[2]) & (xyz[:, 1] >= box[1]) & (xyz[:, 1] < box[3])
                P.append(xyz[m])
                C.append(c[m])
                RN.append(rn[m] == 1)
                if i % 50 == 0:
                    log(f"  read {i + 1}/{len(keys)}")
        return np.concatenate(P), np.concatenate(C), np.concatenate(RN)


def to_local(xyz, lat0, lon0):
    """Web mercator to metres east/north of the site centre (mercator is stretched by 1/cos(lat))."""
    mx0, my0 = merc(lat0, lon0)
    k = math.cos(math.radians(lat0))
    out = xyz.copy()
    out[:, 0] = (xyz[:, 0] - mx0) * k
    out[:, 1] = (xyz[:, 1] - my0) * k
    return out


def box_merc(lat0, lon0, east0, north0, size_m):
    """Mercator box of a size_m square whose centre is east0/north0 metres from (lat0, lon0)."""
    mx0, my0 = merc(lat0, lon0)
    k = math.cos(math.radians(lat0))
    h = size_m / 2
    return ((east0 - h) / k + mx0, (north0 - h) / k + my0, (east0 + h) / k + mx0, (north0 + h) / k + my0)


def grid_index(p, east0, north0, size_m, cell):
    n = int(round(size_m / cell))
    j = np.floor((p[:, 0] - (east0 - size_m / 2)) / cell).astype(int)
    i = np.floor(((north0 + size_m / 2) - p[:, 1]) / cell).astype(int)  # row 0 = north
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    return i, j, ok, n


def fill_nan(a, iters=200):
    """Fill gaps by repeated neighbour averaging."""
    a = a.copy()
    for _ in range(iters):
        m = np.isnan(a)
        if not m.any():
            break
        p = np.pad(a, 1, mode="edge")
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = (~np.isnan(nb)).sum(0)
        s = np.nansum(nb, 0)
        upd = m & (cnt > 0)
        a[upd] = s[upd] / cnt[upd]
    return a


def stat_grid(i, j, v, n, how):
    out = np.full(n * n, np.nan)
    if len(i) == 0:
        return np.zeros((n, n)) if how == "count" else out.reshape(n, n)
    flat = i * n + j
    if how == "min":
        np.fmin.at(out, flat, v)
    elif how == "max":
        np.fmax.at(out, flat, v)
    elif how == "count":
        out = np.bincount(flat, minlength=n * n).astype(float)
    elif how == "mean":
        s = np.bincount(flat, weights=v, minlength=n * n)
        c = np.bincount(flat, minlength=n * n)
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    elif how.startswith("p"):
        q = float(how[1:]) / 100
        order = np.lexsort((v, flat))
        fs, vs = flat[order], v[order]
        starts = np.r_[0, np.flatnonzero(np.diff(fs)) + 1]
        ends = np.r_[starts[1:], len(fs)]
        idx = starts + ((ends - starts - 1) * q).astype(int)
        out[fs[starts]] = vs[idx]
    return out.reshape(n, n)


def layers(p, cls, first, east0, north0, size_m, cell):
    """Rasters at `cell` metres from local points."""
    i, j, ok, n = grid_index(p, east0, north0, size_m, cell)
    p, cls, first, i, j = p[ok], cls[ok], first[ok], i[ok], j[ok]
    keep = ~np.isin(cls, (7, 18))
    p, cls, first, i, j = p[keep], cls[keep], first[keep], i[keep], j[keep]
    g = cls == 2
    dtm = stat_grid(i[g], j[g], p[g, 2], n, "mean")
    dtm = fill_nan(dtm)
    hag = p[:, 2] - dtm[i, j]
    veg = np.isin(cls, VEG) & (hag > 2)
    chm = np.nan_to_num(stat_grid(i[veg], j[veg], hag[veg], n, "p95"), nan=0.0)
    fr = first
    n_first = stat_grid(i[fr], j[fr], None, n, "count")
    n_hi = stat_grid(i[fr & (hag > 2) & ~(cls == 6)], j[fr & (hag > 2) & ~(cls == 6)], None, n, "count")
    cover = np.where(n_first > 0, n_hi / np.maximum(n_first, 1), 0.0)
    b = cls == 6
    n_b = stat_grid(i[fr & b], j[fr & b], None, n, "count")
    bldg = np.where(n_first > 0, n_b / np.maximum(n_first, 1), 0.0)
    bh = np.nan_to_num(stat_grid(i[b], j[b], hag[b], n, "p90"), nan=0.0)
    w = cls == 9
    water = np.where(n_first > 0, stat_grid(i[fr & w], j[fr & w], None, n, "count") / np.maximum(n_first, 1), 0.0)
    return dict(dtm=dtm, chm=chm, cover=cover, bldg=bldg, bldg_h=bh, water=water, cell=cell, size_m=size_m,
                east0=east0, north0=north0), (p, cls, hag)


def crown_base(p, hag, cls, east0, north0, size_m, cell=1.0, min_pts=2):
    """Per cell: top of the vegetation (m above ground) and the bottom of the
    lowest dense vegetation layer above 3 m (np.inf where there is none). The
    gap from the ground (after the undergrowth below 3 m) to that bottom is
    what a drone can fly through."""
    i, j, ok, n = grid_index(p, east0, north0, size_m, cell)
    veg = ok & np.isin(cls, VEG) & (hag > 3.0) & (hag < 60)
    top = np.nan_to_num(stat_grid(i[veg], j[veg], hag[veg], n, "max"), nan=0.0)
    # bottom of the crown: the 5th percentile of vegetation heights above 3 m in a 3 x 3 neighbourhood
    flat = (i[veg] * n + j[veg])
    h = hag[veg]
    base = np.full((n, n), np.inf)
    cnt = np.bincount(flat, minlength=n * n).reshape(n, n)
    b5 = stat_grid(i[veg], j[veg], h, n, "p5")
    has = cnt >= min_pts
    base[has] = b5[has]
    return top, base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["coarse", "fine"])
    ap.add_argument("--site", default="sites/catoctin")
    ap.add_argument("--project", default="MD_Western_2_D21")
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--east", type=float, default=0.0)
    ap.add_argument("--north", type=float, default=0.0)
    ap.add_argument("--size-m", type=float, default=1000.0)
    a = ap.parse_args()
    site = Path(a.site)
    meta = json.load(open(site / "meta.json"))
    lat0, lon0 = meta["lat"], meta["lon"]
    ept = EPT(a.project)
    t0 = time.time()
    if a.what == "coarse":
        size = meta["size_m"]
        box = box_merc(lat0, lon0, 0.0, 0.0, size)
        cache = site / f"coarse_points_d{a.depth}.npz"
        if cache.exists():
            d = np.load(cache)
            xyz, cls, first = d["xyz"], d["cls"], d["first"]
        else:
            xyz, cls, first = ept.points(box, a.depth)
            np.savez(cache, xyz=xyz, cls=cls, first=first)
        print("classes:", dict(zip(*np.unique(cls, return_counts=True))))
        p = to_local(xyz, lat0, lon0)
        L, _ = layers(p, cls, first, 0.0, 0.0, size, 10.0)
        np.savez_compressed(site / "lidar10.npz", **L)
        print(f"{len(p):,} points ({len(p) / size ** 2:.3f}/m2) in {time.time() - t0:.0f}s; "
              f"canopy cover {L['cover'].mean():.0%}, canopy height median {np.median(L['chm'][L['cover'] > 0.5]):.0f} m, "
              f"buildings {(L['bldg'] > 0.3).mean():.2%} of area")
    else:
        box = box_merc(lat0, lon0, a.east, a.north, a.size_m)
        cache = site / f"fine_raw_{a.east:.0f}_{a.north:.0f}_{a.size_m:.0f}.npz"
        if cache.exists():
            d = np.load(cache)
            xyz, cls, first = d["xyz"], d["cls"], d["first"]
        else:
            xyz, cls, first = ept.points(box, 30)
            np.savez(cache, xyz=xyz, cls=cls, first=first)
        print("classes:", dict(zip(*np.unique(cls, return_counts=True))))
        p = to_local(xyz, lat0, lon0)
        L, (p, cls, hag) = layers(p, cls, first, a.east, a.north, a.size_m, 1.0)
        top, base = crown_base(p, hag, cls, a.east, a.north, a.size_m)
        np.savez_compressed(site / "fine1.npz", **L, veg_top=top, crown_base=base)
        np.savez_compressed(site / "fine_points.npz", xyz=p.astype(np.float32), hag=hag.astype(np.float32),
                            cls=cls.astype(np.uint8))
        under = np.isfinite(base) & (base > 6)
        print(f"{len(p):,} points ({len(p) / a.size_m ** 2:.1f}/m2) in {time.time() - t0:.0f}s; canopy cover "
              f"{L['cover'].mean():.0%}; where there are trees, {under[top > 3].mean():.0%} of cells have >6 m clear under the crown")


if __name__ == "__main__":
    sys.exit(main())
