"""Trails, roads, streams and lakes for a site, from The National Map (USGS).

    python scripts/real_site/fetch_vectors.py --site sites/catoctin

Sources (public, no key):
- Trails and roads: USGS National Transportation Dataset (carto.nationalmap.gov).
- Streams and lakes: National Hydrography Dataset, large scale (hydro.nationalmap.gov).

Writes vectors.json (lines and polygons in metres east/north of the site
centre) and masks on the site's 30 m DEM grid: trail.npy (trails and small
roads: where lost people walk), road.npy, stream.npy, lake.npy.
"""

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 110_540.0
LAYERS = {
    "trail": ("https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer", [37, 35]),
    "road": ("https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer", [29, 30, 31, 32]),
    "stream": ("https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer", [6]),
    "lake": ("https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer", [12]),
}


def get_json(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def query(service, layer, bbox):
    feats, off = [], 0
    while True:
        q = dict(geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326,
                 spatialRel="esriSpatialRelIntersects", outFields="*", returnGeometry="true", f="geojson",
                 resultOffset=off, resultRecordCount=1000, where="1=1")
        d = get_json(f"{service}/{layer}/query?" + urllib.parse.urlencode(q))
        got = d.get("features", [])
        feats += got
        if len(got) < 1000 and not d.get("exceededTransferLimit"):
            return feats
        off += len(got)


def to_local(lon, lat, lat0, lon0):
    return ((lon - lon0) * 111_320.0 * math.cos(math.radians(lat0)), (lat - lat0) * M_PER_DEG_LAT)


def parts(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "LineString":
        return [c]
    if t == "MultiLineString":
        return c
    if t == "Polygon":
        return c[:1]
    if t == "MultiPolygon":
        return [p[0] for p in c]
    return []


def rasterize_lines(lines, n, size_m, step):
    m = np.zeros((n, n), bool)
    cell = size_m / n
    for ln in lines:
        pts = np.asarray(ln, float)
        for a, b in zip(pts[:-1], pts[1:]):
            k = max(2, int(np.hypot(*(b - a)) / (cell / 3)) + 1)
            t = np.linspace(0, 1, k)[:, None]
            q = a * (1 - t) + b * t
            j = np.floor((q[:, 0] + size_m / 2) / cell).astype(int)
            i = np.floor((size_m / 2 - q[:, 1]) / cell).astype(int)
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            m[i[ok], j[ok]] = True
    return m


def rasterize_polys(polys, n, size_m):
    from matplotlib.path import Path as MPath
    cell = size_m / n
    c = (np.arange(n) + 0.5) * cell - size_m / 2
    X, Y = np.meshgrid(c, -c)
    pts = np.c_[X.ravel(), Y.ravel()]
    m = np.zeros(n * n, bool)
    for pg in polys:
        if len(pg) >= 3:
            m |= MPath(np.asarray(pg)).contains_points(pts)
    return m.reshape(n, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin")
    a = ap.parse_args()
    site = Path(a.site)
    meta = json.load(open(site / "meta.json"))
    lat0, lon0, size = meta["lat"], meta["lon"], meta["size_m"]
    n = int(round(size / meta["dem_m"]))
    bbox = meta["bbox"]
    out = {}
    for name, (svc, ids) in LAYERS.items():
        lines = []
        for lid in ids:
            for f in query(svc, lid, bbox):
                if f.get("geometry"):
                    for pt in parts(f["geometry"]):
                        lines.append([to_local(x, y, lat0, lon0) for x, y, *_ in pt])
        out[name] = lines
        if name == "lake":
            mask = rasterize_polys(lines, n, size)
        else:
            mask = rasterize_lines(lines, n, size, size / n)
        np.save(site / f"{name}.npy", mask)
        km = sum(np.hypot(*np.diff(np.asarray(ln), axis=0).T).sum() for ln in lines if len(ln) > 1) / 1000
        print(f"{name}: {len(lines)} pieces, {km:.0f} km of line, {mask.sum()} of {n * n} grid cells")
    json.dump(out, open(site / "vectors.json", "w"))


if __name__ == "__main__":
    main()
