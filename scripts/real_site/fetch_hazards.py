"""Wires and tall structures for a site: what lidar and footprints miss.

    python scripts/real_site/fetch_hazards.py --site sites/catoctin_west

- Towers, masts, poles, stacks: FAA Digital Obstacle File (aeronav.faa.gov,
  every registered obstacle in the US with its height above ground). Guyed
  structures (towers, antennas) have guy wires reaching out to about 0.8 of
  their height; the whole cone is treated as blocked.
- High-voltage lines: HIFLD Electric Power Transmission Lines. Conductors sag
  between pylons, so the corridor (15 m each side) is blocked from 8 m to 45 m
  above the ground.
- Local power lines are not in public data. Assumed along every road (most
  rural distribution lines are): wires from 5 m to 13 m above the ground,
  6 m either side of the road; and a service wire from the nearest road to
  each building (within 80 m), 3.5 m to 8 m above the ground.

Writes hazards.json (metres east/north of the site centre). The heights are
assumptions where noted; see docs/REALISM.md.
"""

import argparse
import io
import json
import math
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 110_540.0
DOF_URL = "https://aeronav.faa.gov/Obst_Data/DAILY_DOF_DAT.ZIP"
HIFLD = ("https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
         "Electric_Power_Transmission_Lines/FeatureServer/0/query")
GUYED = ("TOWER", "ANTENNA", "T-L TWR", "WINDMILL")
TRANSMISSION = dict(z0=8.0, z1=45.0, half_w=15.0)
DISTRIBUTION = dict(z0=5.0, z1=13.0, half_w=6.0)
SERVICE = dict(z0=3.5, z1=8.0, half_w=1.5, max_len=80.0)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def dms(deg, mins, sec, hemi):
    v = float(deg) + float(mins) / 60 + float(sec) / 3600
    return -v if hemi in "SW" else v


def towers(bbox, cache: Path, lat0, lon0):
    if not cache.exists():
        cache.write_bytes(get(DOF_URL))
    z = zipfile.ZipFile(cache)
    out = []
    kx = 111_320.0 * math.cos(math.radians(lat0))
    with z.open([n for n in z.namelist() if n.upper().endswith(".DAT")][0]) as f:
        for raw in io.TextIOWrapper(f, encoding="latin1"):
            if len(raw) < 90 or not raw[:2].isdigit():
                continue
            try:
                p = raw[35:47].split()
                q = raw[48:61].split()
                lat = dms(p[0], p[1], p[2][:-1], p[2][-1])
                lon = dms(q[0], q[1], q[2][:-1], q[2][-1])
            except (IndexError, ValueError):
                continue
            if not (bbox[1] <= lat <= bbox[3] and bbox[0] <= lon <= bbox[2]):
                continue
            kind = raw[62:80].strip()
            agl_m = float(raw[83:88]) * 0.3048
            out.append(dict(x=(lon - lon0) * kx, y=(lat - lat0) * M_PER_DEG_LAT, h=agl_m, kind=kind,
                            guy_r=0.8 * agl_m if kind in GUYED else 3.0))
    return out


def lines(bbox, lat0, lon0):
    q = dict(geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326,
             spatialRel="esriSpatialRelIntersects", outFields="VOLTAGE,TYPE", returnGeometry="true", f="geojson",
             where="1=1")
    d = json.loads(get(HIFLD + "?" + urllib.parse.urlencode(q)))
    kx = 111_320.0 * math.cos(math.radians(lat0))
    out = []
    for f in d.get("features", []):
        g = f.get("geometry") or {}
        parts = g.get("coordinates", [])
        if g.get("type") == "LineString":
            parts = [parts]
        for part in parts:
            out.append(dict(kv=(f.get("properties") or {}).get("VOLTAGE"),
                            pts=[((x - lon0) * kx, (y - lat0) * M_PER_DEG_LAT) for x, y, *_ in part]))
    return out


def service_drops(roads, buildings):
    """A wire from the nearest point of a road to each building's centre."""
    segs = [(np.asarray(a), np.asarray(b)) for ln in roads for a, b in zip(ln[:-1], ln[1:])]
    if not segs:
        return []
    A = np.array([s[0] for s in segs])
    B = np.array([s[1] for s in segs])
    out = []
    for pg in buildings:
        c = np.mean(np.asarray(pg, float), axis=0)
        ab = B - A
        u = np.clip(np.einsum("ij,ij->i", c - A, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        q = A + u[:, None] * ab
        d = np.hypot(*(q - c).T)
        k = int(np.argmin(d))
        if d[k] <= SERVICE["max_len"]:
            out.append([q[k].tolist(), c.tolist()])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    a = ap.parse_args()
    site = Path(a.site)
    meta = json.load(open(site / "meta.json"))
    lat0, lon0 = meta["lat"], meta["lon"]
    b = meta["bbox"]
    m = 0.02  # look a little outside the site: guy wires and lines reach in
    bbox = (b[0] - m, b[1] - m, b[2] + m, b[3] + m)
    tw = towers(bbox, site.parent / "_cache" / "DOF.zip", lat0, lon0)
    tl = lines(bbox, lat0, lon0)
    vec = json.load(open(site / "vectors.json")) if (site / "vectors.json").exists() else {}
    bl = json.load(open(site / "buildings.json")) if (site / "buildings.json").exists() else []
    roads = vec.get("road", [])
    drops = service_drops(roads, bl)
    out = dict(towers=tw, transmission=tl, distribution=roads, service=drops,
               bands=dict(transmission=TRANSMISSION, distribution=DISTRIBUTION, service=SERVICE),
               sources=dict(towers="FAA Digital Obstacle File (daily)", transmission="HIFLD transmission lines",
                            distribution="assumed along USGS roads", service="assumed: nearest road to each building"))
    json.dump(out, open(site / "hazards.json", "w"))
    print(f"{len(tw)} registered obstacles ({', '.join(sorted(set(t['kind'] for t in tw))) or 'none'}; "
          f"tallest {max((t['h'] for t in tw), default=0):.0f} m), {len(tl)} transmission line pieces, "
          f"{len(roads)} road pieces with assumed distribution lines, {len(drops)} service wires")


if __name__ == "__main__":
    main()
