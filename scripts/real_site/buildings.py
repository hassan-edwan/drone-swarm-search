"""Building footprints for a site (Microsoft US Building Footprints, ODbL).

    python scripts/real_site/buildings.py --site sites/catoctin --state Maryland

Downloads the state's footprints once (sites/_cache/), keeps those inside the
site, and writes buildings.json: polygons in metres east/north of the site
centre. Heights come later from the lidar (lidar.py), which in many 3DEP
projects does not label buildings itself.
"""

import argparse
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

URL = "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/{state}.geojson.zip"
M_PER_DEG_LAT = 110_540.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin")
    ap.add_argument("--state", default="Maryland")
    a = ap.parse_args()
    site = Path(a.site)
    meta = json.load(open(site / "meta.json"))
    lat0, lon0 = meta["lat"], meta["lon"]
    b = meta["bbox"]
    cache = site.parent / "_cache" / f"{a.state}.geojson.zip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(URL.format(state=a.state), timeout=600) as r:
            cache.write_bytes(r.read())
    zf = zipfile.ZipFile(cache)
    name = zf.namelist()[0]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    out = []
    with zf.open(name) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            line = line.strip().rstrip(",")
            if not line.startswith('{"type":"Feature"') and not line.startswith('{ "type": "Feature"'):
                continue
            feat = json.loads(line)
            ring = feat["geometry"]["coordinates"][0]
            x0 = ring[0][0]
            y0 = ring[0][1]
            if not (b[0] <= x0 <= b[2] and b[1] <= y0 <= b[3]):
                continue
            out.append([((x - lon0) * kx, (y - lat0) * M_PER_DEG_LAT) for x, y in ring])
    json.dump(out, open(site / "buildings.json", "w"))
    area = sum(abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]))) / 2 for p in out)
    print(f"{len(out)} buildings, {area / 1e4:.1f} ha of roof in the site")


if __name__ == "__main__":
    main()
