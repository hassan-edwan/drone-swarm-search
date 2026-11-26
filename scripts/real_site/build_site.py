"""Combine a site's real layers into the simulator's world (site.npz).

    python scripts/real_site/build_site.py --site sites/catoctin

Uses, when present (else the photo-based fallbacks from fetch_site.py):
- lidar10.npz (lidar.py coarse): bare-earth elevation and tree cover.
- trail.npy, road.npy (fetch_vectors.py): trails and roads, which lost people
  tend to follow; lake.npy: lakes and ponds. Streams are not passed: the
  simulator derives drainages from the elevation.
Writes the inputs on the 30 m DEM grid, then runs import_site.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def to_grid(a, n):
    """Resample a square raster to n x n by block mean (nearest block edges)."""
    m = a.shape[0]
    e = np.linspace(0, m, n + 1).round().astype(int)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.nanmean(a[e[i]:max(e[i] + 1, e[i + 1]), e[j]:max(e[j] + 1, e[j + 1])])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin")
    ap.add_argument("--season", choices=["leaf-on", "leaf-off"], default="leaf-on",
                    help="tree cover for summer (area under crowns) or winter (lidar returns blocked)")
    ap.add_argument("--cell-out", type=float, default=0.0, help="search grid cell size (default: from the category)")
    a = ap.parse_args()
    site = Path(a.site)
    meta = json.load(open(site / "meta.json"))
    n = int(round(meta["size_m"] / meta["dem_m"]))
    dem, canopy = np.load(site / "dem.npy"), np.load(site / "canopy.npy")
    src = "photo"
    if (site / "lidar10.npz").exists():
        L = np.load(site / "lidar10.npz")
        key = "crown" if a.season == "leaf-on" and "crown" in L else "cover"
        dem, canopy = to_grid(L["dtm"], n), to_grid(L[key], n)
        src = f"lidar ({a.season}: {key})"
    np.save(site / "in_dem.npy", dem)
    np.save(site / "in_canopy.npy", canopy)
    trail = np.zeros((n, n), bool)
    for f in ("trail.npy", "road.npy"):
        if (site / f).exists():
            trail |= np.load(site / f)
    np.save(site / "in_trail.npy", trail)
    water = np.load(site / "lake.npy") if (site / "lake.npy").exists() else np.load(site / "water.npy")
    np.save(site / "in_water.npy", water)
    c = (n - 1) / 2
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "import_site.py"), "--dem", str(site / "in_dem.npy"),
           "--canopy", str(site / "in_canopy.npy"), "--trails", str(site / "in_trail.npy"),
           "--water", str(site / "in_water.npy"), "--cell-m", str(meta["dem_m"]), "--ipp", str(c + 0.5), str(c + 0.5),
           "--out", str(site / "site.npz")]
    if a.cell_out:
        cmd += ["--cell-out", str(a.cell_out)]
    print(f"elevation and tree cover from the {src}")
    subprocess.run(cmd, check=True)
    if (site / "lidar10.npz").exists() and a.season == "leaf-on":
        # the same site with leaf-off cover, for winter missions (field env loads it by itself)
        np.save(site / "in_canopy_leafoff.npy", to_grid(np.load(site / "lidar10.npz")["cover"], n))
        cmd2 = list(cmd)
        cmd2[cmd2.index("--canopy") + 1] = str(site / "in_canopy_leafoff.npy")
        cmd2[cmd2.index("--out") + 1] = str(site / "site_leafoff.npz")
        subprocess.run(cmd2, check=True)


if __name__ == "__main__":
    main()
