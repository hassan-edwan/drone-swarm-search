"""What the 3D obstacles mean for the search grid: per-cell flight limits for the simulator.

    python scripts/real_site/flight_limits.py --site sites/catoctin_west --fine-from sites/catoctin/fine1.npz \\
        --shift 2000 0 --season leaf-on --drone sar

For each cell of the site's search grid (site.npz):
- min_level: the lowest of the family's search heights that clears every crown,
  roof, wire and tower in the cell by CLEAR_M (3 = none does: the cell is a
  no-fly cell). From the 10 m lidar layers and hazards.json, everywhere.
- under_p, under_s: for forest cells inside the area mapped at full lidar
  density, the share of under-canopy checks that get within REACH_M of a spot
  under the crowns, and how long the whole check takes (from the cell's
  centre at the lowest search height: there, down through a gap, under the
  crowns, look, back, up), from the 3D planner on sampled spots. -1 where not
  known (the simulator then keeps its default).

Writes flight_limits.npz next to site.npz; the field environment loads it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch.field.params import FAMILY  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--fine-from", default="", help="lidar.py fine output (none: only min_level)")
    ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--season", default="leaf-on", choices=["leaf-on", "leaf-off"])
    ap.add_argument("--drone", default="sar", choices=list(f3.DRONES))
    ap.add_argument("--samples", type=int, default=3, help="spots per forest cell")
    a = ap.parse_args()
    site = Path(a.site)
    S = np.load(site / "site.npz")
    g = S["elev"].shape[0]
    cell = float(S["cell_m"])
    fam = FAMILY["wilderness"]
    hz = site / "hazards.json"
    outer = f3.CoarseTerrain(site / "lidar10.npz", hazards=hz)
    # per-cell heights: measured crown tops (95th percentile per 10 m), roofs, wires and towers, without
    # the cruise planner's extra inflation (the maximum over a whole search cell is conservative already)
    L = np.load(site / "lidar10.npz")
    raw_top = np.maximum(L["chm"], np.where(L["bldg"] > 0.1, np.maximum(L["bldg_h"], 6.0), 0.0))
    if hz.exists():
        _, hz_hi = f3.hazard_bands(json.load(open(hz)), outer.e0, outer.n0, outer.n, outer.cell)
        raw_top = np.maximum(raw_top, np.where(np.isfinite(hz_hi), hz_hi, 0.0))
    span = f3.DRONES[a.drone]["span"]

    def cxy(i, j):  # cell centre, metres east/north of the site centre (row 0 = north)
        return (j + 0.5 - g / 2) * cell, (g / 2 - i - 0.5) * cell

    # minimum safe search height per cell
    min_level = np.zeros((g, g), np.int8)
    top_cell = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            x, y = cxy(i, j)
            ii0, jj0 = outer.ij(x - cell / 2 + 1, y + cell / 2 - 1)
            ii1, jj1 = outer.ij(x + cell / 2 - 1, y - cell / 2 + 1)
            t = float(raw_top[ii0:ii1 + 1, jj0:jj1 + 1].max())
            top_cell[i, j] = t
            need = t + f3.CLEAR_M + f3.DRONES[a.drone]["height"] / 2
            lv = [k for k, h in enumerate(fam.alt_m) if h >= need]
            min_level[i, j] = lv[0] if lv else 3
    under_p = np.full((g, g), -1.0)
    under_s = np.full((g, g), -1.0)
    if a.fine_from:
        s3 = f3.Site3D.load(a.fine_from, shift=tuple(a.shift), hazards=hz, season=a.season)
        P = f3.Planner(s3, outer, drone=a.drone)
        rng = np.random.default_rng(0)
        for i in range(g):
            for j in range(g):
                x, y = cxy(i, j)
                if not s3.inside(x, y, cell / 2 + 20):
                    continue
                ci, cj = s3.ij(x - cell / 2, y + cell / 2)
                win = s3.veg_top[ci:ci + int(cell), cj:cj + int(cell)]
                if (win > 8).mean() < 0.3:
                    continue  # not forest: no under-canopy checks are triggered here
                spots = np.argwhere(win > 8)
                ok, secs = 0, []
                for k in rng.choice(len(spots), min(a.samples, len(spots)), replace=False):
                    tx, ty = s3.xy(ci + spots[k][0], cj + spots[k][1])
                    agl = fam.alt_m[max(0, min(int(min_level[i, j]), 2))]
                    tr, info = P.under_check((x, y), (tx, ty), agl)
                    if tr is not None and info["close_enough"]:
                        ok += 1
                        secs.append(info["total_s"])
                n = min(a.samples, len(spots))
                under_p[i, j] = ok / n
                under_s[i, j] = float(np.mean(secs)) if secs else -1.0
    np.savez(site / "flight_limits.npz", min_level=min_level, top_m=top_cell, under_p=under_p, under_s=under_s,
             season=a.season, drone=a.drone, cell_m=cell)
    known = under_p >= 0
    print("highest obstacle per cell (m above ground): median %.0f, 90%% %.0f, max %.0f"
          % (np.median(top_cell), np.percentile(top_cell, 90), top_cell.max()))
    print(f"{site}: cells needing {fam.alt_m[1]:.0f} m or more: {(min_level >= 1).mean():.1%}, {fam.alt_m[2]:.0f} m: "
          f"{(min_level >= 2).mean():.1%}, no-fly: {(min_level >= 3).mean():.1%}; under-canopy checks known in "
          f"{known.sum()} forest cells: success {under_p[known].mean() if known.any() else float('nan'):.0%}, "
          f"time {np.mean(under_s[under_s > 0]) if (under_s > 0).any() else float('nan'):.0f} s")


if __name__ == "__main__":
    main()
