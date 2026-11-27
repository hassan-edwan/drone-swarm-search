"""Under-canopy checks in GENERATED worlds, flown in 3D: per-family numbers for the field simulator.

    python scripts/family_exec.py --worlds 8 --spots 10 --workers 4

The real sites measure under-canopy checks on lidar (scripts/real_site/
flight_limits.py, surprise_eval.py). Generated worlds have no lidar, so for
each family this builds a 3D patch (1 m voxels, swarmsearch.world3d) around
sampled dense-forest cells of generated worlds:

- ground from the world's elevation;
- trees: crowns (porous ellipsoids) and trunks, as many as the cell's canopy
  density covers, as tall as the world's tree heights (field/hazards.py);
- buildings from the world's building heights (urban blocks fill their
  cells; flood houses are 10 x 12 m);
- power lines, masts and cranes from the world's drawn infrastructure
  (field/hazards.py layout), pole by pole with service wires to buildings.

That is the drones' MAP. The real world adds what maps lack
(world3d.make_truth: crown growth, misplaced trunks and buildings, lines on
the other side of the road, unmapped spans, sheds, flagpoles, saplings, dead
branches, felled trees). Then, from 80 m away at a random bearing, the
planner plans the check on the map (flight3d.VoxPlanner) and the executor
flies it in the real world with a forward depth camera and camera navigation
in 5 m/s wind (fly_exec.Executor), as the default variant on the real site.

Writes swarmsearch/field/exec_family.json: per family, under_p (share of
spots the planner gets within sight of), under_s (planned trip, s), and exec
(p_collide, p_abort, p_reach, extra_s) the field environment uses for
generated worlds (field/hazards.py).
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch import fly_exec as fx  # noqa: E402
from swarmsearch import world3d as w3  # noqa: E402

PATCH = 240  # metres a side
UNDER = {"wilderness": 0.85, "flood": 0.85, "urban": 0.3}  # share of the forest floor in undergrowth (see --under)
REAL_UNDER_P = 0.25  # under-canopy checks the planner gets within sight of on the real (lidar) hamlet site, summer


def cell_xy(c, cell):
    """Field cell (x, y) centre in metres: east, north (grid rows grow southwards)."""
    return (c[0] + 0.5) * cell, -(c[1] + 0.5) * cell


def patch(w, spot, rng, under=None):
    """The map (World3D) of a PATCH-metre square around a field cell, and its hazards / buildings in metres."""
    cell, g = w.cell_m, w.g
    X, Y = cell_xy(spot, cell)
    n = PATCH
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    xe = X - n / 2 + jj + 0.5
    yn = Y + n / 2 - ii - 0.5
    fx_ = np.clip(xe / cell - 0.5, 0, g - 1.001)
    fy_ = np.clip(-yn / cell - 0.5, 0, g - 1.001)
    x0, y0 = fx_.astype(int), fy_.astype(int)
    ax, ay = fx_ - x0, fy_ - y0
    E = w.elev

    def bil(A):
        return (A[y0, x0] * (1 - ax) * (1 - ay) + A[y0, np.minimum(x0 + 1, g - 1)] * ax * (1 - ay)
                + A[np.minimum(y0 + 1, g - 1), x0] * (1 - ax) * ay
                + A[np.minimum(y0 + 1, g - 1), np.minimum(x0 + 1, g - 1)] * ax * ay)
    dtm = bil(E)
    cx, cy = np.clip(np.round(fx_).astype(int), 0, g - 1), np.clip(np.round(fy_).astype(int), 0, g - 1)
    W = w3.World3D(np.zeros((n, n, w3.NZ), np.uint8), dtm, X, Y, season="leaf-on")
    # trees: crown radius 2.5-5 m, count from the cover; crowns porous (70% of voxels), trunks to the crown base
    tree_h = w.extra["tree_h"]
    cover = w.map_canopy
    trees = []
    area = np.zeros((g, g))
    np.add.at(area, (cy, cx), 1.0)
    for (y, x), a in zip(np.argwhere(area > 0), area[area > 0]):
        c, h = float(cover[y, x]), float(tree_h[y, x])
        if c < 0.15 or h < 4:
            continue
        sel = np.argwhere((cy == y) & (cx == x))
        k = rng.poisson(c * a / (math.pi * 3.8 ** 2))
        for p in sel[rng.integers(len(sel), size=k)]:
            tx, ty = W.xy(*p)
            hh = float(np.clip(h * rng.uniform(0.75, 1.15), 4, w3.NZ - 2))
            cr = float(rng.uniform(2.5, 5.0))
            cb = hh * rng.uniform(0.35, 0.6)
            trees.append((tx, ty, hh, cr, cb, max(8.0, 2.0 * (hh - 1.3)) / 200.0))
    for tx, ty, hh, cr, cb, tr in trees:
        i, j = W.ij(tx, ty)
        R = int(math.ceil(cr))
        rz = (hh - cb) / 2
        zc = cb + rz
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                if not (0 <= i + di < n and 0 <= j + dj < n):
                    continue
                q = 1 - (di * di + dj * dj) / (cr * cr)
                if q <= 0:
                    continue
                half = rz * math.sqrt(q)
                ks = np.arange(max(1, int(zc - half)), min(w3.NZ, int(math.ceil(zc + half))))
                keep = ks[rng.random(len(ks)) < 0.7]
                W.occ[i + di, j + dj, keep] |= w3.VEG
        W.put_segment((tx, ty, 0.0), (tx, ty, cb), w3.TRUNK, radius=0 if tr < 0.5 else 1)
    W.trees = np.array(trees) if trees else np.zeros((0, 6))
    # undergrowth and low branches under the crowns: clustered patches 1.5-4 m tall covering a share of the
    # forest floor. Even 95% leaves synthetic forest easier to get into than the real lidar forest (the planner
    # gets within sight of 50% of spots, against 25% on the hamlet site), so the reach the field simulator uses
    # is taken from the real site (main); these patches set the clutter the executor meets
    from scipy import ndimage
    share = UNDER[w.family] if under is None else under
    if share > 0:
        f = ndimage.gaussian_filter(rng.random((n, n)), 4.0)
        crowned = (W.occ[:, :, 6:] & w3.VEG).any(axis=2)
        thr = np.quantile(f[crowned], 1 - share) if crowned.any() else 1.0
        bush = crowned & (f >= thr)
        hb = 1.5 + 2.5 * ndimage.gaussian_filter(rng.random((n, n)), 3.0)
        hb = (hb - hb.min()) / max(np.ptp(hb), 1e-9) * 2.5 + 1.5
        ks = np.arange(w3.NZ)[None, None, :]
        W.occ[bush[:, :, None] & (ks >= 0) & (ks < hb[:, :, None])] |= w3.VEG
    # buildings
    B = []
    sh = w.struct_h
    for y, x in np.argwhere((area > 0) & (sh > 0)):
        bx, by = cell_xy((x, y), cell)
        if w.family == "urban":
            wd = ln = cell
        else:
            wd, ln = 10.0, 12.0
        W.put_box(bx, by, wd, ln, float(sh[y, x]), w3.BLDG)
        B.append([[bx - wd / 2, by - ln / 2], [bx + wd / 2, by - ln / 2], [bx + wd / 2, by + ln / 2],
                  [bx - wd / 2, by + ln / 2], [bx - wd / 2, by - ln / 2]])
    # infrastructure (metres)
    Hz = w.extra["hazards"]
    H = dict(distribution=[[list(cell_xy(c, cell)) for c in line] for line in Hz["distribution"] if len(line) > 1],
             transmission=[dict(kv=115, pts=[list(cell_xy(c, cell)) for c in line]) for line in Hz["transmission"]],
             towers=[dict(x=cell_xy((t["x"], t["y"]), cell)[0], y=cell_xy((t["x"], t["y"]), cell)[1], h=t["h"],
                          guy_r=t["guy_m"]) for t in Hz["towers"]])
    W.lines = w3.put_power(W, H, B, side=+1, rng=None)
    for c in Hz["cranes"]:
        px, py = cell_xy((c["x"], c["y"]), cell)
        W.put_pole(px, py, min(c["h"], w3.NZ - 1), w3.STRUCT)
    return W, H, B


def run(job):
    family, seed, spots, under = job
    from swarmsearch.field.params import FieldSampler
    from swarmsearch.field.terrain import make_world
    from swarmsearch.windfield import WindField
    rng = np.random.default_rng(seed)
    sc = FieldSampler(families={family: 1.0}).sample(rng)
    from dataclasses import replace
    sc = replace(sc, season="summer")
    w = make_world(np.random.default_rng(sc.seed_world), sc)
    dense = np.argwhere(w.map_canopy >= 0.5)
    out = []
    if not len(dense):
        return out
    for s_ in range(spots):
        y, x = dense[rng.integers(len(dense))]
        M, H, B = patch(w, (int(x), int(y)), rng, under)
        T = w3.make_truth(M, H, B, seed=int(rng.integers(1 << 30)))
        wf = WindField(T.dtm, T.top() * 0.9, 1.0, 5.0, float(rng.uniform(0, 360)), upwind_m=150.0)
        base = f3.VoxPlanner(M, None)
        factory = lambda extra: base if extra is None else f3.VoxPlanner(M, None, known_extra=extra)  # noqa: E731
        # the spot: a forested point near the patch centre
        top = M.top()
        cand = np.argwhere(top[PATCH // 2 - 25:PATCH // 2 + 25, PATCH // 2 - 25:PATCH // 2 + 25] > 6)
        if not len(cand):
            continue
        ci, cj = cand[rng.integers(len(cand))] + PATCH // 2 - 25
        tx, ty = M.xy(int(ci), int(cj))
        ang = rng.uniform(0, 2 * np.pi)
        start = (tx + 80 * math.cos(ang), ty + 80 * math.sin(ang))
        agl = float(sc.fam.alt_m[0])
        tr, info = base.under_check(start, (tx, ty), agl)
        row = dict(family=family, world=seed, planned=tr is not None, close=bool(tr is not None and info["close_enough"]),
                   planned_s=float(info["total_s"]) if tr is not None else -1.0)
        if tr is not None:
            ex = fx.Executor(M, T, factory, wind=wf, nav="vio", sensing=True, seed=int(rng.integers(1 << 30)),
                             wind_ij=M.ij)
            r = ex.fly_check(start, (tx, ty), agl)
            row.update(outcome=r.outcome, hit=r.hit_kind, reached=bool(r.reached), time_s=float(r.time_s),
                       exec_planned_s=float(r.planned_s), replans=int(r.replans))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["wilderness", "urban", "flood"])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--spots", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--under", type=float, default=None, help="undergrowth share for every family (calibration)")
    ap.add_argument("--out", default=str(ROOT / "swarmsearch" / "field" / "exec_family.json"))
    a = ap.parse_args()
    jobs = [(f, 1000 * k + i, a.spots, a.under) for k, f in enumerate(a.families) for i in range(a.worlds)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run, jobs):
            rows += r
            if r:
                print(f"{r[0]['family']} world {r[0]['world']}: {len(r)} spots ({time.time() - t0:.0f}s)", flush=True)
    res = {}
    for f in a.families:
        R = [r for r in rows if r["family"] == f]
        P = [r for r in R if r["planned"]]
        if not R:
            continue
        flown = [r for r in P if "outcome" in r]
        ok = [r for r in flown if r["outcome"] == "ok"]
        n = max(1, len(flown))
        st = dict(n=len(flown), p_collide=sum(r["outcome"] in ("collided", "bird") for r in flown) / n,
                  p_abort=sum(r["outcome"] == "aborted" for r in flown) / n,
                  p_reach=sum(r["reached"] for r in flown) / n,
                  extra_s=float(np.mean([r["time_s"] - r["exec_planned_s"] for r in ok])) if ok else 0.0)
        res[f] = dict(n_spots=len(R), under_p_3d=sum(r["close"] for r in R) / len(R),
                      under_s=float(np.mean([r["planned_s"] for r in P])) if P else -1.0, exec=st,
                      hits=sorted({r["hit"] for r in flown if r.get("hit")}))
        print(f"{f}: {len(R)} spots, planner gets within sight of {res[f]['under_p_3d']:.0%}, trip {res[f]['under_s']:.0f} s; "
              f"flown {st['n']}: collided {st['p_collide']:.0%} ({', '.join(res[f]['hits'])}), turned back "
              f"{st['p_abort']:.0%}, reached {st['p_reach']:.0%}, extra {st['extra_s']:+.0f} s")
    # reach: the real site's measured share, scaled per family by its 3D reach relative to generated wilderness
    if "wilderness" in res:
        w0 = max(res["wilderness"]["under_p_3d"], 1e-6)
        for f in a.families:
            if f in res:
                res[f]["under_p"] = float(min(1.0, REAL_UNDER_P * res[f]["under_p_3d"] / w0))
    res["_about"] = ("scripts/family_exec.py: under-canopy checks planned on synthetic 3D maps of generated worlds and "
                     "flown in a real-world draw (forward depth camera, camera navigation, 5 m/s wind). "
                     f"{a.worlds} worlds x {a.spots} spots per family. under_p: the real hamlet site's measured "
                     f"{REAL_UNDER_P:.2f}, scaled by each family's 3D reach relative to generated wilderness "
                     "(synthetic forest is more open than the lidar forest); under_p_3d: the 3D reach itself.")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
