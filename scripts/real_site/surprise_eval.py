"""How under-canopy checks planned on the MAP fare in the REAL world (detect and avoid at scale).

    python scripts/real_site/surprise_eval.py --worlds 3 --checks 40 --workers 3 --out results/surprises

For each of --worlds real worlds (world3d.make_truth, a different draw of what
the map lacks each time) and --checks random spots under the trees, plan the
check on the map and fly it in the real world (swarmsearch.fly_exec) for every
variant: onboard sensing off/on (forward camera, or all-round), navigation
under the crowns by GPS or by camera (visual-inertial odometry), wind, drone
size, and season (summer, and winter with snow on the branches: the map then
uses leaf-off foliage). Reports the share of checks that collide (with what),
turn back, or reach the spot, the extra time, and what the drones detected;
writes results.md / results.json and the per-site numbers the field
simulator uses (exec_stats.json: the default variant, sensing on with camera
navigation).
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch import fly_exec as fx  # noqa: E402
from swarmsearch import world3d as w3  # noqa: E402
from swarmsearch.windfield import WindField  # noqa: E402

VARIANTS = [  # (name, sensing, nav, wind m/s, drone, season)
    ("no sensing, camera nav", "off", "vio", 5.0, "sar", "leaf-on"),
    ("forward sensing, camera nav", "forward", "vio", 5.0, "sar", "leaf-on"),
    ("all-round sensing, camera nav", "all", "vio", 5.0, "sar", "leaf-on"),
    ("forward sensing, GPS only", "forward", "gps", 5.0, "sar", "leaf-on"),
    ("all-round sensing, GPS only", "all", "gps", 5.0, "sar", "leaf-on"),
    ("forward sensing, camera nav, 10 m/s wind", "forward", "vio", 10.0, "sar", "leaf-on"),
    ("forward sensing, camera nav, small drone", "forward", "vio", 5.0, "small", "leaf-on"),
    ("forward sensing, camera nav, winter snow", "forward", "vio", 5.0, "sar", "snow"),
]
DEFAULT = "forward sensing, camera nav"

_G: dict = {}


def _worlds(site, fine_from, shift, seed, season):
    key = (seed, season)
    if key in _G:
        return _G[key]
    if len(_G) >= 2:  # a world pair is ~150 MB: keep only the last two
        _G.pop(next(iter(_G)))
    S = Path(site)
    pts = Path(fine_from).with_name("fine_points.npz")
    map_season = "leaf-off" if season == "snow" else season
    M = w3.build_map(fine_from, pts, S / "buildings.json", S / "hazards.json", shift=shift, season=map_season)
    H = json.load(open(S / "hazards.json"))
    B = json.load(open(S / "buildings.json"))
    T = w3.make_truth(M, H, B, seed=seed, season=season)
    _G[key] = (M, T)
    return M, T


def run(job):
    site, fine_from, shift, wseed, targets, variant = job
    name, sensing, nav, wind, drone, season = variant
    M, T = _worlds(site, fine_from, tuple(shift), wseed, season)
    S = Path(site)
    outer = f3.CoarseTerrain(S / "lidar10.npz", hazards=S / "hazards.json")
    L = np.load(S / "lidar10.npz")
    wf = WindField(L["dtm"], L["chm"], 10.0, wind, 270.0)
    roads = json.load(open(S / "hazards.json")).get("distribution", [])
    cache = {}

    def factory(extra):
        if extra is None:
            if "b" not in cache:
                cache["b"] = f3.VoxPlanner(M, outer, drone=drone)
            return cache["b"]
        return f3.VoxPlanner(M, outer, drone=drone, known_extra=extra)

    rows = []
    for n, (tx, ty, sx, sy) in enumerate(targets):
        ex = fx.Executor(M, T, factory, wind=wf, roads=roads, drone=drone, nav=nav, sensing=sensing != "off",
                         seed=1000 * wseed + n, wind_ij=outer.ij)
        if sensing == "all":
            ex.all_round = True
        r = ex.fly_check((sx, sy), (tx, ty), 40.0)
        rows.append(dict(variant=name, world=wseed, outcome=r.outcome, hit=r.hit_kind, reached=r.reached,
                         time_s=r.time_s, planned_s=r.planned_s, replans=r.replans, detected=r.detected,
                         nav_err_max=r.nav_err_max, min_clear={k: (v if np.isfinite(v) else None)
                                                               for k, v in r.min_clear.items()}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0))
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--checks", type=int, default=40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/surprises")
    a = ap.parse_args()
    M = w3.build_map(a.fine_from, Path(a.fine_from).with_name("fine_points.npz"), Path(a.site) / "buildings.json",
                     Path(a.site) / "hazards.json", shift=tuple(a.shift))
    rng = np.random.default_rng(0)
    forest = np.argwhere(M.top() > 10)
    targets = []
    while len(targets) < a.checks:
        tx, ty = M.xy(*forest[rng.integers(len(forest))])
        if M.inside(tx, ty, 100):
            ang = rng.uniform(0, 2 * np.pi)
            targets.append((tx, ty, tx + 80 * np.cos(ang), ty + 80 * np.sin(ang)))
    del M
    jobs = [(a.site, a.fine_from, list(a.shift), w, targets, v) for w, v in itertools.product(range(1, a.worlds + 1),
                                                                                          VARIANTS)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run, jobs, chunksize=len(VARIANTS) if len(jobs) > a.workers * len(VARIANTS) else 1):
            rows += r
            print(f"{r[0]['variant']} world {r[0]['world']} done ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "results.json", "w"))
    lines = ["| Variant | Checks | Collided | With what | Turned back | Reached the spot | Extra time (s) | "
             "Re-plans / check | Seen: solid / foliage / wire |", "|---|---|---|---|---|---|---|---|---|"]
    stats = {}
    for v in [x[0] for x in VARIANTS]:
        R = [r for r in rows if r["variant"] == v and r["outcome"] != "no route"]
        if not R:
            continue
        col = [r for r in R if r["outcome"] in ("collided", "bird")]
        ab = [r for r in R if r["outcome"] == "aborted"]
        ok = [r for r in R if r["outcome"] == "ok"]
        what = Counter(r["hit"] for r in col)
        det = Counter()
        for r in R:
            det.update(r["detected"])
        extra = np.mean([r["time_s"] - r["planned_s"] for r in ok]) if ok else float("nan")
        n = len(R)
        se = lambda p: np.sqrt(p * (1 - p) / n)  # noqa: E731
        pc = len(col) / n
        lines.append(f"| {v} | {n} | {pc:.0%} ± {se(pc):.0%} | {', '.join(f'{k} {c}' for k, c in what.most_common())} | "
                     f"{len(ab) / n:.0%} | {sum(r['reached'] for r in R) / n:.0%} | {extra:+.0f} | "
                     f"{np.mean([r['replans'] for r in R]):.1f} | {det['solid'] / n:.0f} / {det['foliage'] / n:.0f} / "
                     f"{det['wire'] / n:.1f} |")
        stats[v] = dict(n=n, p_collide=pc, p_abort=len(ab) / n, p_reach=sum(r["reached"] for r in R) / n,
                        extra_s=float(extra) if ok else 0.0)
    txt = "\n".join(lines)
    (out / "results.md").write_text(txt + "\n")
    json.dump(dict(default=stats.get(DEFAULT), variants=stats), open(Path(a.site) / "exec_stats.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
