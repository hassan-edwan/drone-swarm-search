"""Many missions flown in ArduPilot SITL, in parallel: tracking and clearance statistics.

    python scripts/real_site/sitl_batch.py --seeds 0-99 --winds 0 5 10 --slots 24 --out runs/sitl

For each seed: fly_mission.py (the search on the hamlet site, every move turned
into a 3D path through the lidar obstacles, plus one under-canopy check to a
random spot under the trees), then sitl_fly.py for each wind speed (one
ArduPilot copter per drone). Runs `--slots` flights at once; each flight uses one
SITL process per drone and its own block of ports.

Writes per-flight reports and summary.md: how far the copters strayed from the
paths (overall and under the canopy), how close they came to the ground,
trunks, buildings and crowns, and how much longer than the search grid's clock
the missions take when flown at real speeds.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAX_DRONES = 6  # drones per mission (3-4 typical) + the test drone, ports reserved per slot


def seeds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--winds", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--steps", type=int, default=60, help="simulator steps per mission (5 s each on this site)")
    ap.add_argument("--slots", type=int, default=4, help="flights at once")
    ap.add_argument("--speedup", type=float, default=10.0)
    ap.add_argument("--gps", nargs="+", default=["degraded", "good"],
                    help="GPS under the canopy: degraded (GPS only) and/or good (camera-based navigation)")
    ap.add_argument("--season", default="leaf-on", choices=["leaf-on", "leaf-off"])
    ap.add_argument("--drone", default="sar", choices=["sar", "small"])
    ap.add_argument("--ardupilot", default="~/ardupilot")
    ap.add_argument("--out", default="runs/sitl")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds_of(a.seeds)
    py = sys.executable
    t0 = time.time()

    # 1. plans (cheap; in parallel)
    def plan(seed):
        f = out / f"flight_seed{seed}.json"
        if f.exists():
            return seed, 0
        return seed, run([py, str(HERE / "fly_mission.py"), "--site", a.site, "--seed", str(seed), "--steps",
                          str(a.steps), "--test", "random", "--out-dir", str(out), "--season", a.season,
                          "--drone", a.drone], out / f"plan_seed{seed}.log")

    with ThreadPoolExecutor(max(1, a.slots)) as ex:
        bad = [s for s, rc in ex.map(plan, seeds) if rc]
    print(f"plans: {len(seeds) - len(bad)} ok, {len(bad)} failed {bad[:10]} ({time.time() - t0:.0f}s)", flush=True)

    # 2. flights
    jobs = [(s, w, gp) for s in seeds if s not in bad for w in a.winds for gp in a.gps]

    def fly(job_slot):
        (seed, wind, gps), slot = job_slot
        rep = out / f"sitl_seed{seed}_wind{wind:.0f}_gps{gps}.json"
        if rep.exists():
            return 0
        return run([py, str(HERE / "sitl_fly.py"), "--flight", str(out / f"flight_seed{seed}.json"), "--site", a.site,
                    "--ardupilot", a.ardupilot, "--wind", str(wind), "--speedup", str(a.speedup),
                    "--instance0", str(slot * MAX_DRONES), "--workdir", str(out / f"sitl_work_{slot}"),
                    "--gps", gps, "--season", a.season, "--drone", a.drone,
                    "--out", str(rep)], out / f"sitl_seed{seed}_wind{wind:.0f}_gps{gps}.log")

    free = list(range(a.slots))
    import threading
    lock = threading.Lock()

    def with_slot(job):
        with lock:
            slot = free.pop()
        try:
            return fly((job, slot))
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(a.slots) as ex:
        rcs = list(ex.map(with_slot, jobs))
    print(f"flights: {sum(r == 0 for r in rcs)} ok, {sum(r != 0 for r in rcs)} failed ({time.time() - t0:.0f}s)",
          flush=True)
    summarise(out, a.winds, a.gps)


def summarise(out: Path, winds, gps_modes=("degraded", "good")):
    rows = []
    for f in sorted(out.glob("sitl_seed*_wind*.json")):
        d = json.load(open(f))
        for name, r in d["report"].items():
            base = dict(wind=d["wind"], gps=d.get("gps", "good"), name=name)
            if "error" in r:
                rows.append(dict(base, error=r["error"]))
                continue
            rows.append(dict(base, **{k: v for k, v in r.items() if k != "clearance_flown"},
                             **{f"clear_{k}": v for k, v in r["clearance_flown"].items()}))
    json.dump(rows, open(out / "summary.json", "w"))
    lines = ["| Wind | GPS under canopy | Flights | Failed | Off the path: mean / 95% / max (m) | Under the canopy: "
             "max off path / max height error (m) | Closest, edge to edge: ground / trunk / building / crown above / "
             "wire (m) | Too close* | Extra time vs grid clock: flyable plan / ArduPilot |",
             "|---|---|---|---|---|---|---|---|---|"]
    for w, gm in [(w, gm) for w in winds for gm in gps_modes]:
        R = [r for r in rows if abs(r["wind"] - w) < 1e-6 and r["gps"] == gm]
        ok = [r for r in R if "error" not in r]
        if not ok:
            lines.append(f"| {w:.0f} m/s | {gm} | {len(R)} | {len(R)} | | | | | |")
            continue
        g = lambda k: np.array([r[k] for r in ok if r.get(k) is not None and np.isfinite(r[k])])  # noqa: E731
        under = [r for r in ok if "under_cross_max" in r]
        close = sum(1 for r in ok if r.get("clear_ground", 9) < 1.0 or r.get("clear_trunk", 9) < 0.5
                    or r.get("clear_building", 9) < 1.0 or r.get("clear_under_crown", 9) < 0.5
                    or r.get("clear_wire", 9) < 0.5)
        mn = lambda k: g(k).min() if len(g(k)) else float("nan")  # noqa: E731
        plan_s = g("plan_s")
        lines.append(
            f"| {w:.0f} m/s | {gm} | {len(R)} | {len(R) - len(ok)} | {g('cross_err_mean').mean():.2f} / "
            f"{np.percentile(g('cross_err_p95'), 95):.1f} / {g('cross_err_max').max():.1f} | "
            + (f"{max(r['under_cross_max'] for r in under):.2f} / {max(r['under_vert_max'] for r in under):.2f} "
               f"({len(under)} flights)" if under else "none")
            + f" | {mn('clear_ground'):.1f} / {mn('clear_trunk'):.1f} / {mn('clear_building'):.1f} / "
              f"{mn('clear_under_crown'):.1f} / {mn('clear_wire'):.1f} | {close} | "
              f"{100 * np.sum(g('behind_grid_clock_s')) / max(np.sum(plan_s - g('behind_grid_clock_s')), 1):.0f}% / "
              f"{100 * np.sum(g('lag_s')) / max(np.sum(plan_s), 1):+.0f}% |")
    lines.append("\n*Too close (edge to edge, from the TRUE track): under 1 m from the ground or a building, under 0.5 m "
                 "from a trunk, the crown above or a wire band (negative: inside it).")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
