"""Fly a simulated mission through a real site's obstacles in 3D.

    python scripts/real_site/fly_mission.py --site sites/catoctin_west --seed 103 --hour 9 --steps 60

1. Runs the search (the rule, BeliefGreedy) on the site's coarse grid and
   records every drone's cell, height and mode per step.
2. Turns each step into a 3D path with swarmsearch.flight3d: cruise over
   the terrain and every crown and roof; when the simulator sends a drone to
   check an alert in dense forest from under the trees, the planner picks a
   real gap in the canopy, flies down through it, routes between the trunks
   and around buildings at 3 m to the spot, looks, and climbs back out.
3. Writes flight_seed<seed>.json (trajectories, events, clearances) and
   mission_seed<seed>.pkl next to the site.

The fine obstacle map comes from fine1.npz / fine_points.npz (lidar.py fine);
--fine-from and --shift reuse those made for a neighbouring site.
"""

import argparse
import dataclasses
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch.baselines import BeliefGreedy  # noqa: E402
from swarmsearch.evaluate import FIELD_CONDITIONS  # noqa: E402
from swarmsearch.field.env import FieldEnv  # noqa: E402


def record(sc, seed, steps):
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    pol = BeliefGreedy()
    pol.reset(env)
    frames = []
    alerts = []
    q_add = env.queue.add

    def add(it):
        alerts.append(dict(t=env.t, cell=tuple(int(c) for c in it["cell"]), true=it["true"] is not None,
                           drone=it.get("drone"), under=bool(it.get("under"))))
        q_add(it)

    env.queue.add = add
    while True:
        frames.append(dict(t=env.t, hour=env.hour(), people=env.people.pos.copy(), found=env.found.copy(),
                           drones={a: dict(cell=tuple(int(v) for v in env.true_pos[a]), alt_m=float(env.alt_now[a]),
                                           mode=env.mode.get(a), target=env.mode_target.get(a),
                                           mode_t=env.mode_t.get(a), bat=float(env.battery[a]))
                                   for a in env.possible_agents[:sc.n_drones] if env.alive.get(a)}))
        if not env.agents or env.t >= steps:
            break
        env.step(pol.act(env))
    return env, frames, alerts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--hour", type=float, default=9.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cell-m", type=float, default=50.0)
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--test-target", type=float, nargs=2, default=(-140.5, 225.5),
                    help="planner test after the mission: a spot under trees behind houses (x y), or none")
    ap.add_argument("--test-start", type=float, nargs=2, default=(-80.5, 285.5))
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0),
                    help="add to the fine data's coordinates to put them in this site's frame")
    a = ap.parse_args()
    site = Path(a.site)
    sc = dataclasses.replace(FIELD_CONDITIONS["wild_mix"](a.seed), site=str(site / "site.npz"), cell_m=a.cell_m,
                             hour=a.hour, light="day")
    env, frames, alerts = record(sc, a.seed, a.steps)
    fam, g = env.fam, env.w.g
    rng = np.random.default_rng(a.seed)

    def cxy(c):
        return (c[0] + 0.5 - g / 2) * fam.cell_m, (g / 2 - c[1] - 0.5) * fam.cell_m

    s3 = f3.Site3D.load(a.fine_from, shift=tuple(a.shift))
    P = f3.Planner(s3, f3.CoarseTerrain(site / "lidar10.npz"))
    # where the person really is inside their cell: somewhere under the crowns
    people_xy = []
    for i, pc in enumerate(frames[0]["people"]):
        x0, y0 = cxy(pc)
        best = None
        for _ in range(200):
            x, y = x0 + rng.uniform(-0.45, 0.45) * fam.cell_m, y0 + rng.uniform(-0.45, 0.45) * fam.cell_m
            if s3.inside(x, y, 5):
                ii, jj = s3.ij(x, y)
                if s3.veg_top[ii, jj] > 8 and not s3.bldg[ii, jj]:
                    best = (x, y)
                    break
        people_xy.append(best or (x0, y0))
    names = sorted(frames[0]["drones"])
    trajs = {d: [] for d in names}
    events = []
    under_done = {d: -1 for d in names}
    for k in range(len(frames) - 1):
        f0, f1 = frames[k], frames[k + 1]
        t0 = k * fam.step_s
        for d in names:
            a0, a1 = f0["drones"].get(d), f1["drones"].get(d)
            if a0 is None or a1 is None or a0["mode"] == "landed":
                continue
            z0 = trajs[d][-1][3] if trajs[d] else None
            if a1["mode"] == "under" and a0["mode"] != "under":
                # the check: from here to the alert spot under the trees, then back up
                cell = a1["target"]
                tgt = None
                for i, pc in enumerate(f1["people"]):
                    if tuple(int(v) for v in pc) == tuple(cell):
                        tgt = people_xy[i]
                if tgt is None:
                    x0, y0 = cxy(cell)
                    tgt = (x0 + rng.uniform(-15, 15), y0 + rng.uniform(-15, 15))
                start = cxy(a0["cell"])
                tr, info = P.under_check(start, tgt, fam.alt_m[0], z0=z0)
                n_steps = 1
                while k + n_steps < len(frames) - 1 and frames[k + n_steps]["drones"].get(d, {}).get("mode") == "under":
                    n_steps += 1
                slot = n_steps * fam.step_s
                if tr is None:
                    events.append(dict(t=t0, drone=d, kind="under_fail", reason=info))
                    continue
                scale = slot / max(tr[-1][0], 1e-6)
                for t, x, y, z, m in tr:
                    trajs[d].append((t0 + t * scale, x, y, z, m))
                under_done[d] = k + n_steps
                events.append(dict(t=t0, drone=d, kind="under", gap=info["gap"], spot=tgt, look=info["look"],
                                   reach_m=info["reach_m"], under_m=info["under_m"], planned_s=info["total_s"],
                                   slot_s=slot, sim_steps=n_steps))
                continue
            if k < under_done[d]:
                continue
            p1 = cxy(a1["cell"])
            p0 = (trajs[d][-1][1], trajs[d][-1][2]) if trajs[d] else cxy(a0["cell"])  # from where it really is
            tr = P.cruise(p0, p1, a1["alt_m"], z0=z0, agl0=a0["alt_m"])
            dur = tr[-1][0] if tr[-1][0] > 0 else fam.step_s
            for t, x, y, z, m in (tr[1:] if trajs[d] else tr):
                trajs[d].append((t0 + t * fam.step_s / max(dur, 1e-6), x, y, z, m))
            if p0 == p1:  # hovering: keep the time line going
                x, y = p0
                zz = trajs[d][-1][3]
                trajs[d].append((t0 + fam.step_s, x, y, zz, "hover"))
    test = None
    if a.test_target:
        t_end = max(tr[-1][0] for tr in trajs.values() if tr) + 5.0
        tr, tinfo = P.under_check(tuple(a.test_start), tuple(a.test_target), fam.alt_m[0])
        if tr is not None:
            trajs["test"] = [(t_end + t, x, y, z, m) for t, x, y, z, m in tr]
            test = dict(t=t_end, gap=tinfo["gap"], spot=tuple(a.test_target), look=tinfo["look"],
                        under_m=tinfo["under_m"], total_s=tinfo["total_s"], reach_m=tinfo["reach_m"])
    clear = {d: f3.clearance(s3, tr) for d, tr in trajs.items()}
    found_t = next((f["t"] for f in frames if f["found"].any()), None)
    info = dict(seed=a.seed, hour=a.hour, category=sc.category, n_drones=sc.n_drones, cell_m=fam.cell_m,
                step_s=fam.step_s, base=cxy(env.w.base), people=people_xy, found_step=found_t, events=events,
                alerts=alerts, clearance=clear, alt_m=list(fam.alt_m), fine=dict(east0=s3.east0, north0=s3.north0,
                                                                                    size=s3.n),
                n_trees=len(s3.trees), policy="the rule (BeliefGreedy)", test=test)
    out = site / f"flight_seed{a.seed}.json"
    f3.save_traj(out, trajs, info)
    pickle.dump(dict(frames=frames, alerts=alerts, info=info), open(site / f"mission3d_seed{a.seed}.pkl", "wb"))
    print(f"{out}: {len(frames)} steps, found at step {found_t}; events: "
          + "; ".join(f"{e['drone']} {e['kind']} at {e['t']:.0f}s" + (f" (gap {e['gap'][0]:.0f},{e['gap'][1]:.0f}; "
                      f"{e['under_m']:.0f} m under the crowns; got within {e['reach_m']:.1f} m; planned "
                      f"{e['planned_s']:.0f}s vs {e['slot_s']:.0f}s in the simulator)" if e['kind'] == 'under' else
                      f" ({e.get('reason')})") for e in events))
    for d, c in clear.items():
        print(d, {k: round(v, 1) for k, v in c.items()})


if __name__ == "__main__":
    main()
