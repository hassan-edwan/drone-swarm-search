"""M3: fly the search policy (or a baseline) in 3D and compare with 2D.

    python scripts/run3d.py --model models/m2.pt --backend pybullet --episodes 20
    python scripts/run3d.py --model models/m2.pt --backend pointmass --wind 3
    python scripts/run3d.py --model models/m2.pt --backend pybullet --gui --episodes 1   # watch it
    python scripts/run3d.py --model models/m2.pt --plot results/m3/flight.png --episodes 1

For each condition and method it reports the 2D result and the same
missions flown in 3D, plus flight metrics: share of moves that arrived late
(no search that step), mean tracking error, and near misses.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import Lawnmower  # noqa: E402
from swarmsearch.env import GridSearchEnv  # noqa: E402
from swarmsearch.evaluate import ALL_CONDITIONS, run_episode  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402
from swarmsearch.sim3d import Swarm3D, run_episode_3d  # noqa: E402

DEFAULT_CONDITIONS = ["drift_unknown", "to_safety_unknown", "obs20_drift_unknown", "obs_random_mix"]


def summarise(ms: list[dict]) -> dict:
    t90 = [m["time_to_90"] if m["time_to_90"] is not None else m["steps"] for m in ms]
    out = {"reached_90": float(np.mean([m["time_to_90"] is not None for m in ms])),
           "found": float(np.mean([m["found_frac"] for m in ms])),
           "t90": float(np.mean(t90)), "coverage": float(np.mean([m["coverage"] for m in ms]))}
    if "late_frac" in ms[0]:
        out.update(late=float(np.mean([m["late_frac"] for m in ms])),
                   track_err_m=float(np.mean([m["track_err_m"] for m in ms])),
                   near_misses=float(np.mean([m["near_misses"] for m in ms])))
    return out


def plot_flight(sim: Swarm3D, path: str):
    import matplotlib.pyplot as plt
    tr = np.array(sim.trace)  # (T, n, 3)
    env = sim.env
    g = env.grid_size
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    oy, ox = np.nonzero(env.obstacles)
    if len(ox):
        ax.bar3d(ox, g - 1 - oy, np.zeros(len(ox)), 1, 1, 0.6, color="0.3", alpha=0.5, shade=False)
    for j in range(tr.shape[1]):
        ax.plot(tr[:, j, 0], tr[:, j, 1], tr[:, j, 2], lw=0.8)
    s = env.survivors
    ax.scatter(s[:, 0] + 0.5, g - 1 - s[:, 1] + 0.5, np.zeros(len(s)), c=np.where(env.found, "tab:green", "tab:red"), s=12)
    ax.set_xlim(0, g); ax.set_ylim(0, g); ax.set_zlim(0, 3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
    m = sim.metrics()
    ax.set_title(f"3D flight ({sim.backend}) · found {m['found_frac']:.0%} · late moves {m['late_frac']:.0%}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="trained model.pt (omit to fly only the lawnmower)")
    p.add_argument("--backend", choices=["pybullet", "pointmass"], default="pybullet")
    p.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS, choices=list(ALL_CONDITIONS))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--wind", type=float, default=0.0, help="pointmass gust strength (m/s^2)")
    p.add_argument("--gui", action="store_true", help="PyBullet window (pybullet backend, 1 episode)")
    p.add_argument("--plot", default=None, help="save a 3D plot of the first episode's flight")
    p.add_argument("--out", default="results/m3")
    a = p.parse_args()

    methods = {"lawnmower": Lawnmower}
    if a.model:
        net, cfg = load(a.model)
        methods[cfg["algo"]] = lambda: LearnedPolicy(net)

    res, rows = {}, []
    for cond in a.conditions:
        res[cond] = {}
        for name, make in methods.items():
            m2d, m3d = [], []
            for ep in range(a.episodes):
                sc = ALL_CONDITIONS[cond](ep)
                m2d.append(run_episode(GridSearchEnv(sc), make(), sc, seed=ep))
                sim = Swarm3D(GridSearchEnv(sc), backend=a.backend, gui=a.gui, wind=a.wind,
                              record=bool(a.plot) and ep == 0)
                m3d.append(run_episode_3d(sim, make(), sc, seed=ep))
                if a.plot and ep == 0 and cond == a.conditions[0] and name == list(methods)[-1]:
                    plot_flight(sim, a.plot)
                sim.close()
            r2, r3 = summarise(m2d), summarise(m3d)
            res[cond][name] = {"2d": r2, "3d": r3}
            rows.append(f"| {cond} | {name} | {r2['reached_90']:.0%} | {r3['reached_90']:.0%} | "
                        f"{r2['found']:.0%} | {r3['found']:.0%} | {r3['late']:.0%} | "
                        f"{r3['track_err_m']:.2f} | {r3['near_misses']:.1f} |")
            print(rows[-1], flush=True)

    head = ("| Condition | Method | Reached 90% (2D) | Reached 90% (3D) | Found (2D) | Found (3D) | "
            "Late moves | Track error (m) | Near misses |\n|---|---|---|---|---|---|---|---|---|")
    md = head + "\n" + "\n".join(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{a.backend}.json").write_text(json.dumps(res, indent=2))
    (out / f"results_{a.backend}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
