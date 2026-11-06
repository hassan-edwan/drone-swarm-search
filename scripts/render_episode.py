"""Draw one episode as a top-down map: prior heat, obstacles, searched cells,
drone paths and survivors found or missed.

    python scripts/render_episode.py --policy lawnmower --out lawnmower.png
    python scripts/render_episode.py --model runs/mappo/model.pt --out mappo.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import Lawnmower, RandomWalk  # noqa: E402
from swarmsearch.env import GridSearchEnv  # noqa: E402
from swarmsearch.evaluate import CONDITIONS  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402

COLORS = ["#2D5F8B", "#2A8C82", "#7A4FA0", "#A9481A", "#4A7A2A", "#8B2D5F", "#5F5F5F", "#B08A1A"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["lawnmower", "random"], default="lawnmower")
    p.add_argument("--model", default=None)
    p.add_argument("--condition", default="default", choices=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="stop early, e.g. to show 20 moves")
    p.add_argument("--out", default="episode.png")
    a = p.parse_args()

    if a.model:
        net, cfg = load(a.model)
        policy, name = LearnedPolicy(net), cfg["algo"].upper()
    else:
        policy = Lawnmower() if a.policy == "lawnmower" else RandomWalk(a.seed)
        name = a.policy
    sc = CONDITIONS[a.condition](a.seed)
    env = GridSearchEnv(sc)
    env.reset(seed=a.seed, options={"scenario": sc})
    policy.reset(env)
    prior = env.prior.copy()
    paths = {ag: [tuple(env.pos[ag])] for ag in env.agents}
    while env.agents and (a.steps is None or env.t < a.steps):
        env.step(policy.act(env))
        for ag in env.agents:
            paths[ag].append(tuple(env.pos[ag]))

    g = env.grid_size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(prior * env.n_free, cmap="Oranges", vmin=0, vmax=4, origin="upper")
    vy, vx = np.nonzero(env.visited)
    ax.scatter(vx, vy, marker="s", s=120, c="#8FA9C4", alpha=0.5, linewidths=0)
    oy, ox = np.nonzero(env.obstacles)
    ax.scatter(ox, oy, marker="s", s=160, c="#333333")
    for i, (ag, pts) in enumerate(paths.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[i % len(COLORS)], lw=2)
        ax.scatter([xs[0]], [ys[0]], marker="s", s=80, color=COLORS[i % len(COLORS)])
    for (x, y), f in zip(env.survivors, env.found):
        ax.scatter([x], [y], s=70, facecolors="#15233B" if f else "none", edgecolors="#15233B", linewidths=2)
    m = env.metrics()
    t90 = m["time_to_90"] if m["time_to_90"] is not None else "not reached"
    ax.set_title(f"{name} · {a.condition} · found {m['found_frac']:.0%} · t90 {t90} · {env.t} steps")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, g - 0.5); ax.set_ylim(g - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out, m)


if __name__ == "__main__":
    main()
