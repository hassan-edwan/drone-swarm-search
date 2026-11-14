"""Video of one mission flown by every method side by side.

    python scripts/render_video.py --condition obs20_to_safety_unknown --seed 1 --out results/figures/all_models.mp4

Each panel shows what the team believes (heat = expected survivors per cell),
obstacles, drones with short trails, and the true survivors: hollow = not yet
found, filled = found. The star is the safe point survivors head for (the
drones are not told it unless the condition says so). Learned policies take
their top action. Needs ffmpeg.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import BeliefGreedy, Lawnmower  # noqa: E402
from swarmsearch.env import GridSearchEnv  # noqa: E402
from swarmsearch.evaluate import ALL_CONDITIONS as CONDITIONS  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402
from swarmsearch.memory import GRUPolicy, RecurrentActor  # noqa: E402

DRONE_COLORS = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#008300", "#e34948", "#52514e"]
TRAIL = 12


def methods():
    out = [("Lawnmower sweep", Lawnmower()), ("Belief-greedy rule", BeliefGreedy())]
    for path, label in [("models/m1_infer_2m.pt", "MAPPO M1 (open areas)"),
                        ("models/m2_2m.pt", "MAPPO M2 (obstacles)"),
                        ("models/bc_greedy.pt", "Copy of the rule"),
                        ("models/bc_rl_2m.pt", "Copy + RL"),
                        ("models/bc_greedy_team.pt", "Copy + team layers")]:
        out.append((label, LearnedPolicy(load(path)[0], temperature=0)))
    gru = RecurrentActor()
    gru.load_state_dict(torch.load("models/bc_greedy_gru.pt", weights_only=False)["state_dict"])
    out.append(("Copy + memory (GRU)", GRUPolicy(gru)))
    return out


def record(sc, seed, policy):
    env = GridSearchEnv(sc)
    env.reset(seed=seed, options={"scenario": sc})
    policy.reset(env)
    frames = []

    def snap():
        frames.append(dict(t=env.t, belief=env.belief * env.n_free,
                           pos={a: tuple(env.pos[a]) for a in env.possible_agents[: sc.n_drones] if env.alive.get(a)},
                           surv=np.array(env.survivors, dtype=float).copy(), found=env.found.copy()))
    snap()
    while env.agents:
        env.step(policy.act(env))
        snap()
    return env, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="obs20_to_safety_unknown", choices=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default="results/figures/all_models.mp4")
    a = p.parse_args()

    sc = CONDITIONS[a.condition](a.seed)
    runs = []
    for name, pol in methods():
        env, frames = record(sc, a.seed, pol)
        m = env.metrics()
        runs.append((name, env, frames))
        print(f"{name:24s} found {m['found_frac']:.0%}  mean find time {m['find_time']:.0f} steps", flush=True)

    g, n_frames = sc.grid_size, max(len(f) for _, _, f in runs)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.2), facecolor="#fcfcfb")
    known = "told" if sc.profile_known else "not told"
    fig.suptitle(f"{sc.n_drones} drones, {sc.n_survivors} survivors, {sc.obstacle_frac:.0%} obstacles, "
                 f"survivors move: {sc.movement.kind} ({known})", fontsize=13)
    clock = fig.text(0.5, 0.925, "", ha="center", fontsize=11, color="#52514e")
    fig.text(0.5, 0.015, "Heat: where the team thinks survivors are   ·   ○ survivor not found   ·   ● found   ·   "
             "★ safe point survivors head for   ·   squares: drones with their last 12 moves",
             ha="center", fontsize=10, color="#52514e")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.06, wspace=0.08, hspace=0.22)
    arts = []
    for ax, (name, env, frames) in zip(axes.flat, runs):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, g - 0.5); ax.set_ylim(g - 0.5, -0.5)
        heat = ax.imshow(frames[0]["belief"], cmap="Oranges", vmin=0, vmax=3, origin="upper")
        oy, ox = np.nonzero(env.obstacles)
        ax.scatter(ox, oy, marker="s", s=55, c="#333333")
        if sc.movement.kind == "to_safety":
            ax.scatter(*sc.movement.target, marker="*", s=220, c="#0b0b0b", zorder=5)
        trails = [ax.plot([], [], color=c, lw=2, alpha=0.7)[0] for c in DRONE_COLORS[: sc.n_drones]]
        drones = ax.scatter([], [], s=90, marker="s", edgecolors="white", linewidths=1.2, zorder=6)
        surv = ax.scatter([], [], s=45, linewidths=1.6, zorder=7)
        title = ax.set_title(name, fontsize=11)
        arts.append((name, frames, heat, trails, drones, surv, title))

    writer = FFMpegWriter(fps=a.fps, bitrate=2400)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, a.out, dpi=90):
        for k in range(n_frames + 2 * a.fps):  # hold the last frame for 2 s
            clock.set_text(f"step {min(k, n_frames - 1)}")
            for name, frames, heat, trails, drones, surv, title in arts:
                i = min(k, len(frames) - 1)
                f = frames[i]
                heat.set_data(f["belief"])
                agents = list(f["pos"])
                for j, tr in enumerate(trails):
                    ag = f"drone_{j}"
                    pts = [fr["pos"][ag] for fr in frames[max(0, i - TRAIL): i + 1] if ag in fr["pos"]]
                    tr.set_data([p[0] for p in pts], [p[1] for p in pts]) if pts else tr.set_data([], [])
                drones.set_offsets(np.array([f["pos"][ag] for ag in agents]) if agents else np.empty((0, 2)))
                drones.set_color([DRONE_COLORS[int(ag.split("_")[1])] for ag in agents])
                drones.set_edgecolor("white")
                surv.set_offsets(f["surv"])
                surv.set_facecolors(["#15233B" if x else "none" for x in f["found"]])
                surv.set_edgecolors("#15233B")
                done = " · done" if i == len(frames) - 1 and k >= len(frames) - 1 and f["found"].all() else ""
                title.set_text(f"{name}\nfound {int(f['found'].sum())}/{len(f['found'])}{done}")
            writer.grab_frame()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
