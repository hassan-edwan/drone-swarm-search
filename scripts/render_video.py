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
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch import camera as cam  # noqa: E402
from swarmsearch.baselines import BeliefGreedy, Lawnmower  # noqa: E402
from swarmsearch.env import GridSearchEnv  # noqa: E402
from swarmsearch.evaluate import ALL_CONDITIONS as CONDITIONS  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402
from swarmsearch.memory import GRUPolicy, RecurrentActor  # noqa: E402

DRONE_COLORS = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#008300", "#e34948", "#52514e"]
TRAIL = 12


def camera_methods(models):
    out = [("Lawnmower sweep (mid, then confirm)", Lawnmower()), ("Belief-greedy rule", BeliefGreedy())]
    for path in models:
        net = load(path)[0]
        out.append((f"MAPPO camera ({Path(path).stem})", LearnedPolicy(net, temperature=0)))
    return out


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
                           alt=dict(env.alt), cand=env.candidates() if env.camera else None,
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
    p.add_argument("--models", nargs="*", default=["models/m6_cam_2m.pt"],
                   help="camera conditions: learned models to show next to the baselines")
    p.add_argument("--out", default="results/figures/all_models.mp4")
    a = p.parse_args()

    sc = CONDITIONS[a.condition](a.seed)
    runs = []
    for name, pol in (camera_methods(a.models) if sc.camera else methods()):
        env, frames = record(sc, a.seed, pol)
        m = env.metrics()
        runs.append((name, env, frames))
        print(f"{name:24s} found {m['found_frac']:.0%}  mean find time {m['find_time']:.0f} steps", flush=True)

    g, n_frames = sc.grid_size, max(len(f) for _, _, f in runs)
    if sc.camera:
        fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 5.6), facecolor="#fcfcfb", squeeze=False)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.2), facecolor="#fcfcfb")
    known = "told" if sc.profile_known else "not told"
    light = f", {sc.lighting.replace('_', ' ')}, {sc.cover_frac:.0%} canopy" if sc.camera else ""
    fig.suptitle(f"{sc.n_drones} drones, {sc.n_survivors} survivors, {sc.obstacle_frac:.0%} obstacles{light}, "
                 f"survivors move: {sc.movement.kind} ({known})", fontsize=13)
    clock = fig.text(0.5, 0.925, "", ha="center", fontsize=11, color="#52514e")
    key = ("Heat: where the team thinks survivors are   ·   ○ survivor not found   ·   ● found   ·   "
           "★ safe point survivors head for   ·   squares: drones with their last 12 moves")
    if sc.camera:
        key = ("Heat: where the team thinks survivors are   ·   ○ not found   ·   ● found   ·   ★ safe point\n"
               "Squares: drones (bigger = higher), outline = camera footprint   ·   ◆ unconfirmed detection\n"
               "Green = tree canopy   ·   grey dots = rubble   ·   darker building = taller")
        clock.set_y(0.875)
    fig.text(0.5, 0.015, key, ha="center", fontsize=9.5 if sc.camera else 10, color="#52514e", linespacing=1.5)
    if sc.camera:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.17, wspace=0.06)
    else:
        fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.06, wspace=0.08, hspace=0.22)
    arts = []
    for ax, (name, env, frames) in zip(axes.flat, runs):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, g - 0.5); ax.set_ylim(g - 0.5, -0.5)
        heat = ax.imshow(frames[0]["belief"], cmap="Oranges", vmin=0, vmax=3, origin="upper")
        oy, ox = np.nonzero(env.obstacles)
        if sc.camera:
            ax.imshow(np.where(env.cover == cam.CANOPY, 1.0, np.nan), cmap="Greens", vmin=0, vmax=2.2, alpha=0.45)
            ry, rx = np.nonzero(env.cover == cam.RUBBLE)
            ax.scatter(rx, ry, s=6, c="#7a7975", zorder=2)
            shade = 0.75 - 0.6 * env.heights[oy, ox] / cam.BUILDING_M[1]
            ax.scatter(ox, oy, marker="s", s=55, c=[str(v) for v in shade])
        else:
            ax.scatter(ox, oy, marker="s", s=55, c="#333333")
        if sc.movement.kind == "to_safety":
            ax.scatter(*sc.movement.target, marker="*", s=220, c="#0b0b0b", zorder=5)
        trails = [ax.plot([], [], color=c, lw=2, alpha=0.7)[0] for c in DRONE_COLORS[: sc.n_drones]]
        drones = ax.scatter([], [], s=90, marker="s", edgecolors="white", linewidths=1.2, zorder=6)
        surv = ax.scatter([], [], s=45, linewidths=1.6, zorder=7)
        cands = ax.scatter([], [], s=60, marker="D", c="#eda100", edgecolors="#0b0b0b", linewidths=0.8, zorder=8)
        feet = [ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=1.4, zorder=5, visible=False))
                for c in DRONE_COLORS[: sc.n_drones]]
        title = ax.set_title(name, fontsize=11)
        arts.append((name, frames, heat, trails, drones, surv, title, cands, feet))

    writer = FFMpegWriter(fps=a.fps, bitrate=2400)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, a.out, dpi=90):
        for k in range(n_frames + 2 * a.fps):  # hold the last frame for 2 s
            clock.set_text(f"step {min(k, n_frames - 1)}")
            for name, frames, heat, trails, drones, surv, title, cands, feet in arts:
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
                if sc.camera:
                    drones.set_sizes([40 + 60 * f["alt"][ag] for ag in agents])
                    cy, cx = np.nonzero(f["cand"])
                    cands.set_offsets(np.c_[cx, cy] if len(cx) else np.empty((0, 2)))
                    for j, fp in enumerate(feet):
                        ag = f"drone_{j}"
                        if ag in f["pos"]:
                            r = cam.FOOT_R[f["alt"][ag]]
                            x0, y0 = f["pos"][ag]
                            fp.set_bounds(x0 - r - 0.5, y0 - r - 0.5, 2 * r + 1, 2 * r + 1)
                            fp.set_visible(True)
                        else:
                            fp.set_visible(False)
                surv.set_offsets(f["surv"])
                surv.set_facecolors(["#15233B" if x else "none" for x in f["found"]])
                surv.set_edgecolors("#15233B")
                done = " · done" if i == len(frames) - 1 and k >= len(frames) - 1 and f["found"].all() else ""
                title.set_text(f"{name}\nfound {int(f['found'].sum())}/{len(f['found'])}{done}")
            writer.grab_frame()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
