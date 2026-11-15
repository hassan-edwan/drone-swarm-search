"""Video of one field-environment mission (M8), with an event log.

    python scripts/render_field.py --condition wild_mix --seed 8 --out results/figures/field_wilderness.mp4

Left: the true world (shaded terrain, forest, trails, drainages, water,
no-fly zones), people (red = not found, green = found, grey ring = where
they started), drones (size = height, outline = camera footprint, label =
autopilot mode), radio (lines to the base), ground teams (triangles).
Right: the team map at the base (heat = where the person may be), pending
detections (diamonds), the reported last-seen point (x).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import BeliefGreedy, Lawnmower  # noqa: E402
from swarmsearch.evaluate import FIELD_CONDITIONS  # noqa: E402
from swarmsearch.field.env import FieldEnv  # noqa: E402
from swarmsearch.field.sensing import light_level  # noqa: E402

COLORS = ["#2a78d6", "#eda100", "#6250d6", "#e87ba4", "#1baf7a", "#008300", "#e34948", "#52514e"]
MODE = {"rtb": "low battery: home", "landed": "battery swap", "link": "no link: home", "weather": "gusts: home",
        "overwatch": "holding over find", "nofly_exit": "leaving no-fly"}


def run(sc, seed, policy):
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    policy.reset(env)
    frames, events = [], []
    prev = dict(found=0, rescued=0, reviewed=0, modes={})

    def log(msg):
        events.append((env.t, env.hour(), msg))

    log(f"launch: {sc.n_drones} drones, {sc.category} lost {sc.hours_lost:.1f} h ago, "
        f"{'day' if light_level(env.hour()) > 0.5 else 'dusk' if light_level(env.hour()) > 0.1 else 'night'}, "
        f"{sc.air_c:.0f} C, wind {sc.wind_mps:.0f} m/s; person is {env.people.resp[0]}, behaviour '{env.people.kind[0]}'"
        + (", starts OUTSIDE the area" if env.people.outside[0] else ""))
    start = env.people.pos.copy()
    while True:
        base_v = env.views["base"]
        env._v = base_v
        frames.append(dict(
            t=env.t, hour=env.hour(), belief=env.belief * env.n_free, cand=env.candidates(),
            drones={a: (tuple(env.true_pos[a]), env.alt[a], env.mode[a], env.battery[a], "base" in env.group_of.get(a, []))
                    for a in env.possible_agents[: sc.n_drones] if env.alive.get(a)},
            people=env.people.pos.copy(), found=env.found.copy(), outside=env.people.outside.copy(),
            teams=[tuple(t["pos"]) for t in env.teams], nofly=env.nofly.copy(), queue=len(env.queue),
            water=env.water.copy(), smoke=env.smoke.copy(), collapsed_seen=env.collapsed & ~env.unseen))
        if not env.agents:
            break
        env.step(policy.act(env))
        for a in env.possible_agents[: sc.n_drones]:
            m = env.mode.get(a)
            if m != prev["modes"].get(a) and m is not None:
                log(f"{a}: {MODE.get(m, m)} (battery {env.battery[a]:.0%})")
            if not env.alive.get(a) and prev["modes"].get(a) != "lost":
                log(f"{a}: lost (crash or failed launch)")
                m = "lost"
            prev["modes"][a] = m
        if env.found.sum() > prev["found"]:
            log("operator confirms a find -> ground team dispatched")
        if (env.rescued_t >= 0).sum() > prev["rescued"]:
            log("ground team reaches the person: rescued")
        if env.queue.reviewed // 10 > prev["reviewed"] // 10:
            log(f"operator has reviewed {env.queue.reviewed} alerts ({env.queue.missed} real ones missed), "
                f"{len(env.queue)} waiting")
        prev.update(found=int(env.found.sum()), rescued=int((env.rescued_t >= 0).sum()), reviewed=env.queue.reviewed)
    m = env.metrics()
    log(f"end: found {m['found_frac']:.0%}, rescued {m['rescued_frac']:.0%}, {m['alarms']} alerts "
        f"({m['false_alarms']} false), {m['swaps']} battery swaps, area covered {m['coverage']:.0%}")
    return env, frames, events, start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="wild_mix", choices=list(FIELD_CONDITIONS))
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--policy", default="rule", choices=["rule", "sweep"])
    p.add_argument("--every", type=int, default=2, help="steps per frame")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default="results/figures/field_wilderness.mp4")
    a = p.parse_args()

    sc = FIELD_CONDITIONS[a.condition](a.seed)
    env, frames, events, start = run(sc, a.seed, BeliefGreedy() if a.policy == "rule" else Lawnmower())
    w, fam, g = env.w, env.fam, env.grid_size
    log_path = Path(a.out).with_suffix(".txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for t, h, msg in events:
            line = f"{int(h % 24):02d}:{int(h % 1 * 60):02d}  (min {t * fam.step_s / 60:5.1f})  {msg}"
            f.write(line + "\n")
            print(line)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7.6), facecolor="#fcfcfb")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.09, wspace=0.05)
    shade = LightSource(315, 40).hillshade(w.elev, vert_exag=3, dx=fam.cell_m, dy=fam.cell_m)
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1.2)
    ax.imshow(np.where(w.canopy > 0.05, w.canopy, np.nan), cmap="Greens", vmin=0, vmax=1.4, alpha=0.55)
    feats = [(w.stream, "#4f8fd6", 4)] if sc.family != "flood" else []
    if sc.family == "wilderness":
        feats.append((w.trail, "#8a5a2b", 5))
    for mask, c, s in feats:
        yy, xx = np.nonzero(mask)
        ax.scatter(xx, yy, s=s, c=c, marker="s", linewidths=0)
    if w.struct_h.any():  # buildings, darker = taller; collapsed ones as rubble
        bh = np.where(w.struct_h > 0, w.struct_h, np.nan)
        ax.imshow(np.where(env.collapsed, np.nan, bh), cmap="Greys", vmin=0, vmax=60, alpha=0.95)
        cy, cx = np.nonzero(env.collapsed)
        ax.scatter(cx, cy, s=22, c="#a0522d", marker="x", linewidths=1.0)
    water_img = ax.imshow(np.full((g, g), np.nan), cmap="Blues", vmin=0, vmax=1.2, alpha=0.75)
    smoke_img = ax.imshow(np.full((g, g), np.nan), cmap="Greys", vmin=0, vmax=1.3, alpha=0.55)
    nofly_img = ax.imshow(np.zeros((g, g)), cmap="Reds", vmin=0, vmax=1, alpha=0.0)
    ax.scatter(*w.base, s=220, marker="*", c="#fcd116", edgecolors="k", zorder=6)
    ax.scatter(start[:, 0], start[:, 1], s=90, facecolors="none", edgecolors="#7a7975", linewidths=1.2, zorder=5)
    heat = bx.imshow(frames[0]["belief"], cmap="Oranges", vmin=0, vmax=6)
    bx.scatter(*env.rep_ipp, s=110, marker="x", c="k", zorder=6)
    bx.scatter(*w.base, s=220, marker="*", c="#fcd116", edgecolors="k", zorder=6)
    for axis, title in [(ax, "The world (truth)"), (bx, "The team's map at the base")]:
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_xlim(-0.5, g - 0.5); axis.set_ylim(g - 0.5, -0.5)
        axis.set_title(title, fontsize=12)
    people = ax.scatter([], [], s=70, zorder=7, edgecolors="k", linewidths=0.8)
    drones = ax.scatter([], [], marker="s", edgecolors="white", linewidths=1.0, zorder=8)
    teams = ax.scatter([], [], marker="^", s=80, c="k", zorder=7)
    cands = bx.scatter([], [], marker="D", s=50, c="#eda100", edgecolors="k", zorder=7)
    links = [ax.plot([], [], color=COLORS[i], lw=0.8, alpha=0.6, ls="--")[0] for i in range(8)]
    feet = [ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=COLORS[i], lw=1.2, zorder=6, visible=False))
            for i in range(8)]
    labels = [ax.text(0, 0, "", fontsize=7, color=COLORS[i], zorder=9, fontweight="bold") for i in range(8)]
    title = fig.text(0.5, 0.95, "", ha="center", fontsize=13)
    sub = fig.text(0.5, 0.905, "", ha="center", fontsize=10, color="#52514e")
    fig.text(0.5, 0.012, "Shaded relief · green: forest · brown: trails · blue: drainages and water · grey: buildings (x: collapsed) · haze: smoke · star: base · "
             "red/green dot: person (not found / found) · grey ring: where they started\nsquares: drones "
             "(bigger = higher, outline = camera footprint) · dashed line: radio link to base · triangles: ground teams · "
             "x: reported last-seen point · diamonds: alerts waiting for a closer look",
             ha="center", fontsize=9, color="#52514e", linespacing=1.5)
    writer = FFMpegWriter(fps=a.fps, bitrate=2600)
    with writer.saving(fig, a.out, dpi=85):
        for f in frames[:: a.every] + [frames[-1]] * a.fps * 2:
            h = f["hour"]
            title.set_text(f"{sc.family.capitalize()} search, {sc.n_drones} drones · "
                           f"{int(h % 24):02d}:{int(h % 1 * 60):02d} · minute {f['t'] * fam.step_s / 60:.0f} · "
                           f"found {int(f['found'].sum())}/{len(f['found'])}")
            sub.set_text(f"operator queue {f['queue']} · " + " · ".join(
                f"{d.split('_')[1]}: {bat:.0%}" + (f" ({MODE.get(m, m)})" if m else "")
                for d, (_, _, m, bat, _) in f["drones"].items()))
            heat.set_data(f["belief"])
            heat.set_clim(0, max(float(np.quantile(f["belief"], 0.995)), 1e-9))  # relative: where the team looks next
            cy, cx = np.nonzero(f["cand"])
            cands.set_offsets(np.c_[cx, cy] if len(cx) else np.empty((0, 2)))
            nofly_img.set_data(f["nofly"].astype(float))
            water_img.set_data(np.where(f["water"], 1.0, np.nan))
            smoke_img.set_data(np.where(f["smoke"] > 0.05, f["smoke"], np.nan))
            nofly_img.set_alpha(0.35 if f["nofly"].any() else 0.0)
            vis = ~f["outside"]
            people.set_offsets(f["people"][vis] if vis.any() else np.empty((0, 2)))
            people.set_facecolors(["#1baf7a" if x else "#e34948" for x in f["found"][vis]])
            ds = list(f["drones"].items())
            drones.set_offsets(np.array([p for _, (p, *_r) in ds]) if ds else np.empty((0, 2)))
            drones.set_color([COLORS[int(d.split("_")[1])] for d, _ in ds])
            drones.set_sizes([30 + 35 * alt for _, (_, alt, *_r) in ds])
            teams.set_offsets(np.array(f["teams"]) if f["teams"] else np.empty((0, 2)))
            for i in range(8):
                d = f"drone_{i}"
                if d in f["drones"]:
                    (x, y), alt, m, bat, linked = f["drones"][d]
                    r = env.foot_r[alt]
                    feet[i].set_bounds(x - r - 0.5, y - r - 0.5, 2 * r + 1, 2 * r + 1)
                    feet[i].set_visible(True)
                    links[i].set_data([x, w.base[0]], [y, w.base[1]]) if linked else links[i].set_data([], [])
                    labels[i].set_position((x + 0.8, y - 0.8))
                    labels[i].set_text(MODE.get(m, "") if m else "")
                else:
                    feet[i].set_visible(False)
                    links[i].set_data([], [])
                    labels[i].set_text("")
            writer.grab_frame()
    print("wrote", a.out, "and", log_path)


if __name__ == "__main__":
    main()
