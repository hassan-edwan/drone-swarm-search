"""One drone's point of view during a field mission (M8).

    python scripts/render_drone_view.py --condition wild_mix --seed 4 --out results/figures/field_drone_view.mp4

Left: what its downward cameras see, to scale (84 deg field of view, so the
image is about 1.8 x the flying height across): colour on top, thermal below.
The ground is drawn from the simulator's terrain, forest, trails and streams
with added texture; a person is drawn at true size (about 1.7 m) when they are
inside the image. Middle: the drone's own copy of the team map, which is only
the base's map while it has a radio link. Right: where it is on the whole map
and its autopilot state. With --drone auto the drone that made the find is
shown (otherwise drone_0).
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
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import BeliefGreedy, Lawnmower  # noqa: E402
from swarmsearch.evaluate import FIELD_CONDITIONS  # noqa: E402
from swarmsearch.field import sensing as sn  # noqa: E402
from swarmsearch.field.env import CLIMB, DESCEND, HOVER, SPRINT0, FieldEnv  # noqa: E402
from swarmsearch.field.sensing import light_level  # noqa: E402

HALF_FOV = np.radians(42.0)
N_PX = 300
SUB = 3  # frames per environment step (the camera pans between cells)
MODE = {"rtb": "low battery: flying home", "landed": "landed: battery swap", "link": "no radio link: flying home",
        "weather": "gusts: flying home", "overwatch": "holding over the find", "nofly_exit": "leaving a no-fly zone"}
DIRS = ["north-west", "north", "north-east", "west", "east", "south-west", "south", "south-east"]  # swarmsearch.env.MOVES order


def act_name(a) -> str:
    if a is None or a == HOVER:
        return "hover and look"
    if a == CLIMB:
        return "climb"
    if a == DESCEND:
        return "descend for a closer look"
    if a >= SPRINT0:
        return "sprint " + DIRS[a - SPRINT0] if a - SPRINT0 < 8 else "sprint"
    return "fly " + DIRS[a] if a < 8 else str(a)


def bilinear(arr, x, y):
    """Sample a (g, g) grid at fractional cell coordinates (cell centres at integers)."""
    g = arr.shape[0]
    x = np.clip(x, 0, g - 1.001)
    y = np.clip(y, 0, g - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    a = arr[y0, x0] * (1 - fx) + arr[y0, x0 + 1] * fx
    b = arr[y0 + 1, x0] * (1 - fx) + arr[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def fbm(n, rng, octaves=5):
    out = np.zeros((n, n))
    for o in range(octaves):
        k = 2 ** (o + 2)
        coarse = rng.random((k + 1, k + 1))
        xs = np.linspace(0, k, n, endpoint=False)
        out += bilinear(np.pad(coarse, ((0, 0), (0, 0))), *np.meshgrid(xs, xs)) / 2 ** o
    out -= out.min()
    return out / out.max()


def run(sc, seed, policy):
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    policy.reset(env)
    alerts = {a: [] for a in env.agents}
    orig = env._search

    def search(a, act):
        v = env.views[a]
        n_out = len(v.outbox)
        got = []
        q_add = env.queue.add
        env.queue.add = lambda it: (got.append(it), q_add(it))[1]
        try:
            r = orig(a, act)
        finally:
            env.queue.add = q_add
        for it in got + [it for it in v.outbox[n_out:] if it.get("drone") == a]:
            alerts[a].append((env.t, tuple(it["cell"]), it["true"] is not None))
        return r

    env._search = search
    frames = []
    while True:
        ds = {}
        for a in env.possible_agents[: sc.n_drones]:
            if not env.alive.get(a):
                continue
            k = env.alt[a]
            cls = sn.view_class(k, 0, 0)
            ds[a] = dict(true=env.true_pos[a].copy(), est=env.pos[a].copy(), k=k, alt_m=float(env.alt_now[a]),
                         bat=env.battery[a], mode=env.mode[a], act=env.last_act.get(a),
                         linked="base" in env.group_of.get(a, []), belief=env.views[a].belief * env.n_free,
                         occ=[env.people.occ[i].get(cls) for i in range(len(env.people.pos))],
                         los=float(env.G_true[k][tuple(np.r_[env.true_pos[a][::-1], sn.R, sn.R])]
                                   if env.G_true[k].ndim == 4 else 1.0))
        frames.append(dict(t=env.t, hour=env.hour(), drones=ds, people=env.people.pos.copy(),
                           found=env.found.copy(), outside=env.people.outside.copy()))
        if not env.agents:
            break
        env.step(policy.act(env))
    return env, frames, alerts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="wild_mix", choices=list(FIELD_CONDITIONS))
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--policy", default="rule", choices=["rule", "sweep"])
    p.add_argument("--model", default=None)
    p.add_argument("--drone", default="auto")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default="results/figures/field_drone_view.mp4")
    a = p.parse_args()

    sc = FIELD_CONDITIONS[a.condition](a.seed)
    if a.model:
        from swarmsearch.mappo import LearnedPolicy, load
        pol = LearnedPolicy(load(a.model)[0], temperature=0)
    else:
        pol = BeliefGreedy() if a.policy == "rule" else Lawnmower()
    env, frames, alerts = run(sc, a.seed, pol)
    w, fam, g = env.w, env.fam, env.grid_size
    cell = fam.cell_m
    drone = a.drone
    if drone == "auto":
        hits = [d for d, al in alerts.items() if any(tr for *_r, tr in al)]
        drone = hits[0] if hits else "drone_0"
    print("drone:", drone, "alerts:", [(t, tr) for t, _c, tr in alerts[drone]])
    my_alerts = {t: (c, tr) for t, c, tr in alerts[drone]}

    rng = np.random.default_rng(0)
    tex = fbm(512, rng, octaves=7)  # 1 m ground texture, tiled
    crowns = fbm(512, np.random.default_rng(1), octaves=6)  # tree crowns, a few metres
    shade = LightSource(315, 40).hillshade(w.elev, vert_exag=3, dx=cell, dy=cell)
    canopy = w.canopy.astype(float)
    water = w.water.astype(float)
    person_off = np.random.default_rng(seed=7).uniform(-0.12, 0.12, size=(len(env.people.pos), 2))
    segs = []  # trail and stream polylines in cell coordinates
    for mask, col, lw in [(w.trail, "#8a6a3b", 2.2), (w.stream, "#3d6fae", 3.0)]:
        ys, xs = np.nonzero(mask)
        on = set(zip(xs.tolist(), ys.tolist()))
        for x, y in on:
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                if (x + dx, y + dy) in on:
                    segs.append(((x, y), (x + dx, y + dy), col, lw))

    fig = plt.figure(figsize=(14, 7.8), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 3, left=0.02, right=0.98, top=0.87, bottom=0.09, wspace=0.08, hspace=0.18,
                          width_ratios=[1, 1.5, 1])
    ax_rgb, ax_th = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_map, ax_loc = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")
    for ax_ in (ax_rgb, ax_th, ax_map, ax_loc):
        ax_.set_xticks([]); ax_.set_yticks([])
    img_rgb = ax_rgb.imshow(np.zeros((N_PX, N_PX, 3)), extent=(0, 1, 1, 0))
    img_th = ax_th.imshow(np.zeros((N_PX, N_PX)), cmap="inferno", vmin=0, vmax=1, extent=(0, 1, 1, 0))
    for ax_ in (ax_rgb, ax_th):
        ax_.set_xlim(0, 1); ax_.set_ylim(1, 0)
    lines = [ax_rgb.plot([], [], color=c, lw=lw, alpha=0.9, solid_capstyle="round")[0] for *_p, c, lw in segs]
    person_rgb = [ax_rgb.add_patch(Ellipse((0, 0), 0, 0, fc="#e8462e", ec="#3a1a10", lw=0.4, visible=False))
                  for _ in env.people.pos]
    alert_box = [ax_.add_patch(Rectangle((0, 0), 0.1, 0.1, fill=False, ec="#fcd116", lw=2, visible=False))
                 for ax_ in (ax_rgb, ax_th)]
    alert_txt = ax_rgb.text(0.02, 0.97, "", color="#fcd116", fontsize=10, fontweight="bold", va="top",
                            transform=ax_rgb.transAxes)
    blocked_txt = ax_rgb.text(0.5, 0.5, "", color="white", fontsize=11, ha="center", va="center",
                              transform=ax_rgb.transAxes)
    scale_rgb = ax_rgb.text(0.98, 0.03, "", color="white", fontsize=9, ha="right", va="bottom",
                            transform=ax_rgb.transAxes, bbox=dict(fc="black", alpha=0.4, lw=0))
    ax_rgb.set_title("Colour camera, looking down", fontsize=11)
    ax_th.set_title("Thermal camera", fontsize=11)

    heat = ax_map.imshow(np.zeros((g, g)), cmap="Oranges", vmin=0, vmax=1)
    ax_map.scatter(*w.base, s=220, marker="*", c="#fcd116", edgecolors="k", zorder=6)
    me_est = ax_map.scatter([], [], marker="s", s=90, c="#2a78d6", edgecolors="white", zorder=8)
    me_true = ax_map.scatter([], [], marker="x", s=50, c="k", zorder=9)
    trail_ln, = ax_map.plot([], [], color="#2a78d6", lw=1, alpha=0.5)
    map_title = ax_map.set_title("", fontsize=11)
    ax_map.set_xlim(-0.5, g - 0.5); ax_map.set_ylim(g - 0.5, -0.5)

    ax_loc.imshow(shade, cmap="gray", vmin=0, vmax=1.2)
    ax_loc.imshow(np.where(canopy > 0.05, canopy, np.nan), cmap="Greens", vmin=0, vmax=1.4, alpha=0.55)
    ax_loc.scatter(*w.base, s=120, marker="*", c="#fcd116", edgecolors="k", zorder=6)
    others = ax_loc.scatter([], [], marker="s", s=18, c="#7a7975", zorder=7)
    me_loc = ax_loc.scatter([], [], marker="s", s=40, c="#2a78d6", edgecolors="white", zorder=8)
    ppl_loc = ax_loc.scatter([], [], s=25, edgecolors="k", linewidths=0.6, zorder=7)
    ax_loc.set_title("Where it is (the whole area)", fontsize=11)
    status = ax_txt.text(0.02, 0.98, "", fontsize=10.5, va="top", family="DejaVu Sans Mono", color="#2b2a27",
                         linespacing=1.5, transform=ax_txt.transAxes)
    title = fig.text(0.5, 0.95, "", ha="center", fontsize=13)
    sub = fig.text(0.5, 0.91, "", ha="center", fontsize=10, color="#52514e")
    fig.text(0.5, 0.015, "Cameras to scale (84° field of view; the image is about 1.8 × the height across). "
             "Ground drawn from the simulator's terrain, forest, trails and streams, with added texture; "
             "the person is drawn at true size.\nMiddle: this drone's own copy of the map "
             "(square: where it thinks it is, x: where it really is). Yellow box: an alert sent to the operator.",
             ha="center", fontsize=9, color="#52514e", linespacing=1.5)

    seq = []
    for i in range(len(frames) - 1):
        for s in range(SUB):
            seq.append((i, s / SUB))
    seq += [(len(frames) - 1, 0.0)] * a.fps * 2
    hist = []
    writer = FFMpegWriter(fps=a.fps, bitrate=3000)
    with writer.saving(fig, a.out, dpi=85):
        last_i = -1
        for i, frac in seq:
            f = frames[i]
            nxt = frames[min(i + 1, len(frames) - 1)]
            if drone not in f["drones"]:
                continue
            d = f["drones"][drone]
            d2 = nxt["drones"].get(drone, d)
            pos = d["true"] * (1 - frac) + d2["true"] * frac
            alt = d["alt_m"] * (1 - frac) + d2["alt_m"] * frac
            light = light_level(f["hour"])
            width = max(2 * max(alt, 3.0) * np.tan(HALF_FOV), 4.0)  # metres across the image
            u = (np.arange(N_PX) + 0.5) / N_PX - 0.5
            X, Y = np.meshgrid(pos[0] + u * width / cell, pos[1] + u * width / cell)
            mx = ((X * cell) % 512).astype(int); my = ((Y * cell) % 512).astype(int)
            t = tex[my, mx]
            sh = bilinear(shade, X, Y)
            can = bilinear(canopy, X, Y)
            wat = bilinear(water, X, Y) > 0.5
            trees = crowns[my, mx] < can * 0.9
            ground = np.stack([0.42 + 0.35 * t, 0.42 + 0.28 * t, 0.26 + 0.18 * t], -1)
            leaf = np.stack([0.08 + 0.15 * t, 0.20 + 0.25 * t, 0.08 + 0.10 * t], -1)
            rgb = np.where(trees[..., None], leaf, ground) * (0.55 + 0.5 * sh[..., None])
            rgb = np.where(wat[..., None], np.array([0.25, 0.38, 0.52]) * (0.9 + 0.2 * t[..., None]), rgb)
            rgb = np.clip(rgb * (0.08 + 0.92 * light), 0, 1)
            th = 0.30 + 0.25 * t + 0.1 * (sh - 0.5) - 0.12 * trees - 0.2 * wat
            # people inside the image
            for j, pe in enumerate(f["people"]):
                pr = person_rgb[j]
                if f["outside"][j]:
                    pr.set_visible(False)
                    continue
                px = (pe[0] + person_off[j, 0] - pos[0]) * cell / width + 0.5
                py = (pe[1] + person_off[j, 1] - pos[1]) * cell / width + 0.5
                inside = 0 <= px <= 1 and 0 <= py <= 1
                hidden = d["occ"][j] is True
                pr.set_visible(inside and not hidden)
                if inside:
                    pr.center = (px, py)
                    pr.width, pr.height = 0.55 / width, 1.7 / width
                    pr.set_alpha(min(1.0, 0.25 + light))
                    r_px = max(1.2 / width * N_PX, 0.7)
                    yy, xx = np.mgrid[0:N_PX, 0:N_PX]
                    blob = np.exp(-((xx - px * N_PX) ** 2 + (yy - py * N_PX) ** 2) / (2 * r_px ** 2))
                    warm = env.people.warm[j] * (0.3 if hidden else 1.0)
                    th = th + 0.6 * warm * blob
            img_rgb.set_data(rgb)
            img_th.set_data(np.clip(th, 0, 1))
            for ln, (p0, p1, _c, lw) in zip(lines, segs):
                xs = [(p0[0] - pos[0]) * cell / width + 0.5, (p1[0] - pos[0]) * cell / width + 0.5]
                ys = [(p0[1] - pos[1]) * cell / width + 0.5, (p1[1] - pos[1]) * cell / width + 0.5]
                ln.set_data(xs, ys)
                ln.set_linewidth(max(lw * 60 / width, 0.6))
            blocked_txt.set_text("" if d["los"] > 0 else "view blocked by terrain")
            scale_rgb.set_text(f"{width:.0f} m across · {alt:.0f} m up")
            # alerts from this step stay up for two steps
            al = my_alerts.get(f["t"]) or my_alerts.get(f["t"] - 1)
            for b in alert_box:
                b.set_visible(bool(al))
            if al:
                cxy, true = al
                j = next((j for j, pe in enumerate(f["people"]) if true and tuple(pe) == cxy), None)
                if j is not None:
                    bx_ = (f["people"][j][0] + person_off[j, 0] - pos[0]) * cell / width + 0.5
                    by_ = (f["people"][j][1] + person_off[j, 1] - pos[1]) * cell / width + 0.5
                else:
                    r2 = np.random.default_rng(f["t"])
                    bx_, by_ = r2.uniform(0.25, 0.75, 2)
                sz = max(6.0 / width, 0.06)
                for b in alert_box:
                    b.set_bounds(bx_ - sz / 2, by_ - sz / 2, sz, sz)
                alert_txt.set_text("ALERT sent to the operator" + ("" if d["linked"] else " (queued: no link)"))
            else:
                alert_txt.set_text("")
            # its own map
            if i != last_i:
                hist.append(d["est"].copy())
                last_i = i
            b_ = d["belief"]
            heat.set_data(b_)
            heat.set_clim(0, max(float(np.quantile(b_, 0.995)), 1e-9))
            me_est.set_offsets([d["est"]])
            me_true.set_offsets([d["true"]] if np.any(d["est"] != d["true"]) else np.empty((0, 2)))
            h = np.array(hist[-200:])
            trail_ln.set_data(h[:, 0], h[:, 1])
            map_title.set_text(f"{drone.replace('_', ' ')}'s own map · " +
                               ("shared with the base" if d["linked"] else "OUT OF RADIO CONTACT: its own copy"))
            oth = [v["true"] for k_, v in f["drones"].items() if k_ != drone]
            others.set_offsets(np.array(oth) if oth else np.empty((0, 2)))
            me_loc.set_offsets([pos])
            vis = ~f["outside"]
            ppl_loc.set_offsets(f["people"][vis] if vis.any() else np.empty((0, 2)))
            ppl_loc.set_facecolors(["#1baf7a" if x else "#e34948" for x in f["found"][vis]])
            err = np.linalg.norm(d["est"] - d["true"]) * cell
            status.set_text(
                f"{drone.replace('_', ' ')}\n"
                f"height      {alt:5.0f} m\n"
                f"battery     {d['bat']:5.0%}\n"
                f"radio       {'linked to base' if d['linked'] else 'no link'}\n"
                f"autopilot   {MODE.get(d['mode'], 'search policy in control')}\n"
                f"doing       {act_name(d['act']) if not d['mode'] else '-'}\n"
                f"GPS error   {err:5.0f} m\n"
                f"alerts sent {sum(1 for t_ in my_alerts if t_ <= f['t'])}")
            hh = f["hour"]
            title.set_text(f"One drone's view · {sc.family} search, {sc.n_drones} drones · "
                           f"{int(hh % 24):02d}:{int(hh % 1 * 60):02d} · minute {f['t'] * fam.step_s / 60:.0f} · "
                           f"found {int(f['found'].sum())}/{len(f['found'])}")
            sub.set_text(f"{sc.category} lost {sc.hours_lost:.1f} h · "
                         f"{'day' if light > 0.5 else 'dusk' if light > 0.1 else 'night'} · wind {sc.wind_mps:.0f} m/s")
            writer.grab_frame()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
