"""Figure: one under-canopy check meeting what the map did not show, seen from above and from the side.

    python scripts/real_site/plot_surprise.py --out results/figures/surprise_example.png

Searches spots under the trees for a check where the drone stops and re-plans
(something real it was not told about), then draws:
- left, from above at the flying height (1 to 5 m): obstacles on the map (grey),
  obstacles only in the real world (red), wires (black), the planned route
  (blue, dashed), the route flown (orange), where it saw something and stopped (x);
- right, the side view along the flown route: real voxels within 3 m of it,
  coloured by whether the map had them.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch import fly_exec as fx  # noqa: E402
from swarmsearch import world3d as w3  # noqa: E402
from swarmsearch.windfield import WindField  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0))
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--out", default="results/figures/surprise_example.png")
    a = ap.parse_args()
    S = Path(a.site)
    M = w3.build_map(a.fine_from, Path(a.fine_from).with_name("fine_points.npz"), S / "buildings.json",
                     S / "hazards.json", shift=tuple(a.shift))
    H, B = json.load(open(S / "hazards.json")), json.load(open(S / "buildings.json"))
    T = w3.make_truth(M, H, B, seed=a.world)
    outer = f3.CoarseTerrain(S / "lidar10.npz", hazards=S / "hazards.json")
    L = np.load(S / "lidar10.npz")
    wf = WindField(L["dtm"], L["chm"], 10.0, 5.0, 270.0)
    base = f3.VoxPlanner(M, outer)
    factory = lambda extra: base if extra is None else f3.VoxPlanner(M, outer, known_extra=extra)  # noqa: E731
    rng = np.random.default_rng(3)
    forest = np.argwhere(M.top() > 10)
    best = None
    for n in range(200):
        tx, ty = M.xy(*forest[rng.integers(len(forest))])
        if not M.inside(tx, ty, 100):
            continue
        ang = rng.uniform(0, 2 * np.pi)
        start = (tx + 80 * np.cos(ang), ty + 80 * np.sin(ang))
        ex = fx.Executor(M, T, factory, wind=wf, nav="vio", sensing=True, seed=n, wind_ij=outer.ij)
        r = ex.fly_check(start, (tx, ty), 40.0)
        if r.replans and r.outcome == "ok" and r.reached:
            plan, _ = base.under_check(start, (tx, ty), 40.0)
            pl = np.array([(q[1], q[2]) for q in plan if q[4] == "under"])
            fl = np.array([(q[1], q[2]) for q in r.trace if q[4] in ("under", "look")])
            if len(pl) and len(fl):
                dev = max(np.min(np.hypot(*(pl - f).T)) for f in fl)
                if dev > 3.0:  # visibly went around something
                    best = (r, plan, (tx, ty))
                    break
    if best is None:
        print("no check with a re-plan found")
        return
    r, plan, tgt = best
    low = [q for q in r.trace if q[4] in ("under", "look", "descend", "climb")]
    xs = np.array([q[1] for q in low])
    ys = np.array([q[2] for q in low])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(25.0, (max(xs.max() - xs.min(), ys.max() - ys.min())) / 2 + 12)
    i0, j0 = M.ij(cx - half, cy + half)
    i1, j1 = M.ij(cx + half, cy - half)
    band = slice(1, 6)
    mb = M.occ[i0:i1, j0:j1, band].any(axis=2)
    tb = T.occ[i0:i1, j0:j1, band].any(axis=2)
    wire_t = ((T.occ[i0:i1, j0:j1, band] & w3.WIRE) > 0).any(axis=2)
    img = np.ones(mb.shape + (3,))
    img[mb] = (0.65, 0.65, 0.65)
    img[tb & ~mb] = (0.9, 0.25, 0.2)
    img[wire_t] = (0.05, 0.05, 0.05)
    ext = (cx - half, cx + half, cy - half, cy + half)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax.imshow(img, extent=ext, origin="upper", interpolation="nearest")
    pl = [q for q in plan if q[4] in ("under", "look", "descend", "climb")]
    ax.plot([q[1] for q in pl], [q[2] for q in pl], "--", color="#1f6fd1", lw=2, label="planned on the map")
    ax.plot(xs, ys, color="#f08c00", lw=2.2, label="flown in the real world")
    stops = [e for e in r.events if e[0] == "stop and re-plan"]
    for e in stops:
        q = min(r.trace, key=lambda q: abs(q[0] - e[1]))
        ax.plot(q[1], q[2], "x", color="k", ms=12, mew=3)
    ax.plot(*tgt, "*", color="#2b9d3a", ms=18, label="the alert spot")
    # explain what happened
    if stops:
        q = min(r.trace, key=lambda q: abs(q[0] - stops[0][1]))
        h_stop = q[3] - T.ground(q[1], q[2])
        ax.annotate(f"1. planned way down; saw something real here\n   {h_stop:.0f} m up: stopped, re-planned",
                    (q[1], q[2]), xytext=(q[1] + 4, q[2] - 9), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(fc="white", ec="0.6", alpha=0.9))
    first_under = next((q for q in r.trace if q[4] == "under"), None)
    if first_under is not None:
        ax.annotate("2. down through another gap,\n   under the crowns to the spot", (first_under[1], first_under[2]),
                    xytext=(first_under[1] + 4, first_under[2] + 6), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#f08c00"), bbox=dict(fc="white", ec="0.6", alpha=0.9))
    ax.plot([], [], "s", color=(0.65, 0.65, 0.65), label="on the map (1-5 m up)")
    ax.plot([], [], "s", color=(0.9, 0.25, 0.2), label="only in the real world")
    ax.plot([], [], "x", color="k", label="saw it: stop and re-plan")
    ax.plot([], [], "s", color=(0.05, 0.05, 0.05), label="wire at flying height")
    ax.set_xlim(ext[:2])
    ax.set_ylim(ext[2:])
    ax.set_xlabel("m east")
    ax.set_ylabel("m north")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("From above, at the drone's flying height")
    # side view along the flown low route
    P = np.c_[xs, ys]
    s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    out_m, out_t = [], []
    for k in range(0, len(P), 3):
        x, y = P[k]
        for W, store in ((M, out_m), (T, out_t)):
            for kind, d in W.kinds_near(x, y, 4.0, 3.0):
                pass
        i, j = T.ij(x, y)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                ii, jj = i + di, j + dj
                if 0 <= ii < T.n and 0 <= jj < T.n:
                    for kk in np.nonzero(T.occ[ii, jj, :20])[0]:
                        (out_t if not M.occ[ii, jj, kk] else out_m).append((s_[k], kk + 0.5))
    if out_m:
        A = np.array(out_m)
        ax2.scatter(A[:, 0], A[:, 1], s=6, c="#8a8a8a", label="on the map")
    if out_t:
        A = np.array(out_t)
        ax2.scatter(A[:, 0], A[:, 1], s=8, c="#e03c31", label="only in the real world (growth, saplings, ...)")
    hh = [q[3] - T.ground(q[1], q[2]) for q in low]
    ax2.plot(s_, hh, color="#f08c00", lw=2.5, label="drone (flown)")
    ax2.axhline(0, color="#7a5a3a", lw=2)
    ax2.set_ylim(-0.5, 20)
    ax2.set_xlabel("distance along the flown route (m)")
    ax2.set_ylabel("height above ground (m)")
    ax2.set_title("Side view along the route (within 2 m of it)")
    ax2.legend(loc="upper right", fontsize=8)
    det = ", ".join(f"{k}: {v}" for k, v in r.detected.items())
    fig.suptitle(f"An under-canopy check meets what the map did not show (forward depth camera, camera navigation): "
                 f"{r.replans} stop and re-plan, reached the spot, {r.time_s - r.planned_s:+.0f} s against the plan",
                 fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
