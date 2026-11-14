"""Figures for docs/REPORT.md from the saved evaluation results.

    python scripts/make_figures.py      # writes results/figures/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

# Validated categorical slots 1-3 (light surface), fixed per method.
COLORS = {"mappo": "#2a78d6", "belief_greedy": "#eb6834", "lawnmower": "#1baf7a"}
LABELS = {"mappo": "MAPPO (learned)", "belief_greedy": "Belief-greedy rule", "lawnmower": "Lawnmower sweep"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
})


def load(*names) -> dict:
    out = {}
    for n in names:
        out.update(json.loads((ROOT / "results" / n).read_text()))
    return out


def grouped_bars(ax, rows, data, methods, title):
    """Horizontal grouped bars of find time (% of budget) with standard-error whiskers."""
    h = 0.26
    for i, (key, label) in enumerate(rows):
        for j, m in enumerate(methods):
            r = data[key][m]
            y = i + (j - (len(methods) - 1) / 2) * (h + 0.02)
            v = 100 * r["find_time_frac"]
            se = 100 * r["find_time_se"] * r["find_time_frac"] / max(r["find_time"], 1e-9)
            ax.barh(y, v, height=h, color=COLORS[m], label=LABELS[m] if i == 0 else None, zorder=2)
            ax.errorbar(v, y, xerr=se, color=INK2, lw=1, capsize=0, zorder=3)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean time to find a survivor (% of step budget, lower is better)")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


def fig_find_time():
    d = load("m5/results_m1a.json", "m5/results_m1b.json", "m5/results_m2.json")
    open_rows = [("still", "Still"), ("wander", "Wandering"), ("drift_known", "Drift, told"),
                 ("drift_unknown", "Drift, not told"), ("to_safety_known", "To safety, told"),
                 ("to_safety_unknown", "To safety, not told"), ("zones_wander", "Likely zones"),
                 ("drone_lost", "Drone lost"), ("6_drones", "6 drones"), ("random_mix", "Random mix")]
    obs_rows = [("obs10_wander", "10% blocked, wandering"), ("obs30_wander", "30% blocked, wandering"),
                ("obs20_drift_unknown", "20% blocked, drift not told"),
                ("obs20_to_safety_unknown", "20% blocked, to safety not told"),
                ("obs20_drone_lost", "20% blocked, drone lost"), ("obs_random_mix", "Random mix with obstacles")]
    methods = ["mappo", "belief_greedy", "lawnmower"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [len(open_rows), len(obs_rows)]})
    grouped_bars(axes[0], open_rows, d, methods, "Open area, 20×20 grid")
    grouped_bars(axes[1], obs_rows, d, methods, "With obstacles, 20×20 grid")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
    for ax in axes:
        ax.set_xlim(0, 65)
    fig.tight_layout()
    fig.savefig(OUT / "find_time.png", dpi=200)


def fig_sizes():
    d = load("m5/results_m1a.json", "m5/results_sizes30_teams.json", "m5/results_sizes40.json")
    rows = [("drift_unknown", "20×20, 3 drones (training size)"), ("g30_drift_unknown", "30×30"),
            ("g40_drift_unknown", "40×40"), ("2_drones_drift_unknown", "20×20, 2 drones"),
            ("6_drones_drift_unknown", "20×20, 6 drones")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bars(ax, rows, d, ["mappo", "belief_greedy", "lawnmower"],
                 "Drift, direction not told: other grid and team sizes")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    ax.set_xlim(0, 65)
    fig.tight_layout()
    fig.savefig(OUT / "sizes.png", dpi=200)


def fig_wind():
    d = load("m3/results_pybullet_windsweep.json")
    gusts = sorted({float(k.split("@gust")[1]) for k in d})
    conds = sorted({k.split("@gust")[0] for k in d})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for m, label in (("mappo", "MAPPO (learned)"), ("lawnmower", "Lawnmower sweep")):
        def avg(field):
            return [sum(d[f"{c}@gust{g:g}"][m]["3d"][field] for c in conds) / len(conds) for g in gusts]
        axes[0].plot(gusts, [100 * v for v in avg("late")], color=COLORS[m], lw=2, marker="o", ms=6, label=label)
        axes[1].plot(gusts, avg("crashed"), color=COLORS[m], lw=2, marker="o", ms=6, label=label)
    axes[0].set_title("Moves that arrived late (%)", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_title("Drones crashed per mission", loc="left", fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Gust strength (m/s²)")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", colors=INK2, length=0)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "wind.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_find_time()
    fig_sizes()
    fig_wind()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
