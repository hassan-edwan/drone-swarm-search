"""Run every method on the same held-out scenarios and summarise the metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .env import GridSearchEnv
from .scenario import DRIFT_DIRS, MovementProfile, Scenario, ScenarioSampler

# Named test conditions. Each is a function seed -> Scenario so that every
# method sees exactly the same missions.
def _sampled(**kw):
    sampler = ScenarioSampler(**kw)
    return lambda seed: sampler.sample(np.random.default_rng(10_000 + seed))


def _mv(kind: str, speed: float, known: bool, n_zones: int = 0, **kw):
    """Drift direction and safe point vary with the episode seed, so no fixed
    sweep pattern is lined up with the movement by luck."""
    g = kw.get("grid_size", 20)

    def make(seed: int) -> Scenario:
        rng = np.random.default_rng(20_000 + seed)
        drift = DRIFT_DIRS[rng.integers(len(DRIFT_DIRS))] if kind == "drift" else (0, 0)
        target = tuple(int(v) for v in rng.integers(0, g, size=2)) if kind == "to_safety" else (0, 0)
        return Scenario(movement=MovementProfile(kind, speed, drift, target), profile_known=known,
                        n_zones=n_zones, **kw)
    return make


# Most conditions have no likely zones: survivors could be anywhere.
CONDITIONS = {
    "still": _mv("still", 0.0, True),
    "wander": _mv("wander", 1.0, False),
    "drift_known": _mv("drift", 1.0, True),
    "drift_unknown": _mv("drift", 1.0, False),
    "to_safety_known": _mv("to_safety", 1.0, True),
    "to_safety_unknown": _mv("to_safety", 1.0, False),
    "zones_wander": _mv("wander", 0.5, False, n_zones=3),
    "drone_lost": _mv("wander", 1.0, False, drone_failure=True),
    "6_drones": _mv("drift", 1.0, True, n_drones=6),
    "random_mix": _sampled(),
}

# Does the advantage grow with survivor speed? Drift (random direction), known profile.
SPEED_SWEEP = {f"drift_{v:.1f}mps": _mv("drift", v, True) for v in (0.0, 0.5, 1.0, 1.5)}
# M2: block obstacles (buildings, rubble). Movement profile unknown unless stated.
M2_CONDITIONS = {
    "obs10_wander": _mv("wander", 1.0, False, obstacle_frac=0.1),
    "obs30_wander": _mv("wander", 1.0, False, obstacle_frac=0.3),
    "obs20_drift_known": _mv("drift", 1.0, True, obstacle_frac=0.2),
    "obs20_drift_unknown": _mv("drift", 1.0, False, obstacle_frac=0.2),
    "obs20_to_safety_known": _mv("to_safety", 1.0, True, obstacle_frac=0.2),
    "obs20_to_safety_unknown": _mv("to_safety", 1.0, False, obstacle_frac=0.2),
    "obs20_drone_lost": _mv("wander", 1.0, False, obstacle_frac=0.2, drone_failure=True),
    "obs_random_mix": _sampled(obstacle_frac=(0.0, 0.3)),
}
# M4: obstacles are buildings taller than the cruise layers (flown in 3D with
# scripts/run3d.py --m4); no diagonal moves past a building's corner.
def _no_corners(make):
    return lambda seed: replace(make(seed), corner_cutting=False)


M4_CONDITIONS = {
    "m4_drift_unknown": _no_corners(_mv("drift", 1.0, False, obstacle_frac=0.2)),
    "m4_to_safety_unknown": _no_corners(_mv("to_safety", 1.0, False, obstacle_frac=0.2)),
    "m4_dense_wander": _no_corners(_mv("wander", 1.0, False, obstacle_frac=0.3)),
    "m4_random_mix": _no_corners(_sampled(obstacle_frac=(0.1, 0.3))),
}
# M5: other map sizes and team sizes. The time budget scales with the work:
# SWEEPS passes over the map's cells shared between the drones (1.5 sweeps
# is the usual 200 steps for 3 drones on 20x20).
SWEEPS = 1.5


def budget(grid_size: int, n_drones: int, obstacle_frac: float = 0.0, sweeps: float = SWEEPS) -> int:
    return int(round(sweeps * grid_size * grid_size * (1 - obstacle_frac) / n_drones))


def _scaled(kind: str, speed: float, known: bool, grid_size: int = 20, n_drones: int = 3, **kw):
    return _mv(kind, speed, known, grid_size=grid_size, n_drones=n_drones,
               max_steps=budget(grid_size, n_drones, kw.get("obstacle_frac", 0.0)), **kw)


M5_CONDITIONS = {
    "g30_drift_unknown": _scaled("drift", 1.0, False, grid_size=30),
    "g30_still": _scaled("still", 0.0, True, grid_size=30),
    "g40_drift_unknown": _scaled("drift", 1.0, False, grid_size=40),
    "g40_to_safety_unknown": _scaled("to_safety", 1.0, False, grid_size=40),
    "2_drones_drift_unknown": _scaled("drift", 1.0, False, n_drones=2),
    "6_drones_drift_unknown": _scaled("drift", 1.0, False, n_drones=6),
}
ALL_CONDITIONS = {**CONDITIONS, **SPEED_SWEEP, **M2_CONDITIONS, **M4_CONDITIONS, **M5_CONDITIONS}


def run_episode(env: GridSearchEnv, policy, scenario: Scenario, seed: int) -> dict:
    env.reset(seed=seed, options={"scenario": scenario})
    policy.reset(env)
    while env.agents:
        env.step(policy.act(env))
    return env.metrics()


def evaluate(policies: dict, conditions: dict | None = None, n_episodes: int = 50, grid_size: int = 20):
    """Returns {condition: {method: summary}}."""
    conditions = conditions or CONDITIONS
    envs: dict[int, GridSearchEnv] = {}
    results = {}
    for cname, make in conditions.items():
        results[cname] = {}
        g = make(0).grid_size
        env = envs.setdefault(g, GridSearchEnv(Scenario(grid_size=g)))
        for pname, policy in policies.items():
            rows = [run_episode(env, policy, make(s), seed=s) for s in range(n_episodes)]
            results[cname][pname] = summarise(rows, rows and make(0).max_steps)
    return results


def summarise(rows: list[dict], max_steps: int) -> dict:
    t90 = np.array([r["time_to_90"] if r["time_to_90"] is not None else max_steps for r in rows], float)
    ft = np.array([r["find_time"] for r in rows], float)
    return {
        "find_time": float(ft.mean()),
        "find_time_se": float(ft.std() / np.sqrt(max(len(ft), 1))),
        "find_time_frac": float(np.mean([r["find_time_frac"] for r in rows])),
        "time_to_90": float(t90.mean()),
        "time_to_90_std": float(t90.std()),
        "reached_90": float(np.mean([r["time_to_90"] is not None for r in rows])),
        "found_frac": float(np.mean([r["found_frac"] for r in rows])),
        "coverage": float(np.mean([r["coverage"] for r in rows])),
        "repeat_frac": float(np.mean([r["repeat_frac"] for r in rows])),
        "collisions": float(np.mean([r["collisions"] for r in rows])),
        "shield": float(np.mean([r.get("shield", 0) for r in rows])),
    }


def to_markdown(results: dict) -> str:
    lines = ["| Condition | Method | Mean find time (steps) | Find time (% of budget) | Time to 90% (steps) | "
             "Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cname, per in results.items():
        for pname, s in per.items():
            lines.append(
                f"| {cname} | {pname} | {s['find_time']:.0f} ± {s['find_time_se']:.0f} | {s['find_time_frac']:.0%} | "
                f"{s['time_to_90']:.0f} ± {s['time_to_90_std']:.0f} | "
                f"{s['reached_90']:.0%} | {s['found_frac']:.0%} | {s['coverage']:.0%} | "
                f"{s['repeat_frac']:.0%} | {s['collisions']:.1f} | {s.get('shield', 0):.1f} |"
            )
    return "\n".join(lines)
