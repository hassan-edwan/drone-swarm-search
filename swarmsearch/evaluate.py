"""Run every method on the same held-out scenarios and summarise the metrics."""

from __future__ import annotations

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
    def make(seed: int) -> Scenario:
        rng = np.random.default_rng(20_000 + seed)
        drift = DRIFT_DIRS[rng.integers(len(DRIFT_DIRS))] if kind == "drift" else (0, 0)
        target = tuple(int(v) for v in rng.integers(0, 20, size=2)) if kind == "to_safety" else (0, 0)
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
ALL_CONDITIONS = {**CONDITIONS, **SPEED_SWEEP}


def run_episode(env: GridSearchEnv, policy, scenario: Scenario, seed: int) -> dict:
    env.reset(seed=seed, options={"scenario": scenario})
    policy.reset(env)
    while env.agents:
        env.step(policy.act(env))
    return env.metrics()


def evaluate(policies: dict, conditions: dict | None = None, n_episodes: int = 50, grid_size: int = 20):
    """Returns {condition: {method: summary}}."""
    conditions = conditions or CONDITIONS
    env = GridSearchEnv(Scenario(grid_size=grid_size))
    results = {}
    for cname, make in conditions.items():
        results[cname] = {}
        for pname, policy in policies.items():
            rows = [run_episode(env, policy, make(s), seed=s) for s in range(n_episodes)]
            results[cname][pname] = summarise(rows, rows and make(0).max_steps)
    return results


def summarise(rows: list[dict], max_steps: int) -> dict:
    t90 = np.array([r["time_to_90"] if r["time_to_90"] is not None else max_steps for r in rows], float)
    return {
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
    lines = ["| Condition | Method | Time to 90% (steps) | Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cname, per in results.items():
        for pname, s in per.items():
            lines.append(
                f"| {cname} | {pname} | {s['time_to_90']:.0f} ± {s['time_to_90_std']:.0f} | "
                f"{s['reached_90']:.0%} | {s['found_frac']:.0%} | {s['coverage']:.0%} | "
                f"{s['repeat_frac']:.0%} | {s['collisions']:.1f} | {s.get('shield', 0):.1f} |"
            )
    return "\n".join(lines)
