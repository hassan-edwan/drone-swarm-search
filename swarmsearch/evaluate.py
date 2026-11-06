"""Run every method on the same held-out scenarios and summarise the metrics."""

from __future__ import annotations

import numpy as np

from .env import GridSearchEnv
from .scenario import MovementProfile, Scenario, ScenarioSampler

# Named test conditions. Each is a function seed -> Scenario so that every
# method sees exactly the same missions.
def _sampled(**kw):
    sampler = ScenarioSampler(**kw)
    return lambda seed: sampler.sample(np.random.default_rng(10_000 + seed))


CONDITIONS = {
    "default": lambda seed: Scenario(),
    "no_zones": lambda seed: Scenario(n_zones=0),
    "still": lambda seed: Scenario(movement=MovementProfile("still", 0.0)),
    "drift_known": lambda seed: Scenario(movement=MovementProfile("drift", 0.4, (1, 0)), profile_known=True),
    "drift_default": lambda seed: Scenario(movement=MovementProfile("drift", 0.4, (1, 0)), profile_known=False),
    "drone_lost": lambda seed: Scenario(drone_failure=True),
    "6_drones": lambda seed: Scenario(n_drones=6),
    "random_mix": _sampled(),
}


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
    }


def to_markdown(results: dict) -> str:
    lines = ["| Condition | Method | Time to 90% (steps) | Reached 90% | Found | Coverage | Repeats | Collisions |",
             "|---|---|---|---|---|---|---|---|"]
    for cname, per in results.items():
        for pname, s in per.items():
            lines.append(
                f"| {cname} | {pname} | {s['time_to_90']:.0f} ± {s['time_to_90_std']:.0f} | "
                f"{s['reached_90']:.0%} | {s['found_frac']:.0%} | {s['coverage']:.0%} | "
                f"{s['repeat_frac']:.0%} | {s['collisions']:.1f} |"
            )
    return "\n".join(lines)
