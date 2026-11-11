"""Guide that suggests a target region for each drone during a run.

The RL policy gets the suggestion as an extra input (direction to the target
region) and, during training, a potential-based shaping bonus for moving
toward it that fades to zero, so the final policy optimizes the true reward
and can override the guide.

"heuristic": each drone in turn takes the free region with the best value
(expected unfound survivors x staleness) minus a distance cost.
"""

from __future__ import annotations

import numpy as np

from .env import COARSE, GridSearchEnv


def region_of(env: GridSearchEnv, x: int, y: int) -> tuple[int, int]:
    """(row, col) of the coarse region holding cell (x, y)."""
    return int(y * COARSE // env.grid_size), int(x * COARSE // env.grid_size)


def region_center(env: GridSearchEnv, row: int, col: int) -> tuple[int, int]:
    """Cell (x, y) at the centre of a coarse region."""
    edges = np.linspace(0, env.grid_size, COARSE + 1)
    cx = int((edges[col] + edges[col + 1] - 1) / 2)
    cy = int((edges[row] + edges[row + 1] - 1) / 2)
    return cx, cy


def region_stats(env: GridSearchEnv) -> tuple[np.ndarray, np.ndarray]:
    """Per region: expected unfound survivors (belief mass) and recently searched fraction."""
    belief = env._coarse(env.belief, np.sum).reshape(COARSE, COARSE) * max(len(env.found), 1)
    searched = env._coarse(env.freshness(), np.mean).reshape(COARSE, COARSE)
    return belief, searched


def live_drones(env: GridSearchEnv) -> list[str]:
    return [a for a in env.agents if env.alive.get(a)]


class HeuristicGuide:
    """Greedy: each drone in turn takes the free region with the best
    value (unfound mass x staleness) minus a distance cost."""

    name = "heuristic"

    def __init__(self, dist_cost: float = 0.15):
        self.dist_cost = dist_cost

    def plan_one(self, env: GridSearchEnv) -> dict[str, tuple[int, int]]:
        belief, searched = region_stats(env)
        value = belief * (1.0 - searched)
        value = value / (value.max() + 1e-9)
        taken: set[tuple[int, int]] = set()
        out = {}
        for a in live_drones(env):
            r0, c0 = region_of(env, *env.pos[a])
            best, best_s = (r0, c0), -1e9
            for r in range(COARSE):
                for c in range(COARSE):
                    if (r, c) in taken:
                        continue
                    s = value[r, c] - self.dist_cost * max(abs(r - r0), abs(c - c0))
                    if s > best_s:
                        best, best_s = (r, c), s
            taken.add(best)
            out[a] = best
        return out

    def plan(self, envs: list[GridSearchEnv]) -> list[dict[str, tuple[int, int]]]:
        return [self.plan_one(e) for e in envs]


def make_guide(spec: str | None, device: str = "cpu"):
    if not spec or spec == "none":
        return None
    if spec == "heuristic":
        return HeuristicGuide()
    raise ValueError(f"unknown guide {spec!r}")


def apply_plans(envs: list[GridSearchEnv], plans: list[dict[str, tuple[int, int]]]):
    for e, p in zip(envs, plans):
        e.set_goals({a: region_center(e, r, c) for a, (r, c) in p.items()})
