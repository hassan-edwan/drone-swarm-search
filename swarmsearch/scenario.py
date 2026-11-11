"""Scenario variables for one search mission.

Every episode is one scenario. Training samples scenarios across the ranges
below so a single policy has to adapt; evaluation can pin any variable.
The "profile" is what an operator could tell the drones about survivor
movement before the mission. If the operator sets nothing, the drones get
DEFAULT_PROFILE instead of the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# still: trapped or sheltering in place.
# wander: lost or disoriented, random walk.
# drift: carried one way (flood water) or all heading one direction.
# to_safety: walking toward a known safe point (road, shelter, high ground).
MOVEMENTS = ("still", "wander", "drift", "to_safety")
MAX_SPEED = 1.5  # m/s; lost people walk ~0.5-1.3 m/s, flood drift can be faster

# Grid cells are 10 m and one env step is ~1 s, so a survivor moving at
# v m/s changes cell with probability v / 10 per step.
CELL_METERS = 10.0
STEP_SECONDS = 1.0


@dataclass(frozen=True)
class MovementProfile:
    """How survivors move (true) or are expected to move (profile)."""

    kind: str = "wander"  # one of MOVEMENTS
    speed_mps: float = 0.5
    drift_dir: tuple[int, int] = (0, 0)  # (dx, dy) unit step, used by "drift"
    target: tuple[int, int] = (0, 0)  # safe point (x, y) cell, used by "to_safety"

    @property
    def move_prob(self) -> float:
        if self.kind == "still":
            return 0.0
        return min(1.0, self.speed_mps * STEP_SECONDS / CELL_METERS)

    def as_vector(self, known: bool, grid_size: int = 20) -> np.ndarray:
        one_hot = [float(self.kind == k) for k in MOVEMENTS]
        tx, ty = (self.target if self.kind == "to_safety" else (0, 0))
        return np.array(
            one_hot
            + [self.speed_mps / MAX_SPEED, float(self.drift_dir[0]), float(self.drift_dir[1]),
               tx / (grid_size - 1), ty / (grid_size - 1), float(known)],
            dtype=np.float32,
        )


# What drones assume when the operator says nothing: people wandering at a slow walk.
DEFAULT_PROFILE = MovementProfile("wander", 0.5)
PROFILE_DIM = len(MOVEMENTS) + 6


@dataclass(frozen=True)
class Scenario:
    grid_size: int = 20
    n_drones: int = 3
    n_survivors: int = 10
    movement: MovementProfile = field(default_factory=lambda: DEFAULT_PROFILE)
    profile_known: bool = False  # operator told the drones the true profile
    n_zones: int = 3
    zone_accuracy: float = 0.8  # fraction of survivors placed inside zones
    obstacle_frac: float = 0.0  # M1: 0; M2: up to 0.3
    detection_prob: float = 0.8
    drone_failure: bool = False  # one drone is lost at a random step
    max_steps: int = 200

    def observed_profile(self) -> MovementProfile:
        return self.movement if self.profile_known else DEFAULT_PROFILE


DRIFT_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class ScenarioSampler:
    """Draws a random scenario from the ranges in the proposal."""

    grid_size: int = 20
    drones: tuple[int, int] = (2, 8)
    survivors: tuple[int, int] = (1, 20)
    zones: tuple[int, int] = (1, 5)  # when there are zones at all
    p_no_zones: float = 0.6  # most searches have no likely spots
    zone_accuracy: tuple[float, float] = (0.5, 1.0)
    obstacle_frac: tuple[float, float] = (0.0, 0.0)
    detection_prob: tuple[float, float] = (0.6, 1.0)
    speed_mps: tuple[float, float] = (0.3, MAX_SPEED)
    p_profile_known: float = 0.5
    p_drone_failure: float = 0.25
    max_steps: int = 200

    def sample(self, rng: np.random.Generator) -> Scenario:
        kind = MOVEMENTS[rng.integers(len(MOVEMENTS))]
        speed = 0.0 if kind == "still" else float(rng.uniform(*self.speed_mps))
        drift = DRIFT_DIRS[rng.integers(len(DRIFT_DIRS))] if kind == "drift" else (0, 0)
        target = tuple(int(v) for v in rng.integers(0, self.grid_size, size=2)) if kind == "to_safety" else (0, 0)
        return Scenario(
            grid_size=self.grid_size,
            n_drones=int(rng.integers(self.drones[0], self.drones[1] + 1)),
            n_survivors=int(rng.integers(self.survivors[0], self.survivors[1] + 1)),
            movement=MovementProfile(kind, speed, drift, target),
            profile_known=bool(rng.random() < self.p_profile_known),
            n_zones=0 if rng.random() < self.p_no_zones else int(rng.integers(self.zones[0], self.zones[1] + 1)),
            zone_accuracy=float(rng.uniform(*self.zone_accuracy)),
            obstacle_frac=float(rng.uniform(*self.obstacle_frac)),
            detection_prob=float(rng.uniform(*self.detection_prob)),
            drone_failure=bool(rng.random() < self.p_drone_failure),
            max_steps=self.max_steps,
        )


def with_overrides(s: Scenario, **kw) -> Scenario:
    return replace(s, **kw)
