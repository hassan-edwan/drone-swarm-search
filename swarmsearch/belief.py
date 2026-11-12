"""Survivor movement models and online inference of the movement type.

A movement profile is turned into per-cell move probabilities over the 9
moves (8 neighbours + stay). The same model moves the true survivors and the
belief maps, so a belief built from the true profile is the exact expected
survivor distribution.

When the operator gave no profile, the team keeps a bank of hypotheses
(still, wander, drift in each direction, head to one of several safe points,
at a few speeds), one belief map per hypothesis. Searching a cell and finding
nothing removes mass from every map in proportion to what it predicted
there, and finding a survivor reweights each hypothesis by how much mass it
had in that cell. The shared belief the policy sees is the weighted mixture,
so it moves from "could be anything" toward the true movement as evidence
comes in.
"""

from __future__ import annotations

import numpy as np

from .scenario import MovementProfile

# Must match env.MOVES: 0-7 neighbours, 8 = stay.
MOVES = np.array(
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 0)],
    dtype=np.int64,
)
STAY = 8
DIRECTED = 0.8  # drift / to_safety: share of moves along the main direction

HYP_SPEEDS = (0.5, 1.0, 1.5)
HYP_DRIFT_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
HYP_TARGET_STEPS = 4  # safe-point hypotheses on a 4x4 lattice
HYP_DIRECTED = DIRECTED  # hypotheses' directedness (lower = tolerant of a wrong direction or target)


def move_weights(prof: MovementProfile, free: np.ndarray, directed: float = DIRECTED) -> np.ndarray:
    """(9, g, g): probability that a survivor in each cell takes each move.
    Moves into walls or obstacles become staying put."""
    g = free.shape[0]
    W = np.zeros((9, g, g))
    p = prof.move_prob
    if p == 0.0:
        W[STAY] = 1.0
        return W
    yy, xx = np.mgrid[0:g, 0:g]
    base = 1.0 / 8 if prof.kind not in ("drift", "to_safety") else (1 - directed) / 8
    W[:8] = p * base
    W[STAY] = 1.0 - p
    if prof.kind == "drift":
        m = _move_index(*prof.drift_dir)
        W[m] += p * directed
    elif prof.kind == "to_safety":
        tx, ty = prof.target
        sx, sy = np.sign(tx - xx), np.sign(ty - yy)
        for m, (dx, dy) in enumerate(MOVES[:8]):
            W[m] += p * directed * ((sx == dx) & (sy == dy))
        at = (xx == tx) & (yy == ty)
        W[:, at] = 0.0
        W[STAY, at] = 1.0  # arrived: stays at the safe point
    # Fold blocked moves into staying put; obstacle cells never move.
    for m, (dx, dy) in enumerate(MOVES[:8]):
        nx, ny = xx + dx, yy + dy
        ok = (nx >= 0) & (nx < g) & (ny >= 0) & (ny < g)
        ok[ok] = free[ny[ok], nx[ok]]
        W[STAY] += np.where(ok, 0.0, W[m])
        W[m] = np.where(ok, W[m], 0.0)
    W[:, ~free] = 0.0
    W[STAY, ~free] = 1.0
    return W


def _move_index(dx: int, dy: int) -> int:
    return int(np.flatnonzero((MOVES[:, 0] == dx) & (MOVES[:, 1] == dy))[0])


def propagate(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    """One step of movement for K belief maps. B: (K, g, g), W: (K, 9, g, g)."""
    g = B.shape[-1]
    out = W[:, STAY] * B
    for m, (dx, dy) in enumerate(MOVES[:8]):
        src = W[:, m] * B
        out[:, max(0, dy):g + min(0, dy), max(0, dx):g + min(0, dx)] += \
            src[:, max(0, -dy):g - max(0, dy), max(0, -dx):g - max(0, dx)]
    return out


def hypothesis_bank(grid_size: int) -> tuple[list[MovementProfile], np.ndarray]:
    """Candidate movement profiles and their prior (each movement type equally likely)."""
    groups = [
        [MovementProfile("still", 0.0)],
        [MovementProfile("wander", v) for v in HYP_SPEEDS],
        [MovementProfile("drift", v, d) for d in HYP_DRIFT_DIRS for v in HYP_SPEEDS],
    ]
    targets = sorted({int(round(v)) for v in np.linspace(1, grid_size - 2, HYP_TARGET_STEPS)})
    groups.append([MovementProfile("to_safety", v, target=(tx, ty))
                   for ty in targets for tx in targets for v in HYP_SPEEDS])
    profiles, prior = [], []
    for grp in groups:
        profiles += grp
        prior += [1.0 / (len(groups) * len(grp))] * len(grp)
    return profiles, np.array(prior)
