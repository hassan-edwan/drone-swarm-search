"""Sensing for the field environment (register S1-S6, E4).

Detection of a person in a footprint cell by a drone is

    p = G[alt, drone cell, offset] * C[target cell] * motion * person * sensor_scale

- G (geometry): inside the footprint, line of sight clear over the terrain
  and buildings (the drone flies at a set height above the ground under it),
  smaller with height and off-centre.
- C (conditions): dual camera, the better of an ordinary camera (needs
  light; blocked by canopy, smoke, rain) and thermal (contrast falls with
  warm air and midday sun; sees through canopy better).
- person: clothing (ordinary camera), warmth (thermal), posture.
- Correlated looks (S2): from each viewing direction and height a person is
  hidden or not, drawn once (more likely under canopy or rubble). A repeat
  from the same direction mostly repeats a miss; a new direction is a new
  chance. The drones' map models the same thing with look counts.

Detections go to an operator review queue (S5): each item takes a reviewer
`review_s`; a reviewer misses true detections more often when tired or
behind; detections from high up are often not clear enough to confirm and
need a closer look.
"""

from __future__ import annotations

import numpy as np

from .. import camera as cam

R = cam.R
SIDE = cam.SIDE
PERSON_M = 1.5
OFF_NADIR = 0.15
N_CLASS = 9  # viewing directions: 8 sides + straight down
CONFIRM = (0.95, 0.75, 0.35)  # chance a true detection can be confirmed from video at each height
FA_DISMISS = 0.85  # reviewer recognises a false alarm
BLUR = 0.7  # cruise speed: motion blur (D3)
SPRINT_BLUR = 0.15
LIGHT = {"day": 1.0, "dusk": 0.45, "night": 0.04}


def geo_table(surface: np.ndarray, elev: np.ndarray, target_ok: np.ndarray, alt_m, foot_r, alt_detect,
              n_samples: int = 12) -> np.ndarray:
    """G: (3, g, g, SIDE, SIDE). Line of sight from a drone at alt_m above its
    cell's ground to a person on the target cell's ground, over `surface`."""
    g = surface.shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    ts = (np.arange(n_samples) + 0.5) / n_samples
    G = np.zeros((3, g, g, SIDE, SIDE))
    for oy in range(SIDE):
        for ox in range(SIDE):
            dy, dx = oy - R, ox - R
            ring = max(abs(dx), abs(dy))
            tx, ty = xx + dx, yy + dy
            inside = (tx >= 0) & (tx < g) & (ty >= 0) & (ty < g)
            txc, tyc = np.clip(tx, 0, g - 1), np.clip(ty, 0, g - 1)
            ok = inside & target_ok[tyc, txc]
            t_h = elev[tyc, txc] + PERSON_M
            if dx or dy:
                sx = np.floor(xx + 0.5 + dx * ts[:, None, None]).astype(int)
                sy = np.floor(yy + 0.5 + dy * ts[:, None, None]).astype(int)
                s_in = (sx >= 0) & (sx < g) & (sy >= 0) & (sy < g)
                mid = s_in & ~((sx == tx) & (sy == ty)) & ~((sx == xx) & (sy == yy))
                h = np.where(mid, surface[np.clip(sy, 0, g - 1), np.clip(sx, 0, g - 1)], -np.inf)
            for k in range(3):
                if ring > foot_r[k]:
                    continue
                vis = ok
                if dx or dy:
                    d_h = elev + alt_m[k]
                    line = d_h[None] * (1 - ts[:, None, None]) + t_h[None] * ts[:, None, None]
                    vis = ok & ~(h > line).any(axis=0)
                G[k, :, :, oy, ox] = np.where(vis, alt_detect[k] * (1 - OFF_NADIR * ring), 0.0)
    return G


def thermal_contrast(air_c: float, hour: float, rain: bool) -> float:
    """Body (~33 C skin) against the background: weak in warm air and under midday sun."""
    c = np.clip((32.0 - air_c) / 20.0, 0.15, 1.0)
    if 10 <= hour % 24 < 17 and not rain:
        c *= 0.6  # sun-heated ground
    return float(c * (0.6 if rain else 1.0))


def light_level(hour: float) -> float:
    h = hour % 24
    if 7 <= h < 18:
        return LIGHT["day"]
    if 6 <= h < 7 or 18 <= h < 20:
        return LIGHT["dusk"]
    return LIGHT["night"]


def conditions(canopy, rubble, smoke, water, hour: float, air_c: float, rain: bool) -> tuple[np.ndarray, np.ndarray]:
    """(ordinary camera, thermal) detection factors per cell."""
    rgb = light_level(hour) * (1 - 0.85 * canopy) * np.where(rubble, 0.6, 1.0) * (1 - smoke)
    rgb = rgb * np.where(water, 0.7, 1.0) * (0.7 if rain else 1.0)
    # thermal under canopy fitted to Hambrecht et al. 2019 (register source 23)
    th = thermal_contrast(air_c, hour, rain) * (1 - 0.25 * canopy) * np.where(rubble, 0.8, 1.0) * (1 - 0.3 * smoke)
    return rgb, th


def occlusion(canopy, rubble) -> np.ndarray:
    """Chance a person is hidden from a given direction (drawn once per direction)."""
    return np.clip(0.05 + 0.55 * canopy + 0.3 * rubble, 0.0, 0.9)


def view_class(alt: int, dx: int, dy: int) -> int:
    """Direction from the person to the drone (dx, dy is target minus drone)."""
    return alt * N_CLASS + (int(np.sign(-dx)) + 1) * 3 + (int(np.sign(-dy)) + 1)


def false_alarm_rate(family: str, canopy, rubble, hour: float, base: float) -> np.ndarray:
    """Per visible cell and look (S4): animals in forest (thermal), sun-heated
    rocks in the open at midday, debris clutter in rubble."""
    sun = 1.0 if 10 <= hour % 24 < 17 else 0.2
    rate = base * (1.0 + 2.0 * canopy + 1.5 * sun * (1 - canopy) * (family == "wilderness") + 2.0 * rubble)
    return rate


CLUE_CONFIRM = 0.8  # an operator accepts a real clue (assumed; see evidence.py)


class OperatorQueue:
    """Reviewers working through detections in arrival order (S5)."""

    def __init__(self, n_ops: int, review_steps: int, base_miss: float, rng):
        self.items: list[dict] = []
        self.busy = [0] * n_ops  # steps left on the current item
        self.current: list[dict | None] = [None] * n_ops
        self.review_steps = max(1, review_steps)
        self.base_miss = base_miss
        self.rng = rng
        self.reviewed = 0
        self.missed = 0

    def add(self, item: dict):
        self.items.append(item)

    def __len__(self):
        return len(self.items) + sum(c is not None for c in self.current)

    def step(self, hours_on_task: float) -> list[tuple[dict, str]]:
        """Advance one step; returns finished reviews as (item, verdict) with
        verdict in confirm / closer_look / dismiss."""
        out = []
        for k in range(len(self.busy)):
            if self.current[k] is None and self.items:
                self.current[k] = self.items.pop(0)
                self.busy[k] = self.review_steps
            if self.current[k] is None:
                continue
            self.busy[k] -= 1
            if self.busy[k] > 0:
                continue
            item = self.current[k]
            self.current[k] = None
            self.reviewed += 1
            miss = min(0.8, self.base_miss + 0.05 * hours_on_task + 0.01 * max(0, len(self.items) - 3))
            if item["true"] is not None:
                if self.rng.random() < miss:
                    self.missed += 1
                    out.append((item, "dismiss"))
                elif self.rng.random() < CONFIRM[item["alt"]]:
                    out.append((item, "confirm"))
                else:
                    out.append((item, "closer_look"))
            elif item.get("clue") is not None:  # K7: a clue, judged from the image
                out.append((item, "clue" if self.rng.random() < CLUE_CONFIRM else "dismiss"))
            else:  # S8: walkers look like the subject; tracked searchers and animals are easy to dismiss
                dismiss = item.get("dismiss", FA_DISMISS)
                out.append((item, "dismiss" if self.rng.random() < dismiss else "closer_look"))
        return out
