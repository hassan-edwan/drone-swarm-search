"""Fine-scale 3D flight over a real site: over, around and under obstacles.

The search policy works on a coarse grid (tens of metres per cell) and says
where each drone goes next and at what height. This module turns that into
the path a drone would really fly through the site's obstacles, measured by
lidar at 1 m:

- terrain (bare-earth elevation),
- trees: crown top and crown bottom per square metre, and individual trees
  (trunks under crown peaks),
- buildings: footprints (Microsoft) with lidar heights.

Flight modes:
- cruise: at the search height above the ground, following the terrain and
  always at least CLEAR_M above the highest crown or roof within
  SIDE_M of the track (climbing and descending at the drone's rates);
- under the canopy: about LOW_M above the ground, below the crowns, at least
  TRUNK_M from any trunk and SIDE_M from any building, slow. The drone gets
  there by descending through a real gap in the canopy (an opening at least
  GAP_R_M in radius) and leaves the same way.

Under-canopy flight is used to check alerts in dense forest (field env mode
"under"): the planner picks the gap that makes the whole trip (there in
cruise, down, under the crowns to the spot, back, up) quickest.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

LOW_M = 3.0  # flying height under the canopy
CLEAR_M = 5.0  # cruise clearance over crowns and roofs
SIDE_M = 3.0  # horizontal clearance from crowns (cruise) and buildings
TRUNK_M = 1.5  # horizontal clearance from trunks under the canopy
HEAD_M = 2.0  # clearance between the drone and the crown bottom above it
GAP_R_M = 3.0  # radius of an opening the drone can descend through
CRUISE_MPS, LOW_MPS, CLIMB_MPS, DESCEND_MPS = 10.0, 2.0, 4.0, 3.0
REACH_M = 15.0  # a look under the canopy sees about this far between trunks and undergrowth
STEP = 2  # planning grid: metres per cell for under-canopy routing


@dataclass
class Site3D:
    """Obstacles on a 1 m grid; coordinates in metres east/north of the site centre."""
    dtm: np.ndarray
    veg_top: np.ndarray  # crown top above ground (0: no tree)
    crown_base: np.ndarray  # crown bottom above ground (inf: no crown)
    bldg: np.ndarray  # bool
    bldg_h: np.ndarray  # roof height above ground
    east0: float
    north0: float
    trees: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # x, y, height, crown radius, crown base
    clutter: np.ndarray | None = None  # lidar vegetation returns per m2 in the flying band under the canopy

    @property
    def n(self):
        return self.dtm.shape[0]

    @classmethod
    def load(cls, fine_npz, buildings_json=None, shift=(0.0, 0.0)):
        d = np.load(fine_npz)
        bldg = d["bldg"] > 0.3
        top = np.where(bldg, 0.0, d["veg_top"])
        base = np.where(bldg, np.inf, d["crown_base"])
        bh = np.where(bldg, np.maximum(d["bldg_h"], 3.0), 0.0)
        # fill roof heights inside footprints where the lidar had no point above 2 m
        if bldg.any():
            lab, nl = ndimage.label(bldg)
            hs = ndimage.maximum(bh, lab, index=np.arange(1, nl + 1))
            bh = np.where(bldg, np.maximum(bh, np.asarray(hs)[np.maximum(lab - 1, 0)]), 0.0)
        s = cls(d["dtm"].astype(float), top.astype(float), base.astype(float), bldg, bh.astype(float),
                float(d["east0"]) + shift[0], float(d["north0"]) + shift[1])
        pts = Path(fine_npz).with_name("fine_points.npz")
        if pts.exists():
            s.clutter = s.band_counts(np.load(pts), shift)
        s.trees = s.find_trees()
        return s

    def band_counts(self, P, shift=(0.0, 0.0)):
        """Vegetation returns at the under-canopy flying height (from LOW_M - 2 to LOW_M + HEAD_M)."""
        xyz, hag, cls = P["xyz"], P["hag"], P["cls"]
        m = (hag > LOW_M - 2.0) & (hag < LOW_M + HEAD_M) & np.isin(cls, (1, 3, 4, 5))
        x = xyz[m, 0] + shift[0]
        y = xyz[m, 1] + shift[1]
        i = np.floor((self.north0 + self.n / 2) - y).astype(int)
        j = np.floor(x - (self.east0 - self.n / 2)).astype(int)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        c = np.zeros((self.n, self.n))
        np.add.at(c, (i[ok], j[ok]), 1.0)
        return c

    # ---------------------------------------------------------------- geometry
    def ij(self, x, y):
        i = int(np.clip((self.north0 + self.n / 2) - y, 0, self.n - 1))
        j = int(np.clip(x - (self.east0 - self.n / 2), 0, self.n - 1))
        return i, j

    def xy(self, i, j):
        return self.east0 - self.n / 2 + j + 0.5, self.north0 + self.n / 2 - i - 0.5

    def inside(self, x, y, margin=0.0):
        return (abs(x - self.east0) < self.n / 2 - margin) and (abs(y - self.north0) < self.n / 2 - margin)

    def ground(self, x, y):
        return float(self.dtm[self.ij(x, y)])

    def find_trees(self, min_h=5.0):
        """Individual trees: peaks of the smoothed crown-top surface."""
        top = ndimage.gaussian_filter(self.veg_top, 1.0)
        win = ndimage.maximum_filter(top, size=7)
        peaks = (top == win) & (top >= min_h) & ~self.bldg
        ii, jj = np.nonzero(peaks)
        out = []
        for i, j in zip(ii, jj):
            h = float(self.veg_top[i, j])
            r = float(np.clip(0.18 * h + 1.0, 2.0, 7.0))
            b = self.crown_base[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
            b = float(np.median(b[np.isfinite(b)])) if np.isfinite(b).any() else 0.5 * h
            x, y = self.xy(i, j)
            out.append((x, y, h, r, min(b, h - 1.0)))
        return np.array(out) if out else np.zeros((0, 5))

    # ---------------------------------------------------------------- free space
    def obstacle_top(self, side=SIDE_M):
        """Highest crown or roof within `side` metres, above the ground at each cell."""
        top = np.maximum(self.veg_top, self.bldg_h)
        return ndimage.maximum_filter(top, size=int(2 * side + 1))

    def low_free(self):
        """Cells (1 m) where a drone can fly LOW_M above the ground under the canopy."""
        trunk = np.zeros_like(self.bldg)
        for x, y, *_ in self.trees:
            trunk[self.ij(x, y)] = True
        near_trunk = ndimage.binary_dilation(trunk, iterations=int(math.ceil(TRUNK_M)))
        near_bldg = ndimage.binary_dilation(self.bldg, iterations=int(math.ceil(SIDE_M)))
        if self.clutter is not None:
            # branches or undergrowth at flying height: several lidar returns in a 3 x 3 m patch
            headroom = ndimage.uniform_filter(self.clutter, size=3) * 9 < 3
        else:
            headroom = (self.veg_top < 1.0) | (self.crown_base >= LOW_M + HEAD_M)
            headroom = ndimage.minimum_filter(headroom.astype(np.uint8), size=3).astype(bool)
        return headroom & ~near_trunk & ~near_bldg

    def gaps(self):
        """Cells where a drone can go straight down from cruise to LOW_M: open sky above within GAP_R_M."""
        open_sky = (self.veg_top < 1.0) & ~self.bldg
        r = int(math.ceil(GAP_R_M))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r * r
        return ndimage.binary_erosion(open_sky, structure=disk)


class CoarseTerrain:
    """Ground and highest obstacle from 10 m lidar layers (lidar.py coarse), for cruising
    outside the area mapped at full density. Coordinates as Site3D."""

    def __init__(self, lidar10_npz, side=SIDE_M):
        d = np.load(lidar10_npz)
        self.cell = float(d["cell"])
        self.size = float(d["size_m"])
        self.e0, self.n0 = float(d["east0"]), float(d["north0"])
        self.dtm = d["dtm"]
        # the 95th percentile crown height under-reads single tall trees: add a margin
        top = np.maximum(d["chm"] * 1.15, np.where(d["bldg"] > 0.1, np.maximum(d["bldg_h"], 6.0), 0.0))
        self.top = ndimage.maximum_filter(top, size=3)
        self.n = self.dtm.shape[0]

    def ij(self, x, y):
        i = int(np.clip(((self.n0 + self.size / 2) - y) / self.cell, 0, self.n - 1))
        j = int(np.clip((x - (self.e0 - self.size / 2)) / self.cell, 0, self.n - 1))
        return i, j

    def inside(self, x, y):
        return abs(x - self.e0) < self.size / 2 and abs(y - self.n0) < self.size / 2

    def ground(self, x, y):
        return float(self.dtm[self.ij(x, y)])


class Planner:
    def __init__(self, site: Site3D, outer: CoarseTerrain | None = None):
        self.s = site
        self.outer = outer
        self.top = site.obstacle_top()
        free1 = site.low_free()
        gap1 = site.gaps()
        n2 = site.n // STEP
        cut = n2 * STEP
        # a 2 m cell is free when all its 1 m cells are; a gap when any is (and it is free)
        self.free = free1[:cut, :cut].reshape(n2, STEP, n2, STEP).all((1, 3))
        self.gap = gap1[:cut, :cut].reshape(n2, STEP, n2, STEP).any((1, 3)) & self.free
        self.n2 = n2
        # connected open space under the crowns; tiny pockets are not reachable in practice
        lab, nl = ndimage.label(self.free, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        self.big = (lab > 0) & (sizes[lab] >= 100) & np.isin(lab, np.unique(lab[self.gap]))

    # ---------------------------------------------------------------- cruise
    def cruise_alt(self, x, y, agl):
        """Height (above sea level) of a cruise at `agl` above the ground over (x, y)."""
        s = self.s
        if not s.inside(x, y):
            if self.outer is not None and self.outer.inside(x, y):
                i, j = self.outer.ij(x, y)
                g = self.outer.dtm[i, j]
                return float(max(g + agl, g + self.outer.top[i, j] + CLEAR_M))
            return None
        i, j = s.ij(x, y)
        g = s.dtm[i, j]
        return float(max(g + agl, g + self.top[i, j] + CLEAR_M))

    def ground(self, x, y):
        if self.s.inside(x, y):
            return self.s.ground(x, y)
        return self.outer.ground(x, y) if self.outer is not None else 0.0

    def cruise(self, p0, p1, agl, z0=None, dt=0.5, agl0=None):
        """Straight cruise from p0 to p1 (x, y) at agl (changing from agl0 to agl along the way),
        following terrain and obstacles, with the drone's climb and descent rates; starts at
        height z0 when given. Returns [(t, x, y, z, mode)]."""
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(d / (CRUISE_MPS * dt)) + 1)
        xs = np.linspace(p0[0], p1[0], n)
        ys = np.linspace(p0[1], p1[1], n)
        aglv = np.linspace(agl if agl0 is None else agl0, agl, n)
        want = np.array([self.cruise_alt(x, y, h) or np.nan for x, y, h in zip(xs, ys, aglv)])
        want = np.where(np.isnan(want), np.nanmax(want) if np.isfinite(want).any() else 0.0, want)
        seg = d / (n - 1) if n > 1 else 0.0
        up, down = CLIMB_MPS / CRUISE_MPS * seg, DESCEND_MPS / CRUISE_MPS * seg
        # the profile must be at or above `want` everywhere and change at most up/down per sample:
        # climb early enough for what is ahead, then descend no faster than allowed
        z = want.copy()
        for k in range(n - 2, -1, -1):
            z[k] = max(z[k], z[k + 1] - up)
        for k in range(1, n):
            z[k] = max(z[k], z[k - 1] - down)
        if z0 is not None:
            # continue from the current height: no faster than the climb and descent rates,
            # except where an obstacle ahead needs it (then the profile above wins)
            for k in range(n):
                z[k] = max(z[k], z0 - down * k) if z0 > z[k] else max(min(z[k], z0 + up * k), want[k])
        t = np.arange(n) * seg / CRUISE_MPS
        return [(float(t[k]), float(xs[k]), float(ys[k]), float(z[k]), "cruise") for k in range(n)]

    # ---------------------------------------------------------------- under the canopy
    def _c2(self, x, y):
        i, j = self.s.ij(x, y)
        return min(i // STEP, self.n2 - 1), min(j // STEP, self.n2 - 1)

    def _xy2(self, i, j):
        return self.s.xy(i * STEP + STEP / 2 - 0.5, j * STEP + STEP / 2 - 0.5)

    def low_field(self, target, max_m=400.0):
        """Under-canopy travel distance (m) from every free 2 m cell to the target."""
        ti, tj = self._c2(*target)
        if not self.big[ti, tj]:  # the spot may be in undergrowth: start from the nearest open space with a way out
            fi, fj = np.nonzero(self.big)
            k = int(np.argmin((fi - ti) ** 2 + (fj - tj) ** 2))
            ti, tj = int(fi[k]), int(fj[k])
        D = np.full((self.n2, self.n2), np.inf)
        D[ti, tj] = 0.0
        pq = [(0.0, ti, tj)]
        lim = max_m
        moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414),
                 (1, 1, 1.414)]
        while pq:
            d, i, j = heapq.heappop(pq)
            if d > D[i, j] or d > lim:
                continue
            for di, dj, c in moves:
                a, b = i + di, j + dj
                if 0 <= a < self.n2 and 0 <= b < self.n2 and self.free[a, b]:
                    nd = d + c * STEP
                    if nd < D[a, b]:
                        D[a, b] = nd
                        heapq.heappush(pq, (nd, a, b))
        return D, (ti, tj)

    def _descend_path(self, D, start):
        i, j = start
        path = [(i, j)]
        while D[i, j] > 0:
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < self.n2 and 0 <= b < self.n2 and D[a, b] < D[i, j]:
                        if best is None or D[a, b] < D[best]:
                            best = (a, b)
            if best is None:
                break
            i, j = best
            path.append((i, j))
        return path

    def under_check(self, start, target, agl, z0=None, look_s=8.0, dt=0.5):
        """Cruise from start to the best gap, down, under the crowns to target, look, back, up.
        Returns (trajectory [(t, x, y, z, mode)], info) or (None, reason)."""
        if not (self.s.inside(*target, margin=5) and self.s.inside(*start)):
            return None, "outside the detailed area"
        D, t2 = self.low_field(target)
        gi, gj = np.nonzero(self.gap & np.isfinite(D))
        if not len(gi):
            return None, "no gap in the canopy connects to the spot"
        gx = np.array([self._xy2(i, j)[0] for i, j in zip(gi, gj)])
        gy = np.array([self._xy2(i, j)[1] for i, j in zip(gi, gj)])
        cost = (np.hypot(gx - start[0], gy - start[1]) / CRUISE_MPS + 2 * D[gi, gj] / LOW_MPS)
        k = int(np.argmin(cost))
        g = (gi[k], gj[k])
        gxy = (float(gx[k]), float(gy[k]))
        traj = self.cruise(start, gxy, agl, z0=z0, dt=dt)
        t = traj[-1][0]
        ground = self.s.ground(*gxy)
        z_low = ground + LOW_M
        z_top = traj[-1][3]
        # down through the gap
        for z in np.arange(z_top, z_low, -DESCEND_MPS * dt)[1:]:
            t += dt
            traj.append((t, gxy[0], gxy[1], float(z), "descend"))
        path = self._descend_path(D, g)
        pts = [self._xy2(i, j) for i, j in path]
        out_leg = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            m = max(1, int(seg / (LOW_MPS * dt)))
            for u in np.linspace(0, 1, m + 1)[1:]:
                x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                out_leg.append((x, y, self.s.ground(x, y) + LOW_M))
        for x, y, z in out_leg:
            t += dt
            traj.append((t, x, y, z, "under"))
        hx, hy, hz = out_leg[-1] if out_leg else (gxy[0], gxy[1], z_low)
        for _ in range(int(look_s / dt)):
            t += dt
            traj.append((t, hx, hy, hz, "look"))
        for x, y, z in reversed(out_leg[:-1]):
            t += dt
            traj.append((t, x, y, z, "under"))
        t += dt
        traj.append((t, gxy[0], gxy[1], z_low, "under"))
        for z in np.arange(z_low, z_top, CLIMB_MPS * dt)[1:]:
            t += dt
            traj.append((t, gxy[0], gxy[1], float(z), "climb"))
        under_m = 2 * float(D[g])
        reach = math.hypot(hx - target[0], hy - target[1])
        return traj, dict(gap=gxy, under_m=under_m, total_s=t, look=(hx, hy, hz), reach_m=reach,
                          close_enough=reach <= REACH_M)


def clearance(site: Site3D, traj, mode_filter=None):
    """Smallest distances to obstacles along a trajectory: above ground, below crowns
    (under-canopy legs), from trunks, from buildings, and above crowns and roofs (cruise)."""
    trunks = site.trees[:, :2] if len(site.trees) else np.zeros((0, 2))
    out = dict(ground=np.inf, under_crown=np.inf, trunk=np.inf, building=np.inf, over_top=np.inf)
    bpts = np.argwhere(site.bldg)
    for t, x, y, z, mode in traj:
        if mode_filter and mode not in mode_filter:
            continue
        if not site.inside(x, y):
            continue
        i, j = site.ij(x, y)
        h = z - site.dtm[i, j]
        out["ground"] = min(out["ground"], h)
        if mode in ("under", "look"):
            if site.veg_top[i, j] > h:
                out["under_crown"] = min(out["under_crown"], site.crown_base[i, j] - h)
            if len(trunks):
                out["trunk"] = min(out["trunk"], float(np.min(np.hypot(trunks[:, 0] - x, trunks[:, 1] - y))))
            if len(bpts):
                near = bpts[(np.abs(bpts[:, 0] - i) < 30) & (np.abs(bpts[:, 1] - j) < 30)]
                if len(near):
                    out["building"] = min(out["building"], float(np.min(np.hypot(near[:, 0] - i, near[:, 1] - j))))
        elif mode == "cruise":
            top = max(site.veg_top[i, j], site.bldg_h[i, j])
            out["over_top"] = min(out["over_top"], h - top)
    return out


def save_traj(path, trajs: dict, info: dict):
    Path(path).write_text(json.dumps(dict(trajs={a: [list(p) for p in tr] for a, tr in trajs.items()}, info=info)))
