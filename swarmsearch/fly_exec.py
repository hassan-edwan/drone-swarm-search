"""Fly a route planned on the MAP through the REAL world, with onboard sensing (detect and avoid).

The route comes from swarmsearch.flight3d (planned on world3d's map). The drone
flies it in the real world (world3d.make_truth), which has what the map
lacks. Every 0.1 s:

- where it is: the planned point, minus its navigation error (it follows its
  own estimate), plus wind drift. Navigation: "gps" (fine in the open; under
  the crowns 6 satellites and multipath: an offset wandering up to 6 m) or
  "vio" (camera-based visual-inertial odometry under the crowns: about 1%
  drift of the distance flown there).
- what it sees: a forward depth camera (90 deg wide, 70 deg tall). Each real
  voxel inside the cone is detected with a chance per frame that depends on
  what it is and how far: trunks, buildings, poles and structures out to
  15 m in the open / 8 m under the crowns (dark), foliage and branches to 6 m
  with 0.6 a frame, wires only within 4 m (2.5 m under the crowns) with 0.25
  a frame (stereo and time-of-flight cameras barely see thin wires).
  Detections join what the drone knows.
- whether the route ahead is still clear of what it knows (within its stopping
  distance plus 3 m): if not, it stops and re-plans. Outbound under the
  crowns: a new route to the spot through the free space it now knows
  (VoxPlanner with the detections added); if there is none, it gives up and
  returns. Returning: it retraces its own track. In the open: it climbs.
- whether it touches anything real: then it has collided (and what with).

Also: vehicles on roads where a low route crosses one, and birds of prey
above the trees (rates are assumptions, kept small and labelled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import flight3d as f3
from . import world3d as w3

SENSOR = dict(fov_h=math.radians(90), fov_v=math.radians(70))
SOLID = w3.TRUNK | w3.BLDG | w3.POLE | w3.STRUCT
DETECT = {  # kind: (range open, range under the crowns, chance per frame)
    "solid": (15.0, 8.0, 0.9),
    "foliage": (6.0, 4.0, 0.6),
    "wire": (4.0, 2.5, 0.25),
}
BRAKE = 4.0  # m/s2
LATENCY = 0.3  # s from seeing to braking
BIRD_PER_H = 0.02  # raptor attacks per flight hour above the trees (assumed; reported anecdotally)
TRAFFIC_GAP_S = 40.0  # mean time between vehicles on a rural road (assumed)


@dataclass
class Result:
    outcome: str = "ok"  # ok / collided / aborted / bird
    hit_kind: str = ""
    time_s: float = 0.0
    planned_s: float = 0.0
    replans: int = 0
    detected: dict = field(default_factory=dict)
    min_clear: dict = field(default_factory=dict)
    reached: bool = False
    trace: list = field(default_factory=list)
    nav_err_max: float = 0.0
    events: list = field(default_factory=list)


def _kind_class(kind):
    if kind & w3.WIRE:
        return "wire"
    if kind & SOLID:
        return "solid"
    return "foliage"


class Executor:
    def __init__(self, W_map: w3.World3D, W_true: w3.World3D, planner_factory, wind=None, roads=None,
                 drone="sar", nav="vio", sensing=True, seed=0, wind_ij=None):
        self.M, self.T = W_map, W_true
        self.make_planner = planner_factory
        self.wind = wind
        self.roads = roads or []
        self.drone, self.nav, self.sensing = drone, nav, sensing
        self.rng = np.random.default_rng(seed)
        self.r_d = f3.DRONES[drone]["span"] / 2
        self.top_true = W_true.top()
        self.wind_index = wind_ij
        self.all_round = False  # True: sensing all around (several cameras), not only forward
        self.local = []  # recently seen obstacles in the drone's local frame (x, y, height above ground)

    # --------------------------------------------------------------- helpers
    def _under_crowns(self, x, y, h):
        i, j = self.T.ij(x, y)
        return self.top_true[i, j] > h + 1.0

    def _wind_at(self, x, y, h):
        if self.wind is None or self.wind_index is None:
            return 0.0
        i, j = self.wind_index(x, y)
        return self.wind.gust(i, j, h, self.rng)

    def _sense(self, pos, heading, known, under, offset=(0.0, 0.0)):
        """Detect real voxels in the camera cone and add them to `known`, in the drone's estimate frame
        (true position + its navigation error: it measures them relative to itself)."""
        x, y, h = pos
        rmax = max(DETECT["solid"][0], DETECT["foliage"][0], DETECT["wire"][0])
        hits = self.T.kinds_near(x, y, h, rmax)
        if not hits:
            return {}
        i0, j0 = self.T.ij(x, y)
        new = {}
        hx, hy, hz = heading
        hn = math.hypot(hx, hy)
        R = int(math.ceil(rmax)) + 1
        k0, k1 = max(0, int(h - R)), min(w3.NZ, int(h + R) + 1)
        i_lo, j_lo = max(0, i0 - R), max(0, j0 - R)
        blk = self.T.occ[i_lo:min(self.T.n, i0 + R + 1), j_lo:min(self.T.n, j0 + R + 1), k0:k1]
        ii, jj, kk = np.nonzero(blk)
        if not len(ii):
            return {}
        cx = self.T.east0 - self.T.n / 2 + (j_lo + jj) + 0.5
        cy = self.T.north0 + self.T.n / 2 - (i_lo + ii) - 0.5
        cz = k0 + kk + 0.5
        dx, dy, dz = cx - x, cy - y, cz - h
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if self.all_round:
            in_cone = np.ones(len(dist), bool)
        elif hn > 1e-6:
            ang_h = np.abs(np.arctan2(dx * hy - dy * hx, dx * hx + dy * hy))
            ang_v = np.abs(np.arctan2(dz, np.hypot(dx, dy)))
            in_cone = (ang_h <= SENSOR["fov_h"] / 2) & (ang_v <= SENSOR["fov_v"] / 2)
        else:  # moving vertically: a down/up-looking view (we treat it as a cone around the vertical)
            in_cone = np.abs(dz) >= 0.5 * np.hypot(dx, dy)
        kinds = blk[ii, jj, kk]
        for kc in ("solid", "foliage", "wire"):
            r_open, r_under, p = DETECT[kc]
            rng_ = r_under if under else r_open
            sel = in_cone & (dist <= rng_)
            if kc == "wire":
                sel &= (kinds & w3.WIRE) > 0
            elif kc == "solid":
                sel &= (kinds & SOLID) > 0
            else:
                sel &= (kinds & w3.VEG) > 0
            if not sel.any():
                continue
            got = sel & (self.rng.random(len(sel)) < p)
            if got.any():  # the local obstacle layer: relative to the drone, so GPS jumps do not move it
                self.local += list(zip(cx[got], cy[got], cz[got]))
            oi, oj = -int(round(offset[1])), int(round(offset[0]))  # +east = +column, +north = -row
            for a_, b_, c_, kd in zip((i_lo + ii)[got], (j_lo + jj)[got], (k0 + kk)[got], kinds[got]):
                key = (int(np.clip(a_ + oi, 0, self.M.n - 1)), int(np.clip(b_ + oj, 0, self.M.n - 1)), int(c_))
                if key not in known:
                    known[key] = int(kd)
                    new[key] = kc
        return new

    def _conflict(self, path_pts, known, r):
        """First index of path_pts (x, y, h) within r of a known (detected) obstacle voxel."""
        if not known:
            return None
        K = np.array(list(known.keys()), float)
        kx = self.M.east0 - self.M.n / 2 + K[:, 1] + 0.5
        ky = self.M.north0 + self.M.n / 2 - K[:, 0] - 0.5
        kz = K[:, 2] + 0.5
        for idx, (x, y, h) in enumerate(path_pts):
            d = np.sqrt((kx - x) ** 2 + (ky - y) ** 2 + (kz - h) ** 2) - 0.5
            if (d < r).any():
                return idx
        return None

    def _touch(self, x, y, h):
        hits = self.T.kinds_near(x, y, h, self.r_d)
        if hits:
            k = int(hits[int(np.argmin([d for _, d in hits]))][0])
            return w3.KIND_NAMES.get(k & -k, "obstacle")
        if h - f3.DRONES[self.drone]["height"] / 2 < 0:
            return "ground"
        return None

    # --------------------------------------------------------------- flying
    def fly_check(self, start, target, agl, dt=0.1, look_s=8.0):
        """Plan an under-canopy check on the map and fly it in the real world."""
        res = Result()
        P = self.make_planner(None)
        tr, info = P.under_check(start, target, agl)
        if tr is None:
            res.outcome = "no route"
            return res
        res.planned_s = tr[-1][0]
        known = {}
        nav = np.zeros(2)
        wind_err = np.zeros(3)
        pts = []
        tt = np.array([q[0] for q in tr])
        for u in np.arange(0.0, tt[-1] + 1e-9, dt):
            kk_ = min(int(np.searchsorted(tt, u, side="right")) - 1, len(tr) - 2)
            a_, b_ = tr[kk_], tr[kk_ + 1]
            f = (u - a_[0]) / max(b_[0] - a_[0], 1e-9)
            x_, y_, z_ = (a_[1] + f * (b_[1] - a_[1]), a_[2] + f * (b_[2] - a_[2]), a_[3] + f * (b_[3] - a_[3]))
            pts.append((x_, y_, z_ - self.M.ground(x_, y_), b_[4] if f > 0.5 else a_[4]))
        self._climb_out = [q for q in pts if q[3] == "climb"]
        k = 0
        t = 0.0
        breadcrumbs = []
        returning = False
        stuck = 0.0
        hold = 0.0
        look_left = look_s
        clear = {nm: np.inf for nm in list(w3.KIND_NAMES.values()) + ["ground"]}
        prev = None
        while k < len(pts) and t < 4 * res.planned_s + 120:
            x, y, h, mode = pts[k]
            if mode == "look" and look_left > 0 and not returning:
                look_left -= dt
            under = self._under_crowns(x, y, h)
            # navigation error (the drone flies its estimate)
            if under and self.nav == "gps":
                nav = np.clip(nav + self.rng.normal(0, 1.0 * math.sqrt(dt / 2.0), 2), -6, 6)
            elif under:  # visual-inertial odometry: ~1% of the distance flown under the crowns
                step = math.hypot(x - prev[0], y - prev[1]) if prev else 0.0
                nav = nav + self.rng.normal(0, 0.01 * step + 0.002, 2)
            else:
                nav = nav * 0.9 + self.rng.normal(0, 0.05, 2)
            res.nav_err_max = max(res.nav_err_max, float(np.hypot(*nav)))
            u = self._wind_at(x, y, max(h, 1.0)) if self.wind is not None else 0.0
            sig = 0.06 * u
            wind_err = wind_err * math.exp(-dt / 2.0) + self.rng.normal(0, sig * math.sqrt(1 - math.exp(-dt)), 3)
            tx_, ty_, th_ = x - nav[0] + wind_err[0], y - nav[1] + wind_err[1], h + 0.5 * wind_err[2]
            # local safety layer: never move towards an obstacle it has seen (any direction it can see),
            # whatever the global estimate says; stuck too long: give up and turn back
            if self.sensing and self.local and res.trace:
                L_ = np.array(self.local[-4000:])
                d_new = np.min(np.sqrt((L_[:, 0] - tx_) ** 2 + (L_[:, 1] - ty_) ** 2 + (L_[:, 2] - th_) ** 2)) - 0.5
                if d_new < self.r_d + 0.3:
                    px_, py_ = res.trace[-1][1], res.trace[-1][2]
                    ph_ = res.trace[-1][3] - self.T.ground(px_, py_)
                    d_old = np.min(np.sqrt((L_[:, 0] - px_) ** 2 + (L_[:, 1] - py_) ** 2 + (L_[:, 2] - ph_) ** 2)) - 0.5
                    if d_new < d_old:
                        tx_, ty_, th_ = px_, py_, ph_
                        stuck += dt
                        if stuck > 15.0 and not returning:
                            res.outcome = "aborted"
                            res.events.append(("stuck: turned back", t))
                            pts = pts[:k + 1] + list(reversed(breadcrumbs)) + self._climb_out
                            returning = True
                            stuck = 0.0
                else:
                    stuck = 0.0
            res.trace.append((t, tx_, ty_, th_ + self.T.ground(tx_, ty_), mode))
            # collision?
            hit = self._touch(tx_, ty_, th_)
            for kind, d in self.T.kinds_near(tx_, ty_, th_, 6.0):
                ki = int(kind)
                nm = w3.KIND_NAMES.get(ki & -ki, "obstacle")
                clear[nm] = min(clear[nm], float(d) - self.r_d)
            clear["ground"] = min(clear["ground"], th_ - f3.DRONES[self.drone]["height"] / 2)
            if hit:
                res.outcome, res.hit_kind = "collided", hit
                break
            # traffic where a low route crosses a road
            if th_ < 4.5 and self._near_road(tx_, ty_, 4.0) and self.rng.random() < dt / TRAFFIC_GAP_S:
                if self.sensing and self.rng.random() < 0.95:
                    hold = max(hold, 4.0)
                    res.events.append(("vehicle: held", t))
                else:
                    res.outcome, res.hit_kind = "collided", "vehicle"
                    break
            # birds of prey above the trees
            if not under and h > 10 and self.rng.random() < BIRD_PER_H * dt / 3600:
                if self.sensing and self.rng.random() < 0.5:
                    hold = max(hold, 8.0)
                    res.events.append(("bird: evaded", t))
                else:
                    res.outcome, res.hit_kind = "bird", "bird of prey"
                    break
            # sensing and avoiding
            if self.sensing:
                nxt = pts[min(k + 1, len(pts) - 1)]
                heading = (nxt[0] - x, nxt[1] - y, nxt[2] - h)
                new = self._sense((tx_, ty_, th_), heading, known, under, offset=(nav[0] - wind_err[0],
                                                                                   nav[1] - wind_err[1]))
                for kc in new.values():
                    res.detected[kc] = res.detected.get(kc, 0) + 1
                v = (LOW_SPEED if mode in ("under", "look") else f3.CRUISE_MPS)
                look = v * v / (2 * BRAKE) + v * LATENCY + 3.0
                ahead = []
                dist_acc = 0.0
                for q in pts[k:]:
                    ahead.append(q[:3])
                    if len(ahead) > 1:
                        dist_acc += math.dist(ahead[-1], ahead[-2])
                    if dist_acc > look:
                        break
                c = self._conflict(ahead, known, self.r_d + 0.3) if new or k % 5 == 0 else None
                if c is not None:
                    res.replans += 1
                    res.events.append(("stop and re-plan", t, mode))
                    new_pts, why = self._replan(pts, k, known, breadcrumbs, returning, target, agl)
                    if new_pts is None:
                        res.outcome = "aborted"
                        res.events.append(("gave up: " + why, t))
                        # retrace own track to the gap (flown before, so clear) and climb out
                        new_pts = list(reversed(breadcrumbs)) + self._climb_out
                        returning = True
                        if not new_pts:
                            break
                    elif not returning and any(q[3] == "look" for q in new_pts) and not any(
                            q[3] == "climb" for q in new_pts):
                        low = [q for q in new_pts if q[3] == "under"]
                        back = [(a_, b_, c_, "under") for a_, b_, c_, _ in reversed(breadcrumbs + low)]
                        new_pts = new_pts + back + self._climb_out
                    pts = pts[:k] + new_pts
                    t += 1.0  # stopping and re-planning
            if hold > 0:
                hold -= dt
                t += dt
                continue
            if mode in ("under", "look") and not returning:
                breadcrumbs.append((x, y, h, "under"))
            if mode == "look" and not returning:
                res.reached = math.hypot(x - target[0], y - target[1]) <= f3.REACH_M
            if mode == "look" and look_left > 0:
                t += dt
                continue
            if mode == "look":
                returning = True
            prev = (x, y)
            k += 1
            t += dt
        res.time_s = t
        res.min_clear = clear
        return res

    def _near_road(self, x, y, r):
        for P in self.roads:
            for a, b in zip(P[:-1], P[1:]):
                a, b = np.asarray(a), np.asarray(b)
                ab = b - a
                u = np.clip(np.dot(np.array([x, y]) - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
                if math.dist((x, y), a + u * ab) < r:
                    return True
        return False

    def _replan(self, pts, k, known, breadcrumbs, returning, target, agl):
        x, y, h, mode = pts[k]
        if returning:
            # on the way back: retrace the track flown in (known clear), skipping what is ahead
            if breadcrumbs:
                return list(reversed(breadcrumbs)), ""
            return None, "blocked on the way back"
        extra = np.zeros_like(self.M.occ)
        for (a_, b_, c_), kd in known.items():
            extra[a_, b_, c_] |= kd
        P = self.make_planner(extra)
        if mode in ("under", "look") or h < 8:
            D, t2 = P.low_field(target)
            ci, cj = P._c2(x, y)
            if not np.isfinite(D[ci, cj]):
                return None, "no way through under the crowns"
            path = P._descend_path(D, (ci, cj))
            out = []
            for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
                x0, y0 = P._xy2(i0, j0)
                x1, y1 = P._xy2(i1, j1)
                m = max(1, int(math.hypot(x1 - x0, y1 - y0) / (LOW_SPEED * 0.1)))
                for u in np.linspace(0, 1, m + 1)[1:]:
                    out.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, f3.LOW_M, "under"))
            out += [out[-1][:3] + ("look",)] * 80 if out else []
            return (out or None), ("" if out else "no way through under the crowns")
        tr, info = P.under_check((x, y), target, agl)
        if tr is None:
            return None, str(info)
        return [(q[1], q[2], q[3] - self.M.ground(q[1], q[2]), q[4]) for q in tr], ""


LOW_SPEED = f3.LOW_MPS
