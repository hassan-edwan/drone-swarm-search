"""Fly planned 3D paths with ArduPilot's copter firmware in software-in-the-loop (SITL).

    python scripts/real_site/sitl_fly.py --flight sites/catoctin_west/flight_seed103.json \\
        --ardupilot ~/ardupilot --wind 5 --speedup 10

For each drone in the flight file (fly_mission.py) this starts one ArduPilot
SITL copter (the real flight-control code, simulated physics, sensors, GPS
and wind), connects over MAVLink, arms and takes off in GUIDED mode, then
streams position and velocity targets along the planned path, as a ground
station or companion computer would. It logs where the copter really flew,
and reports:

- tracking error: distance between the flown and the planned position at the
  same time (horizontal and vertical), overall and under the canopy;
- clearances of the FLOWN path from the lidar obstacles (ground, crown
  bottom, trunks, buildings, crown tops), the same checks as for the plan.

SITL's ground is flat, so its home is put at the lowest terrain of the site
(minus a margin): real terrain is never below the simulated ground, and
heights are compared against the real terrain from the lidar. Obstacles are
not in SITL; they are checked against the flown track afterwards.

Needs: the arducopter SITL binary and copter.parm in --ardupilot
(scripts/cluster/get_ardupilot.sh), pymavlink.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarmsearch import flight3d as f3  # noqa: E402

M_PER_DEG_LAT = 110_540.0
PARAMS = {
    # a fast SAR quad: 20 m/s (e.g. the DJI M30T is rated 23 m/s), 4 m/s up, 3 m/s down.
    # Names as in ArduCopter 4.7 (SI units; older firmware used WPNAV_SPEED in cm/s, ANGLE_MAX in centidegrees).
    "WP_SPD": 20.0, "WP_SPD_UP": 4.0, "WP_SPD_DN": 3.0, "WP_ACC": 5.0, "WP_ACC_Z": 2.0,
    "ATC_ANGLE_MAX": 45.0, "PSC_NE_JERK": 10.0, "LOIT_SPEED_MS": 20.0,
    "FENCE_ENABLE": 0, "SIM_WIND_TURB": 1.0, "BATT_MONITOR": 0, "FS_GCS_ENABLE": 0, "DISARM_DELAY": 0,
}
# GPS under dense canopy: consumer receivers lose satellites and suffer multipath, typically
# several metres of error that wanders slowly; in the open, normal GPS.
GPS_OPEN = {"SIM_GPS1_NOISE": 0.0, "SIM_GPS1_NUMSATS": 10, "SIM_GPS1_GLTCH_X": 0.0, "SIM_GPS1_GLTCH_Y": 0.0}
GPS_CANOPY = {"SIM_GPS1_NOISE": 3.0, "SIM_GPS1_NUMSATS": 6}
GPS_BIAS_MAX = 6.0  # metres: the slowly wandering multipath offset under the crowns
MASK_POS = 0b110111111000  # position only; ArduPilot makes its own velocity (feeding the plan's velocity
# forward made the copter run on past a target the follower was still waiting at)


def ll(meta, x, y):
    lat0, lon0 = meta["lat"], meta["lon"]
    return lat0 + y / M_PER_DEG_LAT, lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))


def plan_arrays(tr):
    a = np.array([q[:4] for q in tr], float)
    o = np.argsort(a[:, 0], kind="stable")
    return a[o], [tr[i][4] for i in o]


def retime(P, vh=20.0, vup=4.0, vdn=3.0, acc=4.0, lat_acc=4.0):
    """A flyable clock for a planned path. The search grid moves one cell per step
    (diagonals and sprints included) with corners at full speed; a real copter has a
    top speed, climb and descent rates, and must slow for sharp turns. Speed at each
    point: at most vh, at most sqrt(lat_acc * turn radius) at corners, reachable with
    acceleration acc from the points before and after, and never faster than the plan.
    Returns the new plan and how much longer than the original clock it takes."""
    n = len(P)
    xy = P[:, 1:3]
    ds = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    vmax = np.full(n, vh)
    for k in range(1, n - 1):
        a, b = xy[k] - xy[k - 1], xy[k + 1] - xy[k]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la > 1e-6 and lb > 1e-6:
            th = math.acos(float(np.clip(np.dot(a, b) / (la * lb), -1, 1)))
            if th > 0.15:
                r = min(la, lb) / (2 * math.tan(th / 2)) if th < math.pi - 1e-3 else 0.5
                vmax[k] = min(vh, max(2.0, math.sqrt(lat_acc * max(r, 0.5))))
    # the plan's own speed (hovering stays hovering)
    dtp = np.r_[1.0, np.diff(P[:, 0])]
    vplan = np.where(dtp > 1e-6, ds / np.maximum(dtp, 1e-6), vh)
    vmax = np.minimum(vmax, np.maximum(vplan, 0.5))
    v = vmax.copy()
    v[0] = 0.0
    for k in range(1, n):
        v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2 * acc * ds[k]))
    v[-1] = min(v[-1], 0.0) if n > 1 else 0.0
    for k in range(n - 2, -1, -1):
        v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2 * acc * ds[k + 1]))
    Q = P.copy()
    for k in range(1, n):
        dt = P[k, 0] - P[k - 1, 0]
        vm = max(0.5 * (v[k] + v[k - 1]), 0.3)
        dz = P[k, 3] - P[k - 1, 3]
        need = max(dt, ds[k] / vm if ds[k] > 1e-6 else 0.0, dz / vup if dz > 0 else -dz / vdn)
        Q[k, 0] = Q[k - 1, 0] + need
    return Q, float(Q[-1, 0] - P[-1, 0])


def interp(P, t):
    T = P[:, 0]
    if t <= T[0]:
        return P[0, 1:4], np.zeros(3)
    if t >= T[-1]:
        return P[-1, 1:4], np.zeros(3)
    k = int(np.searchsorted(T, t)) - 1
    dt = max(T[k + 1] - T[k], 1e-6)
    u = (t - T[k]) / dt
    pos = P[k, 1:4] * (1 - u) + P[k + 1, 1:4] * u
    vel = (P[k + 1, 1:4] - P[k, 1:4]) / dt
    return pos, vel


class Copter:
    def __init__(self, idx, binary, parm, home, workdir, speedup, wind, wind_dir):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.idx = idx
        workdir.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [str(binary), "-w", "-S", "--model", "+", "--speedup", str(speedup), "-I", str(idx),
             "--home", f"{home[0]:.7f},{home[1]:.7f},{home[2]:.2f},0", "--defaults", str(parm)],
            cwd=workdir, stdout=open(workdir / "sitl.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(2.0)
        for _ in range(30):
            try:
                self.m = mavutil.mavlink_connection(f"tcp:127.0.0.1:{5760 + 10 * idx}", source_system=255)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
        self.m.wait_heartbeat(timeout=60)
        self.params = {**PARAMS, "SIM_WIND_SPD": wind, "SIM_WIND_DIR": wind_dir}
        # only what we need: position at 10 Hz (and heartbeats); decoding everything at speed-up starves the client
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 0, 0)
        for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10),
                        (mavutil.mavlink.MAVLINK_MSG_ID_SYSTEM_TIME, 5),
                        (mavutil.mavlink.MAVLINK_MSG_ID_SIMSTATE, 10)):
            self.cmd(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, mid, 1e6 / hz)
        self.t_boot = 0.0
        self.pos = None
        self.truth = None  # simulator's true latitude, longitude (the estimate is what GPS and the EKF think)

    def poll(self):
        """Read waiting messages; keep the latest local position (metres NED) and boot time."""
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            tp = msg.get_type()
            if tp == "LOCAL_POSITION_NED":
                self.pos = (msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz)
                self.t_boot = msg.time_boot_ms / 1000.0
            elif tp == "SIMSTATE":
                self.truth = (msg.lat * 1e-7, msg.lng * 1e-7)
            elif tp == "SYSTEM_TIME":
                self.t_boot = max(self.t_boot, msg.time_boot_ms / 1000.0)
            elif hasattr(msg, "time_boot_ms"):
                self.t_boot = max(self.t_boot, msg.time_boot_ms / 1000.0)

    def send_params(self, d):
        for k, v in d.items():
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component, k.encode(), float(v),
                                      self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def set_params(self, tries=5):
        """Set and read back every parameter (sets sent before the parameter system is up are lost)."""
        todo = dict(self.params)
        for _ in range(tries):
            for k, v in todo.items():
                self.m.mav.param_set_send(self.m.target_system, self.m.target_component, k.encode(), float(v),
                                          self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            t_end = time.time() + 3.0
            while todo and time.time() < t_end:
                msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
                if msg is not None:
                    pid = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
                    if pid in todo and abs(msg.param_value - todo[pid]) <= 1e-3 * max(1.0, abs(todo[pid])):
                        del todo[pid]
            if not todo:
                return []
        return sorted(todo)

    def wait_until(self, cond, timeout_sim, step=0.02):
        t0 = self.t_boot
        wall = time.time()
        while True:
            self.poll()
            if cond():
                return True
            if self.t_boot - t0 > timeout_sim or time.time() - wall > timeout_sim:
                return False
            time.sleep(step)

    def cmd(self, command, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, command, 0, *p)

    def set_mode(self, name):
        mode_id = self.m.mode_mapping()[name]
        self.m.mav.set_mode_send(self.m.target_system, self.mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)

    def target(self, ned, vel):
        self.m.mav.set_position_target_local_ned_send(
            int(self.t_boot * 1000) & 0xFFFFFFFF, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, MASK_POS, ned[0], ned[1], ned[2], vel[0], vel[1], vel[2],
            0, 0, 0, 0, 0)

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def fly_one(args):
    """Fly one drone's planned path; return the flown track and errors."""
    (idx, name, plan, modes, home_xy, home_alt, a, meta) = args
    mav = None
    w0 = time.time()

    def log(msg):
        if a.verbose:
            print(f"[{name} {time.time() - w0:6.1f}s] {msg}", flush=True)
    try:
        from pymavlink import mavutil
        lat, lon = ll(meta, *home_xy)
        c = Copter(idx, Path(a.ardupilot).expanduser() / "arducopter", Path(a.ardupilot).expanduser() / "copter.parm",
                   (lat, lon, home_alt), Path(a.workdir) / f"sitl_{idx}", a.speedup, a.wind, a.wind_dir)
        mav = mavutil.mavlink
        # EKF and GPS ready: wait for a local position, then arm in GUIDED and take off
        if not c.wait_until(lambda: c.pos is not None, 120):
            return dict(name=name, error="no position from SITL")
        log(f"position at sim {c.t_boot:.0f}s")
        missing = c.set_params()
        if missing:
            print(f"[{name}] warning: parameters not accepted by this firmware: {missing}", flush=True)
        c.wait_until(lambda: False, 15)  # let the EKF settle
        first = plan[0, 1:4]
        climb_to = float(first[2] - home_alt)
        up = False
        for attempt in range(12):  # GUIDED, arm, take off; repeat until it is really climbing
            c.set_mode("GUIDED")
            c.wait_until(lambda: c.m.flightmode == "GUIDED", 3)
            if not c.m.motors_armed():
                c.cmd(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1, 21196)
                c.wait_until(lambda: c.m.motors_armed(), 3)
            c.cmd(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, climb_to)
            if c.wait_until(lambda: c.pos is not None and -c.pos[2] > 3.0, 15):
                up = True
                break
            log(f"take-off attempt {attempt + 1} did not climb (mode {c.m.flightmode}, armed {c.m.motors_armed()})")
        if not up:
            return dict(name=name, error="did not take off")
        log("armed; taking off to %.0f m" % climb_to)
        c.wait_until(lambda: c.pos is not None and -c.pos[2] > 0.95 * climb_to, 240)
        def ned_of(p):  # plan (x east, y north, z MSL) -> SITL local NED (origin at home)
            return (p[1] - home_xy[1], p[0] - home_xy[0], -(p[2] - home_alt))

        first = plan[0, 1:4]
        log("armed; taking off to %.0f m" % (first[2] - home_alt))
        c.cmd(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, float(first[2] - home_alt))
        c.wait_until(lambda: c.pos is not None and -c.pos[2] > 0.95 * (first[2] - home_alt), 240)
        # go to the start of the plan and settle
        n0 = ned_of(first)
        c.wait_until(lambda: (c.target(n0, (0, 0, 0)) or True) and c.pos is not None and
                     math.dist(c.pos[:3], n0) < 1.5, 240, step=0.1)
        log(f"at the start point (sim {c.t_boot:.0f}s, off by {math.dist(c.pos[:3], n0):.1f} m)")
        # follow the path: a target a short way ahead of the closest point on the plan
        # (pure pursuit), so a drone that falls behind the plan's clock catches up along the
        # path instead of cutting corners; the lead is shorter under the canopy.
        t_start = c.t_boot
        T = plan[:, 0]
        tau = T[0]
        flown = []
        last_send = -1.0
        stall = 0.0
        gps_bad = False
        bias = np.zeros(2)
        last_bias = 0.0
        rng = np.random.default_rng(idx)
        lat0, lon0 = meta["lat"], meta["lon"]
        kx = 111_320.0 * math.cos(math.radians(lat0))
        under_m = np.isin(modes, ["descend", "under", "look", "climb"])
        while tau < T[-1]:
            c.poll()
            now = c.t_boot - t_start
            if now > (T[-1] - T[0]) * 3 + 120:
                log("gave up: far behind the plan")
                break
            if c.pos is not None and now - last_send >= 0.1:
                here = np.array((c.pos[1] + home_xy[0], c.pos[0] + home_xy[1], -c.pos[2] + home_alt))
                ts = np.arange(tau, min(T[-1], tau + 3.0) + 1e-9, 0.1)
                pts = np.array([interp(plan, u)[0] for u in ts])
                k = int(np.argmin(np.sum((pts - here) ** 2, axis=1)))
                if ts[k] > tau:
                    tau, stall = float(ts[k]), 0.0
                ki = min(int(np.searchsorted(T, tau)), len(T) - 1)
                under_now = bool(under_m[ki])
                if a.gps == "degraded":  # GPS goes bad below the crowns (descent, under, look, climb)
                    if under_now and not gps_bad:
                        c.send_params(GPS_CANOPY)
                        gps_bad = True
                    elif not under_now and gps_bad:
                        c.send_params(GPS_OPEN)
                        gps_bad, bias = False, np.zeros(2)
                    if gps_bad and now - last_bias >= 2.0:
                        bias = np.clip(bias + rng.normal(0, 1.0, 2), -GPS_BIAS_MAX, GPS_BIAS_MAX)
                        # the glitch offsets are in degrees of latitude (X) and longitude (Y)
                        c.send_params({"SIM_GPS1_GLTCH_X": bias[1] / M_PER_DEG_LAT, "SIM_GPS1_GLTCH_Y": bias[0] / kx})
                        last_bias = now
                # the clock runs on by itself while the drone keeps up; after a long stall it runs anyway
                close = math.dist(here, interp(plan, tau)[0]) < (2.0 if under_now else 8.0)
                stall = stall + 0.1 if not close else 0.0
                if close or stall > 10.0:
                    tau = min(T[-1], tau + 0.1 * (0.7 if under_now else 1.0))
                lead = 0.6 if under_now else 1.5
                p, _ = interp(plan, tau + lead)
                c.target(ned_of(p), (0.0, 0.0, 0.0))
                last_send = now
                true_xy = ((c.truth[1] - lon0) * kx, (c.truth[0] - lat0) * M_PER_DEG_LAT) if c.truth else here[:2]
                flown.append((now, true_xy[0], true_xy[1], here[2], tau, here[0], here[1]))
            time.sleep(0.005)
        log(f"done: {len(flown)} samples")
        c.close()
        F = np.array(flown)
        F[:, 0] = F[:, 0] + plan[0, 0]  # flown clock, starting with the plan
        return dict(name=name, flown=F.tolist(), lag_s=float((F[-1, 0] - F[0, 0]) - (plan[-1, 0] - plan[0, 0])))
    except Exception as e:  # noqa: BLE001
        return dict(name=name, error=repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flight", default="sites/catoctin_west/flight_seed103.json")
    ap.add_argument("--site", default="", help="site folder (default: the flight file's)")
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0))
    ap.add_argument("--ardupilot", default="~/ardupilot")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--speedup", type=float, default=10.0)
    ap.add_argument("--wind", type=float, default=5.0, help="mean wind, m/s")
    ap.add_argument("--wind-dir", type=float, default=270.0)
    ap.add_argument("--instance0", type=int, default=0, help="first SITL instance (ports 5760 + 10 * instance)")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="fly only this drone")
    ap.add_argument("--gps", default="degraded", choices=["degraded", "good"],
                    help="GPS under the canopy: degraded (noise, few satellites, wandering offset) or good (as if "
                         "the drone navigated by camera there: visual-inertial odometry)")
    ap.add_argument("--drone", default="sar", choices=list(f3.DRONES))
    ap.add_argument("--season", default="leaf-on", choices=["leaf-on", "leaf-off"])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    flight = Path(a.flight)
    site = Path(a.site) if a.site else flight.parent
    a.workdir = a.workdir or str(site / "sitl_runs")
    meta = json.load(open(site / "meta.json"))
    D = json.load(open(flight))
    info = D["info"]
    hz = site / "hazards.json"
    s3 = f3.Site3D.load(a.fine_from, shift=tuple(a.shift), hazards=hz, season=a.season)
    outer = f3.CoarseTerrain(site / "lidar10.npz", hazards=hz)
    home_xy = tuple(info["base"])
    home_alt = float(min(s3.dtm.min(), outer.dtm.min()) - 5.0)
    jobs, delays = [], {}
    for k, (name, tr) in enumerate(sorted(D["trajs"].items())):
        if a.only and name != a.only:
            continue
        P, modes = plan_arrays(tr)
        P, behind = retime(P)
        delays[name] = behind
        start = home_xy if name != "test" else tuple(P[0, 1:3])
        jobs.append((a.instance0 + k, name, P, modes, start, home_alt, a, meta))
    t0 = time.time()
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(fly_one, jobs))
    report = {}
    for (idx, name, P, modes, *_), r in zip(jobs, res):
        if "error" in r:
            report[name] = dict(error=r["error"])
            print(f"{name}: {r['error']}")
            continue
        F = np.array(r["flown"])  # t, true x, y, height, plan time reached, estimated x, y
        est_err = np.hypot(F[:, 1] - F[:, 5], F[:, 2] - F[:, 6]) if F.shape[1] >= 7 else np.zeros(len(F))
        dense = np.array([interp(P, t)[0] for t in np.arange(P[0, 0], P[-1, 0], 0.1)])
        cross, vert, mode_at = [], [], []
        for t, x, y, z, tau, *_ in F:
            lo = int(np.searchsorted(np.arange(P[0, 0], P[-1, 0], 0.1), tau - 3.0))
            seg = dense[max(0, lo):lo + 120]
            dd = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
            j = int(np.argmin(dd))
            cross.append(dd[j])
            vert.append(z - seg[j, 2])
            k = min(int(np.searchsorted(P[:, 0], tau)), len(modes) - 1)
            mode_at.append(modes[k])
        cross, vert = np.array(cross), np.array(vert)
        under = np.isin(mode_at, ["under", "look"])
        flown_tr = [(row[0], row[1], row[2], row[3], m) for row, m in zip(F, mode_at)]
        clr = f3.clearance(s3, flown_tr, drone=a.drone)
        rep = dict(points=len(F), plan_s=float(P[-1, 0] - P[0, 0]), lag_s=r.get("lag_s"),
                   behind_grid_clock_s=delays.get(name, 0.0),
                   cross_err_mean=float(cross.mean()), cross_err_p95=float(np.percentile(cross, 95)),
                   cross_err_max=float(cross.max()), vert_err_mean=float(np.abs(vert).mean()),
                   vert_err_max=float(np.abs(vert).max()), clearance_flown={k: float(v) for k, v in clr.items()})
        rep["gps_err_max_under"] = float(est_err[under].max()) if under.any() else 0.0
        if under.any():
            rep.update(under_cross_mean=float(cross[under].mean()), under_cross_max=float(cross[under].max()),
                       under_vert_max=float(np.abs(vert[under]).max()))
        report[name] = rep
        r["flown"] = F.tolist()
        print(f"{name}: off the path {rep['cross_err_mean']:.2f} m mean / {rep['cross_err_p95']:.2f} m 95% / "
              f"{rep['cross_err_max']:.1f} m max, height {rep['vert_err_mean']:.2f} m mean / {rep['vert_err_max']:.1f} m max"
              + (f"; under the canopy {rep['under_cross_mean']:.2f} m mean / {rep['under_cross_max']:.2f} m max off the path, "
                 f"{rep['under_vert_max']:.2f} m max in height, GPS off by up to {rep['gps_err_max_under']:.1f} m"
                 if under.any() else "")
              + f"; the flyable plan takes {rep['behind_grid_clock_s']:.0f} s longer than the search grid's clock, "
                f"and ArduPilot {rep['lag_s']:+.0f} s on top"
              + "; flown clearances " + ", ".join(f"{k} {v:.1f}" for k, v in clr.items() if np.isfinite(v)))
    out = Path(a.out) if a.out else flight.with_name(flight.stem + f"_sitl_wind{a.wind:.0f}_gps{a.gps}.json")
    json.dump(dict(report=report, wind=a.wind, wind_dir=a.wind_dir, home_alt=home_alt, gps=a.gps, drone=a.drone,
                   season=a.season,
                   flown={r["name"]: r.get("flown") for r in res}), open(out, "w"))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
