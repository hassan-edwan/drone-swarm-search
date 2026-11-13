"""M3: the 2D search policy flown by quadrotors in 3D.

The 2D GridSearchEnv stays the mission layer (belief map, survivors, search,
metrics). Each policy step:

1. the policy picks a move per drone from the same observation as in 2D;
2. the env's shield turns the moves into target cells (env.targets);
3. a PID controller flies every drone from its cell to the target cell
   centre, following a smooth reference along the segment, for one step
   period;
4. a drone that is not within the camera footprint of its target at the end
   of the period does not search this step (it keeps flying there).

Drones cruise at different altitudes (one layer each) so paths that cross
on the map never meet in the air; the shield already keeps them out of each
other's cells.

Flight backends:
- "pybullet": gym-pybullet-drones (Crazyflie 2.x, PyBullet physics,
  DSLPIDControl). Needs `pip install -r requirements-3d.txt` (Python <= 3.12
  for the PyBullet wheel).
- "pointmass": a 3D double integrator with acceleration and speed limits
  and the same PID-style tracking, plus optional wind gusts. Runs anywhere;
  used by the tests and as a quick check without PyBullet.

The world is scaled so one grid cell is CELL_M metres (Crazyflie scale); the
grid step lasts STEP_S seconds.
"""

from __future__ import annotations

import numpy as np

from .env import GridSearchEnv
from .scenario import Scenario

CELL_M = 1.0  # metres per grid cell in the 3D world
STEP_S = 2.0  # seconds of flight per policy step (1 s flips a Crazyflie on diagonal moves)
CTRL_HZ = 48  # control rate
ALT_BASE = 1.0  # cruise altitude of the lowest layer (m)
ALT_GAP = 0.25  # vertical gap between drones' layers (m)
FOOTPRINT = 0.35  # horizontal distance (m) from the cell centre that still counts as over it
ALT_TOL = 0.3  # altitude error that still gives a usable camera view (m)
TRAVEL_FRAC = 0.7  # reference reaches the target after this share of the period, then holds
NEAR_MISS_M = 0.2  # two drones closer than this count as a near miss


def cell_to_xyz(cell, layer: int, g: int) -> np.ndarray:
    """Cell (x, y) centre in metres; the grid's y axis maps to world y."""
    x, y = cell
    return np.array([(x + 0.5) * CELL_M, (g - 1 - y + 0.5) * CELL_M, ALT_BASE + layer * ALT_GAP])


# ------------------------------------------------------------------ backends
class PointMassFlight:
    """3D double integrator per drone tracking a position/velocity reference
    with a PD controller, acceleration and speed limits, and optional wind."""

    def __init__(self, start_xyz: np.ndarray, ctrl_hz: int = CTRL_HZ, kp: float = 12.0, kd: float = 6.0,
                 a_max: float = 8.0, v_max: float = 3.0, wind: float = 0.0, seed: int = 0):
        self.dt = 1.0 / ctrl_hz
        self.kp, self.kd, self.a_max, self.v_max, self.wind = kp, kd, a_max, v_max, wind
        self.pos = np.array(start_xyz, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.rng = np.random.default_rng(seed)
        self._gust = np.zeros_like(self.pos)

    def step(self, ref_pos: np.ndarray, ref_vel: np.ndarray) -> np.ndarray:
        acc = self.kp * (ref_pos - self.pos) + self.kd * (ref_vel - self.vel)
        if self.wind:
            # Gusts: slowly varying random acceleration (Ornstein-Uhlenbeck).
            self._gust += -0.5 * self._gust * self.dt + self.wind * np.sqrt(self.dt) * self.rng.normal(size=self._gust.shape)
            acc = acc + self._gust
        n = np.linalg.norm(acc, axis=1, keepdims=True)
        acc = acc * np.minimum(1.0, self.a_max / np.maximum(n, 1e-9))
        self.vel += acc * self.dt
        n = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel *= np.minimum(1.0, self.v_max / np.maximum(n, 1e-9))
        self.pos += self.vel * self.dt
        return self.pos.copy()

    def close(self):
        pass


class PyBulletFlight:
    """gym-pybullet-drones CtrlAviary with one DSLPIDControl per drone."""

    def __init__(self, start_xyz: np.ndarray, ctrl_hz: int = CTRL_HZ, gui: bool = False):
        try:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError("backend 'pybullet' needs gym-pybullet-drones: pip install -r requirements-3d.txt") from e
        n = len(start_xyz)
        self.env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=n, initial_xyzs=np.array(start_xyz),
                              initial_rpys=np.zeros((n, 3)), physics=Physics.PYB, pyb_freq=240,
                              ctrl_freq=ctrl_hz, gui=gui, record=False, obstacles=False, user_debug_gui=False)
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(n)]
        self.dt = self.env.CTRL_TIMESTEP
        obs, _ = self.env.reset()
        self.obs = obs
        self.action = np.zeros((n, 4))

    @property
    def pos(self) -> np.ndarray:
        return np.array([o[0:3] for o in self.obs])

    def step(self, ref_pos: np.ndarray, ref_vel: np.ndarray) -> np.ndarray:
        for j, c in enumerate(self.ctrl):
            self.action[j], _, _ = c.computeControlFromState(
                control_timestep=self.dt, state=self.obs[j], target_pos=ref_pos[j],
                target_rpy=np.zeros(3), target_vel=ref_vel[j])
        self.obs, *_ = self.env.step(self.action)
        return self.pos

    def close(self):
        self.env.close()


def make_flight(backend: str, start_xyz: np.ndarray, gui: bool = False, wind: float = 0.0, seed: int = 0):
    if backend == "pointmass":
        return PointMassFlight(start_xyz, wind=wind, seed=seed)
    if backend == "pybullet":
        return PyBulletFlight(start_xyz, gui=gui)
    raise ValueError(f"unknown backend {backend!r}")


# --------------------------------------------------------------- the mission
class Swarm3D:
    """Runs a GridSearchEnv mission with every move flown in 3D."""

    def __init__(self, env: GridSearchEnv, backend: str = "pointmass", gui: bool = False,
                 wind: float = 0.0, record: bool = False):
        self.env = env
        self.backend = backend
        self.gui = gui
        self.wind = wind
        self.record = record
        self.flight = None

    def reset(self, scenario: Scenario | None = None, seed: int | None = None):
        if self.flight is not None:
            self.flight.close()
        obs, infos = self.env.reset(seed=seed, options={"scenario": scenario} if scenario else None)
        g = self.env.grid_size
        self.agents_all = list(self.env.agents)
        self.layer = {a: i for i, a in enumerate(self.agents_all)}
        start = np.array([cell_to_xyz(self.env.pos[a], self.layer[a], g) for a in self.agents_all])
        start[:, 2] = 0.1  # on the ground at the base
        self.flight = make_flight(self.backend, start, self.gui, self.wind, seed or 0)
        self.n_late = 0
        self.n_moves = 0
        self.n_near_miss = 0
        self.track_err: list[float] = []
        self.trace: list[np.ndarray] = [start.copy()] if self.record else []
        # Take off and settle over the start cells before the mission clock starts.
        self._fly({a: self.env.pos[a] for a in self.agents_all}, duration=3.0, takeoff=True)
        return obs, infos

    def step(self, actions):
        env = self.env
        live = [a for a in env.agents if env.alive[a]]
        target = env.targets(actions)
        arrived = self._fly(target, duration=STEP_S)
        self.n_moves += len(live)
        self.n_late += sum(not arrived[a] for a in live)
        return env.step(actions, searching=arrived)

    def _fly(self, target: dict, duration: float, takeoff: bool = False) -> dict:
        env, g = self.env, self.env.grid_size
        idx = {a: i for i, a in enumerate(self.agents_all)}
        p0 = self.flight.pos.copy()
        p1 = p0.copy()
        for a, cell in target.items():
            p1[idx[a]] = cell_to_xyz(cell, self.layer[a], g)
        # Lost drones hover where they are.
        steps = max(1, int(round(duration * CTRL_HZ)))
        travel = max(1, int(round(TRAVEL_FRAC * steps)))
        pos = p0
        close = np.zeros((len(p0), len(p0)), dtype=bool)
        for k in range(1, steps + 1):
            # Smooth (cosine) speed profile: starts and ends at rest.
            u = min(1.0, k / travel)
            ref = p0 + (p1 - p0) * (0.5 - 0.5 * np.cos(np.pi * u))
            vel = (p1 - p0) * (0.5 * np.pi * np.sin(np.pi * u)) / (travel / CTRL_HZ)
            pos = self.flight.step(ref, vel)
            if self.record and k % 4 == 0:
                self.trace.append(pos.copy())
            if not takeoff:
                d = np.linalg.norm(pos[:, None] - pos[None], axis=2) + np.eye(len(pos)) * 1e9
                close |= d < NEAR_MISS_M
        self.n_near_miss += int(close.sum() // 2)  # pairs that came too close this step
        err_xy = np.linalg.norm(pos[:, :2] - p1[:, :2], axis=1)
        err_z = np.abs(pos[:, 2] - p1[:, 2])
        if not takeoff:
            self.track_err += [float(err_xy[idx[a]]) for a in target]
        return {a: bool(err_xy[idx[a]] <= FOOTPRINT and err_z[idx[a]] <= ALT_TOL) for a in target}

    def metrics(self) -> dict:
        m = self.env.metrics()
        m.update(late_frac=self.n_late / max(1, self.n_moves), near_misses=self.n_near_miss,
                 track_err_m=float(np.mean(self.track_err)) if self.track_err else 0.0)
        return m

    def close(self):
        if self.flight is not None:
            self.flight.close()
            self.flight = None


def run_episode_3d(sim: Swarm3D, policy, scenario: Scenario, seed: int) -> dict:
    sim.reset(scenario, seed)
    policy.reset(sim.env)
    while sim.env.agents:
        sim.step(policy.act(sim.env))
    return sim.metrics()
