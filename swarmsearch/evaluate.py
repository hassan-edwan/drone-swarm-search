"""Run every method on the same held-out scenarios and summarise the metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .env import GridSearchEnv
from .scenario import CAMERA_STEPS, DRIFT_DIRS, MovementProfile, Scenario, ScenarioSampler

# Named test conditions. Each is a function seed -> Scenario so that every
# method sees exactly the same missions.
def _sampled(**kw):
    sampler = ScenarioSampler(**kw)
    return lambda seed: sampler.sample(np.random.default_rng(10_000 + seed))


def _mv(kind: str, speed: float, known: bool, n_zones: int = 0, **kw):
    """Drift direction and safe point vary with the episode seed, so no fixed
    sweep pattern is lined up with the movement by luck."""
    g = kw.get("grid_size", 20)

    def make(seed: int) -> Scenario:
        rng = np.random.default_rng(20_000 + seed)
        drift = DRIFT_DIRS[rng.integers(len(DRIFT_DIRS))] if kind == "drift" else (0, 0)
        target = tuple(int(v) for v in rng.integers(0, g, size=2)) if kind == "to_safety" else (0, 0)
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
# M2: block obstacles (buildings, rubble). Movement profile unknown unless stated.
M2_CONDITIONS = {
    "obs10_wander": _mv("wander", 1.0, False, obstacle_frac=0.1),
    "obs30_wander": _mv("wander", 1.0, False, obstacle_frac=0.3),
    "obs20_drift_known": _mv("drift", 1.0, True, obstacle_frac=0.2),
    "obs20_drift_unknown": _mv("drift", 1.0, False, obstacle_frac=0.2),
    "obs20_to_safety_known": _mv("to_safety", 1.0, True, obstacle_frac=0.2),
    "obs20_to_safety_unknown": _mv("to_safety", 1.0, False, obstacle_frac=0.2),
    "obs20_drone_lost": _mv("wander", 1.0, False, obstacle_frac=0.2, drone_failure=True),
    "obs_random_mix": _sampled(obstacle_frac=(0.0, 0.3)),
}
# M4: obstacles are buildings taller than the cruise layers (flown in 3D with
# scripts/run3d.py --m4); no diagonal moves past a building's corner.
def _no_corners(make):
    return lambda seed: replace(make(seed), corner_cutting=False)


M4_CONDITIONS = {
    "m4_drift_unknown": _no_corners(_mv("drift", 1.0, False, obstacle_frac=0.2)),
    "m4_to_safety_unknown": _no_corners(_mv("to_safety", 1.0, False, obstacle_frac=0.2)),
    "m4_dense_wander": _no_corners(_mv("wander", 1.0, False, obstacle_frac=0.3)),
    "m4_random_mix": _no_corners(_sampled(obstacle_frac=(0.1, 0.3))),
}
# M5: other map sizes and team sizes. The time budget scales with the work:
# SWEEPS passes over the map's cells shared between the drones (1.5 sweeps
# is the usual 200 steps for 3 drones on 20x20).
SWEEPS = 1.5


def budget(grid_size: int, n_drones: int, obstacle_frac: float = 0.0, sweeps: float = SWEEPS) -> int:
    return int(round(sweeps * grid_size * grid_size * (1 - obstacle_frac) / n_drones))


def _scaled(kind: str, speed: float, known: bool, grid_size: int = 20, n_drones: int = 3, **kw):
    return _mv(kind, speed, known, grid_size=grid_size, n_drones=n_drones,
               max_steps=budget(grid_size, n_drones, kw.get("obstacle_frac", 0.0)), **kw)


M5_CONDITIONS = {
    "g30_drift_unknown": _scaled("drift", 1.0, False, grid_size=30),
    "g30_still": _scaled("still", 0.0, True, grid_size=30),
    "g40_drift_unknown": _scaled("drift", 1.0, False, grid_size=40),
    "g40_to_safety_unknown": _scaled("to_safety", 1.0, False, grid_size=40),
    "2_drones_drift_unknown": _scaled("drift", 1.0, False, n_drones=2),
    "6_drones_drift_unknown": _scaled("drift", 1.0, False, n_drones=6),
}


# M6: geometry-aware camera (altitude, line of sight, cover, false alarms,
# light). 100-step missions on 20x20 (see scenario.CAMERA_STEPS).
def _cam(kind: str, speed: float, known: bool, lighting: str = "day_rgb", cover: float = 0.0, **kw):
    return _mv(kind, speed, known, camera=True, lighting=lighting, cover_frac=cover, max_steps=CAMERA_STEPS, **kw)


CAMERA_CONDITIONS = {
    "cam_open_wander": _cam("wander", 1.0, False),
    "cam_canopy_wander": _cam("wander", 1.0, False, cover=0.4),
    "cam_city_drift_unknown": _cam("drift", 1.0, False, cover=0.2, obstacle_frac=0.2),
    "cam_city_to_safety_unknown": _cam("to_safety", 1.0, False, cover=0.2, obstacle_frac=0.2),
    "cam_dusk_city_wander": _cam("wander", 1.0, False, lighting="dusk_rgb", cover=0.3, obstacle_frac=0.2),
    "cam_night_thermal": _cam("to_safety", 1.0, False, lighting="night_thermal", cover=0.3, obstacle_frac=0.2),
    "cam_midday_thermal": _cam("drift", 1.0, False, lighting="midday_thermal", cover=0.2, obstacle_frac=0.1),
    "cam_zones_still": _cam("still", 0.0, True, cover=0.3, n_zones=3),
    "cam_random_mix": _sampled(obstacle_frac=(0.0, 0.3), camera=True, max_steps=CAMERA_STEPS),
}
# M7: the real-world effects (realism.py), one at a time on the same city
# mission, then all together, then randomly drawn. Each single-effect
# condition shows how much that gap between model and world costs.
SENSOR = dict(sensor_scale=0.6, fa_scale=2.0, vis_sd=0.4)
COLLAPSE = dict(collapse_frac=0.4)
RADIO = dict(comm_range=12.0, comm_loss=0.1)
POSITION = dict(wind_mps=8.0, gps_err=0.06)
PEOPLE = dict(behaviour="mixed")


def _real(**kw):
    return _cam("to_safety", 1.0, False, cover=0.2, obstacle_frac=0.25, real=True, **kw)


REAL_CONDITIONS = {
    "real_none": _real(),
    "real_sensor": _real(**SENSOR),
    "real_collapse": _real(**COLLAPSE),
    "real_radio": _real(**RADIO),
    "real_position": _real(**POSITION),
    "real_people": _real(**PEOPLE),
    "real_all": _real(**SENSOR, **COLLAPSE, **RADIO, **POSITION, **PEOPLE),
    "real_random_mix": _sampled(obstacle_frac=(0.0, 0.3), camera=True, real=True, max_steps=CAMERA_STEPS),
}
# M8: field environment (swarmsearch/field). Each family drawn with every
# effect randomised, then the hard test cases from docs/ASSUMPTIONS.md on
# wilderness missions. Flood is held out of training (generalisation test).
def _field(family: str, **override):
    from .field.params import sample_scenario
    return lambda seed: sample_scenario(np.random.default_rng(30_000 + seed), family, **override)


FIELD_CONDITIONS = {
    "wild_mix": _field("wilderness"),
    "urban_mix": _field("urban"),
    "flood_mix_heldout": _field("flood"),
    "wild_mix_adaptive_map": _field("wilderness", adaptive_map=True),
    "wild_hard_unresponsive": _field("wilderness", force="unresponsive"),
    "wild_hard_evasive": _field("wilderness", force="evasive"),
    "wild_hard_wrong_ipp": _field("wilderness", ipp_wrong=True),
    "wild_hard_outside": _field("wilderness", force="outside"),
    "wild_hard_weak_sensor": _field("wilderness", sensor_scale=0.4),
    "wild_hard_late_start": _field("wilderness", hours_lost=12.0),
    "wild_hard_radio": _field("wilderness", comm_range_m=300.0, rtb_on_link_loss=True),
    "wild_hard_off_model": _field("wilderness", force="off_model"),
}
# M8 mission suite for Phase 2 (docs/MISSIONS.md): one effect turned up at a
# time on top of the randomised family, so a result can be traced to a cause.
# Same seeds as FIELD_CONDITIONS. Flood missions stay held out of training.
_LONG = 4 * 3600.0
FIELD_MISSIONS = {
    # Who is lost (ISRID categories: distance, behaviour, response, survival)
    "wild_hiker": _field("wilderness", category="hiker"),
    "wild_hunter": _field("wilderness", category="hunter"),
    "wild_child": _field("wilderness", category="child_7_12"),
    "wild_dementia": _field("wilderness", category="dementia"),
    "wild_despondent": _field("wilderness", category="despondent"),
    "wild_group": _field("wilderness", n_people=3, group_size=3),
    "wild_phone_ping": _field("wilderness", phone_ping=True),
    # Conditions
    "wild_night": _field("wilderness", light="night", hour=2.0),
    "wild_cold_night": _field("wilderness", light="night", hour=3.0, air_c=-8.0, hypothermic=0.5),
    "wild_rain": _field("wilderness", rain=0.8),
    "wild_wind": _field("wilderness", wind_mps=11.0, gust=0.3),
    "wild_dense_forest": _field("wilderness", map_error=0.4, fa_scale=2.5),
    # Fleet and airspace
    "wild_two_drones": _field("wilderness", n_drones=2),
    "wild_eight_drones": _field("wilderness", n_drones=8),
    "wild_short_battery": _field("wilderness", endurance_min=25.0, swap_s=300.0),
    "wild_airspace": _field("wilderness", n_nofly=3, p_tfr=1.0),
    # Long operations and a growing area (K2)
    "wild_long_fixed": _field("wilderness", mission_s=_LONG, expand=True, expand_at=9.0),
    "wild_long_expand": _field("wilderness", mission_s=_LONG, expand=True, expand_at=0.5),
    "wild_long_expand_2h": _field("wilderness", mission_s=_LONG, expand=True, expand_at=9.0, expand_min=120.0),
    # Evidence from outside the drones, and others in the area (K7, S8)
    "wild_no_evidence": _field("wilderness", clue_rate=0.0, p_report=0.0, n_walkers=0, n_animals=0),
    "wild_clues": _field("wilderness", clue_rate=1.0),
    "wild_late_report": _field("wilderness", p_report=1.0),
    "wild_busy_trails": _field("wilderness", n_walkers=12, n_animals=10),
    # Urban collapse
    "urban_heavy_collapse": _field("urban", collapse_frac=0.6),
    "urban_aftershock": _field("urban", aftershock=1.0),
    "urban_smoke": _field("urban", smoke=1.0),
    "urban_night": _field("urban", light="night", hour=2.0),
    "urban_radio": _field("urban", comm_range_m=200.0, tether_relay=False),
    # Flood (held out of training)
    "flood_rising_heldout": _field("flood", water_rise=2.0),
    "flood_night_heldout": _field("flood", light="night", hour=2.0),
}
ALL_CONDITIONS = {**CONDITIONS, **SPEED_SWEEP, **M2_CONDITIONS, **M4_CONDITIONS, **M5_CONDITIONS, **CAMERA_CONDITIONS,
                  **REAL_CONDITIONS, **FIELD_CONDITIONS, **FIELD_MISSIONS}


def run_episode(env: GridSearchEnv, policy, scenario: Scenario, seed: int) -> dict:
    env.reset(seed=seed, options={"scenario": scenario})
    policy.reset(env)
    while env.agents:
        env.step(policy.act(env))
    return env.metrics()


_EVAL: tuple | None = None  # (policies, conditions) shared with forked workers
_ENVS: dict = {}


def _env_for(sc0):
    from .envs import is_field
    if is_field(sc0):
        from .field.env import FieldEnv
        return _ENVS.setdefault("field", FieldEnv(sc0))
    g = sc0.grid_size
    return _ENVS.setdefault((g, sc0.camera, sc0.real),
                            GridSearchEnv(Scenario(grid_size=g, camera=sc0.camera, real=sc0.real)))


def _eval_task(task):
    cname, pname, seed = task
    policies, conditions = _EVAL
    make = conditions[cname]
    return cname, pname, seed, run_episode(_env_for(make(0)), policies[pname], make(seed), seed=seed)


def _worker_init():
    import torch
    torch.set_num_threads(1)


def evaluate(policies: dict, conditions: dict | None = None, n_episodes: int = 50, grid_size: int = 20,
             workers: int = 1):
    """Returns {condition: {method: summary}}. With workers > 1 the missions
    run in that many processes (forked, so policies and conditions are shared
    as they are); results are the same as with one."""
    conditions = conditions or CONDITIONS
    if workers > 1:
        import multiprocessing as mp
        global _EVAL
        _EVAL = (policies, conditions)
        tasks = [(c, p, s) for c in conditions for p in policies for s in range(n_episodes)]
        with mp.get_context("fork").Pool(workers, initializer=_worker_init) as pool:
            out = pool.map(_eval_task, tasks, chunksize=1)
        rows: dict = {}
        for c, p, s, r in sorted(out, key=lambda x: x[2]):
            rows.setdefault(c, {}).setdefault(p, []).append(r)
        return {c: {p: summarise(rows[c][p], conditions[c](0).max_steps) for p in policies} for c in conditions}
    envs: dict[tuple[int, bool], GridSearchEnv] = {}
    results = {}
    for cname, make in conditions.items():
        results[cname] = {}
        sc0 = make(0)
        g = sc0.grid_size
        from .envs import is_field
        if is_field(sc0):
            from .field.env import FieldEnv
            env = envs.setdefault("field", FieldEnv(sc0))
        else:
            env = envs.setdefault((g, sc0.camera, sc0.real),
                                  GridSearchEnv(Scenario(grid_size=g, camera=sc0.camera, real=sc0.real)))
        for pname, policy in policies.items():
            rows = [run_episode(env, policy, make(s), seed=s) for s in range(n_episodes)]
            results[cname][pname] = summarise(rows, rows and make(0).max_steps)
    return results


def summarise(rows: list[dict], max_steps: int) -> dict:
    t90 = np.array([r["time_to_90"] if r["time_to_90"] is not None else max_steps for r in rows], float)
    ft = np.array([r["find_time"] for r in rows], float)
    return {
        "find_time": float(ft.mean()),
        "find_time_se": float(ft.std() / np.sqrt(max(len(ft), 1))),
        "find_time_frac": float(np.mean([r["find_time_frac"] for r in rows])),
        "time_to_90": float(t90.mean()),
        "time_to_90_std": float(t90.std()),
        "reached_90": float(np.mean([r["time_to_90"] is not None for r in rows])),
        "found_frac": float(np.mean([r["found_frac"] for r in rows])),
        "coverage": float(np.mean([r["coverage"] for r in rows])),
        "repeat_frac": float(np.mean([r["repeat_frac"] for r in rows])),
        "collisions": float(np.mean([r["collisions"] for r in rows])),
        "shield": float(np.mean([r.get("shield", 0) for r in rows])),
        "alarms": float(np.mean([r.get("alarms", 0) for r in rows])),
        "false_alarms": float(np.mean([r.get("false_alarms", 0) for r in rows])),
        "bumps": float(np.mean([r.get("bumps", 0) for r in rows])),
        "linked": float(np.mean([r.get("linked", 1.0) for r in rows])),
        "revealed": float(np.mean([r.get("revealed", 1.0) for r in rows])),
        **({k: float(np.mean([r[k] for r in rows])) for k in FIELD_KEYS} if rows and "rescue_time" in rows[0] else {}),
    }


FIELD_KEYS = ("found_inside_frac", "found_by_half", "rescue_time", "rescued_frac", "outside_frac", "op_missed",
              "crashes", "near_misses", "swaps", "minutes")


def field_markdown(results: dict) -> str:
    lines = ["| Condition | Method | Mean find time (steps) | Found | Found (inside area) | Found by half time | "
             "Rescued | Mean rescue time | Coverage | Alerts (false) | Operator misses | Crashes | Linked |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cname, per in results.items():
        for pname, s in per.items():
            lines.append(
                f"| {cname} | {pname} | {s['find_time']:.0f} ± {s['find_time_se']:.0f} | {s['found_frac']:.0%} | "
                f"{s['found_inside_frac']:.0%} | {s['found_by_half']:.0%} | {s['rescued_frac']:.0%} | "
                f"{s['rescue_time']:.0f} | {s['coverage']:.0%} | {s['alarms']:.0f} ({s['false_alarms']:.0f}) | "
                f"{s['op_missed']:.1f} | {s['crashes']:.2f} | {s['linked']:.2f} |")
    return "\n".join(lines)


def to_markdown(results: dict) -> str:
    first = next(iter(next(iter(results.values())).values()), {}) if results else {}
    if "rescue_time" in first:
        return field_markdown(results)
    lines = ["| Condition | Method | Mean find time (steps) | Find time (% of budget) | Time to 90% (steps) | "
             "Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops | Alarms (false) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cname, per in results.items():
        for pname, s in per.items():
            lines.append(
                f"| {cname} | {pname} | {s['find_time']:.0f} ± {s['find_time_se']:.0f} | {s['find_time_frac']:.0%} | "
                f"{s['time_to_90']:.0f} ± {s['time_to_90_std']:.0f} | "
                f"{s['reached_90']:.0%} | {s['found_frac']:.0%} | {s['coverage']:.0%} | "
                f"{s['repeat_frac']:.0%} | {s['collisions']:.1f} | {s.get('shield', 0):.1f} | "
                f"{s.get('alarms', 0):.1f} ({s.get('false_alarms', 0):.1f}) |"
            )
    return "\n".join(lines)
