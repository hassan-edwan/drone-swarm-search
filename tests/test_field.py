"""Field environment (M8): each test checks one item of docs/ASSUMPTIONS.md."""

import numpy as np
import pytest

from swarmsearch.baselines import BeliefGreedy, Lawnmower
from swarmsearch.field import sensing as sn
from swarmsearch.field.env import (FIELD_OBS_DIM, FIELD_STATE_DIM, HOVER, N_ACTIONS_FIELD, FieldEnv)
from swarmsearch.field.params import FAMILY, FieldSampler, sample_scenario


def _sc(family="wilderness", seed=0, **kw):
    return sample_scenario(np.random.default_rng(seed), family, **kw)


def _run(env, sc, pol, seed=0, steps=None):
    env.reset(seed=seed, options={"scenario": sc})
    pol.reset(env)
    n = 0
    while env.agents and (steps is None or n < steps):
        env.step(pol.act(env))
        n += 1
    return env.metrics()


def _random_steps(env, n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        if not env.agents:
            break
        env.step({a: int(rng.choice(np.flatnonzero(env.action_mask(a)))) for a in env.agents})


@pytest.mark.parametrize("family", ["wilderness", "urban", "flood"])
def test_families_run_with_fixed_dims(family):
    from swarmsearch.mappo import ActorCritic
    env = FieldEnv()
    obs, info = env.reset(seed=1, options={"scenario": _sc(family, 1)})
    assert env.grid_size == FAMILY[family].grid
    for a in env.agents:
        assert obs[a].shape == (FIELD_OBS_DIM,) and np.isfinite(obs[a]).all()
        assert info[a]["action_mask"].shape == (N_ACTIONS_FIELD,)
    assert env.state().shape == (FIELD_STATE_DIM,)  # same size for every family (critic)
    net = ActorCritic(48, True, field=True)
    assert net.vec == FIELD_OBS_DIM and net.n_act == N_ACTIONS_FIELD
    _random_steps(env, 40)


def test_terrain_blocks_line_of_sight():  # S6
    g = 10
    elev = np.zeros((g, g))
    elev[:, 5] = 200.0  # a ridge between x=4 and x=6
    walk = np.ones((g, g), dtype=bool)
    fam = FAMILY["wilderness"]
    G = sn.geo_table(elev, elev, walk, fam.alt_m, (2, 2, 2), fam.alt_detect)
    # drone at (4, 5), target two cells right at (6, 5): behind the ridge
    assert G[2, 5, 4, 2, 4] == 0.0
    assert G[2, 5, 4, 2, 0] > 0.0  # two cells left: open ground


def test_repeated_looks_from_one_direction_add_little():  # S2 (the map's model)
    env = FieldEnv()
    env.reset(seed=0, options={"scenario": _sc("wilderness", 0, sensor_scale=1.0)})
    a = env.agents[0]
    env.use_view(a)
    x, y = env.pos[a]
    drops = []
    for _ in range(4):
        b0 = env.belief[y, x]
        env._search(a, HOVER)
        drops.append(b0 - env.belief[y, x])
    assert drops[0] > 0 and drops[-1] < 0.5 * drops[0]


def test_operator_queue_delay_misses_and_dismissals():  # S5
    rng = np.random.default_rng(0)
    q = sn.OperatorQueue(1, review_steps=3, base_miss=0.0, rng=rng)
    q.add(dict(cell=(1, 1), alt=0, true=0, t=0))
    out = [q.step(0.0) for _ in range(3)]
    assert out[0] == [] and out[1] == [] and out[2][0][1] in ("confirm", "closer_look")
    fa = sn.OperatorQueue(1, 1, 0.0, rng)
    verdicts = []
    for _ in range(200):
        fa.add(dict(cell=(0, 0), alt=1, true=None, t=0))
        verdicts += [v for _, v in fa.step(0.0)]
    assert 0.75 < np.mean([v == "dismiss" for v in verdicts]) < 0.95
    tired = sn.OperatorQueue(1, 1, 0.1, rng)
    for _ in range(300):
        tired.add(dict(cell=(0, 0), alt=0, true=0, t=0))
    for _ in range(300):
        tired.step(4.0)  # four hours in, long backlog
    assert tired.missed / tired.reviewed > 0.3


def test_battery_return_swap_and_relaunch():  # D1
    env = FieldEnv()
    sc = _sc("urban", 2, endurance_min=6.0, swap_s=60.0, fail_per_h=0.0, gust=0.0, rtb_on_link_loss=False)
    m = _run(env, sc, BeliefGreedy(), seed=2)
    assert m["swaps"] >= 1 and m["crashes"] == 0


def test_link_loss_return_keeps_drones_in_range():  # D6, C1
    env = FieldEnv()
    sc = _sc("wilderness", 3, comm_range_m=300.0, comm_loss=0.0, rtb_on_link_loss=True, tether_relay=False)
    env.reset(seed=3, options={"scenario": sc})
    pol = Lawnmower()
    pol.reset(env)
    for _ in range(150):
        if not env.agents:
            break
        env.step(pol.act(env))
    assert env.metrics()["linked"] > 0.7  # planners stay within radio coverage


def test_no_fly_zones_are_masked():  # D8
    env = FieldEnv()
    env.reset(seed=4, options={"scenario": _sc("wilderness", 4, n_nofly=2)})
    env.nofly[:] = False
    a = env.agents[0]
    x, y = env.pos[a]
    env.nofly[y, x + 1] = True
    assert env.action_mask(a)[4] == 0  # (+1, 0) closed
    assert env.action_mask(a)[11 + 4] == 0  # sprinting through it too


def test_person_outside_is_never_found_and_mission_runs_out():  # K2
    env = FieldEnv()
    sc = _sc("wilderness", 5, force="outside")
    m = _run(env, sc, BeliefGreedy(), seed=5)
    assert m["outside_frac"] == 1.0 and m["found_frac"] == 0.0
    assert m["steps"] == sc.max_steps  # nobody tells the team the person is gone
    assert env.p_outside() > 0.0


@pytest.mark.parametrize("force", ["unresponsive", "evasive"])
def test_forced_responsiveness(force):  # P1
    env = FieldEnv()
    env.reset(seed=6, options={"scenario": _sc("wilderness", 6, force=force)})
    assert (env.people.resp == force).all() or env.people.dead.any()


def test_people_stay_on_walkable_ground_and_trapped_stay_put():  # P3, P4
    env = FieldEnv()
    sc = _sc("urban", 7, collapse_frac=1.0)
    env.reset(seed=7, options={"scenario": sc})
    ppl = env.people
    held = (ppl.trapped | ppl.injured) & ~ppl.outside
    start = ppl.pos[held].copy()
    _random_steps(env, 120)
    assert np.array_equal(ppl.pos[held], start)
    inside = ~ppl.outside
    assert env.w.walk[ppl.pos[inside, 1], ppl.pos[inside, 0]].all()


def test_prior_has_mass_outside_and_reported_point_can_be_wrong():  # K1, K2
    env = FieldEnv()
    env.reset(seed=8, options={"scenario": _sc("wilderness", 8, ipp_wrong=True)})
    assert env.p_outside() > 0.0
    d = np.hypot(env.rep_ipp[0] - env.w.ipp[0], env.rep_ipp[1] - env.w.ipp[1]) * env.fam.cell_m
    assert d > 400


def test_light_and_thermal_change_with_the_clock():  # E4, S3
    assert sn.light_level(12) == 1.0 and sn.light_level(23) < 0.1
    assert sn.thermal_contrast(5.0, 2.0, False) > sn.thermal_contrast(30.0, 13.0, False)


def test_collapse_discoveries_travel_with_radio_groups():  # C2
    env = FieldEnv()
    sc = _sc("urban", 9, collapse_frac=1.0, comm_range_m=30.0, comm_loss=0.0)
    env.reset(seed=9, options={"scenario": sc})
    _random_steps(env, 150, seed=9)
    views = env._distinct_views()
    rev = [v.map.revealed.sum() for v in views]
    if len(views) > 1 and len(set(rev)) > 1:
        m = env._merge(views)
        assert m.map.revealed.sum() >= max(rev)


def test_adaptive_map_learns_a_weak_sensor():  # M3
    env = FieldEnv()
    sc = _sc("urban", 10, sensor_scale=0.4, adaptive_map=True)
    _run(env, sc, BeliefGreedy(), seed=10)
    assert env.views["base"].s_est < 1.0


def test_sampler_and_imitation_labels():
    from swarmsearch.imitation import _rollouts
    sampler = FieldSampler(families={"urban": 1.0})
    O, M, A, ms = _rollouts((sampler, 1, 0, None, 1.0, False))
    assert O.shape[1] == FIELD_OBS_DIM and M.shape[1] == N_ACTIONS_FIELD
    assert M[np.arange(len(A)), A].all()  # every label is a legal action


def test_m7_uses_a_head_count_estimate():  # P6 (leak fix)
    from swarmsearch.env import GridSearchEnv
    from swarmsearch.scenario import Scenario
    diffs = 0
    for s in range(10):
        sc = Scenario(camera=True, real=True, n_survivors=10, max_steps=20)
        env = GridSearchEnv(sc)
        env.reset(seed=s, options={"scenario": sc})
        diffs += env.n_est != 10
    assert diffs > 0


def test_real_site_import_round_trip(tmp_path):  # E2, M1: real map layers
    import subprocess
    import sys
    rng = np.random.default_rng(0)
    dem = np.cumsum(rng.normal(0, 1, (800, 800)), axis=0) + np.linspace(0, 300, 800)[None]
    trails = np.zeros((800, 800), dtype=bool)
    trails[400, :] = True
    np.save(tmp_path / "dem.npy", dem)
    np.save(tmp_path / "canopy.npy", rng.uniform(0, 100, (800, 800)))
    np.save(tmp_path / "trails.npy", trails)
    out = tmp_path / "site.npz"
    subprocess.run([sys.executable, "scripts/import_site.py", "--dem", str(tmp_path / "dem.npy"), "--cell-m", "10",
                    "--canopy", str(tmp_path / "canopy.npy"), "--trails", str(tmp_path / "trails.npy"),
                    "--ipp", "400", "400", "--cell-out", "100", "--out", str(out)], check=True, capture_output=True)
    env = FieldEnv()
    sc = _sc("wilderness", 11, site=str(out), cell_m=100.0)
    env.reset(seed=11, options={"scenario": sc})
    assert env.w.trail.sum() > 0 and env.w.g == 64 and env.fam.cell_m == 100.0
    _random_steps(env, 20)


def test_isrid_distance_model_matches_published_quartiles():  # P5
    from swarmsearch.field import lpb
    q = lpb.CATEGORIES["hiker"]["dist_km"]
    assert np.allclose(lpb.quantile_cdf(np.array(q), q), lpb.Q)
    rng = np.random.default_rng(0)
    d = np.array([lpb.sample_distance(rng, q) for _ in range(20000)])
    assert np.allclose(np.quantile(d, lpb.Q), q, rtol=0.06)


def test_search_area_sized_to_the_category():  # E1: 75% distance at the map edge
    from swarmsearch.field import lpb
    for cat in ("hiker", "dementia"):
        fam = _sc("wilderness", 0, category=cat, cell_m=0.0).fam
        q75 = lpb.CATEGORIES[cat]["dist_km"][2] * 1000
        assert abs(fam.cell_m * fam.grid / 2 - q75) / q75 < 0.05 or fam.cell_m in (50.0, 200.0)


def _grown_env(seed=1):
    from dataclasses import replace
    from swarmsearch.evaluate import FIELD_CONDITIONS
    sc = replace(FIELD_CONDITIONS["wild_mix"](seed), expand=True, expand_at=9.0)
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    return env


def test_expand_keeps_probability_and_positions():
    env = _grown_env()
    pol = BeliefGreedy()
    pol.reset(env)
    for _ in range(20):
        env.step(pol.act(env))
    v = env.views["base"]
    mass0 = v.B.sum(axis=(1, 2)) + v.out
    g0, (ox, oy) = env.grid_size, env.w.extra["origin"]
    pos0 = {a: env.pos[a].copy() for a in env.agents}
    inside0 = ~env.people.outside
    ppl0 = env.people.pos.copy()
    env._expand()
    v = env.views["base"]
    assert env.grid_size == 2 * g0 and env.grid_epoch == 1
    np.testing.assert_allclose(v.B.sum(axis=(1, 2)) + v.out, mass0, rtol=1e-9)
    for a in pos0:
        assert tuple(env.pos[a]) == (pos0[a][0] + ox, pos0[a][1] + oy)
    assert (env.people.pos[inside0] == ppl0[inside0] + [ox, oy]).all()
    assert v.B.shape[-1] == env.grid_size and env.G_true.shape[-3] == env.grid_size


def test_expand_mid_mission_runs_to_the_end():
    env = _grown_env(3)
    pol = Lawnmower()
    pol.reset(env)
    while env.agents:
        if env.t == 10:
            env._expand()
        obs, *_ = env.step(pol.act(env))
    assert env.grid_size == 128 and env.metrics()["expanded_min"] is not None


def test_people_outside_come_back_when_the_area_grows():
    from dataclasses import replace
    from swarmsearch.evaluate import FIELD_CONDITIONS
    for seed in range(60):
        sc = replace(FIELD_CONDITIONS["wild_mix"](seed), expand=True)
        env = FieldEnv()
        env.reset(seed=seed, options={"scenario": sc})
        G, o = env.w.extra["big"].g, np.array(env.w.extra["origin"])
        f = env.people.far + o
        back = env.people.outside & np.isfinite(f[:, 0]) & ((f >= 0) & (f < G)).all(1)
        if back.any():
            env._expand()
            assert not env.people.outside[back].any()
            for x, y in env.people.pos[back]:
                assert env.w.walk[y, x]
            return
    raise AssertionError("no seed had someone outside who the grown area covers")


def _evidence_env(seed=1, **kw):
    from dataclasses import replace
    from swarmsearch.evaluate import FIELD_CONDITIONS
    sc = replace(FIELD_CONDITIONS["wild_mix"](seed), **kw)
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    return env


def test_clue_moves_probability_towards_it_and_keeps_mass():
    env = _evidence_env()
    v = env.views["base"]
    mass0 = (v.B.sum(axis=(1, 2)) + v.out).copy()
    g = env.grid_size
    cell = (g // 4, g // 4)
    near0 = env.belief[cell[1], cell[0]]
    env._use_clue(v, cell)
    np.testing.assert_allclose(v.B.sum(axis=(1, 2)) + v.out, mass0, rtol=1e-9)
    env._v = v
    assert env.belief[cell[1], cell[0]] > near0


def test_ground_team_clears_the_cells_it_walks():
    env = _evidence_env()
    v = env.views["base"]
    x, y = (int(c) for c in env.teams[0]["pos"])
    b0 = v.B[:, y, x].copy()
    env._team_looks([(x, y)])
    assert (v.B[:, y, x] < b0 + 1e-15).all() and (v.B[:, y, x] < b0).any()


def test_others_raise_alerts_the_operator_can_dismiss():
    env = _evidence_env(3, n_walkers=6, n_animals=8)
    pol = BeliefGreedy()
    pol.reset(env)
    while env.agents:
        env.step(pol.act(env))
    assert env.metrics()["other_alarms"] > 0


def test_without_evidence_settings_nothing_new_happens():
    env = _evidence_env(2, clue_rate=0.0, p_report=0.0, n_walkers=0, n_animals=0)
    pol = BeliefGreedy()
    pol.reset(env)
    while env.agents:
        env.step(pol.act(env))
    m = env.metrics()
    assert len(env.clues.pos) == 0 and m["clues_seen"] == 0 and m["reports"] == 0
    assert m["other_alarms"] <= sum(1 for _ in env.teams) * 1000  # only the team's own searchers can appear
