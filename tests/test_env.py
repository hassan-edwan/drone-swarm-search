import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from swarmsearch.baselines import Lawnmower, RandomWalk
from swarmsearch.env import HOVER, GridSearchEnv, obs_dim, state_dim
from swarmsearch.evaluate import run_episode
from swarmsearch.scenario import DEFAULT_PROFILE, MovementProfile, Scenario, ScenarioSampler


def test_parallel_api():
    parallel_api_test(GridSearchEnv(seed=0), num_cycles=200)


def test_shapes():
    env = GridSearchEnv(Scenario(n_drones=4), seed=0)
    obs, infos = env.reset(seed=0)
    assert len(obs) == 4
    for a, o in obs.items():
        assert o.shape == (obs_dim(),)
        assert infos[a]["action_mask"][HOVER] == 1
    assert env.state().shape == (state_dim(20),)


def test_default_profile_when_unknown():
    sc = Scenario(movement=MovementProfile("drift", 0.4, (1, 0)), profile_known=False)
    assert sc.observed_profile() == DEFAULT_PROFILE
    assert Scenario(movement=sc.movement, profile_known=True).observed_profile() == sc.movement


def test_obstacles_block_moves():
    env = GridSearchEnv(Scenario(obstacle_frac=0.3), seed=1)
    env.reset(seed=1)
    assert 0.25 < env.obstacles.mean() < 0.35
    for _ in range(50):
        env.step(RandomWalk(0).act(env))
        for a in env.agents:
            x, y = env.pos[a]
            assert not env.obstacles[y, x]


def test_still_cells_stay_fresh_moving_cells_go_stale():
    still = GridSearchEnv(Scenario(movement=MovementProfile("still", 0.0), profile_known=True), seed=0)
    still.reset(seed=0)
    moving = GridSearchEnv(Scenario(movement=MovementProfile("wander", 0.5), profile_known=True), seed=0)
    moving.reset(seed=0)
    for _ in range(40):
        still.step({a: HOVER for a in still.agents})
        moving.step({a: HOVER for a in moving.agents})
    x, y = still.starts[0]
    assert still.freshness()[y, x] == 1.0
    # A cell searched at t=0 and not since has gone stale.
    moving.last_searched[0, 0] = 0
    assert moving.freshness()[0, 0] < 0.2


def test_survivors_stay_on_free_cells():
    env = GridSearchEnv(Scenario(obstacle_frac=0.2, movement=MovementProfile("wander", 0.5)), seed=3)
    env.reset(seed=3)
    for _ in range(100):
        if not env.agents:
            break
        env.step({a: HOVER for a in env.agents})
        for x, y in env.survivors:
            assert env.free[y, x]


def test_drone_failure_removes_one_agent():
    env = GridSearchEnv(Scenario(drone_failure=True, n_survivors=20), seed=2)
    env.reset(seed=2)
    n0 = len(env.agents)
    while env.agents and env.t < env.fail_step:
        env.step({a: HOVER for a in env.agents})
    assert len(env.agents) == n0 - 1


@pytest.mark.parametrize("policy", [Lawnmower(), RandomWalk(0)])
def test_baselines_run(policy):
    env = GridSearchEnv(Scenario(), seed=0)
    m = run_episode(env, policy, Scenario(obstacle_frac=0.1), seed=0)
    assert 0.0 <= m["found_frac"] <= 1.0
    assert m["steps"] <= 200


def test_lawnmower_covers_open_grid():
    env = GridSearchEnv(Scenario(), seed=0)
    m = run_episode(env, Lawnmower(), Scenario(n_survivors=20, detection_prob=1.0), seed=0)
    assert m["coverage"] > 0.95 and m["collisions"] == 0


def test_sampler_ranges():
    s = ScenarioSampler()
    rng = np.random.default_rng(0)
    for _ in range(200):
        sc = s.sample(rng)
        assert 2 <= sc.n_drones <= 8 and 1 <= sc.n_survivors <= 20 and 0 <= sc.n_zones <= 5


def test_full_map_obs_and_net():
    import torch

    from swarmsearch.env import map_dim
    from swarmsearch.mappo import ActorCritic, LearnedPolicy

    env = GridSearchEnv(Scenario(n_drones=3), seed=0, full_map=True)
    obs, _ = env.reset(seed=0)
    o = next(iter(obs.values()))
    assert o.shape == (obs_dim() + map_dim(20),)
    for arch in ("cnn", "transformer"):
        for central in (True, False):
            net = ActorCritic(20, central, 64, arch=arch)
            d = net.dist(torch.from_numpy(o[None]), torch.ones(1, 9))
            assert d.probs.shape == (1, 9)
            if not central:
                assert net.value(torch.from_numpy(o[None])).shape == (1,)
    # A full-map policy runs on a plain env (evaluation builds one env for all methods).
    m = run_episode(GridSearchEnv(Scenario()), LearnedPolicy(net), Scenario(), seed=0)
    assert 0.0 <= m["found_frac"] <= 1.0


def test_transformer_patch_tokens_keep_cell_positions():
    """Each map cell lands in the right patch token: marking one cell changes
    only the token that covers it."""
    import torch

    from swarmsearch.env import MAP_CHANNELS
    from swarmsearch.mappo import MapTransformer

    net = MapTransformer(20, 9, patch=4)
    x = torch.zeros(1, obs_dim() + MAP_CHANNELS * 400)
    toks0 = None
    captured = {}
    net.patch_embed.register_forward_hook(lambda m, i, o: captured.__setitem__("in", i[0]))
    net(x)
    toks0 = captured["in"].clone()
    cell_y, cell_x = 6, 13  # tile row 1, tile col 3 -> token 1 * 5 + 3 = 8
    x[0, obs_dim() + 3 * 400 + cell_y * 20 + cell_x] = 1.0  # channel 3 (self)
    net(x)
    changed = (captured["in"] != toks0).any(-1)[0].nonzero().flatten().tolist()
    assert changed == [8]


def test_heuristic_guide_assigns_distinct_regions():
    from swarmsearch.guide import HeuristicGuide

    env = GridSearchEnv(Scenario(n_drones=4), seed=0)
    env.reset(seed=0)
    plan = HeuristicGuide().plan_one(env)
    assert set(plan) == set(env.agents) and len(set(plan.values())) == len(plan)


def test_guide_shaping_rewards_moving_toward_goal():
    from swarmsearch.env import MOVES

    def reward_for(move):
        env = GridSearchEnv(Scenario(n_drones=2, n_survivors=1), seed=0, guided=True)
        env.reset(seed=0)
        x, y = env.pos["drone_0"]
        env.set_goals({"drone_0": (19, int(y)), "drone_1": tuple(env.pos["drone_1"])})
        env.guide_coef = 1.0
        _, r, *_ = env.step({"drone_0": move, "drone_1": HOVER})
        return r["drone_0"]

    right = int(np.flatnonzero((MOVES == [1, 0]).all(1))[0])
    assert reward_for(right) > reward_for(HOVER)
    env = GridSearchEnv(Scenario(), seed=0, guided=True)
    obs, _ = env.reset(seed=0)
    assert next(iter(obs.values())).shape == (obs_dim() + 3,)


def test_shield_prevents_all_collisions():
    for seed in range(5):
        env = GridSearchEnv(Scenario(n_drones=8, obstacle_frac=0.2), seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        while env.agents:
            env.step({a: int(rng.integers(9)) for a in env.agents})  # ignores the mask on purpose
            cells = [tuple(env.pos[a]) for a in env.agents if env.alive[a]]
            assert len(cells) == len(set(cells))
        assert env.metrics()["collisions"] == 0


def test_shield_blocks_head_on_swap():
    from swarmsearch.env import MOVES

    env = GridSearchEnv(Scenario(n_drones=2, n_survivors=1), seed=0)
    env.reset(seed=0)
    env.pos["drone_0"], env.pos["drone_1"] = np.array([5, 5]), np.array([7, 5])
    right = int(np.flatnonzero((MOVES == [1, 0]).all(1))[0])
    left = int(np.flatnonzero((MOVES == [-1, 0]).all(1))[0])
    env.step({"drone_0": right, "drone_1": left})  # both claim (6, 5): only one gets it
    assert tuple(env.pos["drone_0"]) != tuple(env.pos["drone_1"])
    a, b = sorted(env.agents, key=lambda d: env.pos[d][0])
    env.pos[a], env.pos[b] = np.array([5, 5]), np.array([6, 5])
    env.step({a: right, b: left})  # head-on swap: both cancelled
    assert tuple(env.pos[a]) == (5, 5) and tuple(env.pos[b]) == (6, 5)


def test_sampler_mostly_no_zones():
    s = ScenarioSampler()
    rng = np.random.default_rng(0)
    frac = np.mean([s.sample(rng).n_zones == 0 for _ in range(2000)])
    assert 0.55 < frac < 0.65


@pytest.mark.parametrize("prof", [
    MovementProfile("wander", 1.0),
    MovementProfile("drift", 1.5, (1, 0)),
    MovementProfile("to_safety", 1.0, target=(15, 4)),
])
def test_belief_tracks_true_survivor_movement(prof):
    """With the true profile known, the belief map is the exact expected
    distribution of survivors: check against Monte Carlo."""
    sc = Scenario(n_drones=2, n_survivors=20, n_zones=0, movement=prof, profile_known=True,
                  obstacle_frac=0.1, detection_prob=0.6)
    env = GridSearchEnv(sc, seed=0)
    env.reset(seed=0)
    assert np.allclose(env._W.sum(1), 1.0) and len(env._w) == 1
    g = env.grid_size
    start = np.zeros(g * g)
    x0, y0 = 4, 10
    assert env.free[y0, x0]
    start[y0 * g + x0] = 1.0
    env._B = start.reshape(1, g, g).copy()
    for _ in range(30):
        env._spread_belief()
    counts = np.zeros(g * g)
    env.found[:] = False
    for _ in range(300):
        env.survivors[:] = (x0, y0)
        for _ in range(30):
            env._move_survivors()
        for x, y in env.survivors:
            counts[y * g + x] += 1
    emp = counts / counts.sum()
    assert 0.5 * np.abs(emp - env.belief.ravel()).sum() < 0.08  # total variation distance


def test_to_safety_survivors_end_at_safe_point():
    sc = Scenario(n_survivors=10, n_zones=0, movement=MovementProfile("to_safety", 1.5, target=(15, 4)))
    env = GridSearchEnv(sc, seed=1)
    env.reset(seed=1)
    d0 = np.abs(env.survivors - [15, 4]).max(1).mean()
    for _ in range(150):
        env._move_survivors()
    assert np.abs(env.survivors - [15, 4]).max(1).mean() < d0 / 3


def test_reward_is_detected_belief_mass():
    env = GridSearchEnv(Scenario(n_drones=2, n_survivors=1, n_zones=0, detection_prob=0.8), seed=0)
    env.reset(seed=0)
    x, y = env.pos["drone_0"]
    env._B[:, y, x] = 0.01
    env._mix()
    r, _ = env._search("drone_0")
    assert np.isclose(r, 0.01 * 0.8 * env.n_free) and np.isclose(env.belief[y, x], 0.002)


def test_unknown_profile_is_inferred_from_finds():
    """Without a profile, finds and empty searches shift the posterior toward
    the true movement type."""
    from swarmsearch.scenario import MOVEMENTS
    prof = MovementProfile("drift", 1.5, (1, 0))
    sc = Scenario(n_drones=6, n_survivors=20, n_zones=0, movement=prof, profile_known=False)
    env = GridSearchEnv(sc, seed=3)
    env.reset(seed=3)
    kinds = np.array([MOVEMENTS.index(k) for k in ("still", "wander", "drift", "to_safety")])
    p0 = env.profile_posterior() @ env._V[:, :4]
    rng = np.random.default_rng(0)
    while env.agents:
        env.step({a: int(rng.integers(9)) for a in env.agents})
    p1 = env.profile_posterior() @ env._V[:, :4]
    assert p1[kinds[2]] > 0.5 > p0[kinds[2]]
    assert np.isclose(env._w.sum(), 1.0) and np.isclose(p1.sum(), 1.0)


def test_mixture_belief_is_weighted_hypotheses():
    env = GridSearchEnv(Scenario(n_zones=0, profile_known=False), seed=0)
    env.reset(seed=0)
    for _ in range(5):
        env.step({a: 4 for a in env.agents})
    assert np.allclose(env.belief, np.tensordot(env._w, env._B, axes=1))
    assert len(env._w) > 50


# ---------------------------------------------------------------- M2 obstacles
@pytest.mark.parametrize("layout", ["blocks", "scatter"])
def test_obstacle_layouts_leave_every_free_cell_reachable(layout):
    from swarmsearch.obstacles import reachable
    for seed in range(10):
        sc = Scenario(obstacle_frac=0.3, obstacle_layout=layout,
                      movement=MovementProfile("to_safety", 1.0, target=(12, 7)))
        env = GridSearchEnv(sc, seed=seed)
        env.reset(seed=seed)
        assert 0.25 < env.obstacles.mean() < 0.4
        assert reachable(env.free, env.starts[0])[env.free].all()
        assert env.free[7, 12] and all(env.free[y, x] for x, y in env.starts)
        assert all(env.free[y, x] for x, y in env.survivors)


def test_distance_field_goes_around_wall():
    from swarmsearch.obstacles import distance_fields
    free = np.ones((10, 10), dtype=bool)
    free[1:10, 5] = False  # wall with a gap at the top
    d = distance_fields(free, [(8, 8)])[0]
    assert d[8, 2] == 16  # 8 moves up to the gap at (5, 0), 8 back down
    assert np.isinf(d[5, 5])


def test_to_safety_survivors_path_around_obstacles():
    """Survivors heading to a safe point behind a wall walk round it instead
    of piling up against it."""
    from swarmsearch.belief import move_weights
    g = 20
    free = np.ones((g, g), dtype=bool)
    free[2:g, 10] = False
    prof = MovementProfile("to_safety", 1.5, target=(15, 15))
    from swarmsearch.obstacles import distance_fields
    W = move_weights(prof, free, dist=distance_fields(free, [(15, 15)])[0])
    assert np.allclose(W.sum(0), 1.0)
    B = np.zeros((1, g, g))
    B[0, 15, 3] = 1.0
    Wk = W[None]
    from swarmsearch.belief import propagate
    for _ in range(1500):
        B = propagate(B, Wk)
    assert B[0, :, 11:].sum() > 0.9  # made it past the wall
    # Without path-following they get stuck on the near side.
    W0 = move_weights(prof, free)
    B0 = np.zeros((1, g, g))
    B0[0, 15, 3] = 1.0
    for _ in range(1500):
        B0 = propagate(B0, W0[None])
    assert B0[0, :, 11:].sum() < 0.5


def test_path_following_matches_straight_line_on_open_map():
    from swarmsearch.belief import move_weights
    from swarmsearch.obstacles import distance_fields
    free = np.ones((20, 20), dtype=bool)
    for target in [(3, 17), (10, 10), (19, 0)]:
        prof = MovementProfile("to_safety", 1.0, target=target)
        assert np.allclose(move_weights(prof, free), move_weights(prof, free, dist=distance_fields(free, [target])[0]))


def test_hypotheses_skip_safe_points_in_obstacles():
    from swarmsearch.belief import hypothesis_bank
    free = np.ones((20, 20), dtype=bool)
    free[:, :10] = False
    profs, prior = hypothesis_bank(20, free)
    assert all(free[p.target[1], p.target[0]] for p in profs if p.kind == "to_safety")
    assert np.isclose(prior.sum(), 1.0)


# ------------------------------------------------------------------- M3 3D
def test_3d_without_disturbance_matches_2d():
    """Point-mass flight with no wind reaches every cell in time, so the 3D
    mission is the 2D mission."""
    from swarmsearch.sim3d import Swarm3D, run_episode_3d
    sc = Scenario(n_drones=4, movement=MovementProfile("drift", 1.0, (0, 1)), n_zones=0,
                  obstacle_frac=0.2, max_steps=60)
    m2 = run_episode(GridSearchEnv(sc), Lawnmower(), sc, seed=1)
    sim = Swarm3D(GridSearchEnv(sc), backend="pointmass")
    m3 = run_episode_3d(sim, Lawnmower(), sc, seed=1)
    assert m3["late_frac"] == 0.0 and m3["near_misses"] == 0 and m3["track_err_m"] < 0.2
    for k in ("found_frac", "coverage", "time_to_90", "steps"):
        assert m2[k] == m3[k]


def test_3d_late_drones_do_not_search():
    from swarmsearch.sim3d import Swarm3D, run_episode_3d
    sc = Scenario(n_drones=3, n_zones=0, max_steps=60)
    sim = Swarm3D(GridSearchEnv(sc), backend="pointmass", wind=6.0)
    m = run_episode_3d(sim, Lawnmower(), sc, seed=0)
    assert m["late_frac"] > 0.05
    assert sim.env.n_visits <= 3 + (1 - m["late_frac"]) * 3 * 60 + 1e-9


def test_env_targets_is_pure():
    env = GridSearchEnv(Scenario(n_drones=8), seed=0)
    env.reset(seed=0)
    acts = {a: 4 for a in env.agents}
    before = {a: env.pos[a].copy() for a in env.agents}
    n = env.n_shield
    t1 = env.targets(acts)
    assert env.n_shield == n and all((env.pos[a] == before[a]).all() for a in env.agents)
    env.step(acts)
    assert all((env.pos[a] == t1[a]).all() for a in t1)


# --------------------------------------------------------------- M4 buildings
def test_no_corner_cutting_masks_diagonals_past_blocked_cells():
    env = GridSearchEnv(Scenario(n_drones=2, corner_cutting=False), seed=0)
    env.reset(seed=0)
    a = "drone_0"
    env.pos[a] = np.array([5, 5])
    env.pos["drone_1"] = np.array([15, 15])
    env.obstacles[:] = False
    env.obstacles[5, 6] = True  # cell east of the drone
    env.free = ~env.obstacles
    m = env.action_mask(a)
    # (1, -1) and (1, 1) pass the blocked east cell's corner; (-1, *) do not.
    assert m[2] == 0 and m[7] == 0 and m[0] == 1 and m[5] == 1 and m[4] == 0


def test_m4_lawnmower_flies_round_buildings_without_touching():
    from swarmsearch.sim3d import Swarm3D, run_episode_3d
    sc = Scenario(n_drones=4, n_zones=0, obstacle_frac=0.25, corner_cutting=False, max_steps=80)
    sim = Swarm3D(GridSearchEnv(sc), backend="pointmass", physical_obstacles=True)
    m = run_episode_3d(sim, Lawnmower(), sc, seed=2)
    assert sim.boxes and m["obstacle_hits"] == 0 and m["crashed"] == 0 and m["late_frac"] == 0.0
    assert m["coverage"] > 0.2


def test_m4_corner_cut_would_hit_building():
    """Sanity check of the hit detector: a diagonal past a building's corner touches it."""
    from swarmsearch.sim3d import Swarm3D
    sc = Scenario(n_drones=1, n_zones=0)
    sim = Swarm3D(GridSearchEnv(sc), backend="pointmass", physical_obstacles=True)
    sim.reset(sc, 0)
    sim._box_arr = np.array([[5.5, 5.5, 0.5, 0.5, 3.0]])
    corner = np.array([[6.0, 6.0, 1.0]])
    assert sim._touching_obstacle(corner).all()
    assert not sim._touching_obstacle(np.array([[6.5, 6.5, 1.0]])).any()


def test_wind_pushes_drones_off_their_cells():
    from swarmsearch.sim3d import Swarm3D, run_episode_3d
    sc = Scenario(n_drones=3, n_zones=0, max_steps=40)
    calm = run_episode_3d(Swarm3D(GridSearchEnv(sc), backend="pointmass"), Lawnmower(), sc, seed=0)
    windy = run_episode_3d(Swarm3D(GridSearchEnv(sc), backend="pointmass", wind=3.0, wind_mean=2.0),
                           Lawnmower(), sc, seed=0)
    assert windy["track_err_m"] > 3 * calm["track_err_m"]


def test_find_time_metric_counts_unfound_as_budget():
    from swarmsearch.scenario import Scenario
    env = GridSearchEnv(Scenario(n_survivors=4, max_steps=10))
    env.reset(seed=0)
    env.found_times = [2, 4]
    m = env.metrics()
    assert m["find_time"] == (2 + 4 + 10 + 10) / 4
    assert abs(m["find_time_frac"] - m["find_time"] / 10) < 1e-12


def test_belief_cache_separates_grid_sizes():
    from swarmsearch.scenario import Scenario
    for g in (20, 30):
        env = GridSearchEnv(Scenario(grid_size=g))
        env.reset(seed=0)
        assert env._W.shape[-1] == g


def test_belief_greedy_and_lookahead_run_on_any_size():
    import torch
    from swarmsearch.baselines import BeliefGreedy
    from swarmsearch.evaluate import M5_CONDITIONS, run_episode
    from swarmsearch.mappo import ActorCritic, LearnedPolicy
    net = ActorCritic(20, True)
    for cond in ("g30_still", "2_drones_drift_unknown"):
        sc = M5_CONDITIONS[cond](0)
        for pol in (BeliefGreedy(), LearnedPolicy(net, lookahead=3), LearnedPolicy(net, temperature=0.5)):
            m = run_episode(GridSearchEnv(sc), pol, sc, seed=0)
            assert m["steps"] <= sc.max_steps and m["collisions"] == 0


def test_belief_greedy_beats_random_on_still_survivors():
    from swarmsearch.baselines import BeliefGreedy, RandomWalk
    from swarmsearch.evaluate import CONDITIONS, run_episode
    sc = CONDITIONS["still"](0)
    bg = np.mean([run_episode(GridSearchEnv(sc), BeliefGreedy(), sc, seed=s)["find_time"] for s in range(3)])
    rw = np.mean([run_episode(GridSearchEnv(sc), RandomWalk(0), sc, seed=s)["find_time"] for s in range(3)])
    assert bg < rw


def test_path_value_prefers_mass_and_discounts_revisits():
    from swarmsearch.lookahead import path_value
    gain = np.zeros((5, 5)); gain[0, 4] = 1.0
    free = np.ones((5, 5), bool)
    assert path_value(gain, free, (2, 2), 3, 0.8) == 1.0  # reaches the mass in 2 moves
    assert path_value(gain, free, (0, 4), 3, 0.8) == 0.0  # too far
    assert abs(path_value(gain, free, (4, 0), 3, 0.8) - 1.24) < 1e-12  # search there three times: 1 + 0.2 + 0.04


def test_imitation_collect_and_fit():
    from swarmsearch.imitation import collect, fit
    from swarmsearch.mappo import ActorCritic
    sampler = ScenarioSampler(obstacle_frac=(0.0, 0.2), max_steps=30)
    O, M, A, ms = collect(sampler, 2, seed=0, workers=1)
    assert len(O) == len(M) == len(A) and len(ms) == 2
    assert all(M[i, A[i]] for i in range(len(A)))  # expert labels are valid moves
    net = ActorCritic(20, True)
    acc = fit(net, O, M, A, epochs=2, log=lambda *_: None)
    assert 0.0 <= acc <= 1.0
    O2, _, A2, _ = collect(sampler, 1, seed=1, net=net, beta=0.0, workers=1)
    assert len(O2) == len(A2) > 0


def test_team_layers():
    from swarmsearch.env import TEAM_DIM, obs_dim
    from swarmsearch.scenario import Scenario
    env = GridSearchEnv(Scenario(n_drones=4), seed=0, team=True)
    env.reset(seed=0)
    a, b = env.agents[0], env.agents[1]
    assert env._obs(a).shape == (obs_dim() + TEAM_DIM,)
    assert env._obs(a, team=False).shape == (obs_dim(),)
    # b flies east for 6 steps: its trail fades behind it, intent points east.
    env.pos[b] = np.array([5, 10])
    env.history[b] = [env.pos[b].copy()]
    for _ in range(6):
        acts = {x: 8 for x in env.agents}
        env.pos[b] = env.pos[b] + [1, 0]
        env.t += 1
        env.trail_t[b][env.pos[b][1], env.pos[b][0]] = env.t
        env.history[b] = (env.history[b] + [env.pos[b].copy()])[-6:]
    trail, intent = env.team_maps(a)
    assert trail[10, 11] == 1.0 and 0 < trail[10, 6] < trail[10, 10] < 1.0
    assert intent[10, 12] == 1.0 and intent[10, 16] > 0 and intent[10, 17] == 0
    assert env.team_maps(b)[0][10, 11] < 1.0  # own trail excluded
    for _ in range(3):
        env.step({x: 8 for x in env.agents})
    assert np.isfinite(env._obs(a)).all()
