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
