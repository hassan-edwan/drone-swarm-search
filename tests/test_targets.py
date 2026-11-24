"""The target-choosing policy, its training, and the rollout planner."""

import numpy as np
import torch

from swarmsearch.baselines import BeliefGreedy
from swarmsearch.envs import make_env
from swarmsearch.field.params import FieldSampler
from swarmsearch.planner import PlannerPolicy
from swarmsearch import targets as tg
from swarmsearch import targets_train as tt


def _env(fam="urban", seed=0):
    env = make_env(sampler=FieldSampler(families={fam: 1.0}), seed=seed)
    env.reset(seed=seed + 40)
    return env


def _actions(policy, fam, seed, steps):
    env = _env(fam, seed)
    policy.reset(env)
    out = []
    while env.agents and len(out) < steps:
        a = policy.act(env)
        out.append(tuple(sorted(a.items())))
        env.step(a)
    return out


def test_untrained_greedy_is_the_rule():
    torch.manual_seed(0)
    net = tg.TargetNet()
    for fam in ("urban", "wilderness"):
        assert _actions(tg.TargetPolicy(net, temperature=0), fam, 1, 60) == _actions(BeliefGreedy(), fam, 1, 60)


def test_recorder_advantages():
    rec = tg.Recorder(gamma=0.5, lam=1.0, keep=1.0, rng=np.random.default_rng(0))
    x, m = np.zeros((tg.C_IN, 2, 2), np.float32), np.ones((3, 2, 2), bool)
    rec.decide("drone_0", 0, x, m, 0, -1.0, 1.0)
    rec.reward({"drone_0": 1.0}, 0)
    rec.reward({"drone_0": 1.0}, 1)  # discounted by 0.5
    rec.decide("drone_0", 2, x, m, 0, -1.0, 2.0)
    rec.reward({"drone_0": 4.0}, 2)
    s = rec.finish(3)
    assert len(s) == 2
    assert s[1]["adv"] == 4.0 - 2.0  # last decision: no value after it
    # first: R = 1 + 0.5 = 1.5, then 0.5^2 * V(next) = 0.5; delta = 2 - 1 = 1; plus 0.25 * next adv (2)
    assert abs(s[0]["adv"] - (1.0 + 0.25 * 2.0)) < 1e-9
    assert abs(s[0]["ret"] - (s[0]["adv"] + 1.0)) < 1e-9


def test_planner_look_matches_the_map_and_keeps_the_rule_choice():
    env = _env("wilderness", 2)
    p = PlannerPolicy()
    p.reset(env)
    a = env.agents[0]
    pd, G, b = p._model(env, a)
    V = np.stack([env.footprint_values(k) for k in range(3)])
    for (x, y, k) in [(5, 7, 0), (20, 30, 1), (40, 12, 2)]:
        got = p._look(b.copy(), V.copy(), pd, G, x, y, k, 1.0)
        assert abs(got - V[k, y, x]) < 1e-9 * max(1.0, V[k, y, x]) + 1e-12
    for _ in range(15):
        env.step(p.act(env))
    assert p.n_decisions > 0
    for c in getattr(p, "last_values", {}):
        assert len(c) == 3


def test_training_and_planner_imitation_run(tmp_path):
    cfg = tt.TConfig(families=("urban",), iters=1, episodes=1, workers=0, critic_warmup=0, val_every=1,
                     val_episodes=1, epochs=1, keep=0.5)
    tt.train(cfg, str(tmp_path / "rl"), log=lambda *_: None)
    net, d = tg.load(tmp_path / "rl" / "model.pt")
    assert d["kind"] == "targets" and (tmp_path / "rl" / "best.pt").exists()
    assert tg.is_targets(tmp_path / "rl" / "model.pt")
    data, m = tt.label_mission((None, cfg, 3, 1.0, 0.3, 20))
    assert data and 0 <= m["changed"] <= 1
    acc = tt.fit_labels(tg.TargetNet(), data, epochs=1, log=lambda *_: None)
    assert 0 <= acc <= 1


def test_travel_distance_plan_features():
    env = make_env(sampler=FieldSampler(families={"urban": 1.0}), seed=0, plan_feats=2)
    obs, _ = env.reset(seed=3)
    o = obs[env.agents[0]]
    from swarmsearch.field.env import PLAN_DIM, field_obs_dim
    assert o.shape == (field_obs_dim(2),)
    f = o[-PLAN_DIM:]
    assert np.all(f[:8] >= 0) and np.all(f[:8] <= 1.0 + 1e-6) and f[:8].max() > 0.99
