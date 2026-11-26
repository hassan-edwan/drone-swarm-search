"""Drones are always in contact by default; the radio model is opt-in."""

import dataclasses

from swarmsearch.baselines import BeliefGreedy
from swarmsearch.evaluate import FIELD_CONDITIONS
from swarmsearch.field.env import FieldEnv


def _linked_share(sc, steps=120):
    env = FieldEnv()
    env.reset(seed=1, options={"scenario": sc})
    pol = BeliefGreedy()
    pol.reset(env)
    seen = []
    for _ in range(steps):
        if not env.agents:
            break
        env.step(pol.act(env))
        up = [a for a in env.agents if env.alive[a] and env.mode[a] != "landed"]
        seen += ["base" in env.group_of.get(a, []) for a in up]
        assert not any(env.mode[a] == "link" for a in env.agents)
    return sum(seen) / max(1, len(seen))


def test_always_linked_by_default():
    sc = dataclasses.replace(FIELD_CONDITIONS["wild_mix"](3), comm_range_m=300.0, rtb_on_link_loss=True)
    assert sc.always_linked
    assert _linked_share(sc) == 1.0


def test_radio_model_still_available():
    sc = FIELD_CONDITIONS["wild_hard_radio"](3)
    assert not sc.always_linked


def test_under_canopy_check_happens_in_dense_forest():
    """An alert over dense forest is checked from under the trees, which takes time."""
    import numpy as np
    from swarmsearch.field import env as fe
    sc = dataclasses.replace(FIELD_CONDITIONS["wild_mix"](5), n_drones=2)
    env = FieldEnv()
    env.reset(seed=2, options={"scenario": sc})
    a = env.agents[0]
    env.w.map_canopy[:] = 0.9
    x, y = (int(v) for v in env.pos[a])
    env.use_view(a)
    env._v.cand_t[y, x] = env.t
    env.alt[a] = 0
    env._maybe_go_under(a, 0, x, y)
    assert env.mode[a] == "under" and env.mode_t[a] >= 1
    steps = 0
    while env.mode.get(a) == "under" and steps < 50:
        env.step({b: fe.HOVER for b in env.agents})
        steps += 1
    assert steps >= 1 and env.mode[a] != "under"
    assert env._v.cand_t[y, x] < 0
    assert env.n_under == 1
