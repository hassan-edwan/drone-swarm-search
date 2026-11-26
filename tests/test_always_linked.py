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
