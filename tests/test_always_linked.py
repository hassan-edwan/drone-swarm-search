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


def test_safe_heights_make_drones_climb_and_the_rule_respects_them():
    import numpy as np
    from swarmsearch.field import env as fe
    sc = dataclasses.replace(FIELD_CONDITIONS["wild_mix"](4), n_drones=2)
    env = FieldEnv()
    env.reset(seed=3, options={"scenario": sc})
    a = env.agents[0]
    x, y = (int(v) for v in env.pos[a])
    env.min_level[:] = 0
    env.min_level[:, x + 1:] = 1  # everything east needs the middle height
    env.alt[a] = env.alt_tgt[a] = 0
    env.alt_now[a] = env.fam.alt_m[0]
    east = 4  # MOVES order: nw, n, ne, w, e, sw, s, se
    assert env.action_mask(a)[east] == 0
    env.step({b: (east if b == a else fe.HOVER) for b in env.agents})
    assert env.alt_tgt[a] >= 1 and int(env.pos[a][0]) == x  # climbed instead of entering
    pol = BeliefGreedy()
    pol.reset(env)
    S, taken, D = pol.rule_scores(env, a, [env.footprint_values(k) for k in range(3)], [a])
    assert np.all(S[0][:, x + 1:] <= 0)


def test_real_site_loads_flight_limits():
    env = FieldEnv()
    env.reset(seed=0, options={"scenario": FIELD_CONDITIONS["wild_real_hamlet"](0)})
    assert env.min_level.max() >= 1 and env.under_p is not None and (env.under_p >= 0).any()
