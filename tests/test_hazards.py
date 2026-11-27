"""Generated-world obstacles and conditions (swarmsearch/field/hazards.py)."""

from dataclasses import replace

import numpy as np
import pytest

from swarmsearch.field.params import FieldSampler
from swarmsearch.field.terrain import make_world


def _sc(family, k=0, **kw):
    sc = FieldSampler(families={family: 1.0}).sample(np.random.default_rng(k))
    return replace(sc, **kw)


@pytest.mark.parametrize("family", ["wilderness", "urban", "flood"])
def test_layers_present_and_consistent(family):
    sc = _sc(family)
    w = make_world(np.random.default_rng(sc.seed_world), sc)
    g = w.g
    for k in ("min_level", "land_dist", "top_m", "tree_h"):
        assert w.extra[k].shape == (g, g)
    assert w.extra["wind_mult"].shape == (8, g, g)
    assert np.isfinite(w.extra["wind_mult"]).all() and (w.extra["wind_mult"] > 0).all()
    lv = w.extra["min_level"]
    assert lv.min() >= 0 and lv.max() <= 3
    assert w.fly_block[lv >= 3].all()  # no-fly cells are blocked
    # the lowest allowed height clears the highest obstacle by the margin
    alt = np.array(sc.fam.alt_m)
    ok = lv < 3
    if family == "urban":
        ok &= ~w.fly_block
    assert (alt[lv[ok]] >= w.extra["top_m"][ok] + 5.0).all()


@pytest.mark.parametrize("family", ["wilderness", "urban", "flood"])
def test_off_gives_the_old_world_and_on_keeps_its_layers(family):
    sc = _sc(family, 3)
    on = make_world(np.random.default_rng(sc.seed_world), sc)
    off = make_world(np.random.default_rng(sc.seed_world), replace(sc, hazards=False))
    assert "min_level" not in off.extra and "wind_mult" not in off.extra
    assert np.array_equal(on.elev, off.elev) and np.array_equal(on.walk, off.walk)
    assert on.ipp == off.ipp and on.base == off.base
    again = make_world(np.random.default_rng(sc.seed_world), sc)
    assert np.array_equal(on.extra["min_level"], again.extra["min_level"])
    assert np.array_equal(on.extra["land_dist"], again.extra["land_dist"])


def test_growing_area_keeps_the_limits():
    from swarmsearch.field.env import FieldEnv
    sc = _sc("wilderness", 5, expand=True, expand_at=0.0, expand_min=0.01, mission_s=1800.0)
    env = FieldEnv(sc)
    env.reset(seed=0, options={"scenario": sc})
    big = env.w.extra["big"]
    env._expand()
    assert env.min_level.shape == (big.g, big.g)
    assert np.array_equal(env.min_level, big.extra["min_level"])
    assert env._wind_mult_all.shape == (8, big.g, big.g)
