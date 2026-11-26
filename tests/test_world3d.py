"""Map vs real world, and flying a map-planned route through the real world with and without sensing."""

import numpy as np

from swarmsearch import flight3d as f3
from swarmsearch import fly_exec as fx
from swarmsearch import world3d as w3


def _worlds():
    n = 160
    occ = np.zeros((n, n, w3.NZ), np.uint8)
    occ[20:150, 20:150, 8:22] = w3.VEG  # a forest canopy from 8 to 22 m
    occ[75:89, 30:44, :] = 0  # a clearing to descend through (14 m across)
    M = w3.World3D(occ, np.zeros((n, n)), 0.0, 0.0, season="leaf-off")
    T = M.copy()
    return M, T


def _route(M):
    P = f3.VoxPlanner(M, None)
    start, target = M.xy(82, 10), M.xy(82, 110)
    tr, info = P.under_check(start, target, 40.0)
    assert tr is not None and info["close_enough"], info
    return P, tr, start, target


def test_planned_route_is_clear_on_the_map():
    M, _ = _worlds()
    P, tr, *_ = _route(M)
    c = f3.clearance_world(M, tr)
    assert min(v for k, v in c.items() if k != "ground") >= f3.SAFETY_M - 1.0  # voxel resolution
    assert c["ground"] > 1.0


def test_sensing_avoids_what_the_map_lacks():
    M, T = _worlds()
    P, tr, start, target = _route(M)
    # the real world has a sapling right on the route under the canopy (not on the map)
    under = [q for q in tr if q[4] == "under"]
    x, y = under[len(under) // 3][1], under[len(under) // 3][2]
    i, j = T.ij(x, y)
    T.occ[i - 1:i + 2, j - 1:j + 2, 0:6] |= w3.VEG
    factory = lambda extra: P if extra is None else f3.VoxPlanner(M, None, known_extra=extra)  # noqa: E731
    blind = fx.Executor(M, T, factory, nav="vio", sensing=False, seed=0).fly_check(start, target, 40.0)
    assert blind.outcome == "collided" and blind.hit_kind == "foliage"
    seeing = fx.Executor(M, T, factory, nav="vio", sensing=True, seed=0).fly_check(start, target, 40.0)
    assert seeing.outcome in ("ok", "aborted") and seeing.replans >= 1
    assert seeing.detected.get("foliage", 0) > 0


def test_truth_adds_unmapped_things():
    M, _ = _worlds()
    T = w3.make_truth(M, {}, [], seed=0)
    assert ((T.occ > 0) & (M.occ == 0)).sum() > 0
    assert any(s["kind"] in ("saplings", "dead branches at flying height") for s in T.surprises)
