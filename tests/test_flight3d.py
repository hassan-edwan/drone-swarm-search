"""3D flight planning: cruise clears obstacles; under-canopy routes use a gap and keep clear of trunks and buildings."""

import numpy as np

from swarmsearch import flight3d as f3


def _site():
    n = 200
    dtm = np.add.outer(np.zeros(n), np.linspace(0, 20, n))  # slope up to the east
    top = np.zeros((n, n))
    base = np.full((n, n), np.inf)
    top[40:160, 60:200] = 20.0  # a forest block, crowns from 10 to 20 m
    base[40:160, 60:200] = 10.0
    top[95:105, 95:105] = 0.0  # a clearing inside the forest
    base[95:105, 95:105] = np.inf
    bldg = np.zeros((n, n), bool)
    bldg[60:80, 30:50] = True  # a house outside the forest
    bh = np.where(bldg, 8.0, 0.0)
    s = f3.Site3D(dtm, top, base, bldg, bh, 0.0, 0.0)
    # trunks on a 9 m grid in the forest
    s.trees = np.array([(*s.xy(i, j), 20.0, 4.0, 10.0) for i in range(44, 160, 9) for j in range(64, 200, 9)
                        if not (93 <= i <= 107 and 93 <= j <= 107)])
    return s


def test_cruise_clears_terrain_and_crowns():
    s = _site()
    P = f3.Planner(s)
    tr = P.cruise(s.xy(100, 5), s.xy(100, 195), 40.0)
    c = f3.clearance(s, tr)
    assert c["over_top"] >= f3.CLEAR_M - 1e-6
    assert c["ground"] >= 40.0 - f3.DRONES["sar"]["height"] / 2 - 1e-6


def test_under_check_goes_down_a_gap_and_keeps_clear():
    s = _site()
    P = f3.Planner(s)
    target = s.xy(130, 140)  # under the crowns, south-east of the clearing
    tr, info = P.under_check(s.xy(10, 10), target, 40.0)
    assert tr is not None, info
    modes = [q[4] for q in tr]
    assert "descend" in modes and "under" in modes and "climb" in modes
    gi, gj = s.ij(*info["gap"])
    assert s.veg_top[gi, gj] < 1.0  # descended where there is open sky
    c = f3.clearance(s, tr)
    # edge to edge: the drone's size and the trunk radius are taken off
    assert c["trunk"] >= f3.SAFETY_M - 1e-6
    assert c["under_crown"] >= f3.SAFETY_M - 1e-6
    assert c["ground"] >= f3.LOW_M - f3.DRONES["sar"]["height"] / 2 - 1e-6
    assert info["reach_m"] < f3.REACH_M


def test_wires_block_low_flight_and_raise_cruise():
    s = _site()
    H = dict(transmission=[dict(pts=[s.xy(100, 0), s.xy(100, 199)])], distribution=[], service=[],
             towers=[], bands=dict(transmission=dict(z0=8, z1=45, half_w=15)))
    s.hz_lo, s.hz_hi = f3.hazard_bands(H, s.east0, s.north0, s.n, 1.0)
    P = f3.Planner(s)
    tr = P.cruise(s.xy(10, 100), s.xy(190, 100), 40.0)  # crosses the line at 40 m: must go over it
    c = f3.clearance(s, tr)
    assert c["wire"] >= 0.0
    assert max(q[3] - s.ground(q[1], q[2]) for q in tr) >= 45.0 + f3.CLEAR_M
    # no descent through the corridor
    gi, gj = np.nonzero(P.gap)
    xs = [P._xy2(i, j) for i, j in zip(gi, gj)]
    assert all(abs(y - s.xy(100, 0)[1]) > 15 for _, y in xs)
