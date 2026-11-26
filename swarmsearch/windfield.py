"""Wind shaped by the terrain and the forest, for one mean wind (speed at 10 m over open ground, direction).

- Over hills: speed-up at crests and on upwind slopes, slow-down and gustier
  air in the lee. Rule-of-thumb from the Jackson-Hunt hill theory as used in
  wind engineering: fractional speed-up at a crest about 1.6 H / L (H hill
  height, L half-width), taken here from the ground's height above the
  average of the upwind terrain within 500 m, clipped to 0.6 to 1.6 times.
- With height: logarithmic profile above the canopy (roughness 0.1 x canopy
  height, displacement 0.67 x canopy height; 0.03 m over open ground).
- Under the canopy: exponential decay to about 20% at mid-trunk (typical
  forest measurements).
- Gusts: turbulence intensity about 0.2 in the open, 0.35 in the lee of
  higher ground and over rough forest.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


class WindField:
    def __init__(self, dtm, canopy_h, cell, speed10, direction_deg, upwind_m=500.0):
        """dtm, canopy_h: (n, n) rasters (row 0 = north) of `cell` metres; direction: where the wind comes FROM
        (meteorological, degrees clockwise from north)."""
        self.cell, self.U, self.dir = cell, float(speed10), float(direction_deg)
        n = dtm.shape[0]
        th = math.radians(direction_deg)
        # the upwind direction in grid steps (rows grow southwards)
        ux, uy = math.sin(th), math.cos(th)  # towards where it comes from: east, north components
        steps = max(1, int(upwind_m / cell))
        acc = np.zeros_like(dtm, float)
        cnt = 0
        for s in range(1, steps + 1, max(1, steps // 10)):
            sh = ndimage.shift(dtm, (s * uy, -s * ux), order=1, mode="nearest")  # value s cells upwind
            acc += sh
            cnt += 1
        upwind = acc / cnt
        rel = dtm - upwind
        half_w = upwind_m
        self.mult = np.clip(1.0 + 1.6 * rel / half_w * 4.0, 0.6, 1.6)  # x4: rel is over the half-width, H/L
        self.lee = rel < -0.02 * upwind_m
        self.ti = np.where(self.lee, 0.35, 0.2) + 0.1 * np.clip(canopy_h / 30.0, 0, 1)
        self.canopy_h = canopy_h
        self.n = n

    def speed(self, i, j, h_agl):
        """Mean wind speed (m/s) at a cell and height above the ground. All surfaces share the wind at a
        blending height of 100 m (open-ground profile from the 10 m speed); below it, a log profile
        for the surface (forest: displaced and rougher), and inside the canopy an exponential decay."""
        hc = float(self.canopy_h[i, j])
        u100 = self.U * float(self.mult[i, j]) * math.log(100.0 / 0.03) / math.log(10.0 / 0.03)
        if hc > 2.0:
            z0, d = 0.1 * hc, 0.67 * hc
            prof = lambda z: math.log(max(z - d, z0 * 1.01) / z0) / math.log((100.0 - d) / z0)  # noqa: E731
            if h_agl >= hc:
                return u100 * prof(min(h_agl, 100.0))
            return u100 * prof(hc) * math.exp(-2.0 * (1.0 - h_agl / hc))
        z0 = 0.03
        return u100 * math.log(max(min(h_agl, 100.0), 0.5) / z0) / math.log(100.0 / z0)

    def gust(self, i, j, h_agl, rng):
        u = self.speed(i, j, h_agl)
        return max(0.0, u * (1.0 + float(self.ti[i, j]) * rng.normal()))

    def grid_multiplier(self, block):
        """Mean multiplier per block x block cells (for the search grid)."""
        n = self.n // block * block
        return self.mult[:n, :n].reshape(self.n // block, block, self.n // block, block).mean((1, 3))
