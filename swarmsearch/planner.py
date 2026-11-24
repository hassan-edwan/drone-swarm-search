"""A rollout planner on the search map (field environment, camera mode).

The rule picks the target with the most expected detections per step of
travel: it looks one target ahead. The planner looks further. For each drone
that needs a target it takes the rule's best few candidates and, for each,
simulates the drone on its own copy of the map for `horizon` steps: fly
there (sprinting while far, as the rule does), then keep following the rule.
Every simulated look removes the expected detections it would make from the
map, exactly as the map's own update does for a look that finds nobody, so
the simulation needs no knowledge of where anyone really is. The candidate
whose simulation collects the most expected detections (discounted by time)
is chosen. The rule's own choice is always among the candidates, so under
the map's model the planner can only gain. This is policy improvement by
rollout of a base policy.

Simplifications: teammates' future looks are not simulated (their current
targets are already excluded, as for the rule), and the map does not spread
during the short horizon.
"""

from __future__ import annotations

import numpy as np

from . import camera as cam
from .baselines import BeliefGreedy
from .env import MOVES
from .field import sensing as sn

R = cam.R


class PlannerPolicy(BeliefGreedy):
    def __init__(self, candidates: int = 6, horizon: int = 60, gamma: float = 0.99, sep: int = 2, **kw):
        super().__init__(**kw)
        self.K, self.H, self.gamma, self.sep = candidates, horizon, gamma, sep
        self.n_changed = 0
        self.n_decisions = 0

    # -------------------------------------------------------------- the model
    def _model(self, env, a):
        """Per height: detection rate at each cell for one look (per unit of
        footprint weight), the footprint weights, and the starting map."""
        env.use_view(a)
        o = env.occ_model
        pl = np.minimum(0.98, np.clip(0.5 * env.C_model, 1e-3, 0.98) / np.maximum(1e-6, 1 - o))
        f = lambda m: o + (1 - o) * (1 - pl) ** m  # noqa: E731
        pd = []
        for k in range(cam.N_ALT):
            n = env._v.N[k * sn.N_CLASS + 4]
            disc = (1 - f(n + 1) / f(n)) / np.maximum(1 - f(1), 1e-9)
            pd.append(env.C_model * disc)
        return np.stack(pd), env._v.map.G, env.belief.copy()

    @staticmethod
    def _look(b, fv, pd, G, x, y, k, motion):
        """One look from (x, y) at height k: expected detections, then the map
        and the footprint values nearby as if nobody was seen."""
        g = b.shape[0]
        Gk = G[k][y, x]
        got = 0.0
        y0, y1, x0, x1 = max(0, y - R), min(g, y + R + 1), max(0, x - R), min(g, x + R + 1)
        w = Gk[y0 - y + R:y1 - y + R, x0 - x + R:x1 - x + R]
        q = np.minimum(0.98, pd[k][y0:y1, x0:x1] * w * motion)
        got = float((b[y0:y1, x0:x1] * q).sum())
        b[y0:y1, x0:x1] *= 1 - q
        # footprint values of cells whose footprint touches the looked area
        Y0, Y1, X0, X1 = max(0, y - 2 * R), min(g, y + 2 * R + 1), max(0, x - 2 * R), min(g, x + 2 * R + 1)
        for j in range(fv.shape[0]):
            m = np.pad(b * pd[j], R)
            acc = np.zeros((Y1 - Y0, X1 - X0))
            for oy in range(2 * R + 1):
                for ox in range(2 * R + 1):
                    acc += m[Y0 + oy:Y1 + oy, X0 + ox:X1 + ox] * G[j][Y0:Y1, X0:X1, oy, ox]
            fv[j, Y0:Y1, X0:X1] = acc
        return got

    def _pick(self, env, fv, taken, x, y, alt):
        D = self._field(env, (x, y))
        S = np.stack([np.where(taken, -1.0, fv[k] / (1.0 + D + abs(k - alt))) for k in range(cam.N_ALT)])
        S[alt, y, x] = -1.0
        S[~np.isfinite(S)] = -1.0
        k, ty, tx = np.unravel_index(int(np.argmax(S)), S.shape)
        return (int(k), int(ty), int(tx)) if S[k, ty, tx] > 0 else None

    def _rollout(self, env, a, first, V, taken, model):
        pd, G, b0 = model
        b = b0.copy()
        fv = np.stack(V).copy()
        x, y = (int(v) for v in env.pos[a])
        alt = int(env.alt[a])
        k, ty, tx = first
        v0 = fv[k, ty, tx]
        total, t = 0.0, 0
        sprint = getattr(env, "has_sprint", False)
        while t < self.H:
            if alt != k:
                alt += 1 if k > alt else -1
                total += self.gamma ** t * self._look(b, fv, pd, G, x, y, alt, 1.0)
                t += 1
                continue
            if (x, y) == (tx, ty) or fv[k, ty, tx] < self.drop * v0:
                nxt = self._pick(env, fv, taken, x, y, alt)
                if nxt is None:
                    break
                k, ty, tx = nxt
                v0 = fv[k, ty, tx]
                if alt != k:
                    continue
                if (x, y) == (tx, ty):  # hover and look again
                    total += self.gamma ** t * self._look(b, fv, pd, G, x, y, alt, 1.0)
                    t += 1
                    continue
            Df = self._field(env, (tx, ty))
            n_cells = 2 if sprint and Df[y, x] > 4 else 1
            moved = False
            for _ in range(n_cells):
                best = None
                for dx, dy in MOVES[:8]:
                    nx, ny = x + int(dx), y + int(dy)
                    if 0 <= nx < Df.shape[1] and 0 <= ny < Df.shape[0] and np.isfinite(Df[ny, nx]):
                        key = (Df[ny, nx], -fv[alt, ny, nx])
                        if best is None or key < best[0]:
                            best = (key, nx, ny)
                if best is None or best[0][0] >= Df[y, x]:
                    break
                x, y = best[1], best[2]
                moved = True
            if not moved:  # stuck (no way closer): the rule would re-pick; stop the rollout here
                break
            motion = sn.SPRINT_BLUR if n_cells == 2 else sn.BLUR
            total += self.gamma ** t * self._look(b, fv, pd, G, x, y, alt, motion)
            t += 1
        return total

    # -------------------------------------------------------------- choosing
    def candidates(self, S):
        flat = np.argsort(S, axis=None)[::-1]
        out = []
        for i in flat:
            k, ty, tx = np.unravel_index(int(i), S.shape)
            if S[k, ty, tx] <= 0:
                break
            if all(max(abs(ty - c[1]), abs(tx - c[2])) > self.sep for c in out):
                out.append((int(k), int(ty), int(tx)))
            if len(out) >= self.K:
                break
        return out

    def choose(self, env, a, S, V, taken, D):
        rule = super().choose(env, a, S, V, taken, D)
        cands = self.candidates(S)
        if rule not in cands:
            cands = [rule] + cands[: self.K - 1]
        if len(cands) <= 1:
            return rule
        model = self._model(env, a)
        vals = [self._rollout(env, a, c, V, taken, model) for c in cands]
        best = cands[int(np.argmax(vals))]
        self.n_decisions += 1
        self.n_changed += int(best != rule)
        self.last_values = dict(zip(cands, vals))
        return best
