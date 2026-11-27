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

Options added 2026-11-27 (Phase 3 found the plain planner gains nothing):
- team: teammates in the same radio group also fly in the simulation, to
  their current targets and then by the rule, and their looks remove what
  they would find (sequential planning over the team; Dec-MCTS-like intent
  from the targets they have announced).
- objective="time": each expected detection counts by how early it comes in
  the mission (1 - t / mission length, as the find reward), instead of a
  fixed discount: minimum expected time to find.
- robust: score each candidate on this many perturbed copies of the map
  (detection rates x0.6-1.4, a smooth log-normal distortion of the map) and
  take the mean, so a choice that only pays if the map is exactly right
  loses to one that pays either way.
"""

from __future__ import annotations

import numpy as np

from . import camera as cam
from .baselines import BeliefGreedy
from .env import MOVES
from .field import sensing as sn

R = cam.R


class PlannerPolicy(BeliefGreedy):
    def __init__(self, candidates: int = 6, horizon: int = 60, gamma: float = 0.99, sep: int = 2,
                 team: bool = False, objective: str = "discount", robust: int = 0, seed: int = 0, **kw):
        super().__init__(**kw)
        self.K, self.H, self.gamma, self.sep = candidates, horizon, gamma, sep
        self.team, self.objective, self.robust = team, objective, robust
        self.prng = np.random.default_rng(seed)
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

    def _weight(self, env, t):
        if self.objective == "time":
            T = getattr(getattr(env, "fam", None), "max_steps", 0) or 1
            return max(0.0, 1.0 - (env.t + t) / T)
        return self.gamma ** t

    def _mates(self, env, a):
        """Teammates sharing a's map, with the targets they have announced."""
        if not self.team:
            return []
        out = []
        for b in env.agents:
            if b == a or not env.alive.get(b) or env.view_of(b) is not env.view_of(a) or b not in self.cam_targets:
                continue
            (tx, ty), k, v0 = self.cam_targets[b]
            out.append(dict(x=int(env.pos[b][0]), y=int(env.pos[b][1]), alt=int(env.alt[b]), k=k, tx=tx, ty=ty, v0=v0))
        return out

    def _mate_step(self, env, m, b, fv, pd, G, taken, t):
        """One step of a simulated teammate: climb/descend, else a cell towards its target, then look."""
        if m["alt"] != m["k"]:
            m["alt"] += 1 if m["k"] > m["alt"] else -1
        else:
            if (m["x"], m["y"]) == (m["tx"], m["ty"]) or fv[m["k"], m["ty"], m["tx"]] < self.drop * m["v0"]:
                nxt = self._pick(env, fv, taken, m["x"], m["y"], m["alt"])
                if nxt is None:
                    return 0.0
                m["k"], m["ty"], m["tx"] = nxt
                m["v0"] = fv[nxt]
            Df = self._field(env, (m["tx"], m["ty"]))
            best = None
            for dx, dy in MOVES[:8]:
                nx, ny = m["x"] + int(dx), m["y"] + int(dy)
                if 0 <= nx < Df.shape[1] and 0 <= ny < Df.shape[0] and np.isfinite(Df[ny, nx]):
                    if best is None or Df[ny, nx] < best[0]:
                        best = (Df[ny, nx], nx, ny)
            if best is not None and best[0] < Df[m["y"], m["x"]]:
                m["x"], m["y"] = best[1], best[2]
        return self._weight(env, t) * self._look(b, fv, pd, G, m["x"], m["y"], m["alt"], sn.BLUR)

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
        mates = [dict(m) for m in self._mates(env, a)]
        # a teammate's own claim area stays taken for it; the candidate's for us
        team_total = 0.0
        while t < self.H:
            for m in mates:  # teammates look first each step (their removals count for the team, not for a)
                team_total += self._mate_step(env, m, b, fv, pd, G, taken, t)
            if alt != k:
                alt += 1 if k > alt else -1
                total += self._weight(env, t) * self._look(b, fv, pd, G, x, y, alt, 1.0)
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
                    total += self._weight(env, t) * self._look(b, fv, pd, G, x, y, alt, 1.0)
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
            total += self._weight(env, t) * self._look(b, fv, pd, G, x, y, alt, motion)
            t += 1
        # with teammates simulated, the candidate is judged by what the whole team finds
        return total + team_total if mates else total

    def _perturb(self, model):
        """The map as it might really be: detection rates off by a common factor, and a smooth distortion."""
        pd, G, b = model
        from .field.terrain import smooth_noise as smooth_field
        s = float(np.exp(self.prng.normal(0.0, 0.25)))  # about x0.6-x1.6
        f = np.exp(0.5 * (smooth_field(self.prng, b.shape[0], 4) - 0.5) * 2.0)
        b2 = b * f
        b2 *= b.sum() / max(b2.sum(), 1e-12)
        return np.minimum(pd * s, 0.98), G, b2

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
        if self.robust > 0:
            vals = np.zeros(len(cands))
            for _ in range(self.robust):
                m2 = self._perturb(model)
                vals += np.array([self._rollout(env, a, c, V, taken, m2) for c in cands])
            vals = list(vals / self.robust)
        else:
            vals = [self._rollout(env, a, c, V, taken, model) for c in cands]
        best = cands[int(np.argmax(vals))]
        self.n_decisions += 1
        self.n_changed += int(best != rule)
        self.last_values = dict(zip(cands, vals))
        return best
