"""A policy that chooses search targets: the rule's own scores plus a learned
correction (field environment, camera mode).

The rule (BeliefGreedy) picks, for each drone that needs a target, the
(height, cell) with the most expected detections over the footprint per step
of travel around obstacles and height change. This policy scores the same
choices with a fully convolutional network:

    logits = beta * log(rule score / best score) + correction(map)

The correction's last layer starts at zero, so the untrained policy acting
greedily (temperature 0) is the rule, move for move. Everything below the
choice of target (flying there, confirming alerts, the safety shield, battery
and radio returns) stays the rule's. Learning can only change which target
each drone picks, and starts from the rule rather than from an imperfect copy.

The network sees, per cell: the log rule score and the footprint value at
each height, the travel distance from the drone, the cells it may not pick,
its teammates, itself and the base; plus a few scalars (time, height,
battery, family, team size) broadcast over the map.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import camera as cam
from .baselines import BeliefGreedy

LOG_FLOOR = -12.0
FAMS = ("wilderness", "urban", "flood")
N_SCAL = 5 + len(FAMS)
C_IN = 2 * cam.N_ALT + 5 + N_SCAL
NEG = -1e9


def inputs(env, a, S, V, taken, D):
    """(C_IN, g, g) float32 network input and the (N_ALT, g, g) choice mask."""
    g = S.shape[1]
    smax = max(float(S.max()), 1e-12)
    logS = np.where(S > 0, np.log(np.maximum(S, 1e-300) / smax), LOG_FLOOR)
    logS = np.clip(logS, LOG_FLOOR, 0.0)
    Vs = np.stack(V)
    Vn = Vs / max(float(Vs.max()), 1e-12)
    Dn = np.where(np.isfinite(D), D / g, 2.0)
    mates = np.zeros((g, g))
    me = env.view_of(a)
    for b in env.agents:
        if b != a and env.alive.get(b) and env.view_of(b) is me:
            x, y = (int(v) for v in env.pos[b])
            mates[y, x] = 1.0
    own = np.zeros((g, g))
    x, y = (int(v) for v in env.pos[a])
    own[y, x] = 1.0
    base = np.zeros((g, g))
    bx, by = env.w.base
    base[by, bx] = 1.0
    fam = env.fam
    batt = float(getattr(env, "battery", {}).get(a, 1.0))
    scal = [env.t / fam.max_steps, env.alt[a] / (cam.N_ALT - 1), batt,
            float(env._energy_home(a)) if hasattr(env, "_energy_home") else 0.0,
            len(env.agents) / 8.0, *[float(fam.name == f) for f in FAMS]]
    X = np.concatenate([logS, Vn, Dn[None], taken[None].astype(float), mates[None], own[None], base[None],
                        np.broadcast_to(np.array(scal)[:, None, None], (N_SCAL, g, g))]).astype(np.float32)
    return X, S > -0.5


class TargetNet(nn.Module):
    def __init__(self, width: int = 32, beta: float = 5.0):
        super().__init__()
        self.width, self.beta0 = width, beta
        self.inp = nn.Conv2d(C_IN, width, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(width, width, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 1)])
        self.glob = nn.Linear(2 * width, width)
        self.mix = nn.Conv2d(width, width, 1)
        self.head = nn.Conv2d(width, cam.N_ALT, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.value = nn.Sequential(nn.Linear(2 * width + N_SCAL, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))

    def forward(self, x, area, mask):
        """x (B, C, G, G); area (B, 1, G, G), 1 inside each map (maps are padded
        to the batch's largest); mask (B, N_ALT, G, G), choices allowed.
        Returns logits (B, N_ALT * G * G), the rule's prior logits, and values (B,)."""
        h = F.relu(self.inp(x)) * area
        for c in self.blocks:
            h = (h + F.relu(c(h))) * area
        n = area.sum((2, 3)).clamp(min=1.0)
        mean = h.sum((2, 3)) / n
        mx = (h - (1.0 - area) * 1e4).amax((2, 3))
        gf = torch.cat([mean, mx], 1)
        h2 = F.relu(self.mix(h) + self.glob(gf)[:, :, None, None])
        prior = torch.exp(self.log_beta) * x[:, :cam.N_ALT]
        logits = (prior + self.head(h2)).masked_fill(~mask, NEG)
        prior = prior.detach().masked_fill(~mask, NEG)
        scal = x[:, -N_SCAL:, 0, 0]
        v = self.value(torch.cat([gf, scal], 1)).squeeze(-1)
        return logits.flatten(1), prior.flatten(1), v


class Recorder:
    """Decisions of one mission for decision-level PPO. A drone's decision runs
    until its next one; its reward is the discounted sum of the drone's step
    rewards in between. Inputs are kept for a random share `keep` of the
    decisions (they are large); all decisions count for the advantages."""

    def __init__(self, gamma: float, lam: float, keep: float, rng):
        self.gamma, self.lam, self.keep, self.rng = gamma, lam, keep, rng
        self.force_keep = False  # keep the inputs of every decision made while set (counterfactual checks)
        self.open: dict = {}
        self.chains: dict = {}

    def decide(self, a, t, x, mask, idx, logp, v):
        if a in self.open:
            self._close(a, t)
        keep = self.rng.random() < self.keep or self.force_keep
        self.open[a] = dict(t=t, R=0.0, v=float(v), logp=float(logp), idx=int(idx),
                            x=x.astype(np.float16) if keep else None, mask=mask if keep else None)

    def reward(self, rew: dict, t: int):
        for a, r in rew.items():
            o = self.open.get(a)
            if o is not None:
                o["R"] += self.gamma ** (t - o["t"]) * float(r)

    def _close(self, a, t_next):
        o = self.open.pop(a)
        o["dt"] = max(1, t_next - o["t"])
        self.chains.setdefault(a, []).append(o)

    def finish(self, t_end: int) -> list[dict]:
        """Close every open decision (the mission ended: no value after it;
        time is in the input, so the value can learn that) and compute GAE."""
        for a in list(self.open):
            self._close(a, t_end)
        out = []
        for chain in self.chains.values():
            adv = 0.0
            for i in range(len(chain) - 1, -1, -1):
                o = chain[i]
                last = i == len(chain) - 1
                v_next = 0.0 if last else chain[i + 1]["v"]
                gdt = self.gamma ** o["dt"]
                delta = o["R"] + gdt * v_next - o["v"]
                adv = delta + (0.0 if last else gdt * self.lam * adv)
                o["adv"], o["ret"] = adv, adv + o["v"]
            out.extend(o for o in chain if o["x"] is not None)
        return out


class TargetPolicy(BeliefGreedy):
    """The rule with its choice of target made by a TargetNet."""

    def __init__(self, net: TargetNet, temperature: float = 1.0, recorder: Recorder | None = None,
                 seed: int = 0, **kw):
        super().__init__(**kw)
        self.net, self.temperature, self.recorder, self.seed = net, temperature, recorder, seed
        self.rng = np.random.default_rng(seed)
        self.rule_for = None  # a drone whose next choice is the rule's own (counterfactual branches)

    def reset(self, env):
        super().reset(env)
        # the same draws for the same mission, however missions are spread over processes
        self.rng = np.random.default_rng([self.seed, int(getattr(env.scenario, "seed_world", 0))])

    def choose(self, env, a, S, V, taken, D):
        if a == self.rule_for:
            self.rule_for = None
            return super().choose(env, a, S, V, taken, D)
        X, mask = inputs(env, a, S, V, taken, D)
        if not mask.any():
            return super().choose(env, a, S, V, taken, D)
        g = S.shape[1]
        with torch.no_grad():
            logits, _, v = self.net(torch.from_numpy(X)[None], torch.ones(1, 1, g, g),
                                    torch.from_numpy(mask)[None])
        lg = logits[0].double()
        if self.temperature == 0:
            idx = int(torch.argmax(lg))
        else:
            p = torch.softmax(lg / self.temperature, 0).numpy()
            idx = int(self.rng.choice(p.size, p=p / p.sum()))
        if self.recorder is not None:
            logp = float(torch.log_softmax(lg, 0)[idx])
            self.recorder.decide(a, env.t, X, mask, idx, logp, float(v[0]))
        k, rem = divmod(idx, g * g)
        ty, tx = divmod(rem, g)
        return int(k), int(ty), int(tx)


def save(path, net: TargetNet, extra: dict | None = None):
    import os
    d = dict(kind="targets", state=net.state_dict(), width=net.width, beta=net.beta0, **(extra or {}))
    tmp = f"{path}.tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)


def load(path) -> tuple[TargetNet, dict]:
    d = torch.load(path, weights_only=False, map_location="cpu")
    net = TargetNet(d.get("width", 32), d.get("beta", 5.0))
    net.load_state_dict(d["state"])
    net.eval()
    return net, d


def is_targets(path) -> bool:
    d = torch.load(path, weights_only=False, map_location="cpu")
    return isinstance(d, dict) and d.get("kind") == "targets"
