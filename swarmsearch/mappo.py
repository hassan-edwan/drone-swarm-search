"""MAPPO and IPPO for the grid search env.

With full_map=True the actor also sees the whole shared team map, read by a
small CNN (fix for the coverage plateau of the vector-only policy).

Both use ONE actor network shared by every drone (parameter sharing); each
drone runs it on its own local observation. They differ only in the critic:

- MAPPO: centralized critic on [global state, drone's own obs] (training only).
- IPPO:  each drone's critic sees only its own local obs.

The critic outputs a value estimate, never actions, and is not used at
deployment.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env import MAP_CHANNELS, MAX_DRONES, N_ACTIONS, GridSearchEnv, map_dim, obs_dim, state_dim
from .scenario import ScenarioSampler


def mlp(inp: int, out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)
    )


class MapNet(nn.Module):
    """Vector obs through an MLP branch, full map through a CNN branch."""

    def __init__(self, grid_size: int, out: int, hidden: int = 256):
        super().__init__()
        self.g = grid_size
        self.vec = obs_dim()
        self.cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, MAP_CHANNELS, grid_size, grid_size)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(self.vec + n_cnn, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        m = x[:, self.vec :].reshape(-1, MAP_CHANNELS, self.g, self.g)
        h = torch.cat([x[:, : self.vec], self.cnn(m)], dim=-1)
        return self.head(h).reshape(*lead, -1)


class ActorCritic(nn.Module):
    def __init__(self, grid_size: int, centralized: bool, hidden: int = 256, full_map: bool = False):
        super().__init__()
        self.centralized = centralized
        self.full_map = full_map
        o = obs_dim()
        if full_map:
            self.actor = MapNet(grid_size, N_ACTIONS, hidden)
            # MAPPO's critic already gets the full global state; IPPO's reads the map with its own CNN.
            self.critic = mlp(o + state_dim(grid_size), 1, hidden) if centralized else MapNet(grid_size, 1, hidden)
        else:
            self.actor = mlp(o, N_ACTIONS, hidden)
            self.critic = mlp(o + state_dim(grid_size) if centralized else o, 1, hidden)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor(obs).masked_fill(mask == 0, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, critic_in: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_in).squeeze(-1)


@dataclass
class Config:
    algo: str = "mappo"  # or "ippo"
    total_env_steps: int = 2_000_000
    n_envs: int = 8
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 5
    minibatches: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    hidden: int = 256
    full_map: bool = False
    seed: int = 0
    out_dir: str = "runs/mappo"
    sampler: ScenarioSampler = field(default_factory=ScenarioSampler)


def critic_input(env: GridSearchEnv, obs: np.ndarray, centralized: bool, state=None) -> np.ndarray:
    if not centralized:
        return obs
    # The centralized critic takes the global state plus the vector part of the obs.
    return np.concatenate([env.state() if state is None else state, obs[: obs_dim()]])


def train(cfg: Config, log_every: int = 1) -> ActorCritic:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    centralized = cfg.algo == "mappo"
    envs = [GridSearchEnv(sampler=cfg.sampler, seed=int(rng.integers(1 << 31)), full_map=cfg.full_map)
            for _ in range(cfg.n_envs)]
    g = envs[0].grid_size
    net = ActorCritic(g, centralized, cfg.hidden, cfg.full_map)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, E, A = cfg.rollout_len, cfg.n_envs, MAX_DRONES
    o_dim = obs_dim() + (map_dim(g) if cfg.full_map else 0)
    c_dim = obs_dim() + state_dim(g) if centralized else o_dim
    buf = {
        "obs": np.zeros((T, E, A, o_dim), np.float32),
        "cin": np.zeros((T, E, A, c_dim), np.float32),
        "mask": np.zeros((T, E, A, N_ACTIONS), np.float32),
        "act": np.zeros((T, E, A), np.int64),
        "logp": np.zeros((T, E, A), np.float32),
        "val": np.zeros((T, E, A), np.float32),
        "rew": np.zeros((T, E, A), np.float32),
        "term": np.zeros((T, E, A), np.float32),  # no bootstrap after this step
        "boot": np.zeros((T, E, A), np.float32),  # value to bootstrap on truncation
        "valid": np.zeros((T, E, A), np.float32),
    }

    cur = []
    for e in envs:
        obs, infos = e.reset()
        cur.append((obs, infos))

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["update", "env_steps", "episodes", "found_frac", "time_to_90", "coverage",
                  "repeat_frac", "collisions", "policy_loss", "value_loss", "entropy", "secs"])

    n_updates = cfg.total_env_steps // (T * E)
    env_steps, t0, ep_metrics = 0, time.time(), []
    for update in range(1, n_updates + 1):
        for k in buf:
            buf[k].fill(0)
        for t in range(T):
            for ei, env in enumerate(envs):
                obs, infos = cur[ei]
                agents = list(env.agents)
                idx = [int(a.split("_")[1]) for a in agents]
                o = np.stack([obs[a] for a in agents])
                m = np.stack([infos[a]["action_mask"] for a in agents]).astype(np.float32)
                state = env.state() if centralized else None
                c = np.stack([critic_input(env, obs[a], centralized, state) for a in agents])
                with torch.no_grad():
                    d = net.dist(torch.from_numpy(o), torch.from_numpy(m))
                    act = d.sample()
                    logp = d.log_prob(act)
                    val = net.value(torch.from_numpy(c))
                acts = {a: int(x) for a, x in zip(agents, act)}
                nobs, rew, term, trunc, ninfos = env.step(acts)
                for j, (a, i) in enumerate(zip(agents, idx)):
                    buf["obs"][t, ei, i] = o[j]
                    buf["cin"][t, ei, i] = c[j]
                    buf["mask"][t, ei, i] = m[j]
                    buf["act"][t, ei, i] = act[j]
                    buf["logp"][t, ei, i] = logp[j]
                    buf["val"][t, ei, i] = val[j]
                    buf["rew"][t, ei, i] = rew[a] * cfg.reward_scale
                    buf["valid"][t, ei, i] = 1.0
                    if term[a]:
                        buf["term"][t, ei, i] = 1.0
                    elif trunc[a]:
                        buf["term"][t, ei, i] = 1.0
                        nstate = env.state() if centralized else None
                        cin = critic_input(env, nobs[a], centralized, nstate)
                        with torch.no_grad():
                            buf["boot"][t, ei, i] = float(net.value(torch.from_numpy(cin[None]))[0])
                env_steps += 1
                if not env.agents:
                    ep_metrics.append(env.metrics())
                    cur[ei] = env.reset()
                else:
                    cur[ei] = (nobs, ninfos)

        # Bootstrap values for agents still running at the end of the rollout.
        last_val = np.zeros((E, A), np.float32)
        for ei, env in enumerate(envs):
            obs, _ = cur[ei]
            state = env.state() if centralized else None
            for a in env.agents:
                cin = critic_input(env, obs[a], centralized, state)
                with torch.no_grad():
                    last_val[ei, int(a.split("_")[1])] = float(net.value(torch.from_numpy(cin[None]))[0])

        adv = np.zeros((T, E, A), np.float32)
        gae = np.zeros((E, A), np.float32)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else buf["val"][t + 1]
            nvalid = np.ones((E, A), np.float32) if t == T - 1 else buf["valid"][t + 1]
            term = buf["term"][t]
            next_v = np.where(term > 0, buf["boot"][t], nv * nvalid)
            delta = buf["rew"][t] + cfg.gamma * next_v - buf["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - term) * nvalid * gae
            gae *= buf["valid"][t]
            adv[t] = gae
        ret = adv + buf["val"]

        sel = buf["valid"].reshape(-1) > 0
        flat = lambda x: torch.from_numpy(x.reshape(-1, *x.shape[3:])[sel])
        b_obs, b_cin, b_mask = flat(buf["obs"]), flat(buf["cin"]), flat(buf["mask"])
        b_act, b_logp = flat(buf["act"]), flat(buf["logp"])
        b_adv, b_ret = flat(adv), flat(ret)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = len(b_obs)
        mb = max(1, n // cfg.minibatches)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                j = perm[s : s + mb]
                d = net.dist(b_obs[j], b_mask[j])
                logp = d.log_prob(b_act[j])
                ratio = (logp - b_logp[j]).exp()
                pl = -torch.min(ratio * b_adv[j], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * b_adv[j]).mean()
                vl = ((net.value(b_cin[j]) - b_ret[j]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        if update % log_every == 0 and ep_metrics:
            def avg(k, default=None):
                vals = [m[k] if m[k] is not None else default for m in ep_metrics]
                return float(np.mean(vals))
            row = [update, env_steps, len(ep_metrics), avg("found_frac"),
                   avg("time_to_90", cfg.sampler.max_steps), avg("coverage"), avg("repeat_frac"),
                   avg("collisions"), float(pl), float(vl), float(ent), round(time.time() - t0)]
            log.writerow(row)
            log_f.flush()
            print(f"[{cfg.algo}] upd {update}/{n_updates} steps {env_steps} eps {len(ep_metrics)} "
                  f"found {row[3]:.2f} t90 {row[4]:.0f} cov {row[5]:.2f} rep {row[6]:.2f} "
                  f"coll {row[7]:.2f} ent {row[10]:.2f} {row[11]}s", flush=True)
            ep_metrics = []
        if update % 50 == 0 or update == n_updates:
            save(net, cfg, out / "model.pt")
    log_f.close()
    return net


def save(net: ActorCritic, cfg: Config, path: Path):
    c = asdict(cfg)
    c["sampler"] = asdict(cfg.sampler)
    torch.save({"state_dict": net.state_dict(), "config": c}, path)


def load(path: str | Path) -> tuple[ActorCritic, dict]:
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    net = ActorCritic(cfg["sampler"]["grid_size"], cfg["algo"] == "mappo", cfg["hidden"], cfg.get("full_map", False))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, cfg


class LearnedPolicy:
    """Wraps a trained actor with the same reset/act interface as the baselines."""

    def __init__(self, net: ActorCritic, greedy: bool = False, seed: int = 0):
        self.net = net
        self.greedy = greedy
        torch.manual_seed(seed)

    def reset(self, env):
        pass

    def act(self, env, obs=None) -> dict:
        agents = list(env.agents)
        if not agents:
            return {}
        o = torch.from_numpy(np.stack([env._obs(a, full_map=self.net.full_map) for a in agents]))
        m = torch.from_numpy(np.stack([env.action_mask(a) for a in agents]).astype(np.float32))
        with torch.no_grad():
            d = self.net.dist(o, m)
            act = d.probs.argmax(-1) if self.greedy else d.sample()
        return {a: int(x) for a, x in zip(agents, act)}
