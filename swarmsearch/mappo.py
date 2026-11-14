"""MAPPO and IPPO for the grid search env.

Actor architectures (Config.arch):
- "mlp":         local vector obs only.
- "cnn":         + the whole shared team map, read by a small CNN.
- "transformer": + the whole map, cut into patch tokens; a transformer
                 encoder lets the drone's own token attend over every map
                 region and its own state before choosing an action.

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

from .env import (CAMERA_DIM, GUIDE_DIM, MAP_CHANNELS, MAX_DRONES, TEAM_DIM, GridSearchEnv, map_dim, n_actions,
                  obs_dim, state_dim)
from .guide import apply_plans, make_guide
from .lookahead import path_value
from .scenario import ScenarioSampler


def mlp(inp: int, out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)
    )


class MapNet(nn.Module):
    """Vector obs through an MLP branch, full map through a CNN branch."""

    def __init__(self, grid_size: int, out: int, hidden: int = 256, vec: int | None = None):
        super().__init__()
        self.g = grid_size
        self.vec = vec or obs_dim()
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


class MapTransformer(nn.Module):
    """Transformer over [self token, map patch tokens].

    The g x g map (MAP_CHANNELS channels) is cut into patch x patch tiles; each
    tile becomes one token (linear embedding + learned position). The drone's
    vector obs becomes the self token. After the encoder, the self token,
    which has attended over every region, feeds the output head.
    """

    def __init__(self, grid_size: int, out: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, patch: int = 4, vec: int | None = None):
        super().__init__()
        assert grid_size % patch == 0, "grid size must be a multiple of the patch size"
        self.g, self.p = grid_size, patch
        self.vec = vec or obs_dim()
        n_tok = (grid_size // patch) ** 2
        self.self_embed = nn.Linear(self.vec, d_model)
        self.patch_embed = nn.Linear(MAP_CHANNELS * patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        b, g, p = x.shape[0], self.g, self.p
        m = x[:, self.vec :].reshape(b, MAP_CHANNELS, g // p, p, g // p, p)
        m = m.permute(0, 2, 4, 1, 3, 5).reshape(b, (g // p) ** 2, -1)  # tiles, row-major
        tok = torch.cat([self.self_embed(x[:, None, : self.vec]), self.patch_embed(m)], dim=1)
        h = self.encoder(tok + self.pos)
        return self.head(self.norm(h[:, 0])).reshape(*lead, -1)


def make_body(arch: str, grid_size: int, out: int, hidden: int, tf: dict | None = None,
              vec: int | None = None) -> nn.Module:
    vec = vec or obs_dim()
    if arch == "mlp":
        return mlp(vec, out, hidden)
    if arch == "cnn":
        return MapNet(grid_size, out, hidden, vec)
    if arch == "transformer":
        return MapTransformer(grid_size, out, vec=vec, **(tf or {}))
    raise ValueError(arch)


class ActorCritic(nn.Module):
    def __init__(self, grid_size: int, centralized: bool, hidden: int = 256, arch: str = "mlp",
                 tf: dict | None = None, guided: bool = False, team: bool = False, camera: bool = False):
        super().__init__()
        self.centralized = centralized
        self.arch = arch
        self.full_map = arch != "mlp"
        self.guided = guided
        self.team = team
        self.camera = camera
        o = self.vec = (obs_dim() + (GUIDE_DIM if guided else 0) + (TEAM_DIM if team else 0)
                        + (CAMERA_DIM if camera else 0))
        self.actor = make_body(arch, grid_size, n_actions(camera), hidden, tf, o)
        # MAPPO's critic gets the full global state plus the vector obs; IPPO's sees only the local obs.
        self.critic = (mlp(o + state_dim(grid_size, camera), 1, hidden) if centralized
                       else make_body(arch, grid_size, 1, hidden, tf, o))

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
    arch: str = "mlp"  # mlp | cnn | transformer
    tf_d_model: int = 64
    tf_layers: int = 2
    tf_heads: int = 4
    tf_patch: int = 4
    device: str = "cpu"
    guide: str = "none"  # none | heuristic
    guide_every: int = 25  # env steps between guide queries
    guide_coef: float = 1.0  # initial shaping weight
    guide_anneal: float = 0.6  # fraction of training over which the weight fades to 0
    seed: int = 0
    init_from: str | None = None  # start from a trained model (e.g. M1 -> M2 fine-tuning)
    # RL fine-tuning of an imitation policy (M5): keep the actor close to a
    # frozen anchor actor (KL, fading to 0) and train only the critic at first.
    bc_anchor: str | None = None
    bc_coef: float = 1.0
    bc_anneal: float = 0.5  # fraction of training over which bc_coef fades to 0
    critic_warmup: int = 0  # updates with the actor frozen while the critic learns
    team: bool = False  # team layers in the observation (all teammates' positions, trails, intent)
    out_dir: str = "runs/mappo"
    sampler: ScenarioSampler = field(default_factory=ScenarioSampler)


def tf_kwargs(c) -> dict:
    get = c.get if isinstance(c, dict) else lambda k, d=None: getattr(c, k, d)
    return dict(d_model=get("tf_d_model", 64), n_layers=get("tf_layers", 2),
                n_heads=get("tf_heads", 4), patch=get("tf_patch", 4))


def critic_input(env: GridSearchEnv, obs: np.ndarray, centralized: bool, state=None, vec: int | None = None) -> np.ndarray:
    if not centralized:
        return obs
    # The centralized critic takes the global state plus the vector part of the obs.
    return np.concatenate([env.state() if state is None else state, obs[: vec or obs_dim()]])


def train(cfg: Config, log_every: int = 1) -> ActorCritic:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    centralized = cfg.algo == "mappo"
    guide = make_guide(cfg.guide)
    guided = guide is not None
    envs = [GridSearchEnv(sampler=cfg.sampler, seed=int(rng.integers(1 << 31)), full_map=cfg.arch != "mlp",
                         guided=guided, team=cfg.team)
            for _ in range(cfg.n_envs)]
    g = envs[0].grid_size
    dev = torch.device(cfg.device)
    camera = cfg.sampler.camera
    net = ActorCritic(g, centralized, cfg.hidden, cfg.arch, tf_kwargs(cfg), guided, cfg.team, camera).to(dev)
    vec = net.vec
    T_ = lambda a: torch.from_numpy(a).to(dev)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, weights_only=False, map_location="cpu")
        net.load_state_dict(ck["state_dict"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    anchor = None
    if cfg.bc_anchor:
        anchor = ActorCritic(g, centralized, cfg.hidden, cfg.arch, tf_kwargs(cfg), guided, cfg.team, camera).to(dev)
        anchor.load_state_dict(torch.load(cfg.bc_anchor, weights_only=False, map_location="cpu")["state_dict"])
        anchor.eval()

    T, E, A = cfg.rollout_len, cfg.n_envs, MAX_DRONES
    o_dim = vec + (map_dim(g) if cfg.arch != "mlp" else 0)
    c_dim = vec + state_dim(g, camera) if centralized else o_dim
    buf = {
        "obs": np.zeros((T, E, A, o_dim), np.float32),
        "cin": np.zeros((T, E, A, c_dim), np.float32),
        "mask": np.zeros((T, E, A, n_actions(camera)), np.float32),
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
                  "repeat_frac", "collisions", "policy_loss", "value_loss", "entropy", "secs",
                  "guide_coef"])

    n_updates = cfg.total_env_steps // (T * E)
    env_steps, t0, ep_metrics = 0, time.time(), []
    def replan():
        """Query the guide for every env due a new plan; refresh their obs."""
        due = [ei for ei, e in enumerate(envs) if e.agents and e.t % cfg.guide_every == 0]
        if not due:
            return
        apply_plans([envs[ei] for ei in due], guide.plan([envs[ei] for ei in due]))
        for ei in due:
            e, (_, infos) = envs[ei], cur[ei]
            cur[ei] = ({a: e._obs(a) for a in e.agents}, infos)

    for update in range(1, n_updates + 1):
        for k in buf:
            buf[k].fill(0)
        if guided:
            frac = (update - 1) / max(1, int(cfg.guide_anneal * n_updates))
            coef = cfg.guide_coef * max(0.0, 1.0 - frac)
            for e in envs:
                e.guide_coef = coef
        for t in range(T):
            if guided:
                replan()
            for ei, env in enumerate(envs):
                obs, infos = cur[ei]
                agents = list(env.agents)
                idx = [int(a.split("_")[1]) for a in agents]
                o = np.stack([obs[a] for a in agents])
                m = np.stack([infos[a]["action_mask"] for a in agents]).astype(np.float32)
                state = env.state() if centralized else None
                c = np.stack([critic_input(env, obs[a], centralized, state, vec) for a in agents])
                with torch.no_grad():
                    d = net.dist(T_(o), T_(m))
                    act = d.sample()
                    logp = d.log_prob(act).cpu().numpy()
                    val = net.value(T_(c)).cpu().numpy()
                    act = act.cpu().numpy()
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
                        cin = critic_input(env, nobs[a], centralized, nstate, vec)
                        with torch.no_grad():
                            buf["boot"][t, ei, i] = float(net.value(T_(cin[None]))[0])
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
                cin = critic_input(env, obs[a], centralized, state, vec)
                with torch.no_grad():
                    last_val[ei, int(a.split("_")[1])] = float(net.value(T_(cin[None]))[0])

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
        flat = lambda x: T_(np.ascontiguousarray(x.reshape(-1, *x.shape[3:])[sel]))
        b_obs, b_cin, b_mask = flat(buf["obs"]), flat(buf["cin"]), flat(buf["mask"])
        b_act, b_logp = flat(buf["act"]), flat(buf["logp"])
        b_adv, b_ret = flat(adv), flat(ret)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = len(b_obs)
        mb = max(1, n // cfg.minibatches)
        warm = update <= cfg.critic_warmup
        kl_coef = 0.0
        if anchor is not None:
            kl_coef = cfg.bc_coef * max(0.0, 1.0 - (update - 1) / max(1, int(cfg.bc_anneal * n_updates)))
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                j = perm[s : s + mb]
                d = net.dist(b_obs[j], b_mask[j])
                logp = d.log_prob(b_act[j])
                ratio = (logp - b_logp[j]).exp()
                pl = -torch.min(ratio * b_adv[j], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * b_adv[j]).mean()
                vl = ((net.value(b_cin[j]) - b_ret[j]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                if kl_coef > 0:
                    with torch.no_grad():
                        q = anchor.dist(b_obs[j], b_mask[j]).probs
                    kl = (q * (torch.log(q + 1e-8) - torch.log(d.probs + 1e-8))).sum(-1).mean()
                    loss = loss + kl_coef * kl
                if warm:  # critic only: the actor keeps the imitation policy
                    loss = cfg.vf_coef * vl
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
                   avg("collisions"), float(pl), float(vl), float(ent), round(time.time() - t0),
                   envs[0].guide_coef]
            log.writerow(row)
            log_f.flush()
            print(f"[{cfg.algo}] upd {update}/{n_updates} steps {env_steps} eps {len(ep_metrics)} "
                  f"found {row[3]:.2f} t90 {row[4]:.0f} cov {row[5]:.2f} rep {row[6]:.2f} "
                  f"coll {row[7]:.2f} ent {row[10]:.2f} {row[11]}s"
                  + (f" guide {row[12]:.2f}" if guided else ""), flush=True)
            ep_metrics = []
        if update % 50 == 0 or update == n_updates:
            save(net, cfg, out / "model.pt")
    log_f.close()
    return net


def save(net: ActorCritic, cfg: Config, path: Path):
    c = asdict(cfg)
    c["sampler"] = asdict(cfg.sampler)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "config": c}, path)


def load(path: str | Path) -> tuple[ActorCritic, dict]:
    ck = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    arch = cfg.get("arch") or ("cnn" if cfg.get("full_map") else "mlp")
    guided = cfg.get("guide", "none") not in (None, "none")
    net = ActorCritic(cfg["sampler"]["grid_size"], cfg["algo"] == "mappo", cfg["hidden"], arch, tf_kwargs(cfg), guided,
                       cfg.get("team", False), cfg["sampler"].get("camera", False))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, cfg


class LearnedPolicy:
    """Wraps a trained actor with the same reset/act interface as the baselines.

    temperature < 1 sharpens the action distribution (0 = always the top
    action). lookahead = H > 0 re-ranks each drone's top_k moves by the
    expected detections on the best H-step path from the cell each leads to
    (see lookahead.py); drones choose in turn and later ones do not count the
    cells earlier ones just claimed. Both apply at run time only.
    """

    def __init__(self, net: ActorCritic, greedy: bool = False, seed: int = 0, guide=None,
                 guide_every: int = 25, temperature: float = 1.0, lookahead: int = 0, top_k: int = 3,
                 min_prob: float = 0.05):
        self.net = net
        self.greedy = greedy or temperature == 0
        self.temperature = temperature
        self.lookahead = lookahead
        self.top_k = top_k
        self.min_prob = min_prob
        self.guide = guide
        self.guide_every = guide_every
        if net.guided and guide is None:
            raise ValueError("this policy was trained with a guide; pass one (see make_guide)")
        torch.manual_seed(seed)

    def reset(self, env):
        pass

    def act(self, env, obs=None) -> dict:
        agents = list(env.agents)
        if not agents:
            return {}
        if self.guide is not None and env.t % self.guide_every == 0:
            apply_plans([env], self.guide.plan([env]))
        o = torch.from_numpy(np.stack([env._obs(a, full_map=self.net.full_map, guided=self.net.guided, team=self.net.team)
                                       for a in agents]))
        m = torch.from_numpy(np.stack([env.action_mask(a) for a in agents]).astype(np.float32))
        with torch.no_grad():
            d = self.net.dist(o, m)
            if self.lookahead:
                return self._rerank(env, agents, d.probs.numpy())
            if self.greedy:
                act = d.probs.argmax(-1)
            elif self.temperature != 1.0:
                act = torch.distributions.Categorical(logits=d.logits / self.temperature).sample()
            else:
                act = d.sample()
        return {a: int(x) for a, x in zip(agents, act)}

    def _rerank(self, env, agents, probs) -> dict:
        from .env import MOVES
        sc = env.scenario
        gain = env.belief * sc.detection_prob
        out = {}
        for a, p in zip(agents, probs):
            cand = [int(c) for c in np.argsort(-p)[: self.top_k] if p[c] >= self.min_prob * p.max()]
            nxt = [env.pos[a] + MOVES[c] for c in cand]
            val = [path_value(gain, env.free, n, self.lookahead, sc.detection_prob, sc.corner_cutting) for n in nxt]
            best = max(range(len(cand)), key=lambda i: (val[i], p[cand[i]]))
            out[a] = cand[best]
            bx, by = nxt[best]
            gain = gain.copy()
            gain[by, bx] *= 1.0 - sc.detection_prob  # this drone searches it next step
        return out
