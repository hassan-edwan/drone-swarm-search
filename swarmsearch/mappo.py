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
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env import (CAMERA_DIM, GUIDE_DIM, MAP_CHANNELS, MAX_DRONES, REAL_DIM, TEAM_DIM, GridSearchEnv, map_dim,
                  n_actions, obs_dim, state_dim)
from .guide import apply_plans, make_guide
from .lookahead import path_value
from .scenario import ScenarioSampler


def mlp(inp: int, out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)
    )


class MapNet(nn.Module):
    """Vector obs through an MLP branch, full map through a CNN branch."""

    def __init__(self, grid_size: int, out: int, hidden: int = 256, vec: int | None = None,
                 channels: int = MAP_CHANNELS):
        super().__init__()
        self.g = grid_size
        self.c = channels
        self.vec = vec or obs_dim()
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, channels, grid_size, grid_size)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(self.vec + n_cnn, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        m = x[:, self.vec :].reshape(-1, self.c, self.g, self.g)
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
                 tf: dict | None = None, guided: bool = False, team: bool = False, camera: bool = False,
                 real: bool = False, field: bool = False, plan: bool = False):
        super().__init__()
        self.centralized = centralized
        self.arch = arch
        self.full_map = arch != "mlp"
        self.guided = guided
        self.team = team
        self.camera = camera
        self.real = real
        o = self.vec = (obs_dim() + (GUIDE_DIM if guided else 0) + (TEAM_DIM if team else 0)
                        + (CAMERA_DIM if camera else 0) + (REAL_DIM if real else 0))
        self.n_act = n_actions(camera)
        self.state_in = state_dim(grid_size, camera, real)
        self.field = field
        self.plan = plan
        self.map_in = map_dim(grid_size) if self.full_map else 0  # size of the map part of the obs
        if field:  # M8: its own observation, 19 actions, fixed-size critic state
            from .field.env import (FIELD_MAP_CHANNELS, FIELD_MAP_DIM, FIELD_MAP_RES, FIELD_OBS_DIM,
                                    FIELD_STATE_DIM, N_ACTIONS_FIELD, PLAN_DIM)
            o = self.vec = FIELD_OBS_DIM + (PLAN_DIM if plan else 0)
            self.n_act, self.state_in = N_ACTIONS_FIELD, FIELD_STATE_DIM
            if arch == "cnn":  # the whole map at a fixed size, whatever the grid
                self.map_in = FIELD_MAP_DIM
                self.actor = MapNet(FIELD_MAP_RES, self.n_act, hidden, o, FIELD_MAP_CHANNELS)
                self.critic = mlp(o + self.state_in, 1, hidden) if centralized else MapNet(
                    FIELD_MAP_RES, 1, hidden, o, FIELD_MAP_CHANNELS)
                return
            if arch != "mlp":
                raise ValueError("field environment: arch must be mlp or cnn")
        self.actor = make_body(arch, grid_size, self.n_act, hidden, tf, o)
        # MAPPO's critic gets the full global state plus the vector obs; IPPO's sees only the local obs.
        self.critic = (mlp(o + self.state_in, 1, hidden) if centralized
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
    reward: str = "count"  # "count" or "find_time" (see env.FIND_TIME_REWARD)
    team: bool = False  # team layers in the observation (all teammates' positions, trails, intent)
    out_dir: str = "runs/mappo"
    ckpt_every: int = 25  # updates between resumable checkpoints (model, optimizer, counters, RNG)
    resume: bool = False  # continue from out_dir/ckpt.pt if it exists
    workers: int = 0  # processes running the environments (0: in this process, the original loop)
    plan_feats: int = 0  # field: planning features (1: straight-line distance, 2: the rule's score, travel distance)
    bc_floor: float = 0.0  # the KL pull towards the anchor never falls below this weight
    reward_norm: bool = False  # scale rewards per scenario family by a running return spread
    val_every: int = 0  # updates between validation runs (0: none); the best model is kept as best.pt
    val_episodes: int = 8  # validation missions per family
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
    from .envs import is_field, make_env
    field = is_field(cfg.sampler)
    seeds = [int(rng.integers(1 << 31)) for _ in range(cfg.n_envs)]

    def make(seed):
        e = make_env(sampler=cfg.sampler, seed=seed, full_map=cfg.arch != "mlp", guided=guided, team=cfg.team,
                     plan_feats=cfg.plan_feats)
        e.reward = cfg.reward
        return e

    use_vec = cfg.workers > 0
    if use_vec and guided:
        raise ValueError("worker processes do not support a guide")
    envs = [make(seeds[0])] if use_vec else [make(s) for s in seeds]  # with workers: one probe env here
    g = envs[0].grid_size
    dev = torch.device(cfg.device)
    camera, real = cfg.sampler.camera, cfg.sampler.real
    net = ActorCritic(g, centralized, cfg.hidden, cfg.arch, tf_kwargs(cfg), guided, cfg.team, camera, real,
                      field, cfg.plan_feats).to(dev)
    vec = net.vec
    T_ = lambda a: torch.from_numpy(a).to(dev)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, weights_only=False, map_location="cpu")
        net.load_state_dict(ck["state_dict"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    anchor = None
    if cfg.bc_anchor:
        anchor = ActorCritic(g, centralized, cfg.hidden, cfg.arch, tf_kwargs(cfg), guided, cfg.team, camera,
                             real, field, cfg.plan_feats).to(dev)
        anchor.load_state_dict(torch.load(cfg.bc_anchor, weights_only=False, map_location="cpu")["state_dict"])
        anchor.eval()

    T, E, A = cfg.rollout_len, cfg.n_envs, MAX_DRONES
    o_dim = vec + net.map_in
    c_dim = vec + net.state_in if centralized else o_dim
    buf = {
        "obs": np.zeros((T, E, A, o_dim), np.float32),
        "cin": np.zeros((T, E, A, c_dim), np.float32),
        "mask": np.zeros((T, E, A, net.n_act), np.float32),
        "act": np.zeros((T, E, A), np.int64),
        "logp": np.zeros((T, E, A), np.float32),
        "val": np.zeros((T, E, A), np.float32),
        "rew": np.zeros((T, E, A), np.float32),
        "term": np.zeros((T, E, A), np.float32),  # no bootstrap after this step
        "boot": np.zeros((T, E, A), np.float32),  # value to bootstrap on truncation
        "valid": np.zeros((T, E, A), np.float32),
    }

    from .vec import VecEnvs
    venv = VecEnvs(make, seeds, centralized, vec, cfg.workers) if use_vec else None
    if use_vec:
        pks = venv.reset()
    else:
        cur = []
        for e in envs:
            obs, infos = e.reset()
            cur.append((obs, infos))

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start, env_steps, secs0 = 1, 0, 0.0
    ck_path = out / "ckpt.pt"
    if cfg.resume and ck_path.exists():
        ck = torch.load(ck_path, weights_only=False, map_location="cpu")
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        start, env_steps, secs0 = ck["update"] + 1, ck["env_steps"], ck["secs"]
        torch.set_rng_state(ck["torch_rng"])
        rng.bit_generator.state = ck["np_rng"]
        if use_vec:  # episodes in flight are not saved: start fresh ones
            venv.set_rng_states(ck["env_rngs"])
            pks = venv.reset()
        else:
            for e, st in zip(envs, ck["env_rngs"]):
                e.rng.bit_generator.state = st
            cur = [e.reset() for e in envs]
        print(f"resumed from {ck_path} at update {ck['update']} ({env_steps} steps)", flush=True)
    elif cfg.resume:
        print(f"no checkpoint at {ck_path}: starting from scratch", flush=True)
    resumed = start > 1
    log_f = open(out / "train_log.csv", "a" if resumed else "w", newline="")
    log = csv.writer(log_f)
    if not resumed:
        log.writerow(["update", "env_steps", "episodes", "found_frac", "time_to_90", "coverage",
                  "repeat_frac", "collisions", "policy_loss", "value_loss", "entropy", "secs",
                  "guide_coef"])

    n_updates = cfg.total_env_steps // (T * E)
    t0, ep_metrics = time.time() - secs0, []
    def replan():
        """Query the guide for every env due a new plan; refresh their obs."""
        due = [ei for ei, e in enumerate(envs) if e.agents and e.t % cfg.guide_every == 0]
        if not due:
            return
        apply_plans([envs[ei] for ei in due], guide.plan([envs[ei] for ei in due]))
        for ei in due:
            e, (_, infos) = envs[ei], cur[ei]
            cur[ei] = ({a: e._obs(a) for a in e.agents}, infos)

    # Reward scale per scenario family (Config.reward_norm): urban missions pay many
    # finds, wilderness ones rarely any, so one scale lets urban dominate the updates.
    # Each family's rewards are divided by the running spread of its discounted returns.
    rstats: dict = {}
    ret_acc = np.zeros(E)

    def family_of(env):
        return getattr(getattr(env, "scenario", None), "family", "grid")

    def rscale(ei, fam, r, done) -> float:
        if not cfg.reward_norm:
            return 1.0
        ret_acc[ei] = ret_acc[ei] * cfg.gamma + r
        n, mean, m2 = rstats.get(fam, (0, 0.0, 0.0))
        n += 1
        d = ret_acc[ei] - mean
        mean += d / n
        m2 += d * (ret_acc[ei] - mean)
        rstats[fam] = (n, mean, m2)
        if done:
            ret_acc[ei] = 0.0
        return 1.0 / float(np.sqrt(m2 / n + 1e-8)) if n > 100 else 1.0

    # Validation (Config.val_every): fixed missions, not the test seeds, flown
    # greedily; the best model so far is kept as best.pt.
    best_val = -1.0
    if cfg.val_every and field:
        from .evaluate import evaluate as run_eval
        from .field.params import sample_scenario
        val_conds = {f: (lambda f_: lambda s_: sample_scenario(np.random.default_rng(90_000 + s_), f_))(f)
                     for f in cfg.sampler.families}
        val_f = open(out / "val_log.csv", "a" if resumed else "w", newline="")
        val_log = csv.writer(val_f)
        if not resumed:
            val_log.writerow(["update", "env_steps", *[f"found_{f}" for f in val_conds], "score"])
        if resumed and (out / "best.pt").exists():
            best_val = float(torch.load(out / "best.pt", weights_only=False).get("val", -1.0))

    def validate(update):
        nonlocal best_val
        net.eval()
        res = run_eval({"p": LearnedPolicy(net, temperature=0)}, val_conds, n_episodes=cfg.val_episodes,
                       workers=max(1, cfg.workers))
        net.train()
        found = [res[f]["p"]["found_frac"] for f in val_conds]
        score = float(np.mean(found))
        val_log.writerow([update, env_steps, *[round(x, 4) for x in found], round(score, 4)])
        val_f.flush()
        print(f"[val] upd {update} found " + " ".join(f"{f} {x:.2f}" for f, x in zip(val_conds, found))
              + f" score {score:.3f} (best {max(best_val, score):.3f})", flush=True)
        if score > best_val:
            best_val = score
            save(net, cfg, out / "best.pt", extra={"val": score, "update": update})

    def vec_rollout(pks):
        """One rollout over all environments in the worker processes: every
        drone of every environment in one network call per step."""
        mets, n = [], 0
        for t in range(T):
            live = [i for i, pk in enumerate(pks) if pk["agents"]]
            cuts = np.cumsum([len(pks[i]["agents"]) for i in live])[:-1]
            O = np.concatenate([pks[i]["o"] for i in live])
            Mk = np.concatenate([pks[i]["m"] for i in live])
            C = np.concatenate([pks[i]["c"] for i in live])
            with torch.no_grad():
                d = net.dist(T_(O), T_(Mk))
                act = d.sample()
                logp = d.log_prob(act).cpu().numpy()
                val = net.value(T_(C)).cpu().numpy()
                act = act.cpu().numpy()
            per = {i: (a_, l_, v_) for i, a_, l_, v_ in
                   zip(live, np.split(act, cuts), np.split(logp, cuts), np.split(val, cuts))}
            out = venv.step([per[i][0] if i in per else np.zeros(0, int) for i in range(len(pks))])
            boots = []
            for ei, (res, _npk) in enumerate(out):
                if ei not in per:
                    continue
                pk, (a_, l_, v_) = pks[ei], per[ei]
                rs = rscale(ei, pk["fam"], float(np.mean(res["rew"])), res["metrics"] is not None)
                for j, i in enumerate(pk["idx"]):
                    buf["obs"][t, ei, i] = pk["o"][j]
                    buf["cin"][t, ei, i] = pk["c"][j]
                    buf["mask"][t, ei, i] = pk["m"][j]
                    buf["act"][t, ei, i] = a_[j]
                    buf["logp"][t, ei, i] = l_[j]
                    buf["val"][t, ei, i] = v_[j]
                    buf["rew"][t, ei, i] = res["rew"][j] * cfg.reward_scale * rs
                    buf["valid"][t, ei, i] = 1.0
                    if res["term"][j] or res["trunc"][j]:
                        buf["term"][t, ei, i] = 1.0
                    if j in res["boot"]:
                        boots.append((ei, i, res["boot"][j]))
                n += 1
                if res["metrics"] is not None:
                    mets.append(res["metrics"])
            if boots:
                with torch.no_grad():
                    bv = net.value(T_(np.stack([b_[2] for b_ in boots]))).cpu().numpy()
                for (ei, i, _), v_ in zip(boots, bv):
                    buf["boot"][t, ei, i] = v_
            pks = [npk for _, npk in out]
        last = np.zeros((E, A), np.float32)
        live = [i for i, pk in enumerate(pks) if pk["agents"]]
        if live:
            with torch.no_grad():
                lv = net.value(T_(np.concatenate([pks[i]["c"] for i in live]))).cpu().numpy()
            k = 0
            for i in live:
                for idx in pks[i]["idx"]:
                    last[i, idx] = lv[k]
                    k += 1
        return pks, last, n, mets

    for update in range(start, n_updates + 1):
        for k in buf:
            buf[k].fill(0)
        if guided:
            frac = (update - 1) / max(1, int(cfg.guide_anneal * n_updates))
            coef = cfg.guide_coef * max(0.0, 1.0 - frac)
            for e in envs:
                e.guide_coef = coef
        if use_vec:
            pks, last_val, n_steps, mets = vec_rollout(pks)
            env_steps += n_steps
            ep_metrics += mets
        else:
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
                    rs = rscale(ei, family_of(env), float(np.mean([rew[a] for a in agents])), not env.agents)
                    for j, (a, i) in enumerate(zip(agents, idx)):
                        buf["obs"][t, ei, i] = o[j]
                        buf["cin"][t, ei, i] = c[j]
                        buf["mask"][t, ei, i] = m[j]
                        buf["act"][t, ei, i] = act[j]
                        buf["logp"][t, ei, i] = logp[j]
                        buf["val"][t, ei, i] = val[j]
                        buf["rew"][t, ei, i] = rew[a] * cfg.reward_scale * rs
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
            kl_coef = cfg.bc_coef * max(cfg.bc_floor, 1.0 - (update - 1) / max(1, int(cfg.bc_anneal * n_updates)))
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
        if cfg.val_every and field and (update % cfg.val_every == 0 or update == n_updates):
            validate(update)
        if update % cfg.ckpt_every == 0 or update == n_updates:
            env_rngs = venv.rng_states() if use_vec else [e.rng.bit_generator.state for e in envs]
            save_checkpoint(ck_path, net, opt, cfg, update, env_steps, time.time() - t0, rng, env_rngs)
    log_f.close()
    if venv is not None:
        venv.close()
    return net


def save_checkpoint(path: Path, net, opt, cfg: Config, update: int, env_steps: int, secs: float, rng, env_rngs):
    """Everything needed to continue training exactly where it stopped (except
    the episodes in flight). Written to a temporary file and renamed, so a
    crash mid-write never leaves a broken checkpoint."""
    c = asdict(cfg)
    c["sampler"] = asdict(cfg.sampler)
    tmp = Path(str(path) + ".tmp")
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "optimizer": opt.state_dict(),
                "config": c, "update": update, "env_steps": env_steps, "secs": secs,
                "torch_rng": torch.get_rng_state(), "np_rng": rng.bit_generator.state,
                "env_rngs": env_rngs}, tmp)
    os.replace(tmp, path)


def save(net: ActorCritic, cfg: Config, path: Path, extra: dict | None = None):
    c = asdict(cfg)
    c["sampler"] = asdict(cfg.sampler)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "config": c, **(extra or {})}, path)


def load(path: str | Path) -> tuple[ActorCritic, dict]:
    ck = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    arch = cfg.get("arch") or ("cnn" if cfg.get("full_map") else "mlp")
    guided = cfg.get("guide", "none") not in (None, "none")
    net = ActorCritic(cfg["sampler"]["grid_size"], cfg["algo"] == "mappo", cfg["hidden"], arch, tf_kwargs(cfg), guided,
                       cfg.get("team", False), cfg["sampler"].get("camera", False), cfg["sampler"].get("real", False),
                       cfg["sampler"].get("field", False), cfg.get("plan_feats", False))
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
        if getattr(self.net, "field", False):
            env.plan_feats = self.net.plan

    def act(self, env, obs=None) -> dict:
        if getattr(self.net, "field", False):
            env.plan_feats = self.net.plan
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
