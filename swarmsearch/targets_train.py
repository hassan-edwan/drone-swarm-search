"""Decision-level PPO for the target-choosing policy (swarmsearch.targets).

Each iteration flies `episodes` missions with the current policy in worker
processes, records every target decision (a drone's reward runs until its
next decision, discounted by the steps in between), and updates the network
on those decisions with the clipped PPO loss. A KL term towards the rule's
own choice distribution (the policy with no correction) can hold learning
near the rule. Resumable: model, optimizer, iteration and random state are
checkpointed atomically.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import targets as tg


@dataclass
class TConfig:
    families: tuple = ("wilderness", "urban")
    iters: int = 100
    episodes: int = 16
    workers: int = 4
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    kl_rule: float = 0.02
    vf_coef: float = 0.5
    keep: float = 0.25
    reward_scale: float = 0.05  # a find is worth up to 20: about 1 after scaling
    temperature: float = 1.0
    critic_warmup: int = 3
    width: int = 32
    beta: float = 5.0
    expand_share: float = 0.0
    long_share: float = 0.0
    seed: int = 0
    val_every: int = 10
    val_episodes: int = 8
    adv_by_family: bool = True  # normalise advantages within each family (urban pays far more finds)
    pod_weight: float = 0.0  # share of the find reward paid per look as the true chance of a detection (env.reward_pod)
    # counterfactual rule baseline: at this many sampled steps per mission, copy the mission, give one deciding
    # drone the rule's target instead, fly both copies cf_horizon steps with the same random numbers, and use
    # the difference in that drone's discounted return as the decision's advantage (blended by cf_weight)
    cf_decisions: int = 0
    cf_horizon: int = 120
    cf_weight: float = 1.0
    # prioritised replay of missions (PLR): this share of each iteration's missions are seeds flown before,
    # picked by rank of their score (mean |advantage|: where the value estimate is most wrong) and staleness
    replay_share: float = 0.0
    replay_size: int = 2000
    extra: dict = field(default_factory=dict)


def _sampler(cfg: TConfig):
    from .field.params import FieldSampler
    return FieldSampler(families={f: 1.0 / len(cfg.families) for f in cfg.families},
                        expand_share=cfg.expand_share, long_share=cfg.long_share)


def run_mission(args):
    """One mission with the given weights; returns the recorded decisions and metrics."""
    state, cfg, seed, temperature, record = args
    torch.set_num_threads(1)
    import copy
    from .envs import make_env
    net = tg.TargetNet(cfg.width, cfg.beta)
    net.load_state_dict(state)
    net.eval()
    env = make_env(sampler=_sampler(cfg), seed=seed)
    env.reward_pod = cfg.pod_weight if record else 0.0
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rec = tg.Recorder(cfg.gamma, cfg.lam, cfg.keep, rng) if record else None
    pol = tg.TargetPolicy(net, temperature=temperature, recorder=rec, seed=seed + 2)
    pol.reset(env)
    cf_steps = set()
    if record and cfg.cf_decisions > 0:
        crng = np.random.default_rng([seed, 11])
        cf_steps = set(int(t) for t in crng.choice(max(1, env.fam.max_steps - 1), size=min(
            2 * cfg.cf_decisions, max(1, env.fam.max_steps - 1)), replace=False))
    pending = []  # (decision dict, drone, start step, the rule's branch return)
    streams: dict = {}  # per drone, scaled reward per step (for the main branch's returns)
    n_cf = 0
    while env.agents:
        t = env.t
        snap = None
        if t in cf_steps and n_cf < cfg.cf_decisions:
            snap = copy.deepcopy((env, pol))
            rec.force_keep = True
        acts = pol.act(env)
        if rec is not None:
            rec.force_keep = False
        if snap is not None:
            new = [(a, o) for a, o in rec.open.items() if o["t"] == t and o["x"] is not None]
            if new:
                a, o = new[int(np.random.default_rng([seed, t]).integers(len(new)))]
                env_b, pol_b = snap
                pol_b.recorder, pol_b.rule_for = None, a
                acts_b = pol_b.act(env_b)
                tb, tm = pol_b.cam_targets.get(a), pol.cam_targets.get(a)
                if pol_b.rule_for is None and tb is not None and tm is not None and tb[:2] == tm[:2]:
                    o["cf"] = 0.0  # the rule would have chosen the same
                else:
                    G = 0.0
                    for h in range(cfg.cf_horizon):
                        if not env_b.agents:
                            break
                        _, rb, *_ = env_b.step(acts_b if h == 0 else pol_b.act(env_b))
                        G += cfg.gamma ** h * rb.get(a, 0.0) * cfg.reward_scale
                    pending.append((o, a, t, G))
                n_cf += 1
        _, rew, *_ = env.step(acts)
        if rec is not None:
            rec.reward({a: r * cfg.reward_scale for a, r in rew.items()}, t)
            for a_, r in rew.items():
                streams.setdefault(a_, {})[t] = r * cfg.reward_scale
    for o, a, t0, G_b in pending:  # main branch: the same drone's return over the same horizon
        st = streams.get(a, {})
        G_m = sum(cfg.gamma ** h * st.get(t0 + h, 0.0) for h in range(cfg.cf_horizon))
        o["cf"] = G_m - G_b
    samples = rec.finish(env.t) if rec is not None else []
    for s in samples:
        s["fam"] = env.fam.name
        if "cf" in s:
            s["adv"] = (1.0 - cfg.cf_weight) * s["adv"] + cfg.cf_weight * s["cf"]
    m = env.metrics()
    score = float(np.mean([abs(s["adv"]) for s in samples])) if samples else 0.0
    return samples, dict(found=float(m.get("found_frac", 0.0)), family=env.fam.name, decisions=len(samples),
                         seed=seed, score=score, n_cf=sum("cf" in s for s in samples))


class Replay:
    """Prioritised level replay (Jiang et al. 2021): mission seeds with scores; picks by rank of score,
    mixed with staleness (iterations since last flown) so old scores get refreshed."""

    def __init__(self, size=2000, beta=0.3, rho=0.1):
        self.size, self.beta, self.rho = size, beta, rho
        self.score: dict = {}
        self.last: dict = {}

    def pick(self, rng, n, it):
        if n <= 0 or len(self.score) < 2 * n:
            return []
        seeds = np.array(list(self.score))
        sc = np.array([self.score[s] for s in seeds])
        rank = np.empty(len(sc))
        rank[np.argsort(-sc)] = np.arange(1, len(sc) + 1)
        p_s = (1.0 / rank) ** (1.0 / self.beta)
        p_s /= p_s.sum()
        stale = np.array([it - self.last[s] for s in seeds], float)
        p_c = stale / stale.sum() if stale.sum() > 0 else np.full(len(sc), 1.0 / len(sc))
        p = (1 - self.rho) * p_s + self.rho * p_c
        return [int(x) for x in rng.choice(seeds, size=n, replace=False, p=p)]

    def update(self, seed, score, it):
        self.score[int(seed)] = float(score)
        self.last[int(seed)] = it
        if len(self.score) > self.size:  # drop the lowest scores
            for s in sorted(self.score, key=self.score.get)[:len(self.score) - self.size]:
                del self.score[s], self.last[s]

    def state(self):
        return dict(score=self.score, last=self.last)

    def load(self, d):
        self.score = {int(k): v for k, v in d.get("score", {}).items()}
        self.last = {int(k): v for k, v in d.get("last", {}).items()}


def _batches(samples, size, rng):
    """Minibatches of decisions with the same map size (no padding needed)."""
    by_g: dict = {}
    for s in samples:
        by_g.setdefault(s["x"].shape[1], []).append(s)
    out = []
    for group in by_g.values():
        idx = rng.permutation(len(group))
        for i in range(0, len(idx), size):
            out.append([group[j] for j in idx[i:i + size]])
    rng.shuffle(out)
    return out


def ppo_update(net, opt, samples, cfg: TConfig, rng, critic_only=False) -> dict:
    # advantage scale: per family, so that the family paying the most finds (urban) does not set the
    # direction of every update and drown the other one
    norm = {}
    if cfg.adv_by_family:
        for f in {s.get("fam") for s in samples}:
            a_ = np.array([s["adv"] for s in samples if s.get("fam") == f])
            norm[f] = (a_.mean(), a_.std() + 1e-8)
    adv = np.array([s["adv"] for s in samples])
    glob = (adv.mean(), adv.std() + 1e-8)
    scale = lambda s: norm.get(s.get("fam"), glob)  # noqa: E731
    stats = dict(pl=[], vl=[], ent=[], kl=[])
    for _ in range(cfg.epochs):
        for mb in _batches(samples, cfg.minibatch, rng):
            g = mb[0]["x"].shape[1]
            X = torch.from_numpy(np.stack([s["x"] for s in mb]).astype(np.float32))
            M = torch.from_numpy(np.stack([s["mask"] for s in mb]))
            A = torch.tensor([s["idx"] for s in mb])
            old = torch.tensor([s["logp"] for s in mb])
            ADV = torch.tensor([(s["adv"] - scale(s)[0]) / scale(s)[1] for s in mb], dtype=torch.float32)
            RET = torch.tensor([s["ret"] for s in mb], dtype=torch.float32)
            logits, prior, v = net(X, torch.ones(len(mb), 1, g, g), M)
            lp = torch.log_softmax(logits, 1)
            vl = F.mse_loss(v, RET)
            if critic_only:
                loss = cfg.vf_coef * vl
                pl = ent = kl = torch.zeros(())
            else:
                new = lp.gather(1, A[:, None]).squeeze(1)
                ratio = torch.exp(new - old)
                pl = -torch.min(ratio * ADV, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ADV).mean()
                p = lp.exp()
                ent = -(p * lp.clamp(min=-50)).sum(1).mean()
                kl = (p * (lp - torch.log_softmax(prior, 1)).clamp(min=-50)).sum(1).mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent + cfg.kl_rule * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            for k, x in zip(("pl", "vl", "ent", "kl"), (pl, vl, ent, kl)):
                stats[k].append(float(x.detach()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


VAL_SEED = 90_000


def validate(pool, net, cfg: TConfig, base: dict | None = None) -> dict:
    """Greedy on fixed missions per family (the same every time). With `base` (the rule's per-mission
    results on the same missions), also the paired difference from the rule and its standard error:
    luck that helps or hurts both on a mission cancels out."""
    out, per = {}, {}
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for f in cfg.families:
        c = TConfig(**{**asdict(cfg), "families": (f,), "expand_share": 0.0, "long_share": 0.0})
        jobs = [(state, c, VAL_SEED + i, 0.0, False) for i in range(cfg.val_episodes)]
        res = pool.map(run_mission, jobs) if pool else list(map(run_mission, jobs))
        per[f] = [m["found"] for _, m in res]
        out[f] = float(np.mean(per[f]))
    out["score"] = float(np.mean([out[f] for f in cfg.families]))
    if base is not None:
        # score difference per mission index: the mean over families of (policy - rule) on that mission
        d = np.mean([np.array(per[f]) - np.array(base[f]) for f in cfg.families], axis=0)
        out["vs_rule"] = float(d.mean())
        out["vs_rule_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    out["per"] = per
    return out


def rule_baseline(pool, cfg: TConfig, out_dir: str) -> dict:
    """The rule's per-mission validation results (an untrained net is exactly the rule), cached."""
    import json
    path = os.path.join(out_dir, f"rule_val_{cfg.val_episodes}.json")
    if os.path.exists(path):
        return json.load(open(path))
    per = validate(pool, tg.TargetNet(cfg.width, cfg.beta), cfg)["per"]
    json.dump(per, open(path, "w"))
    return per


def train(cfg: TConfig, out_dir: str, init: str | None = None, resume: bool = False, log=print):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = tg.TargetNet(cfg.width, cfg.beta)
    if init:
        net.load_state_dict(tg.load(init)[0].state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    it0, secs0, best = 0, 0.0, -1.0
    replay_state = None
    ck = os.path.join(out_dir, "ckpt.pt")
    if resume and os.path.exists(ck):
        d = torch.load(ck, weights_only=False)
        net.load_state_dict(d["net"])
        opt.load_state_dict(d["opt"])
        it0, secs0, best = d["iter"], d["secs"], d.get("best", -1.0)
        replay_state = d.get("replay")
        rng.bit_generator.state = d["rng"]
        torch.set_rng_state(d["torch_rng"])
        log(f"[targets] resumed at iteration {it0}")
    new_log = not (resume and os.path.exists(os.path.join(out_dir, "train_log.csv")))
    lf = open(os.path.join(out_dir, "train_log.csv"), "w" if new_log else "a", newline="")
    lw = csv.writer(lf)
    if new_log:
        lw.writerow(["iter", "missions", "decisions", "found", *[f"found_{f}" for f in cfg.families],
                     "pl", "vl", "ent", "kl_rule", "beta", "secs"])
    vf = open(os.path.join(out_dir, "val_log.csv"), "w" if new_log else "a", newline="")
    vw = csv.writer(vf)
    if new_log:
        vw.writerow(["iter", *[f"found_{f}" for f in cfg.families], "score", "vs_rule", "vs_rule_se"])
    ctx = mp.get_context("fork")
    pool = ctx.Pool(cfg.workers) if cfg.workers > 0 else None
    replay = Replay(cfg.replay_size)
    if replay_state:
        replay.load(replay_state)
    t0 = time.time() - secs0
    base = None

    def val_row(it_, v):
        vw.writerow([it_, *[round(v[f], 4) for f in cfg.families], round(v["score"], 4),
                     round(v.get("vs_rule", float("nan")), 4), round(v.get("vs_rule_se", float("nan")), 4)])
        vf.flush()

    def show(v):
        return {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "per"}

    try:
        if it0 < cfg.iters and cfg.val_every:
            base = rule_baseline(pool, cfg, out_dir)
        if it0 == 0 and cfg.val_every:
            v = validate(pool, net, cfg, base)
            val_row(0, v)
            best = v["score"]
            tg.save(os.path.join(out_dir, "best.pt"), net, dict(val=show(v), iter=0, cfg=asdict(cfg)))
            log(f"[targets] validation at start{' (the rule)' if init is None else ''}: {show(v)}")
        for it in range(it0, cfg.iters):
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            n_rep = int(round(cfg.replay_share * cfg.episodes))
            old = replay.pick(rng, n_rep, it)
            seeds = list(old) + [int(x) for x in rng.integers(0, 2**31 - 1, cfg.episodes - len(old))]
            jobs = [(state, cfg, int(s), cfg.temperature, True) for s in seeds]
            res = pool.map(run_mission, jobs) if pool else list(map(run_mission, jobs))
            samples = [s for r, _ in res for s in r]
            mets = [m for _, m in res]
            if cfg.replay_share > 0:
                for m in mets:
                    replay.update(m["seed"], m["score"], it)
            st = ppo_update(net, opt, samples, cfg, rng, critic_only=it < cfg.critic_warmup)
            fam_found = {f: np.mean([m["found"] for m in mets if m["family"] == f] or [np.nan]) for f in cfg.families}
            row = [it + 1, len(mets), len(samples), round(float(np.mean([m["found"] for m in mets])), 4),
                   *[round(float(fam_found[f]), 4) for f in cfg.families],
                   round(st["pl"], 5), round(st["vl"], 5), round(st["ent"], 4), round(st["kl"], 5),
                   round(float(torch.exp(net.log_beta.detach())), 3), round(time.time() - t0)]
            lw.writerow(row)
            lf.flush()
            log(f"[targets] iter {it + 1}/{cfg.iters} decisions {len(samples)} found {row[3]:.2f} "
                f"kl {st['kl']:.4f} ent {st['ent']:.3f} {row[-1]}s")
            if cfg.val_every and (it + 1) % cfg.val_every == 0:
                v = validate(pool, net, cfg, base)
                val_row(it + 1, v)
                if v["score"] > best:
                    best = v["score"]
                    tg.save(os.path.join(out_dir, "best.pt"), net, dict(val=show(v), iter=it + 1, cfg=asdict(cfg)))
                log(f"[targets] validation: {show(v)} (best {best:.3f})")
            tg.save(os.path.join(out_dir, "model.pt"), net, dict(iter=it + 1, cfg=asdict(cfg)))
            tmp = ck + ".tmp"
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), iter=it + 1, secs=time.time() - t0,
                            best=best, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            replay=replay.state()), tmp)
            os.replace(tmp, ck)
    finally:
        if pool:
            pool.close()
            pool.join()
        lf.close()
        vf.close()
    return net


# ------------------------------------------------------------ imitating the planner
def label_mission(args):
    """One mission flown by the planner (beta) or the student, with every
    decision labelled by the planner. Returns (inputs, masks, labels), metrics."""
    state, cfg, seed, beta, keep, horizon = args
    torch.set_num_threads(1)
    from .envs import make_env
    from .planner import PlannerPolicy

    net = tg.TargetNet(cfg.width, cfg.beta)
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    rng = np.random.default_rng(seed + 3)
    data = []

    class Teacher(PlannerPolicy):
        def choose(self, env, a, S, V, taken, D):
            best = super().choose(env, a, S, V, taken, D)
            X, mask = tg.inputs(env, a, S, V, taken, D)
            g = S.shape[1]
            if mask.any() and rng.random() < keep:
                idx = best[0] * g * g + best[1] * g + best[2]
                if mask.flat[idx]:
                    data.append((X.astype(np.float16), mask, idx))
            if state is None or rng.random() < beta or not mask.any():
                return best
            with torch.no_grad():
                lg, _, _ = net(torch.from_numpy(X)[None], torch.ones(1, 1, g, g), torch.from_numpy(mask)[None])
            idx = int(torch.argmax(lg[0]))
            k, rem = divmod(idx, g * g)
            ty, tx = divmod(rem, g)
            return int(k), int(ty), int(tx)

    env = make_env(sampler=_sampler(cfg), seed=seed)
    env.reset(seed=seed)
    pol = Teacher(horizon=horizon)
    pol.reset(env)
    while env.agents:
        env.step(pol.act(env))
    m = env.metrics()
    return data, dict(found=float(m.get("found_frac", 0.0)), family=env.fam.name,
                      changed=pol.n_changed / max(1, pol.n_decisions))


def fit_labels(net, data, epochs=4, lr=1e-3, minibatch=128, seed=0, log=print):
    """Cross-entropy on the teacher's choices; the value head is not trained here."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith("value")], lr=lr)
    samples = [dict(x=x, mask=m, idx=i) for x, m, i in data]
    net.train()
    for ep in range(epochs):
        losses, hits = [], []
        for mb in _batches(samples, minibatch, rng):
            g = mb[0]["x"].shape[1]
            X = torch.from_numpy(np.stack([s["x"] for s in mb]).astype(np.float32))
            M = torch.from_numpy(np.stack([s["mask"] for s in mb]))
            A = torch.tensor([s["idx"] for s in mb])
            logits, _, _ = net(X, torch.ones(len(mb), 1, g, g), M)
            loss = F.cross_entropy(logits, A)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            hits.append(float((logits.argmax(1) == A).float().mean()))
        log(f"  epoch {ep + 1}/{epochs} loss {np.mean(losses):.3f} agreement {np.mean(hits):.3f}")
    net.eval()
    return float(np.mean(hits))


def imitate_planner(cfg: TConfig, out: str, episodes: int, betas=(1.0, 0.5), horizon: int = 60,
                    epochs: int = 4, keep: float = 0.5, log=print):
    """DAgger rounds: the planner flies (beta) or the student does; the planner labels."""
    net = tg.TargetNet(cfg.width, cfg.beta)
    rng = np.random.default_rng(cfg.seed)
    ctx = mp.get_context("fork")
    data: list = []
    with ctx.Pool(cfg.workers) as pool:
        for r, beta in enumerate(betas):
            state = None if r == 0 else {k: v.detach().clone() for k, v in net.state_dict().items()}
            seeds = rng.integers(0, 2**31 - 1, episodes)
            res = pool.map(label_mission, [(state, cfg, int(s), beta, keep, horizon) for s in seeds])
            for d, _ in res:
                data.extend(d)
            mets = [m for _, m in res]
            log(f"round {r} beta {beta}: {len(data)} labels, found {np.mean([m['found'] for m in mets]):.2f}, "
                f"planner changed the rule's choice {np.mean([m['changed'] for m in mets]):.0%}")
            acc = fit_labels(net, data, epochs=epochs, seed=cfg.seed + r, log=log)
            log(f"round {r}: agreement with the planner {acc:.1%}")
            tg.save(out, net, dict(teacher="planner", horizon=horizon, rounds=r + 1, cfg=asdict(cfg)))
    return net
