"""Imitation of the belief-greedy rule (M5 option 3), before RL fine-tuning.

1. Behaviour cloning: run BeliefGreedy on sampled training scenarios and fit
   the shared actor to its actions from each drone's local observation.
2. DAgger rounds: fly the cloned policy (mixed with the expert with
   probability beta), label every visited state with the expert's action,
   add it to the data and refit. Fixes the drift a pure clone shows once it
   leaves the states the expert visits.

The expert is stateful (it keeps a target per drone), so it is stepped
alongside the learner on the same env and keeps updating its targets from
wherever the learner went. Its choice also depends on that target memory,
which the observation does not contain, so a perfect fit is not possible;
RL fine-tuning (mappo.train with a BC anchor) takes over from there.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

from .baselines import BeliefGreedy
from .env import GridSearchEnv
from .mappo import ActorCritic
from .scenario import ScenarioSampler


def _rollouts(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    sampler, n_episodes, seed, state_dict, beta, team, arch, plan, every = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    from .envs import is_field, make_env
    env = make_env(sampler=sampler, seed=seed, team=team, full_map=arch != "mlp", plan_feats=plan)
    net = None
    if state_dict is not None:
        net = ActorCritic(sampler.grid_size, True, arch=arch, team=team, camera=sampler.camera, real=sampler.real,
                          field=is_field(sampler), plan=plan)
        net.load_state_dict(state_dict)
        net.eval()
    O, M, A, S, metrics = [], [], [], [], []
    for _ in range(n_episodes):
        env.reset()
        expert = BeliefGreedy()
        expert.reset(env)
        while env.agents:
            agents = list(env.agents)
            labels = expert.act(env)
            obs = np.stack([env._obs(a) for a in agents])
            mask = np.stack([env.action_mask(a) for a in agents])
            # a drone the autopilot flies can only hover (its mask says so): label it that way
            lab = np.array([labels.get(a, 8) if m[labels.get(a, 8)] else int(np.argmax(m))
                            for a, m in zip(agents, mask)])
            if env.t % every == 0:  # keep every `every`-th step (full-map samples are large)
                O.append(obs.astype(np.float16))
                M.append(mask)
                A.append(lab)
                # every move as good as the rule's (the rule breaks ties arbitrarily)
                sets = np.zeros_like(mask, dtype=np.int8)
                for j, a in enumerate(agents):
                    sets[j, lab[j]] = 1
                    for e in getattr(expert, "equivalent", {}).get(a, []):
                        if mask[j, e]:
                            sets[j, e] = 1
                S.append(sets)
            if net is None or rng.random() < beta:
                acts = labels
            else:
                with torch.no_grad():
                    d = net.dist(torch.from_numpy(obs), torch.from_numpy(mask.astype(np.float32)))
                    acts = {a: int(x) for a, x in zip(agents, d.probs.argmax(-1))}
            env.step(acts)
        metrics.append(env.metrics())
    return (np.concatenate(O).astype(np.float16), np.concatenate(M).astype(np.int8),
            np.concatenate(A).astype(np.int64), np.concatenate(S).astype(np.int8), metrics)


def collect(sampler: ScenarioSampler, n_episodes: int, seed: int, net: ActorCritic | None = None,
            beta: float = 1.0, workers: int = 4, team: bool = False, arch: str = "mlp", plan: bool = False,
            every: int = 1):
    """Returns observations, masks, the rule's actions, the sets of equally good
    actions (multi-hot), and the missions' metrics."""
    sd = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    jobs = [(sampler, n, seed * 1000 + i, sd, beta, team, arch, plan, every) for i, n in enumerate(per) if n]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_rollouts, jobs)
    O, M, A, S = (np.concatenate([p[k] for p in parts]) for k in range(4))
    return O, M, A, S, [m for p in parts for m in p[4]]


def fit(net: ActorCritic, O, M, A, S=None, epochs: int = 4, batch: int = 4096, lr: float = 1e-3,
        log=print) -> float:
    """Fit the actor to the rule. With S (sets of equally good actions), the
    loss is -log of the probability on the whole set, so the copy may pick
    any of them; without, cross-entropy on the rule's one action. Returns the
    share of states where the copy's top action is in the set (or is the action)."""
    opt = torch.optim.Adam(net.actor.parameters(), lr=lr)
    O_t, M_t, A_t = torch.from_numpy(O), torch.from_numpy(M), torch.from_numpy(A)  # converted per batch
    S_t = torch.from_numpy(S) if S is not None else None
    n = len(A_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct, exact = 0.0, 0, 0
        for s in range(0, n, batch):
            j = perm[s:s + batch]
            logits = net.actor(O_t[j].float()).masked_fill(M_t[j] == 0, -1e9)
            if S_t is not None:
                logp = torch.log_softmax(logits, -1)
                loss = -torch.logsumexp(logp.masked_fill(S_t[j] == 0, -1e9), -1).mean()
            else:
                loss = nn.functional.cross_entropy(logits, A_t[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.actor.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(j)
            top = logits.argmax(-1)
            exact += int((top == A_t[j]).sum())
            correct += int(S_t[j].gather(1, top[:, None]).sum()) if S_t is not None else int((top == A_t[j]).sum())
        log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.3f} agreement {correct / n:.3f} (same action {exact / n:.3f})")
    return correct / n
