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


def _rollouts(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    sampler, n_episodes, seed, state_dict, beta = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = GridSearchEnv(sampler=sampler, seed=seed)
    net = None
    if state_dict is not None:
        net = ActorCritic(sampler.grid_size, True)
        net.load_state_dict(state_dict)
        net.eval()
    O, M, A, metrics = [], [], [], []
    for _ in range(n_episodes):
        env.reset()
        expert = BeliefGreedy()
        expert.reset(env)
        while env.agents:
            agents = list(env.agents)
            labels = expert.act(env)
            obs = np.stack([env._obs(a) for a in agents])
            mask = np.stack([env.action_mask(a) for a in agents])
            O.append(obs)
            M.append(mask)
            A.append(np.array([labels.get(a, 8) for a in agents]))
            if net is None or rng.random() < beta:
                acts = labels
            else:
                with torch.no_grad():
                    d = net.dist(torch.from_numpy(obs), torch.from_numpy(mask.astype(np.float32)))
                    acts = {a: int(x) for a, x in zip(agents, d.probs.argmax(-1))}
            env.step(acts)
        metrics.append(env.metrics())
    return (np.concatenate(O).astype(np.float16), np.concatenate(M).astype(np.int8),
            np.concatenate(A).astype(np.int64), metrics)


def collect(sampler: ScenarioSampler, n_episodes: int, seed: int, net: ActorCritic | None = None,
            beta: float = 1.0, workers: int = 4):
    sd = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    jobs = [(sampler, n, seed * 1000 + i, sd, beta) for i, n in enumerate(per) if n]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_rollouts, jobs)
    O, M, A = (np.concatenate([p[k] for p in parts]) for k in range(3))
    return O, M, A, [m for p in parts for m in p[3]]


def fit(net: ActorCritic, O, M, A, epochs: int = 4, batch: int = 4096, lr: float = 1e-3,
        log=print) -> float:
    """Cross-entropy on the expert's actions (masked like the policy). Returns accuracy."""
    opt = torch.optim.Adam(net.actor.parameters(), lr=lr)
    O_t, M_t, A_t = torch.from_numpy(O), torch.from_numpy(M), torch.from_numpy(A)  # converted per batch
    n = len(A_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, correct = 0.0, 0
        for s in range(0, n, batch):
            j = perm[s:s + batch]
            logits = net.actor(O_t[j].float()).masked_fill(M_t[j] == 0, -1e9)
            loss = nn.functional.cross_entropy(logits, A_t[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.actor.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(j)
            correct += int((logits.argmax(-1) == A_t[j]).sum())
        log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.3f} acc {correct / n:.3f}")
    return correct / n
