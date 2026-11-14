"""Memory check (M5): can a recurrent actor copy the belief-greedy rule better?

The rule keeps a target per drone across steps, which a feed-forward actor
cannot see. Here the actor gets a GRU, so it can carry its own "plan" from
step to step. Same BC + DAgger protocol as imitation.py, but data are kept
as one sequence per drone per episode and the fit runs the GRU over whole
episodes (no truncation, so memory can span the full mission).

The head sees both the current-step features and the GRU state, so the
recurrent actor can do anything the feed-forward one can.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

from .baselines import BeliefGreedy
from .env import N_ACTIONS, GridSearchEnv, obs_dim
from .scenario import ScenarioSampler


class RecurrentActor(nn.Module):
    def __init__(self, obs: int | None = None, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(nn.Linear(obs or obs_dim(), hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_ACTIONS))

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None):
        """obs: (batch, time, obs_dim). Returns logits (batch, time, 9) and the last state."""
        e = self.enc(obs)
        out, h = self.gru(e, h)
        return self.head(torch.cat([e, out], -1)), h


def _rollouts(args):
    sampler, n_episodes, seed, state_dict, beta = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = GridSearchEnv(sampler=sampler, seed=seed)
    net = None
    if state_dict is not None:
        net = RecurrentActor()
        net.load_state_dict(state_dict)
        net.eval()
    seqs, metrics = [], []
    for _ in range(n_episodes):
        env.reset()
        expert = BeliefGreedy()
        expert.reset(env)
        buf = {a: ([], [], []) for a in env.agents}
        hid = {a: None for a in env.agents}
        while env.agents:
            agents = list(env.agents)
            labels = expert.act(env)
            obs = np.stack([env._obs(a) for a in agents])
            mask = np.stack([env.action_mask(a) for a in agents])
            for i, a in enumerate(agents):
                buf[a][0].append(obs[i])
                buf[a][1].append(mask[i])
                buf[a][2].append(labels.get(a, 8))
            acts = labels
            if net is not None:
                # The learner's memory must follow every step it sees, whoever acts.
                with torch.no_grad():
                    h0 = torch.cat([hid[a] if hid[a] is not None else torch.zeros(1, 1, net.hidden)
                                    for a in agents], 1)
                    logits, h1 = net(torch.from_numpy(obs).unsqueeze(1), h0)
                    logits = logits[:, 0].masked_fill(torch.from_numpy(mask) == 0, -1e9)
                for i, a in enumerate(agents):
                    hid[a] = h1[:, i : i + 1]
                if rng.random() >= beta:
                    acts = {a: int(x) for a, x in zip(agents, logits.argmax(-1))}
            env.step(acts)
        for o, m, y in buf.values():
            if o:
                seqs.append((np.stack(o).astype(np.float16), np.stack(m).astype(np.int8),
                             np.array(y, dtype=np.int64)))
        metrics.append(env.metrics())
    return seqs, metrics


def collect(sampler: ScenarioSampler, n_episodes: int, seed: int, net: RecurrentActor | None = None,
            beta: float = 1.0, workers: int = 4):
    sd = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    jobs = [(sampler, n, seed * 1000 + i, sd, beta) for i, n in enumerate(per) if n]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_rollouts, jobs)
    return [s for p in parts for s in p[0]], [m for p in parts for m in p[1]]


def fit(net: RecurrentActor, seqs: list, epochs: int = 4, batch: int = 32, lr: float = 1e-3,
        log=print) -> float:
    """Masked cross-entropy over whole episodes, padded per batch. Returns accuracy."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_steps = sum(len(s[2]) for s in seqs)
    order = np.argsort([len(s[2]) for s in seqs])  # similar lengths per batch: little padding
    chunks = [order[i : i + batch] for i in range(0, len(order), batch)]
    for ep in range(epochs):
        tot, correct = 0.0, 0
        for c in np.random.permutation(len(chunks)):
            idx = chunks[c]
            T = max(len(seqs[i][2]) for i in idx)
            O = torch.zeros(len(idx), T, seqs[idx[0]][0].shape[1])
            M = torch.zeros(len(idx), T, N_ACTIONS, dtype=torch.int8)
            Y = torch.full((len(idx), T), -100, dtype=torch.int64)  # -100 = padding, ignored
            for k, i in enumerate(idx):
                o, m, y = seqs[i]
                O[k, : len(y)] = torch.from_numpy(o.astype(np.float32))
                M[k, : len(y)] = torch.from_numpy(m)
                Y[k, : len(y)] = torch.from_numpy(y)
            M[Y == -100] = 1
            logits, _ = net(O)
            logits = logits.masked_fill(M == 0, -1e9)
            loss = nn.functional.cross_entropy(logits.reshape(-1, N_ACTIONS), Y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            valid = Y != -100
            tot += float(loss.detach()) * int(valid.sum())
            correct += int(((logits.argmax(-1) == Y) & valid).sum())
        log(f"  epoch {ep + 1}/{epochs} loss {tot / n_steps:.3f} acc {correct / n_steps:.3f}")
    return correct / n_steps
