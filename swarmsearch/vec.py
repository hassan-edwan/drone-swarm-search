"""Missions for training run in worker processes (Phase 2: many more missions
per update on a many-core machine).

Each worker owns a slice of the environments. Per step it returns, for every
environment, what the learner needs (observations, action masks, critic
inputs, rewards, ends); the learner batches all drones of all environments
into one network call, then sends the actions back. With workers = 0 the same
code runs the environments in this process.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def critic_input(env, obs, centralized: bool, state, vec: int):
    if not centralized:
        return obs
    return np.concatenate([state, obs[:vec]])


def pack(env, obs, infos, centralized: bool, vec: int) -> dict:
    """What the learner needs to act in this environment now."""
    agents = list(env.agents)
    state = env.state() if centralized else None
    return dict(
        fam=getattr(getattr(env, "scenario", None), "family", "grid"),
        agents=agents,
        idx=[int(a.split("_")[1]) for a in agents],
        o=np.stack([obs[a] for a in agents]) if agents else None,
        m=np.stack([infos[a]["action_mask"] for a in agents]).astype(np.float32) if agents else None,
        c=np.stack([critic_input(env, obs[a], centralized, state, vec) for a in agents]) if agents else None,
    )


def step(env, pk: dict, acts: np.ndarray, centralized: bool, vec: int) -> tuple[dict, dict]:
    """Step one environment; returns the step's results and the next pack
    (after a reset when the mission ended)."""
    agents = pk["agents"]
    nobs, rew, term, trunc, ninfos = env.step({a: int(x) for a, x in zip(agents, acts)})
    res = dict(rew=np.array([rew[a] for a in agents], np.float32),
               term=np.array([bool(term[a]) for a in agents]),
               trunc=np.array([bool(trunc[a]) and not term[a] for a in agents]), boot={}, metrics=None)
    if res["trunc"].any():  # truncated: the learner bootstraps on the value of where it stopped
        nstate = env.state() if centralized else None
        for j, a in enumerate(agents):
            if res["trunc"][j]:
                res["boot"][j] = critic_input(env, nobs[a], centralized, nstate, vec)
    if not env.agents:
        res["metrics"] = env.metrics()
        nobs, ninfos = env.reset()
    return res, pack(env, nobs, ninfos, centralized, vec)


def _worker(conn, make, seeds, centralized, vec):
    import torch
    torch.set_num_threads(1)
    envs = [make(s) for s in seeds]
    pks = []
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            pks = [pack(e, *e.reset(), centralized, vec) for e in envs]
            conn.send(pks)
        elif cmd == "step":
            out = []
            for i, (e, a) in enumerate(zip(envs, arg)):
                r, pks[i] = step(e, pks[i], a, centralized, vec)
                out.append((r, pks[i]))
            conn.send(out)
        elif cmd == "get_rng":
            conn.send([e.rng.bit_generator.state for e in envs])
        elif cmd == "set_rng":
            for e, st in zip(envs, arg):
                e.rng.bit_generator.state = st
            conn.send(None)
        elif cmd == "close":
            conn.close()
            return


class VecEnvs:
    """n_envs environments, in `workers` processes (0: in this process)."""

    def __init__(self, make, seeds: list[int], centralized: bool, vec: int, workers: int = 0):
        self.n = len(seeds)
        self.cv = (centralized, vec)
        self.workers = min(workers, self.n)
        if self.workers == 0:
            self.envs = [make(s) for s in seeds]
            self.procs = []
            return
        ctx = mp.get_context("fork")
        self.slices = np.array_split(np.arange(self.n), self.workers)
        self.conns, self.procs = [], []
        for sl in self.slices:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, make, [seeds[i] for i in sl], centralized, vec), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self) -> list[dict]:
        if not self.procs:
            self.pks = [pack(e, *e.reset(), *self.cv) for e in self.envs]
            return self.pks
        for c in self.conns:
            c.send(("reset", None))
        return [pk for c in self.conns for pk in c.recv()]

    def step(self, actions: list[np.ndarray]) -> list[tuple[dict, dict]]:
        if not self.procs:
            out = []
            for i, (e, a) in enumerate(zip(self.envs, actions)):
                r, self.pks[i] = step(e, self.pks[i], a, *self.cv)
                out.append((r, self.pks[i]))
            return out
        for c, sl in zip(self.conns, self.slices):
            c.send(("step", [actions[i] for i in sl]))
        return [x for c in self.conns for x in c.recv()]

    def rng_states(self) -> list:
        if not self.procs:
            return [e.rng.bit_generator.state for e in self.envs]
        for c in self.conns:
            c.send(("get_rng", None))
        return [st for c in self.conns for st in c.recv()]

    def set_rng_states(self, states: list):
        if not self.procs:
            for e, st in zip(self.envs, states):
                e.rng.bit_generator.state = st
            return
        for c, sl in zip(self.conns, self.slices):
            c.send(("set_rng", [states[i] for i in sl]))
            c.recv()

    def close(self):
        for c in getattr(self, "conns", []):
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
