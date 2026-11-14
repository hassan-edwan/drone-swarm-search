"""Memory check: clone the belief-greedy rule into a GRU actor (BC + DAgger).

    python scripts/imitate_gru.py --obstacles 0.3 --out models/bc_greedy_gru.pt

Same protocol and seeds as scripts/imitate.py, so the printed accuracies and
the "clone alone" check compare directly with the feed-forward clone.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.memory import RecurrentActor, collect, fit  # noqa: E402
from swarmsearch.scenario import ScenarioSampler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obstacles", type=float, default=0.3)
    p.add_argument("--episodes", type=int, default=500, help="episodes collected per round")
    p.add_argument("--betas", nargs="*", type=float, default=[1.0, 0.5, 0.25, 0.0])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/bc_greedy_gru.pt")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    sampler = ScenarioSampler(obstacle_frac=(0.0, a.obstacles))
    net = RecurrentActor()
    data: list = []
    t0 = time.time()
    for r, beta in enumerate(a.betas):
        seqs, ms = collect(sampler, a.episodes, a.seed * 100 + r, None if r == 0 else net, beta, a.workers)
        n = sum(len(s[2]) for s in seqs)
        print(f"round {r} beta {beta}: {n} samples, rollout find time "
              f"{np.mean([m['find_time_frac'] for m in ms]):.0%} of budget, "
              f"found {np.mean([m['found_frac'] for m in ms]):.0%} ({time.time() - t0:.0f}s)", flush=True)
        data += seqs
        acc = fit(net, data, epochs=a.epochs)
        print(f"round {r}: fit on {sum(len(s[2]) for s in data)} samples, accuracy {acc:.1%} "
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.save({"state_dict": net.state_dict(), "args": vars(a)}, a.out)
    _, ms = collect(sampler, a.episodes // 2, a.seed * 100 + 99, net, 0.0, a.workers)
    print(f"clone alone: find time {np.mean([m['find_time_frac'] for m in ms]):.0%} of budget, "
          f"found {np.mean([m['found_frac'] for m in ms]):.0%}", flush=True)


if __name__ == "__main__":
    main()
