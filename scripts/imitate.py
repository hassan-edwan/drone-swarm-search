"""Clone the belief-greedy rule into the shared actor (BC + DAgger).

    python scripts/imitate.py --obstacles 0.3 --out models/bc_greedy.pt
    python scripts/train.py --obstacles 0.3 --init models/bc_greedy.pt --bc-anchor models/bc_greedy.pt \\
        --critic-warmup 30 --out runs/bc_rl            # then RL fine-tuning

Round 0 flies the expert; later rounds fly the clone (with the expert taking
over a share beta of steps), label every state with the expert's choice,
and refit on everything collected so far.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.imitation import collect, fit  # noqa: E402
from swarmsearch.mappo import ActorCritic, Config, save  # noqa: E402
from swarmsearch.scenario import CAMERA_STEPS, ScenarioSampler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obstacles", type=float, default=0.3)
    p.add_argument("--episodes", type=int, default=500, help="episodes collected per round")
    p.add_argument("--betas", nargs="*", type=float, default=[1.0, 0.5, 0.25, 0.0],
                   help="share of expert steps per round (first must be 1: pure expert)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--team", action="store_true",
                   help="team layers in the observation (all teammates' positions, trails, intent)")
    p.add_argument("--camera", action="store_true", help="M6 camera mode (100-step missions)")
    p.add_argument("--out", default="models/bc_greedy.pt")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    sampler = ScenarioSampler(obstacle_frac=(0.0, a.obstacles), camera=a.camera,
                              max_steps=CAMERA_STEPS if a.camera else 200)
    cfg = Config(sampler=sampler, seed=a.seed, out_dir=str(Path(a.out).parent), team=a.team)
    net = ActorCritic(sampler.grid_size, True, cfg.hidden, team=a.team, camera=a.camera)
    data = None
    t0 = time.time()
    for r, beta in enumerate(a.betas):
        O, M, A, ms = collect(sampler, a.episodes, a.seed * 100 + r, None if r == 0 else net, beta, a.workers, a.team)
        ft = np.mean([m["find_time_frac"] for m in ms])
        print(f"round {r} beta {beta}: {len(A)} samples, rollout find time {ft:.0%} of budget, "
              f"found {np.mean([m['found_frac'] for m in ms]):.0%} ({time.time() - t0:.0f}s)", flush=True)
        data = (O, M, A) if data is None else tuple(np.concatenate([d, x]) for d, x in zip(data, (O, M, A)))
        acc = fit(net, *data, epochs=a.epochs)
        print(f"round {r}: fit on {len(data[2])} samples, accuracy {acc:.1%}", flush=True)
        save(net, cfg, Path(a.out))
    # Last check: the clone alone.
    _, _, _, ms = collect(sampler, a.episodes // 2, a.seed * 100 + 99, net, 0.0, a.workers, a.team)
    print(f"clone alone: find time {np.mean([m['find_time_frac'] for m in ms]):.0%} of budget, "
          f"found {np.mean([m['found_frac'] for m in ms]):.0%}", flush=True)


if __name__ == "__main__":
    main()
