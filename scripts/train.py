"""Train MAPPO or IPPO on randomized 2D grid scenarios.

    python scripts/train.py --algo mappo --steps 2000000 --out runs/mappo
    python scripts/train.py --algo ippo  --steps 2000000 --out runs/ippo
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.mappo import Config, train  # noqa: E402
from swarmsearch.scenario import ScenarioSampler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["mappo", "ippo"], default="mappo")
    p.add_argument("--steps", type=int, default=2_000_000, help="total env steps")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--obstacles", type=float, default=0.0, help="max obstacle fraction (M2: 0.3)")
    p.add_argument("--full-map", action="store_true", help="actor also sees the whole shared map (CNN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    cfg = Config(
        algo=a.algo,
        total_env_steps=a.steps,
        n_envs=a.envs,
        full_map=a.full_map,
        seed=a.seed,
        out_dir=a.out or f"runs/{a.algo}",
        sampler=ScenarioSampler(grid_size=a.grid, obstacle_frac=(0.0, a.obstacles)),
    )
    train(cfg)


if __name__ == "__main__":
    main()
