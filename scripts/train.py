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
    p.add_argument("--arch", choices=["mlp", "cnn", "transformer"], default="mlp",
                   help="actor: local vector only (mlp), + full map via CNN, or + full map via transformer")
    p.add_argument("--full-map", action="store_true", help="same as --arch cnn (kept for old commands)")
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-dim", type=int, default=64)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--guide", default="none", choices=["none", "heuristic"])
    p.add_argument("--guide-every", type=int, default=25, help="env steps between guide queries")
    p.add_argument("--guide-anneal", type=float, default=0.6,
                   help="fraction of training over which the guide's shaping bonus fades to 0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    cfg = Config(
        algo=a.algo,
        total_env_steps=a.steps,
        n_envs=a.envs,
        arch="cnn" if a.full_map else a.arch,
        tf_layers=a.tf_layers,
        tf_d_model=a.tf_dim,
        device=a.device,
        guide=a.guide,
        guide_every=a.guide_every,
        guide_anneal=a.guide_anneal,
        seed=a.seed,
        out_dir=a.out or f"runs/{a.algo}",
        sampler=ScenarioSampler(grid_size=a.grid, obstacle_frac=(0.0, a.obstacles)),
    )
    train(cfg)


if __name__ == "__main__":
    main()
