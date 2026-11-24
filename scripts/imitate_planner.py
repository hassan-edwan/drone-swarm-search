"""Train the target-choosing policy to imitate the rollout planner (DAgger).

    python scripts/imitate_planner.py --field wilderness urban --episodes 40 --betas 1.0 0.5 --workers 4 --out runs/targets/planner_bc.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.targets_train import TConfig, imitate_planner  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", nargs="+", default=["wilderness", "urban"])
    p.add_argument("--episodes", type=int, default=40, help="missions per round")
    p.add_argument("--betas", nargs="+", type=float, default=[1.0, 0.5])
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--keep", type=float, default=0.5, help="share of decisions kept as labels")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/targets/planner_bc.pt")
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imitate_planner(TConfig(families=tuple(a.field), workers=a.workers, seed=a.seed), a.out, a.episodes,
                    betas=a.betas, horizon=a.horizon, epochs=a.epochs, keep=a.keep)


if __name__ == "__main__":
    main()
