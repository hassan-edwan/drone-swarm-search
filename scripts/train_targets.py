"""Train the target-choosing policy (the rule's scores plus a learned correction)
with decision-level PPO on field missions.

    python scripts/train_targets.py --field wilderness urban --iters 100 --episodes 16 --workers 4 --out runs/targets
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.targets_train import TConfig, train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    d = TConfig()
    p.add_argument("--field", nargs="+", default=list(d.families))
    for k in ("iters", "episodes", "workers", "epochs", "minibatch", "critic_warmup", "width", "seed",
              "val_every", "val_episodes"):
        p.add_argument("--" + k.replace("_", "-"), type=int, default=getattr(d, k))
    for k in ("lr", "gamma", "lam", "clip", "ent_coef", "kl_rule", "vf_coef", "keep", "temperature", "beta",
              "expand_share", "long_share", "reward_scale"):
        p.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(d, k))
    p.add_argument("--init", default=None, help="start from a saved target policy (e.g. one trained to imitate the planner)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=4, help="learner threads")
    p.add_argument("--out", default="runs/targets")
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    kw = {k: getattr(a, k) for k in TConfig.__dataclass_fields__ if hasattr(a, k) and k != "families"}
    cfg = TConfig(families=tuple(a.field), **kw)
    train(cfg, a.out, init=a.init, resume=a.resume)


if __name__ == "__main__":
    main()
