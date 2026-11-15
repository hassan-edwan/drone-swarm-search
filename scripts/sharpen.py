"""Scale a copied policy's action scores so sampling from it behaves like its top action.

    python scripts/sharpen.py models/bc_field.pt models/bc_field_sharp.pt --factor 4
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.mappo import load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--factor", type=float, default=4.0)
    a = p.parse_args()
    ck = torch.load(a.src, weights_only=False, map_location="cpu")
    net = load(a.src)[0]
    last = [n for n, m in net.actor.named_modules() if isinstance(m, torch.nn.Linear)][-1]
    for k in (f"actor.{last}.weight", f"actor.{last}.bias"):
        ck["state_dict"][k] = ck["state_dict"][k] * a.factor
    torch.save(ck, a.dst)
    print(f"scaled actor.{last} by {a.factor}: {a.dst}")


if __name__ == "__main__":
    main()
