"""Compare baselines and trained policies on the same held-out scenarios.

    python scripts/evaluate.py --models runs/mappo/model.pt runs/ippo/model.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import Lawnmower, RandomWalk  # noqa: E402
from swarmsearch.guide import make_guide  # noqa: E402
from swarmsearch.evaluate import CONDITIONS, evaluate, to_markdown  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    p.add_argument("--out", default="results")
    p.add_argument("--guide", default=None, help="override the guide a model was trained with")
    a = p.parse_args()

    policies = {"lawnmower": Lawnmower(), "random": RandomWalk(seed=0)}
    for path in a.models:
        net, cfg = load(path)
        arch = cfg.get("arch") or ("cnn" if cfg.get("full_map") else "mlp")
        name = cfg["algo"] + ("" if arch == "mlp" else f"_{arch}")
        spec = a.guide or cfg.get("guide", "none")
        guide = make_guide(spec) if net.guided else None
        if guide is not None:
            name += "_guided"
        policies[name] = LearnedPolicy(net, guide=guide, guide_every=cfg.get("guide_every", 25))
    conds = {c: CONDITIONS[c] for c in a.conditions}
    res = evaluate(policies, conds, n_episodes=a.episodes)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (out / "results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
