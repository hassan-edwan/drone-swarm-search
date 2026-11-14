"""Compare baselines and trained policies on the same held-out scenarios.

    python scripts/evaluate.py --models runs/mappo/model.pt runs/ippo/model.pt
    python scripts/evaluate.py --models runs/mappo_m2/model.pt --m2   # obstacle conditions
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import BeliefGreedy, Lawnmower, RandomWalk  # noqa: E402
from swarmsearch.guide import make_guide  # noqa: E402
from swarmsearch.evaluate import (ALL_CONDITIONS, CAMERA_CONDITIONS, CONDITIONS, M2_CONDITIONS, M5_CONDITIONS, SPEED_SWEEP,  # noqa: E402
                                  evaluate, to_markdown)  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--conditions", nargs="*", default=None, choices=list(ALL_CONDITIONS),
                   help="default: all main conditions (plus the speed sweep with --sweep)")
    p.add_argument("--sweep", action="store_true", help="also run the survivor-speed sweep")
    p.add_argument("--m2", action="store_true", help="run the M2 obstacle conditions instead of the M1 ones")
    p.add_argument("--m5", action="store_true", help="run the M5 map-size and team-size conditions")
    p.add_argument("--camera", action="store_true", help="run the M6 camera conditions")
    p.add_argument("--temps", nargs="*", type=float, default=[1.0],
                   help="action temperatures for each model (1 = as trained, 0 = always the top action)")
    p.add_argument("--lookahead", type=int, default=0, help="also run each model with H-step lookahead re-ranking")
    p.add_argument("--random", action="store_true", help="include the random-walk baseline")
    p.add_argument("--tag", default="", help="suffix for the results files")
    p.add_argument("--out", default="results")
    p.add_argument("--guide", default=None, help="override the guide a model was trained with")
    a = p.parse_args()

    policies = {"lawnmower": Lawnmower(), "belief_greedy": BeliefGreedy()}
    if a.random:
        policies["random"] = RandomWalk(seed=0)
    for path in a.models:
        net, cfg = load(path)
        arch = cfg.get("arch") or ("cnn" if cfg.get("full_map") else "mlp")
        name = (Path(path).stem if len(a.models) > 1 else cfg["algo"]) + ("" if arch == "mlp" else f"_{arch}")
        spec = a.guide or cfg.get("guide", "none")
        guide = make_guide(spec) if net.guided else None
        if guide is not None:
            name += "_guided"
        kw = dict(guide=guide, guide_every=cfg.get("guide_every", 25))
        for t in a.temps:
            policies[name + ("" if t == 1.0 else f"_T{t:g}")] = LearnedPolicy(net, temperature=t, **kw)
        if a.lookahead:
            policies[name + f"_look{a.lookahead}"] = LearnedPolicy(net, lookahead=a.lookahead, **kw)
    main_set = CAMERA_CONDITIONS if a.camera else M5_CONDITIONS if a.m5 else M2_CONDITIONS if a.m2 else CONDITIONS
    names = a.conditions or list(main_set) + (list(SPEED_SWEEP) if a.sweep else [])
    conds = {c: ALL_CONDITIONS[c] for c in names}
    res = evaluate(policies, conds, n_episodes=a.episodes)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    (out / f"results{tag}.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (out / f"results{tag}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
