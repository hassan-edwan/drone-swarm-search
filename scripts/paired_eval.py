"""Paired comparison of search methods on the same missions: share found, and each method's difference
from the first one with its standard error (luck that helps or hurts both on a mission cancels out).

    python scripts/paired_eval.py --methods rule rule_view planner_team --conditions wild_mix urban_mix \\
        --episodes 100 --workers 8

Methods (name: what it is):
  rule              BeliefGreedy (the reference: list it first)
  rule_view         the rule, valuing looks by viewing direction (env.view_value, S2)
  rule_adaptive     the rule with the map estimating sensor quality online (scenario adaptive_map)
  planner           the rollout planner (Phase 3)
  planner_team      planner, teammates' future looks simulated, expected time to find
  planner_robust    planner_team, scored on 3 perturbed maps (detection rate, map distortion)
  targets:<path>    a saved target policy, greedy
A method can be suffixed "+nohz" to fly the obstacle-free generated worlds (scenario hazards=False).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make(name):
    """(policy, env attributes, scenario overrides) for a method name."""
    from swarmsearch.baselines import BeliefGreedy
    from swarmsearch.planner import PlannerPolicy
    base, _, flag = name.partition("+")
    over = {"hazards": False} if flag == "nohz" else {}
    if base == "rule":
        return BeliefGreedy(), {}, over
    if base == "rule_view":
        return BeliefGreedy(), {"view_value": True}, over
    if base == "rule_adaptive":
        return BeliefGreedy(), {}, {**over, "adaptive_map": True}
    if base == "planner":
        return PlannerPolicy(), {}, over
    if base == "planner_team":
        return PlannerPolicy(team=True, objective="time"), {}, over
    if base == "planner_robust":
        return PlannerPolicy(team=True, objective="time", robust=3), {}, over
    if base.startswith("targets:"):
        from swarmsearch import targets as tg
        return tg.TargetPolicy(tg.load(base[8:])[0], temperature=0), {}, over
    raise ValueError(name)


def one(job):
    import torch
    torch.set_num_threads(1)
    from swarmsearch.evaluate import ALL_CONDITIONS, run_episode
    from swarmsearch.field.env import FieldEnv
    cname, method, seed = job
    pol, attrs, over = make(method)
    sc = ALL_CONDITIONS[cname](seed)
    sc = replace(sc, **over) if over else sc
    env = FieldEnv(sc)
    for k, v in attrs.items():
        setattr(env, k, v)
    t0 = time.time()
    m = run_episode(env, pol, sc, seed)
    return dict(c=cname, m=method, seed=seed, found=float(m.get("found_frac", 0.0)),
                find_time=float(m.get("find_time_frac", np.nan)), secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["rule", "rule_view"])
    ap.add_argument("--conditions", nargs="+", default=["wild_mix", "urban_mix", "flood_mix_heldout"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = [(c, m, s) for c in a.conditions for s in range(a.episodes) for m in a.methods]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    if a.out:
        json.dump(rows, open(a.out, "w"))
    ref = a.methods[0]
    lines = ["| Condition | Method | Found | vs " + ref + " (paired) | Time-weighted find | s / mission |",
             "|---|---|---|---|---|---|"]
    for c in a.conditions:
        R = {(r["m"], r["seed"]): r for r in rows if r["c"] == c}
        for m in a.methods:
            f = np.array([R[(m, s)]["found"] for s in range(a.episodes)])
            d = f - np.array([R[(ref, s)]["found"] for s in range(a.episodes)])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            ft = np.nanmean([R[(m, s)]["find_time"] for s in range(a.episodes)])
            sec = np.mean([R[(m, s)]["secs"] for s in range(a.episodes)])
            vs = "" if m == ref else f"{d.mean():+.3f} ± {se:.3f}"
            lines.append(f"| {c} | {m} | {f.mean():.3f} | {vs} | {ft:.3f} | {sec:.0f} |")
    print("\n".join(lines))
    print(f"({len(rows)} missions, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
