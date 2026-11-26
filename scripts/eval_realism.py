"""Effect of the realism changes on search results (the rule, same missions for every variant).

    python scripts/eval_realism.py --episodes 100 --workers 3 --out results/realism

Variants:
- old sensing (canopy_model="linear"), no under-canopy checks: the earlier simulator
  (with drones always in contact, the new default)
- new sensing (visibility, calibrated), no under-canopy checks
- new sensing with under-canopy checks of alerts in dense forest (the new default)
on generated wilderness (wild_mix) and on the real Catoctin site (lidar, leaf-on cover).
"""

import argparse
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import BeliefGreedy  # noqa: E402
from swarmsearch.evaluate import FIELD_CONDITIONS  # noqa: E402
from swarmsearch.field.env import FieldEnv  # noqa: E402

VARIANTS = {
    "old sensing": dict(canopy_model="linear", under_canopy=False),
    "new sensing": dict(canopy_model="visibility", under_canopy=False),
    "new sensing + under-canopy checks": dict(canopy_model="visibility", under_canopy=True),
}
WORLDS = {"generated wilderness": None, "real site (Catoctin, lidar)": "sites/catoctin/site.npz",
          "real site (hamlet, lidar + 3D flight limits)": "sites/catoctin_west/site.npz"}


def one(job):
    world, variant, seed = job
    sc = FIELD_CONDITIONS["wild_mix"](seed)
    kw = dict(VARIANTS[variant])
    if WORLDS[world]:
        cell = float(np.load(WORLDS[world])["cell_m"])
        kw.update(site=WORLDS[world], cell_m=cell)
    sc = dataclasses.replace(sc, **kw)
    env = FieldEnv()
    env.reset(seed=seed, options={"scenario": sc})
    pol = BeliefGreedy()
    pol.reset(env)
    ft = None
    while env.agents:
        env.step(pol.act(env))
        if ft is None and env.found.any():
            ft = env.t
    m = env.metrics() if hasattr(env, "metrics") else {}
    return dict(world=world, variant=variant, seed=seed, found=float(env.found.mean()),
                find_min=(ft * env.fam.step_s / 60.0) if ft is not None else None,
                under=int(env.n_under), under_finds=int(env.n_under_finds),
                false_alarms=int(env.n_false_alarms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/realism")
    a = ap.parse_args()
    jobs = [(w, v, s) for w in WORLDS for v in VARIANTS for s in range(a.episodes)]
    with Pool(a.workers) as p:
        rows = p.map(one, jobs, chunksize=4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "results.json", "w"))
    lines = ["| World | Variant | Found | Mean time to first find (min) | Under-canopy checks / mission | "
             "Finds from them | False alarms / mission |", "|---|---|---|---|---|---|---|"]
    for w in WORLDS:
        for v in VARIANTS:
            r = [x for x in rows if x["world"] == w and x["variant"] == v]
            f = np.array([x["found"] for x in r])
            ft = [x["find_min"] for x in r if x["find_min"] is not None]
            se = f.std(ddof=1) / np.sqrt(len(f))
            lines.append(f"| {w} | {v} | {f.mean():.0%} ± {se:.0%} | {np.mean(ft):.0f} | "
                         f"{np.mean([x['under'] for x in r]):.1f} | {sum(x['under_finds'] for x in r)} | "
                         f"{np.mean([x['false_alarms'] for x in r]):.1f} |")
    txt = "\n".join(lines)
    (out / "results.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
