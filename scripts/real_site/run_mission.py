"""Fly one simulated lost-person mission on a real site and save every step.

    python scripts/real_site/run_mission.py --site sites/catoctin --seed 4 --policy runs/p3/rl/best.pt

The world (terrain, tree cover, water) comes from the site's real layers
(scripts/real_site/fetch_site.py, scripts/import_site.py). The drones, the
missing person and the detections are the simulator's. --policy is a saved
target policy (learned), or "rule" for BeliefGreedy.
"""

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_drone_view import run  # noqa: E402

from swarmsearch.baselines import BeliefGreedy  # noqa: E402
from swarmsearch.evaluate import FIELD_CONDITIONS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", default="sites/catoctin")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--condition", default="wild_mix")
    p.add_argument("--policy", default="runs/p3/rl/best.pt")
    a = p.parse_args()
    import numpy as np
    site = Path(a.site) / "site.npz"
    cell = float(np.load(site)["cell_m"])
    sc = dataclasses.replace(FIELD_CONDITIONS[a.condition](a.seed), site=str(site), cell_m=cell)
    if a.policy == "rule":
        pol, name = BeliefGreedy(), "the rule (BeliefGreedy)"
    else:
        from swarmsearch import targets as tg
        pol, name = tg.TargetPolicy(tg.load(a.policy)[0], temperature=0), "learned target policy"
    env, frames, alerts = run(sc, a.seed, pol)
    found = [f["t"] for f in frames if f["found"].any()]
    out = Path(a.site) / f"mission_seed{a.seed}.pkl"
    keep = dict(frames=frames, alerts=alerts, cell_m=cell, g=env.w.g, base=env.w.base, ipp=tuple(env.w.ipp)
                if hasattr(env.w, "ipp") else env.w.base, step_s=env.fam.step_s, category=sc.category,
                n_drones=sc.n_drones, policy=name, seed=a.seed, condition=a.condition,
                alt_m=tuple(env.fam.alt_m), outside=env.people.outside.copy())
    pickle.dump(keep, open(out, "wb"))
    print(f"{out}: {len(frames)} steps of {env.fam.step_s:.0f} s, {sc.n_drones} drones, {sc.category}, "
          f"found at step {found[0] if found else None}")


if __name__ == "__main__":
    main()
