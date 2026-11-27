"""Population-based tuning of the target policy (Jaderberg et al. 2017), asynchronous, through files.

Each member is a training run (scripts/train_targets.py) in its own folder with its own settings
(hp.json). Members train in segments; before each segment a member runs `step`: if its latest paired
validation score (vs_rule in val_log.csv) is in the bottom quarter of the population, it copies the
checkpoint of a member in the top quarter (exploit) and perturbs the settings it copied (explore:
x0.8 or x1.25, choices re-drawn a quarter of the time). Every exploit is logged in pbt_log.jsonl.

    python scripts/pbt.py init --pop 8 --out runs/pbt        # draw the members' settings
    python scripts/pbt.py step --member 3 --out runs/pbt     # exploit / explore before a segment
    python scripts/pbt.py args --member 3 --out runs/pbt     # train_targets.py options for the member
    python scripts/pbt.py report --out runs/pbt              # the population

On the cluster: scripts/cluster/realism_pbt.sbatch (one array task per member).
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

# setting: (kind, low, high) for continuous (log scale when kind == "log"), or ("choice", options)
SPACE = {
    "kl_rule": ("log", 0.005, 0.1),
    "ent_coef": ("lin", 0.0, 0.01),
    "lr": ("log", 1e-4, 1e-3),
    "gamma": ("choice", [0.99, 0.995, 0.998]),
    "pod_weight": ("lin", 0.0, 1.0),
    "cf_decisions": ("choice", [0, 2, 4]),
    "replay_share": ("choice", [0.0, 0.25, 0.5]),
}


def draw(rng, k):
    kind = SPACE[k][0]
    if kind == "choice":
        return SPACE[k][1][int(rng.integers(len(SPACE[k][1])))]
    lo, hi = SPACE[k][1], SPACE[k][2]
    v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" else float(rng.uniform(lo, hi))
    return v


def perturb(rng, hp):
    out = dict(hp)
    for k, spec in SPACE.items():
        if spec[0] == "choice":
            if rng.random() < 0.25:
                out[k] = draw(rng, k)
        else:
            lo, hi = spec[1], spec[2]
            out[k] = float(np.clip(out[k] * (0.8 if rng.random() < 0.5 else 1.25), lo, hi))
    return out


def member_dir(out, m):
    return Path(out) / f"m{m:02d}"


def latest(out, m):
    """(iteration, vs_rule) of the member's latest validation, or None."""
    f = member_dir(out, m) / "val_log.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    rows = [r for r in rows if r.get("vs_rule") not in (None, "", "nan")]
    if not rows:
        return None
    r = rows[-1]
    return int(r["iter"]), float(r["vs_rule"])


def cmd_init(a):
    rng = np.random.default_rng(a.seed)
    for m in range(a.pop):
        d = member_dir(a.out, m)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "hp.json").exists():
            json.dump({k: draw(rng, k) for k in SPACE}, open(d / "hp.json", "w"), indent=1)
    print(f"{a.pop} members in {a.out}")


def cmd_step(a):
    me = latest(a.out, a.member)
    pop = sorted(int(p.name[1:]) for p in Path(a.out).glob("m[0-9][0-9]"))
    scores = {m: latest(a.out, m) for m in pop}
    ready = {m: s for m, s in scores.items() if s is not None}
    if me is None or len(ready) < 4:
        print("pbt: nothing to compare yet")
        return
    order = sorted(ready, key=lambda m: ready[m][1])
    q = max(1, len(order) // 4)
    bottom, top = order[:q], order[-q:]
    if a.member not in bottom:
        print(f"pbt: member {a.member} keeps its weights (vs_rule {me[1]:+.3f})")
        return
    rng = np.random.default_rng([a.member, me[0], int(time.time())])
    src = int(rng.choice(top))
    sd, dd = member_dir(a.out, src), member_dir(a.out, a.member)
    for f in ("ckpt.pt", "model.pt"):
        if (sd / f).exists():
            shutil.copyfile(sd / f, dd / (f + ".tmp"))
            (dd / (f + ".tmp")).replace(dd / f)
    hp = perturb(rng, json.load(open(sd / "hp.json")))
    json.dump(hp, open(dd / "hp.json", "w"), indent=1)
    with open(Path(a.out) / "pbt_log.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.time(), member=a.member, iter=me[0], score=me[1], copied=src,
                                src_score=ready[src][1], hp=hp)) + "\n")
    print(f"pbt: member {a.member} ({me[1]:+.3f}) copies member {src} ({ready[src][1]:+.3f}), new settings {hp}")


def cmd_args(a):
    hp = json.load(open(member_dir(a.out, a.member) / "hp.json"))
    print(" ".join(f"--{k.replace('_', '-')} {v}" for k, v in hp.items()))


def cmd_report(a):
    pop = sorted(int(p.name[1:]) for p in Path(a.out).glob("m[0-9][0-9]"))
    print("| Member | Iteration | vs rule | Settings |\n|---|---|---|---|")
    for m in pop:
        s = latest(a.out, m)
        hp = json.load(open(member_dir(a.out, m) / "hp.json"))
        hs = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in hp.items())
        print(f"| {m} | {s[0] if s else '-'} | {f'{s[1]:+.3f}' if s else '-'} | {hs} |")
    log = Path(a.out) / "pbt_log.jsonl"
    if log.exists():
        print(f"\n{sum(1 for _ in open(log))} exploit steps (pbt_log.jsonl)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "step", "args", "report"])
    ap.add_argument("--out", default="runs/pbt")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dict(init=cmd_init, step=cmd_step, args=cmd_args, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
