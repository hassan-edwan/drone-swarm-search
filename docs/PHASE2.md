# Phase 2: running at scale

Phase 1 ran on four CPU cores. Its main limit was the learning signal. A
wilderness mission lasts up to 1,100 steps, so each update saw only 1–4
finished missions, and the found-share per update swung between 0 and 1.
Phase 2 runs the same pipeline with many more missions per update, several
seeds and a larger evaluation.

## Pipeline

1. **Imitate the rule** (`scripts/imitate.py`, DAgger). The rule labels every
   drone's move; later rounds fly the copy and keep the rule's corrections.
   Then the copy is sharpened (`scripts/sharpen.py`, action scores ×4).
2. **Reinforcement learning** (`scripts/train.py`, MAPPO). Training starts
   from the sharpened copy, with a KL pull towards it kept on for the whole
   run. Phase 1 showed that every run drifted below the rule once this pull
   faded. Rewards are the find-time reward and ground cleared.
3. **Evaluate** (`scripts/evaluate.py`). The baselines, the copy and each
   seed's model are run on the core conditions and on the mission suite
   ([MISSIONS.md](MISSIONS.md)), each greedy (T0) and as trained.

## What is new for scale

| Feature | Flag | Notes |
|---|---|---|
| Missions in worker processes | `train.py --workers N` | One network call per step for all drones of all missions. 2.2× faster with 4 workers on 4 cores; scales with cores until the learner update dominates |
| Resumable checkpoints | `--resume`, `--ckpt-every` | Model, optimizer, counters and random state; written atomically. A stopped job continues where it was |
| Parallel evaluation | `evaluate.py --workers N` | Results identical to one process |
| Mission suite | `evaluate.py --missions` | 32 missions, one effect each ([MISSIONS.md](MISSIONS.md)) |
| Growing search area | `--expand-share`, `--long-share` | The area doubles in width when the map says the person is more likely outside (K2) |
| GPU learner | `--device cuda` | The network on the GPU, the missions on CPU workers |

## On UMD Zaratan (or another SLURM cluster)

Jobs should run from scratch, not home. Once, on a login node:

    cd ~/scratch
    git clone https://github.com/hassan-edwan/drone-swarm-search.git
    cd drone-swarm-search
    bash scripts/cluster/setup.sh                      # module load python, venv, PyTorch (CPU), requirements
    sbalance                                           # your account name and remaining SUs

Then a 1.5-hour smoke job on 16 cores (about 24 SUs): the tests, the whole
pipeline at tiny settings, and the training speed:

    sbatch -A <account> scripts/cluster/smoke.sbatch
    squeue --me
    tail logs/swarm-smoke-<jobid>.out                   # ends with "speed on 16 cores: N steps/s"

If it passes, submit the full run:

    bash scripts/cluster/submit.sh -A <account>

`submit.sh` submits three jobs, each waiting for the one before:

| Job | Resources (edit the `#SBATCH` lines) | Rough cost |
|---|---|---|
| imitate | 32 CPUs, 12 h limit | 100–400 SUs |
| train | an array with one task per seed; 64 CPUs, 24 h each | 2,000–3,000 SUs for three seeds |
| evaluate | 64 CPUs, 12 h limit | 250–500 SUs |

The costs assume about 75 steps per second per core, as measured here;
replace them using the smoke job's speed. Zaratan gives each core 4 GB, and asking
for more memory is charged as extra cores, so the jobs ask for no memory
and take the default (128 GB for imitation). Slurm output goes to `logs/`,
results to `runs/phase2/`. Check progress with `squeue --me`, and the
training curves in `runs/phase2/rl_seed*/train_log.csv` and `val_log.csv`.
For SU usage afterwards: `sacct -j <jobid> -o JobID,Elapsed,State,AllocTRES%32`.

Training jobs requeue themselves five minutes before their time limit or a
preemption, and continue from the checkpoint. All settings are in
`scripts/cluster/phase2.env`, and any of them can be overridden from the
environment (for example `SEEDS="0 1" bash scripts/cluster/submit.sh -A <account>`).

## On one large machine

    bash scripts/cluster/run_local.sh

This runs the same pipeline without a scheduler. Run it again after a stop:
finished stages are skipped, and training resumes from its checkpoints.

A quick check of the whole pipeline with tiny settings takes about
15 minutes on 4 cores:

    RUN=/tmp/p2 IMITATE_EPISODES=4 BETAS="1.0 0.0" RL_STEPS=8192 N_ENVS=4 SEEDS=0 EVAL_EPISODES=2 \
      bash scripts/cluster/run_local.sh

## Sizing

Measured on the 4-core Phase 1 machine (field missions, wilderness and urban):

| | Steps per second |
|---|---|
| One process | about 130 |
| 4 workers | about 310 (about 75 per core for the missions) |

On a 64-core node, expect a few thousand steps per second. 100M steps is then
roughly 7–14 hours per seed. Confirm with a short run before booking time:

    python scripts/train.py --field wilderness urban --steps 500000 --envs 128 --workers 60 --threads 4 --out /tmp/bench

The `secs` column of `train_log.csv` gives the rate. Imitation data takes
about 0.65 KB per sample, roughly 2,250 samples per mission. At 2,000
missions per round, four rounds come to about 12 GB in memory.

## Settings (`phase2.env`)

| Setting | Default | Why |
|---|---|---|
| `N_ENVS` | 128 | Missions in flight per update: about 30–120 finished missions per update instead of 1–4 |
| `RL_STEPS` | 100M | 25× Phase 1 |
| `SEEDS` | 0 1 2 | Spread between seeds is part of the result |
| `POLICY_ARCH`, `PLAN` | mlp, `--plan-feats` | The pooled map plus what the rule ranks by (best value per step of travel in each direction). Chosen by the comparison below |
| `IMITATE_EVERY` | 1 | Keep every step for the MLP; use 4 with `POLICY_ARCH=cnn` (a full-map sample is about 12 KB) |
| `BC_COEF`, `BC_ANNEAL`, `BC_FLOOR` | 0.3, 1.0, 0.2 | The pull towards the copy falls over the run to a fifth, never to zero: room to improve on the rule without the Phase 1 drift |
| `GAMMA` | 0.995 | Missions run 300–1,100 steps (800 in 4-hour operations); 0.99 sees about 100 steps ahead |
| `REWARD_NORM` | `--reward-norm` | Rewards scaled per family; urban missions pay many more finds than wilderness ones |
| `VAL_EVERY`, `VAL_EPISODES` | 50, 16 | Fixed validation missions; the best model is kept as `best.pt` and is the one evaluated |
| `EXPAND_SHARE`, `LONG_SHARE` | 0.3, 0.3 | Growing areas and long operations come up often enough to learn |
| `EVAL_EPISODES` | 200 | ±3% on a found-share, against ±6–7% with 50 |

## Choosing what the policy sees

Four imitation set-ups were compared on the same small budget (60 missions per
round, three DAgger rounds), each copy evaluated greedily on 40 missions.
Labels are tie-aware: when the rule has several equally good moves, any of
them counts as right.

| Set-up | Agreement | wild_mix found | wild_drones_only found | urban_mix found |
|---|---|---|---|---|
| Rule (BeliefGreedy) | — | 22% | 19% | 69% |
| A: MLP, one label per step | 54.3% | 24% | 22% | 57% |
| B: MLP, tie-aware labels | 54.4% | 25% | 20% | 56% |
| C: MLP, tie-aware, planning features | 54.5% | 23% | 22% | **67%** |
| D: full-map CNN, planning features (every 2nd step) | **57.2%** | 19% | 14% | 61% |

With 40 missions a found-share is ±6–7%, so the wilderness rows are all ties
with the rule. Urban separates them: without planning features the copies
fall 12 points behind the rule; with them (C) the copy is within 2 points.
The CNN matched the rule's moves best but flew worse, on half as many
samples; at cluster scale it may catch up, so it stays an option
(`POLICY_ARCH=cnn IMITATE_EVERY=4`). Results: `runs/p2_compare/`.

## What to try if the policy still only matches the rule

1. **The full-map network with more data:** `POLICY_ARCH=cnn IMITATE_EVERY=4`.
2. **A lower floor on the pull** (`BC_FLOOR=0.1`), or a larger entropy
   bonus, so reinforcement learning can move further from the rule.
3. **Score on what can be found.** Use "found (inside area)" next to "found";
   no policy can find someone beyond the area.
4. **Not yet built:** a curriculum from easy to hard missions; a reward for
   rescue time rather than find time; clues given to the policy directly
   rather than only through the map.

## Known limits carried into Phase 2

- **Detection rates** come from published trials, not from the drone and
  camera that will fly. Calibrate them in field tests (ASSUMPTIONS.md, last
  section).
- **Obstacle sensing is assumed.** Drones stop before unknown obstacles; there
  is no local avoidance manoeuvre (D5).
- **Mission pacing:** missions in flight at a checkpoint restart on resume.
  That costs at most one mission per environment.
