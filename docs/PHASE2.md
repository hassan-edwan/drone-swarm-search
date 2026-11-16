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

## On a SLURM cluster

    git clone <repo> ~/drone-swarm-search
    bash scripts/cluster/setup.sh                       # venv, requirements, tests
    bash scripts/cluster/submit.sh --account=<acct> --partition=<cpu>

`submit.sh` submits three jobs, each waiting for the one before:

| Job | Resources (edit the `#SBATCH` lines) |
|---|---|
| imitate | 32 CPUs, 64 GB, 12 h |
| train | an array with one task per seed; 64 CPUs, 64 GB, 24 h each |
| evaluate | 64 CPUs, 12 h |

Training jobs requeue themselves five minutes before their time limit or a
preemption, and continue from the checkpoint. All settings are in
`scripts/cluster/phase2.env`, and any of them can be overridden from the
environment.

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
| `BC_COEF`, `BC_ANNEAL` | 0.3, 1e6 | The pull towards the copy stays on (no drift) |
| `EXPAND_SHARE`, `LONG_SHARE` | 0.3, 0.3 | Growing areas and long operations come up often enough to learn |
| `EVAL_EPISODES` | 200 | ±3% on a found-share, against ±6–7% with 50 |

## What to try if the policy still only matches the rule

1. **Give the policy the whole map.** `--arch cnn` reads the full probability
   map rather than its pooled summary; the copy should then match the rule
   more closely. Imitation does not support it yet: `imitate.py` builds the
   MLP only, and the field environment's full-map observation has not been
   tested.
2. **Let the pull shrink to a floor** instead of holding it fixed: for
   example, `--bc-anneal 1.0` with a minimum weight. A minimum weight is not
   implemented; it would be a few lines in `mappo.py` where `kl_coef` is set.
3. **Score on what can be found.** Use "found (inside area)" next to "found";
   no policy can find someone beyond the area.

## Known limits carried into Phase 2

- **Detection rates** come from published trials, not from the drone and
  camera that will fly. Calibrate them in field tests (ASSUMPTIONS.md, last
  section).
- **Obstacle sensing is assumed.** Drones stop before unknown obstacles; there
  is no local avoidance manoeuvre (D5).
- **Mission pacing:** missions in flight at a checkpoint restart on resume.
  That costs at most one mission per environment.
