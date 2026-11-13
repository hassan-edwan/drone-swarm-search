# Cooperative Drone Swarm Search

MSML642 project (Hassan Edwan). A team of drones searches a disaster zone for
survivors. One shared policy, trained with multi-agent reinforcement learning
(MAPPO), has to adapt to each scenario: use likely zones when they exist,
split the area when they don't, re-check where survivors may have moved, and
cover for a lost drone. The goal is to find 90% of survivors faster than a
fixed lawnmower sweep.

## Milestones

| Step | Milestone | Status |
|---|---|---|
| M1 | 2D grid search env, lawnmower and random baselines, MAPPO and IPPO | **In progress** (env, baselines, trainer, evaluation done; training runs next) |
| M2 | 2D grid with obstacles | Set up: block layouts, path-following survivors, `--m2` test conditions; not trained yet |
| M3 | 3D in gym-pybullet-drones, policy waypoints flown by PID | Set up (`scripts/run3d.py`); PyBullet runs need Python ≤ 3.12 |
| M4 | 3D with obstacles | Stretch |
| M5 | 3D camera with visual blockers | Stretch |
| M6 | RL flight policy replaces PID | After the course |

## Layout

```
swarmsearch/
  scenario.py   scenario variables, movement profiles, random sampler
  env.py        2D grid search env (PettingZoo ParallelEnv)
  baselines.py  lawnmower sweep and random walk
  mappo.py      shared-actor PPO with centralized (MAPPO) or local (IPPO) critic
  evaluate.py   named test conditions and metrics
scripts/
  train.py, evaluate.py, render_episode.py
tests/
```

## Setup

```bash
pip install -r requirements.txt   # CPU-only torch is enough
python -m pytest
```

## Running on your own machine

```bash
git clone https://github.com/hassan-edwan/drone-swarm-search && cd drone-swarm-search
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                                        # 11 tests, a few seconds
bash scripts/run_experiments.sh                         # MAPPO + IPPO, 2 seeds each, in parallel, then evaluation
```

The env runs on the CPU, one core per run, so a GPU does not help. Each run
is a separate process: 4 runs in parallel need about 4–8 cores and ~2 GB of
RAM. At ~250 env steps/s per run, 2M steps take about 2–2.5 hours. Results
land in `results/m1/results.md`; models in `runs/`.

## Usage

```bash
# Train (CPU). ~240 env steps/s on 4 cores; use more --envs on a cluster node.
python scripts/train.py --algo mappo --steps 2000000 --out runs/mappo
python scripts/train.py --algo ippo  --steps 2000000 --out runs/ippo

# Actor architecture: local vector only (default), + full map via CNN, or via transformer
python scripts/train.py --algo mappo --arch transformer --steps 2000000 --out runs/mappo_tf

# M2: up to 30% of cells blocked, fine-tuned from the M1 model; evaluate on the obstacle conditions
python scripts/train.py --algo mappo --obstacles 0.3 --init models/m1.pt --out runs/mappo_m2
python scripts/evaluate.py --models runs/mappo_m2/model.pt --m2 --episodes 100

# A second machine: scripts/kaggle/train_kaggle.ipynb runs one training + evaluation on a
# Kaggle CPU session and pushes the model and results to branch kaggle/<TAG>

# M3: fly the policy in 3D (PyBullet Crazyflies + PID, or the point-mass backend anywhere)
pip install -r requirements-3d.txt          # PyBullet wheel needs Python <= 3.12
python scripts/run3d.py --model models/m2.pt --backend pybullet --episodes 20
python scripts/run3d.py --model models/m2.pt --backend pointmass --wind 3 --plot results/m3/flight.png

# Compare everything on the same held-out scenarios
python scripts/evaluate.py --models runs/mappo/model.pt runs/ippo/model.pt --episodes 50

# Picture of one episode
python scripts/render_episode.py --model runs/mappo/model.pt --condition drift_known --out mappo.png
```

## Policy architectures (`--arch`)

| Arch | What the actor sees | How it reads it |
|---|---|---|
| `mlp` | local 5×5 patches, coarse 5×5 map summary, own state, nearest teammates, scenario profile | MLP |
| `cnn` | the above + the full 20×20 shared team map | CNN on the map, MLP on the rest |
| `transformer` | the same as `cnn` | map cut into 4×4 tiles = 25 tokens, plus a self token; a 2-layer transformer encoder lets the drone's token attend over every region before the action head |

All three use the same PPO trainer and the MAPPO/IPPO critics, so results are directly comparable.

## Guidance (`--guide heuristic`)

A rule-based planner gives each drone a target region every 25 steps (best
expected unfound survivors × staleness, minus distance, one drone per
region). The policy gets the direction to its target as input, and during
training a potential-based shaping bonus for moving toward it (Ng et al.
1999) that fades to zero (`--guide-anneal`), so the final policy optimizes
the true reward and can override the guide.

## Running on an HPC cluster (Slurm, e.g. Zaratan)

See the header of `scripts/hpc/train.sbatch` for one-time setup, then:

```bash
ARGS="--guide heuristic" TAG=guided sbatch scripts/hpc/train.sbatch   # set --account first
```

## The environment

- **Grid:** 20 × 20 cells of 10 m (a 200 m × 200 m zone); one step ≈ 1 s.
- **Actions:** move to one of 8 neighbouring cells, or hover. Moves into walls or obstacles are masked.
- **Obstacles (M2):** rectangles of 1–4 cells a side placed until the target fraction is covered (`obstacle_layout="scatter"` gives single cells). Enclosed pockets are filled in, so every open cell is reachable from the base. Survivors heading to safety take shortest paths around obstacles, and so do the movement hypotheses; drifting survivors pile up against them.
- **Search:** the camera searches the cell under the drone and finds each survivor there with the detection probability.
- **Reward (per drone):** the survivor probability mass its search detects (scaled so an unsearched cell with a uniform prior pays about the detection probability); +10 to every drone per survivor found by anyone; −0.01 per step. Collisions cannot happen (shield).
- **Observation (per drone, 167 values):** 5 × 5 local patches of freshness, belief (expected survivors), obstacles and teammates; a coarse 5 × 5 summary of the whole map's freshness and belief (the shared team map); own position, battery, time; the two nearest teammates; the scenario profile.
- **Global state (critic only):** full maps, including true survivor positions, plus scenario info.
- **Belief:** the expected undetected survivor mass per cell. It starts as the prior heat map, drops by the detection probability where a cell is searched, and each step moves with the movement model. Survivors move with the same model under the true profile, so with a known profile the belief is exact. Without a profile, the team keeps one belief map per movement hypothesis (still; wander and drift in 4 directions at 0.5/1.0/1.5 m/s; head to a safe point on a 10 × 10 lattice: 116 in all, `swarmsearch/belief.py`). Empty searches and finds reweight the hypotheses, the shared belief is their mixture, and the profile input is the posterior-expected profile, so the drones work out how survivors move during the search.
- **Movement:** still; wander (random walk); drift (80% of moves one way, e.g. flood water); to_safety (80% of moves toward a safe point). Speed sets how often a survivor changes cell (1 m/s = 10% of steps).

### Scenario variables (`ScenarioSampler`)

| Variable | Range | Default | Do drones know it? |
|---|---|---|---|
| Survivors | 1–20 | 10 | No, only those found |
| Movement | still, wander, drift, to_safety; 0.3–1.5 m/s | — | Only if the operator sets it; otherwise they infer it during the search |
| Likely zones | none in 60% of scenarios, else 1–5 | 3 | Yes, as the prior heat map |
| Zone accuracy | 50–100% of survivors in zones | 80% | No |
| Obstacles | 0–30% of cells, as 1–4 cell blocks (buildings, rubble) | 0 (M1) | Nearby cells only |
| Drones | 2–8, one may fail | 3 | Nearby teammates only |
| Detection per pass | 60–100% | 80% | No |

### Test conditions (`swarmsearch/evaluate.py`)

`still`, `wander`, `drift_known`, `drift_unknown`, `to_safety_known`,
`to_safety_unknown`, `zones_wander`, `drone_lost`, `6_drones`, `random_mix`
(no likely zones unless named; movement at 1 m/s; drift direction and safe
point vary per episode). `--sweep` adds drift at 0, 0.5, 1 and 1.5 m/s.
Every method runs on the same seeds.

Metrics: time to 90% found (steps; the episode length if never reached), share
of episodes reaching 90%, fraction found, coverage, repeat fraction, collisions.

## M3: 3D flight

`swarmsearch/sim3d.py` keeps the 2D environment as the mission layer (belief, survivors, search) and flies every move. Each policy step the shield turns the chosen moves into target cells, then a PID controller flies each drone along a smooth reference to its target cell centre for one step (1 s; one cell = 1 m, Crazyflie scale). A drone that is not within 0.35 m of the cell centre (and 0.3 m of its altitude) at the end of the step does not search that step. Drones cruise on separate altitude layers (1.0 m + 0.25 m each), so paths that cross on the map never meet in the air.

Backends: `pybullet` (gym-pybullet-drones `CtrlAviary`, Crazyflie 2.x, `DSLPIDControl`) and `pointmass` (3D double integrator with acceleration and speed limits, PD tracking and optional wind gusts; runs anywhere and is used by the tests). With no disturbance the point-mass 3D mission reproduces the 2D mission exactly. `scripts/run3d.py` reports 2D vs 3D results on the same missions, plus the share of moves that arrived late, mean tracking error and near misses.
