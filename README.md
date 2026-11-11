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
| M2 | 2D grid with obstacles | Env supports it (`--obstacles 0.3`); not trained yet |
| M3 | 3D in gym-pybullet-drones, policy waypoints flown by PID | Planned |
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

# M2: same, with up to 30% of cells blocked
python scripts/train.py --algo mappo --obstacles 0.3 --out runs/mappo_m2

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

## LLM guidance (`--guide`)

A pretrained language model downloaded from Hugging Face acts as a high-level
planner. Every 25 steps it reads a text summary of the search (5×5 regions with
expected unfound survivors and how recently each was searched, the movement
profile, drone positions, time left) and answers with a target region per
drone as JSON. The policy gets the direction to its target as input, and
during training a potential-based shaping bonus for moving toward it
(Ng et al. 1999) that fades to zero (`--guide-anneal`), so the final policy
optimizes the true reward and can override bad advice. This follows
LLM-guided exploration in RL (e.g. ELLM, Du et al. 2023).

```bash
pip install -r requirements-llm.txt
python scripts/train.py --algo mappo --guide llm:Qwen/Qwen2.5-7B-Instruct --guide-device cuda --envs 32
python scripts/train.py --algo mappo --guide heuristic     # rule-based guide: fast, for development
```

Unparseable LLM answers fall back to the rule-based guide; the training log
records the parse rate. On a CPU the LLM is far too slow (a 1.5B model takes
~70 s per batch of 8 prompts), so guided training needs a GPU.

## Running on an HPC cluster (Slurm, e.g. Zaratan)

See the header of `scripts/hpc/train_guided.sbatch` for one-time setup
(venv, model download to scratch), then:

```bash
sbatch scripts/hpc/train_guided.sbatch     # set --account first
```

## The environment

- **Grid:** 20 × 20 cells of 10 m (a 200 m × 200 m zone); one step ≈ 1 s.
- **Actions:** move to one of 8 neighbouring cells, or hover. Moves into walls or obstacles are masked.
- **Search:** the camera searches the cell under the drone and finds each survivor there with the detection probability.
- **Reward (per drone):** +1 × staleness of the searched cell (1 for a never-searched cell); −0.5 for searching a fresh cell; +10 to every drone per survivor found by anyone; −10 per drone-drone collision; −0.01 per step.
- **Observation (per drone, 167 values):** 5 × 5 local patches of freshness, belief (expected survivors), obstacles and teammates; a coarse 5 × 5 summary of the whole map's freshness and belief (the shared team map); own position, battery, time; the two nearest teammates; the scenario profile.
- **Global state (critic only):** full maps, including true survivor positions, plus scenario info.
- **Belief and staleness:** the belief starts as the prior heat map, drops where a cell is searched, and spreads according to the movement profile the drones were given. Searched cells go stale at the profile's movement rate, so re-checks pay off only when survivors are expected to move.

### Scenario variables (`ScenarioSampler`)

| Variable | Range | Default | Do drones know it? |
|---|---|---|---|
| Survivors | 1–20 | 10 | No, only those found |
| Movement | still, drift, wander; 0–0.5 m/s | slow wander (0.2 m/s) | Only if the operator sets it; otherwise they get the default profile |
| Likely zones | 0–5 | 3 | Yes, as the prior heat map |
| Zone accuracy | 50–100% of survivors in zones | 80% | No |
| Obstacles | 0–30% of cells | 0 (M1) | Nearby cells only |
| Drones | 2–8, one may fail | 3 | Nearby teammates only |
| Detection per pass | 60–100% | 80% | No |

### Test conditions (`swarmsearch/evaluate.py`)

`default`, `no_zones`, `still`, `drift_known`, `drift_default`, `drone_lost`,
`6_drones`, `random_mix`. Every method runs on the same seeds.

Metrics: time to 90% found (steps; the episode length if never reached), share
of episodes reaching 90%, fraction found, coverage, repeat fraction, collisions.
