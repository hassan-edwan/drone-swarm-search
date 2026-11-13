# M5 handoff: generalise the environment and find survivors faster

Brief for the next session working on this repo. Read `README.md` first for the
milestones so far (M1 movement inference, M2 obstacles, M3 3D flight, wind,
M4 buildings).

## Owner's standing preferences
- Keep comparison runs to a minimum; don't chase small percentage gains.
- The owner decides when a milestone is done (no pass/fail threshold scripts).
- Drones cooperate step by step. No hierarchical "RL picks regions" design.
- Most scenarios have **no** likely-zone prior. Moving survivors are the core story.
- Keep the safety shield. No LLM component.
- **Compute: no HPC for now.** One training run at a time on this 4-core machine,
  plus optionally one Kaggle CPU session (4 cores, 12 h; `scripts/kaggle/train_kaggle.ipynb`).
  Never run a CPU-heavy job alongside a training run here (it slows ~10x).
  Detach long jobs (`setsid nohup ... & disown` + a `.done` file).
- Results/models dirs are gitignored: `git add -f`. Run `pytest tests` before committing.

## Where things stand (100-episode evals unless noted)
| Case | MAPPO | Lawnmower |
|---|---|---|
| drift, unknown movement | 83% reached 90% | 41% |
| to_safety, unknown | 78% | 35% |
| obstacles 20%, drift unknown (M2) | 81% | 59% |
| **survivors still** | **38%** | **58%** |
| **drone lost mid-mission** | **4%** | **10%** |
| **obstacles 10%, wander** | **18%** | **19%** |

The bold rows are the weak spots: when survivors barely move, the policy is worse
than systematic sweeping, and it does not reorganise after losing a teammate.
M4 (buildings, PyBullet): MAPPO ahead on all 4 conditions, no hits in calm air;
with 2 m/s² gusts drones occasionally get pushed into walls (flight-layer issue).

## Key fact that saves compute
The actor's observation is already **independent of map size and team size**:
5x5 local patches (`PATCH_RADIUS=2`), a 5x5 pooled coarse map (`COARSE=5`),
normalised position, and the nearest `N_TEAMMATES=2` teammates
(`swarmsearch/env.py`, `_obs`). Only the centralized critic depends on grid size
(`state_dim(grid_size)`), and it is used only in training. So the existing
`models/m2_2m.pt` can be **evaluated on 30x30 maps and 2-6 drones with no
training at all**. Do that first, before building any new architecture.

## Plan, cheapest first
1. **Generalisation check (no training).** Add eval conditions for 30x30 and
   40x40 maps (scale `max_steps` with area) and 2/5/6 drones. Evaluate m2_2m vs
   lawnmower. If it holds up, the "variable map/team size" item is done.
   (The hypothesis bank's safe-point lattice uses `linspace(1, g-2, 10)`, so it scales already.)
2. **Speed metric.** Add area under the found-over-time curve (normalised to [0,1])
   to `evaluate.py` and `run3d.py` tables. This is the "faster" measure.
3. **Fix the weak spots in one training run**, warm-started with `--init models/m2_2m.pt`:
   - more `still` and `drone_lost` scenarios in the training sampler;
   - a small reward for searching cells not visited recently (staleness is already
     in the obs as `freshness()`), so sweeping becomes learnable;
   - optionally weight reward toward early finds (e.g. multiply find bonus by a
     decay in t).
   Train ~2M steps here (or on Kaggle while something else runs here), then one
   comparison vs m2_2m on `--m2` plus the new size/team conditions.
4. **Lookahead re-ranking at run time (no training).** At each step, for each drone,
   take the policy's top-k actions, simulate a few steps on the belief map
   (expected detected mass, with teammates' tentative moves), and pick the best.
   Toggle with a flag; compare with/without. Often a large gain for search, and
   it costs inference time only.
5. **M4 flight safety margin (no training).** Keep drones >= ~0.3 m from walls and
   slow near buildings; rerun only `run3d.py --m4 --wind 2`.

Later (M6), if wanted: limited communication (each drone keeps its own belief,
merges within radio range), terrain-dependent detection probability with false
alarms, battery/return-to-base.

## Useful commands
```
pytest tests
python scripts/evaluate.py --model models/m2_2m.pt --m2 --episodes 100
python scripts/train.py --obstacles 0.3 --init models/m2_2m.pt --steps 2000000 --out models/m5_2m.pt
python scripts/run3d.py --model models/m2_2m.pt --m4 --wind 2 --episodes 10 --out results/m4
```
3D needs the Python 3.12 venv at `/home/user/venv3d` (PyBullet has no 3.13 wheel).
