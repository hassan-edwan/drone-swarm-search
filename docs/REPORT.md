<div align="center">

# Cooperative Drone Swarm Search with Multi-Agent Reinforcement Learning

*Phase 1 Progress Report*

Hassan Edwan · MSML642 · November 2026

</div>

**Summary.** Phase 1 set out to show that one shared policy, trained with
multi-agent reinforcement learning, can coordinate a drone swarm to find 90%
of survivors faster than a fixed lawnmower sweep, first in 2D and then in 3D
flight. Both objectives are met: where survivors move or likely zones exist,
the swarm reaches 90% up to twice as fast and in far more missions, and the
missions fly in a physics simulator. Realistic sensing and limited
communication, planned as later work, were also built, and the simulator was
extended into a field environment grounded in search-and-rescue data in
preparation for large-scale training in Phase 2.

**Setup status.** The simulation, code and compute environment are complete
and operational (Section 1). The search runs in a 2D grid world, in 3D
physics flight, in a field environment built from search-and-rescue data, on
real terrain, and in ArduPilot's flight software; 117 automated tests pass;
and the full pipeline runs on the UMD Zaratan cluster, where four training
and evaluation jobs have completed. Videos of each part are listed in Table 2.
The first cluster results (Section 7) show a learned policy finding more
people than the hand-written rule on real terrain it was never trained on.

## 1. Project Setup

Every part of the planned setup is built, tested and in use. Table 1 lists
each component, and Table 2 the videos that show it working.

*Table 1. Setup components and how each is verified.*

| Component | What is set up | Verified by |
|---|---|---|
| 2D search simulation | Grid world for 2–8 drones (PettingZoo), shared probability map, moving survivors, obstacles, lawnmower / random / rule baselines | Tests; 100-mission evaluations (Table 4); Figure 1 |
| 3D flight | Crazyflie quadrotors in PyBullet (gym-pybullet-drones) under PID control, buildings, wind | 3D missions reproduce the 2D results; Figure 2 |
| Field environment | Urban collapse, wilderness and flood scenarios from lost-person data (ISRID), camera model, operator confirmation, batteries, radio, ground teams | 50–500-mission evaluations (Table 5); Figure 5 |
| Real terrain | Catoctin Mountain Park, Maryland: USGS elevation and lidar, NAIP aerial imagery; 3D obstacle worlds with detect-and-avoid flight | Real-site videos; evaluations on two real sites (Table 6) |
| Autopilot in the loop | ArduPilot copter firmware in software-in-the-loop, over MAVLink, with wind and GPS error | Cluster job flying the planned paths |
| Learning code | MAPPO / IPPO, imitation (behaviour cloning, DAgger), target-choosing policy trained with PPO, rollout planner, population-based tuning | Trained models; cluster runs (Section 7) |
| Tests | 117 automated tests (pytest) | All pass (3 min on 4 cores) |
| Compute | Local Python environment (`requirements.txt`); Slurm pipeline for UMD Zaratan: one-time setup, job scripts, resumable training, status script, results pushed to a git branch | Completed jobs: training (400 iterations, 14.5 h on 128 cores), tuning (8 × 64 cores), two evaluations (3,600 and 7,500 missions) |

*Table 2. Evidence videos (all in `results/figures/`).*

| Video | Shows |
|---|---|
| `all_models.mp4` | One 2D mission flown by eight methods (Figure 1) |
| `flight_3d.mp4`, `flight3d_demo.mp4` | Quadrotors flying the search in PyBullet among buildings (Figure 2) |
| `camera_models.mp4`, `camera_baselines.mp4` | Camera sensing: height, footprints, confirming detections (Figure 3) |
| `real_gaps.mp4` | The world disagreeing with the drones' model (Figure 4) |
| `field_wilderness.mp4`, `field_urban.mp4`, `field_flood.mp4` | The three field scenario families (Figure 5) |
| `field_base_sweep.mp4`, `field_base_rule.mp4`, `field_learned.mp4`, `field_drone_view.mp4` | One wilderness mission flown by the sweep, the rule and the learned policy, and from a drone's cameras |
| `real_site_demo.mp4`, `real_site_cameras.mp4` | A search over real terrain and imagery, and each drone's camera feed |
| `rule_demo.mp4`, `rule_cameras.mp4` | The same real site searched by the rule |
| `avoid_demo.mp4` | A drone detecting and flying around trees and power lines not on its map |

## 2. Objectives

The proposal described a swarm of 2–8 drones searching a disaster area for
survivors under one shared policy trained with MAPPO. The policy should adapt
to each scenario: use likely zones when they exist, split the area when they
do not, re-check where survivors may have moved, and cover for a lost drone.
The measure of success is the time to find 90% of survivors, compared with a
fixed lawnmower sweep. The plan was to show this in a 2D grid world, add
obstacles, fly the missions in 3D with buildings, and then move towards
realistic sensing, limited communication and learned flight control.

*Table 3. Progress against the proposal.*

| Objective | Status | Evidence |
|---|---|---|
| Shared MAPPO policy for 2–8 drones | Achieved | One actor for all drones, any map or team size |
| Find 90% faster than a sweep | Achieved | Where survivors move or zones exist (Table 4) |
| Use likely zones | Achieved | 155 vs 185 steps to 90% |
| Re-check where survivors moved | Achieved | Movement inferred during the search |
| Split the area with no prior | Equal to sweep | Pure coverage, where a sweep is optimal |
| Cover for a lost drone | Equal to sweep | 197 vs 194 steps |
| 2D grid with obstacles | Achieved | Table 4, Figure 1 |
| 3D flight with buildings | Achieved | Figure 2 |
| Realistic sensing | Achieved early | Figure 3 |
| Limited communication | Achieved early | Figure 4 |
| Learned flight control | Phase 2 | Search outputs waypoints for any controller |

## 3. Search Performance

The drones share a probability map of where survivors may be, lowered wherever
they search and shifted each step the way survivors move; when nobody says how
survivors move, the team works it out from what it finds. Table 4 compares the
learned policy with the lawnmower on 100 missions per condition. When
survivors move or the team has likely zones, the swarm reaches 90% up to about
twice as fast and in far more missions. With still survivors and no prior the
task is pure coverage, and the two tie.

*Table 4. Steps to find 90% of survivors (of 200; lower is better) and share
of missions reaching 90%.*

| Condition | MAPPO | Lawnmower |
|---|---|---|
| Walking to a safe point, told | 98 (99%) | 186 (35%) |
| Walking to a safe point, not told | 148 (70%) | 186 (35%) |
| Drifting, told | 156 (79%) | 183 (41%) |
| Likely zones | 155 (56%) | 185 (34%) |
| Obstacles, walking to safety, not told | 125 (82%) | 169 (62%) |
| Random mix of scenarios | 140 (64%) | 151 (57%) |
| Still survivors, no prior | 170 (60%) | 167 (58%) |
| A drone is lost | 197 (6%) | 194 (10%) |

![Figure 1](../results/figures/all_models_frame.png)

*Figure 1. The 2D grid with obstacles: one mission flown by eight methods
([video](../results/figures/all_models.mp4)). Survivors walk to a safe point
the drones are not told about; the map-based methods converge on it.*

## 4. Flight in Three Dimensions

Each step of the policy was flown by quadrotors in the PyBullet physics
simulator under PID control, among buildings taller than the drones. In calm
air the 3D missions reproduce the 2D results, and they hold up to gusts of
about 2 m/s². Because the search layer only chooses the next cell, a learned
flight controller can later replace PID without changing it.

![Figure 2](../results/figures/flight_3d_frame.png)

*Figure 2. A mission flown by Crazyflie quadrotors in PyBullet beside the
team's map ([video](../results/figures/flight_3d.mp4)).*

## 5. Beyond the Plan

A hand-written rule that uses the same map does as well as the learned
policy, and further learning methods (imitation, lookahead, teammate
information, memory) reached the same level. To understand why, the
simulation was made more realistic. A camera model added flying height, sight
lines past buildings, cover and false alarms that must be confirmed from low
altitude (Figure 3); it also showed that the training reward must match the
goal, since early policies hovered high and found nobody until finds were
rewarded for being early. Reinforcement learning on top of a copy of the rule
improved it at first, but drifted away from it once the pull towards the copy
was released; the camera runs ended below the rule.

Next, the world was allowed to disagree with the drones' model: a sensor
weaker than assumed, buildings that had collapsed unseen, radio links that
share maps only within range, and wind and positioning error. Under these
conditions the rule trusts a map that is wrong and falls below a plain sweep
(Figure 4). This is the setting in which a learned policy that adapts to
evidence has room to improve on hand-written strategies.

*Figure 3. Realistic sensing: flying height, footprints and confirmation of
detections ([video](../results/figures/camera_models.mp4)).*
*Figure 4. The world disagrees with the drones' model; the rule loses its
advantage over the sweep ([video](../results/figures/real_gaps.mp4)).*

## 6. Preparation for Phase 2

To give Phase 2 a realistic target, the simulator was extended into a field
environment built from search-and-rescue research, with every assumption
recorded alongside its source ([ASSUMPTIONS.md](ASSUMPTIONS.md)). It covers
urban collapse, wilderness and flood scenarios on real or generated terrain.
Lost-person statistics set where people go and how large the search area is;
people may not respond to drones or may hide; an operator must confirm each
detection; drones swap batteries and lose radio contact behind terrain; and
ground teams carry out the rescue. Realistic wilderness search is hard, mostly because many lost people have
walked beyond any area that can be searched in 90 minutes (Table 5).

A first learned policy was trained here on four CPU cores: a copy of the rule,
refined by 4 million steps of reinforcement learning with the pull towards the
copy kept on. It matches the rule in the wilderness, where both beat the sweep,
does better when radio links fail, and is slightly behind on the unseen flood
scenario. It does not yet improve on the rule overall, which is the target for
Phase 2.

*Table 5. Field environment, 50 missions per condition: share of missions in
which the person is found (in brackets: among people inside the search area).*

| Scenario | Lawnmower | Rule | Learned |
|---|---|---|---|
| Wilderness, nine conditions pooled | 19% (61%) | 24% (66%) | 25% (67%) |
| Wilderness, radio links fail | 15% (57%) | 16% (58%) | 26% (68%) |
| Urban collapse | 75% (81%) | 77% (83%) | 74% (80%) |
| Flood, not seen in training | 56% (68%) | 53% (65%) | 48% (58%) |

Phase 2 will train in this environment at scale: imitation of the rule
followed by reinforcement learning with the find-time reward, across scenario
families, with several seeds and many more missions per update, evaluated on
hard cases and on the unseen flood scenario. This requires more compute than the four CPU cores used in Phase 1,
and will be followed by calibration against field tests and a learned flight
controller.

![Figure 5](../results/figures/field_wilderness_frame.png)

*Figure 5. The field environment: (a) wilderness, a hiker on an 11.6 km map
sized from lost-person data ([video](../results/figures/field_wilderness.mp4));
(b) urban collapse with smoke ([video](../results/figures/field_urban.mp4));
(c) flood, held out for testing ([video](../results/figures/field_flood.mp4)).
The same wilderness mission flown by the sweep
([video](../results/figures/field_base_sweep.mp4)) and the learned policy
([video](../results/figures/field_learned.mp4)), and seen from one drone's
cameras ([video](../results/figures/field_drone_view.mp4)).*

## 7. Progress on the Cluster

Since the field environment, the simulator was made closer to real
operations ([REALISM.md](REALISM.md)): detection under trees from measured
visibility, real map layers, flight among lidar-mapped obstacles, an
autopilot in the loop, and weather and season. Every generated world now has
obstacles, landing spots and terrain wind as well; these cost the rule about
4 points in wilderness and urban search, so the targets are harder than in
Table 5. The training and evaluation then moved to the Zaratan cluster.

- **Learned target policy.** The policy chooses where each drone searches
  next and is trained with PPO, held close to the rule. A 400-iteration run
  (job 24292146) matched or slightly beat the rule in every condition.
- **New training signals and tuning.** Four changes from the literature
  were added: a reward on every look, the rule as a counterfactual baseline,
  replay of hard missions, and population-based tuning
  ([PHASE4.md](PHASE4.md)). Eight policies were tuned together for 200
  iterations, and they converged on the same settings.
- **Search without learning.** Planners that simulate teammates or allow for
  a wrong map did not beat the rule (3,600 paired missions).
- **Result.** The tuned policies were evaluated on 300 fresh missions per
  condition, the same missions for every method (Table 6). On the two real
  sites, which no policy was trained on, every learned policy finds more
  people than the rule. On generated missions they tie it, except member 4,
  which is ahead in every condition.

*Table 6. Share of people found by the rule, and each learned policy's
difference from it in points (± one standard error), 300 paired missions per
condition.*

| Condition | Rule | Member 4 | Member 5 | Member 6 | First run |
|---|---|---|---|---|---|
| Wilderness (generated) | 29.9% | +3.1 ± 2.1 | +0.4 ± 1.9 | +0.7 ± 2.0 | +1.8 ± 2.1 |
| Urban (generated) | 72.0% | +0.8 ± 1.0 | +0.4 ± 1.0 | −0.8 ± 1.1 | −0.6 ± 0.9 |
| Flood (never trained on) | 48.0% | +1.1 ± 1.2 | −0.5 ± 1.2 | −0.9 ± 1.2 | −0.6 ± 1.2 |
| Catoctin (real site) | 9.1% | +1.3 ± 1.0 | +2.3 ± 1.4 | +2.2 ± 1.3 | +0.6 ± 1.1 |
| Hamlet (real site) | 25.0% | +2.3 ± 2.0 | +3.4 ± 1.8 | +3.6 ± 1.6 | +4.4 ± 1.9 |
| **Real sites, pooled** | | +1.8 ± 1.1 | **+2.9 ± 1.1** | **+2.9 ± 1.0** | **+2.5 ± 1.1** |

## 8. Continuity into Phase 2

Phase 2 continues on this setup without changing it:

- **Same environments and missions.** The named evaluation conditions
  ([MISSIONS.md](MISSIONS.md)) stay the same, so every Phase 2 number can be
  compared with the tables above.
- **Same measures.** Share found and time to find, each method compared with
  the rule on the same missions.
- **Opt-in changes.** New options are off by default, so earlier results
  reproduce.
- **Same pipeline.** The cluster scripts and settings stay the same, and
  every run can be resumed.

The next step is a longer training run from the best tuned policy (member 4;
`scripts/cluster/realism_long.sbatch`), followed by the same paired
evaluation. After that come calibration against field data and a learned
flight controller.
