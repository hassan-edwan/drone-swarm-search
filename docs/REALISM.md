# Making the simulator closer to real operations

Changes made to move the simulator toward something that transfers to a real
search, and what each one does to the results.

## 1. Drones are always in contact

Assumption: every drone is always in contact with the command post (cellular,
satellite or a mesh). `FieldScenario.always_linked = True` is the default:
one radio group with the base every step, no returns to regain contact. The
earlier radio model (line-of-sight range, terrain shadowing, link losses) is
still available with `always_linked=False`, and the two radio-limited
conditions (`wild_hard_radio`, `urban_radio`) keep it.

Consequence: the one place where learning beat the rule in Phase 2 (a short
radio range) no longer applies by default.

## 2. Detection under trees, from measured visibility (thermal and colour)

Each drone carries a colour and a thermal camera; the simulator uses
whichever detects better. Under trees, the old model kept thermal at 75% of
its open-ground rate even under full canopy, which the field literature does
not support. The new model (`canopy_model="visibility"`, the default) follows
what the field studies find:

- Detection from the air depends mainly on how much of the ground is
  visible through the canopy: its gap fraction, 1 - cover (thermal drone
  wildlife surveys in forest, *Ecological Informatics* 2023).
- Thermal needs only part of a warm body to show (factor gap^0.5); a
  colour camera needs more of the person and needs light (gap^1.2).
- Calibration: one pass at 40 m over dense leaf-on forest (cover 0.9) finds
  a person with thermal 15% of the time (colour 4%); open ground 73%
  (colour 85% by day). Thermal searches for carcasses in dense forest
  found under 25% (*Animals* 2025). Schedl et al. (2020, airborne optical
  sectioning) report that single-view thermal in dense forest is generally
  not workable, and found 38 of 42 hidden people only by combining many
  viewpoints. The simulator's per-direction hiding rewards looking again
  from a new side for the same reason.
- `canopy_model="linear"` reproduces the old model for earlier results.

### In standard SAR terms (`swarmsearch/field/sar.py`)

Effective sweep width W: the width of strip a single pass clears (as many
people found outside it as missed inside). Coverage = W × track length / area;
probability of detection for an area searched at coverage c is
POD = 1 - exp(-c) (39% at c = 0.5, 63% at 1, 86% at 2). Sweep widths of the
simulator's drone at cruise (10 m/s, motion blur, average sensor quality),
day (10:00, 15 °C) and night:

| Height | Footprint | Tree cover | Sensor | Day W (m) | Day km²/h | Night W (m) | Night km²/h |
|---|---|---|---|---|---|---|---|
| 40 m | 72 m | 0% | colour / thermal / both | 32 / 16 / 32 | 1.16 | 1 / 32 / 32 | 1.16 |
| 40 m | 72 m | 50% | both | 11 | 0.41 | 19 | 0.67 |
| 40 m | 72 m | 80% | both | 5 | 0.19 | 10 | 0.37 |
| 40 m | 72 m | 90% | both | 3 | 0.13 | 7 | 0.25 |
| 70 m | 126 m | 0% | both | 47 | 1.70 | 47 | 1.70 |
| 70 m | 126 m | 50% | both | 17 | 0.60 | 27 | 0.98 |
| 70 m | 126 m | 90% | both | 5 | 0.18 | 10 | 0.36 |
| 120 m | 216 m | 0% | both | 38 | 1.36 | 38 | 1.36 |
| 120 m | 216 m | 90% | both | 4 | 0.15 | 8 | 0.29 |

(`python -c "from swarmsearch.field import sar; print(sar.table())"` prints
every combination.) Two things stand out: at night thermal does all the
work, and in dense forest one drone clears well under 0.5 km² an hour at
any useful POD. That is why real teams combine drones with ground teams
under canopy, and why checking alerts from under the trees (section 4)
matters.

## 3. Real map layers

`scripts/real_site/` builds a site from public data (any US location):

- **Elevation and trees: USGS 3DEP lidar** (`lidar.py`, read straight from the
  public Entwine point tiles on AWS). Bare-earth elevation, crown height,
  and tree cover. This project (MD_Western_2_D21, 2021) was flown leaf-off,
  like most 3DEP lidar: the share of laser returns blocked by branches
  (70% at Catoctin) understates summer cover. The area under crowns (86%)
  is close to leaf-on cover, and is the default (`build_site.py --season`).
  The lidar does not label trees or buildings in this project (both
  "unclassified"), so buildings come from footprints.
- **Buildings: Microsoft US Building Footprints** (`buildings.py`), heights
  from the lidar.
- **Trails and roads: USGS National Transportation Dataset; lakes: National
  Hydrography Dataset** (`fetch_vectors.py`). Lost people tend to follow
  trails and roads. Streams are left to the simulator, which derives
  drainages from the elevation.
- `build_site.py` combines them into the simulator's world.

Two sites: `sites/catoctin` (Catoctin Mountain Park, 11.6 km, sized for a
lost hiker) and `sites/catoctin_west` (3.2 km around a hamlet at the forest
edge: houses, fields, forest and 150 m of relief; 50 m search cells).

## 4. Checking alerts from under the trees

From 40 m above dense forest a drone still cannot see through the leaves, so
a second look from above changes little. When a drone at its lowest search
height reaches an unresolved alert where tree cover is at least 50%, it now
drops through a gap in the canopy and flies under the crowns, about 3 m off
the ground, to the alerted spot. Alerts are located to a few metres in the
image. It looks with nothing between camera and person, then climbs back out
(field env mode `under`). The time is charged: descent, about 100 m under
the crowns at 2 m/s, and the climb. The detection chance up close is 0.9
(trunks and undergrowth can still hide someone), and false alarms
(animals, warm stumps) are easy to dismiss up close. Metrics: `under_checks`,
`under_finds`.

## 5. Flying through real obstacles in 3D (`swarmsearch/flight3d.py`)

The search runs on a coarse grid; `flight3d` turns each move into the path a
drone would fly through the lidar-mapped obstacles at 1 m:

- **Trees:** crown top and crown bottom per square metre, and 7,051 individual
  trees in the 1.2 km core (trunks under crown peaks).
- **Clutter at flying height:** lidar returns between 1 and 5 m, i.e.
  undergrowth and low branches.
- **Buildings:** with their measured heights.

How the drones move:

- **Cruise:** at the search height, following the terrain, always at least
  5 m over any crown or roof within 3 m, climbing early for what is ahead at
  the drone's climb and descent rates.
- **Under the canopy:** 3 m off the ground, at least 2 m below the crowns,
  1.5 m from trunks and 3 m from buildings, at 2 m/s. The drone enters and
  leaves through a real opening at least 3 m in radius, chosen to make the
  whole trip quickest. Where the spot is deep in undergrowth, the planner
  reports how close it could get (15 m or more counts as "could not get
  close").

Demo (`fly_mission.py`, `render_flight3d.py`, results/figures/flight3d_demo.mp4):
`wild_mix` seed 103 on `catoctin_west` at 09:00 (a lost hunter, 3 drones,
the rule).

- **Drone 1 checks an alert from under the trees 10 s in.** It descends
  through a gap 20 m from the spot and flies 45 m under the crowns, getting
  within 1.7 m of the hunter. The planned trip takes 64 s; the simulator
  charged 80 s. The hunter is confirmed at step 29.
- **Clearances:** 3.0 m above ground, 4.4 m below crowns and 3.5 m from
  trunks under the canopy; at least 37 m over crowns and roofs in cruise.
- **Planner test after the mission (labelled in the video):** a route under
  the canopy to a spot behind houses. It passes 5.0 m from a house, 2.9 m
  from trunks and 3.2 m below crowns, over 92 m.

## What the changes do to the results

`scripts/eval_realism.py`: the rule, 100 missions per row, the same missions
in every row (wild_mix scenarios; on the real site the terrain, trees, trails
and lakes are Catoctin's, cover leaf-on). All rows have drones always in
contact. ± is one standard error.

| World | Variant | Found | Mean time to first find (min) | Under-canopy checks / mission | Finds from them | False alarms / mission |
|---|---|---|---|---|---|---|
| generated wilderness | old sensing | 27% ± 4% | 18 | 0.0 | 0 | 22.6 |
| generated wilderness | new sensing | 33% ± 5% | 19 | 0.0 | 0 | 19.3 |
| generated wilderness | new sensing + under-canopy checks | 35% ± 5% | 21 | 0.2 | 3 | 18.6 |
| real site (Catoctin, lidar) | old sensing | 17% ± 4% | 25 | 0.0 | 0 | 3.5 |
| real site (Catoctin, lidar) | new sensing | 10% ± 3% | 20 | 0.0 | 0 | 3.4 |
| real site (Catoctin, lidar) | new sensing + under-canopy checks | 11% ± 3% | 21 | 0.3 | 0 | 3.5 |

- **Dense real forest is much harder.** At Catoctin (86% leaf-on cover) the
  calibrated sensing finds 10% of people within the 90-minute mission, down
  from 17% with the old model. Most of the generated wilderness is less
  densely wooded, and there the new model's lower per-direction hiding
  slightly outweighs the lower visibility (33% against 27%, within about one
  and a half standard errors).
- **Under-canopy checks are rarely triggered,** 0.2 to 0.3 per mission, and
  produced 3 finds in 100 generated missions and none at Catoctin; the gain is
  within noise. The reason is upstream: they only start from an alert, and
  from above in dense forest alerts are rare. A real team would also search
  under the canopy along likely routes (trails, drainages), or send ground
  teams. That is the natural next change if under-canopy flight is to matter.
- **Earlier results no longer apply.** The learned policies were trained
  under the old sensing and radio; the rule needs no retraining, which is
  why it is used here.

## 6. Autopilot in the loop: ArduPilot SITL

`scripts/real_site/sitl_fly.py` flies the planned 3D paths with ArduPilot's
copter firmware in software-in-the-loop: the real flight-control code with
simulated physics, sensors, GPS and wind (`SIM_WIND_SPD`, with turbulence),
one copter process per drone, over MAVLink, as a ground station or companion
computer would. Setup per copter: GUIDED mode, arm, take off, then position
targets.

- **A flyable clock.** The search grid moves one cell per step, diagonals and
  sprints included, with corners at full speed. Before flying, each path gets
  a speed profile a real quad can fly: at most 20 m/s (about what a SAR quad
  like the M30T is rated for), 4 m/s up, 3 m/s down, 4 m/s² of acceleration,
  slowing for sharp turns.
- **Path following.** The target sent to ArduPilot is a short way ahead of
  the closest point on the path: 1.5 s ahead in cruise, 0.6 s under the
  canopy. A copter that falls behind catches up along the path instead of
  cutting corners.
- **Checking the flown track.** It is checked against the lidar obstacles
  afterwards (SITL itself has no trees). SITL's ground is flat, so its home
  is put at the lowest terrain of the site; heights are compared with the
  real terrain.
- **Firmware note.** ArduCopter 4.7 renamed its speed and angle parameters
  to SI units (`WP_SPD` in m/s, `ATC_ANGLE_MAX` in degrees). The script
  reads every parameter back after setting it.

Demo mission (seed 103 on the hamlet site) plus the planner test:

| | Wind 5 m/s | Wind 10 m/s |
|---|---|---|
| Off the path, mean (worst drone) | 0.1–0.5 m (max 2.0 m) | 0.1–3.3 m (max 22 m) |
| Under the canopy: off the path / height, worst | 0.38 m / 0.47 m | 0.49 m / 0.31 m (test drone) |
| Closest flown: trunk / building / crown above | 3.0 / 5.1 / 3.2 m | 3.1 / 5.1 / 3.1 m |
| Extra time: flyable plan vs grid clock; ArduPilot on top | +13–39%; +4–28 s per drone | same; +33–91 s |

What this shows:

- **ArduPilot follows the under-canopy routes to within half a metre.** In
  10 m/s wind the cruise legs drift and lag much more; that is near the
  operating limit of small quads anyway.
- **The search grid's clock is optimistic.** Flown at real speeds, slowing
  for corners, the same moves take 13–39% longer. A grid policy should pay
  for diagonals, sprints and turns as the aircraft does. This is the clearest
  "sim-to-real" gap found so far.
- **A first batch run found one close call.** On two missions it found a
  flight passing 0.6 m under a crown: the crown-bottom estimate from leaf-off
  lidar is noisier than the undergrowth test the planner uses. The cluster
  run measures how often.

## 7. Running at scale on Zaratan

Three independent jobs (scripts/cluster/realism_*.sbatch, settings in
realism.env):

| Job | What | Time on one 128-core node |
|---|---|---|
| realism_eval | the comparison table above with 500 missions per row | about 1–2 h |
| realism_sitl | 100 hamlet missions × winds 0, 5, 10 m/s in ArduPilot, each with a random under-canopy check | about 1 h |
| realism_targets | retrain the target-choosing policy under the new defaults (400 iterations), then evaluate it against the rule on 5 conditions incl. both real sites | about 8–12 h |

On a login node, once (after `setup.sh`):

    bash scripts/cluster/setup_realism.sh           # pymavlink, laspy, scipy; ArduPilot SITL; full-density lidar
    bash scripts/cluster/submit_realism.sh -A msml642-class

Results: `runs/realism/eval/results.md`, `runs/realism/sitl/summary.md`,
`runs/realism/targets/results/results_targets.md`. Re-submitting a job skips
or resumes finished work.

## Limits

- The lidar is from 2021 and leaf-off; summer cover is estimated from crown
  area.
- Trees are modelled as crowns on single trunks; small stems and saplings
  appear only as clutter at flying height.
- Under-canopy flight assumes the drone senses and avoids what the map
  misses (thin branches, wires). Small drones that do this exist, but this
  simulator does not model it.
- The person's detection under trees is calibrated to published bounds, not
  to measured trials of this camera at these heights.
