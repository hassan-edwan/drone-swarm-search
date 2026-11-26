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
a second look from above changes little.

**When a check starts.** A drone at its lowest search height reaches an
unresolved alert where tree cover is at least 50%.

**What the drone does** (field env mode `under`):

- descends through a gap in the canopy;
- flies under the crowns, about 3 m off the ground, to the alerted spot
  (alerts are located to a few metres in the image);
- looks with nothing between camera and person, then climbs back out.

**Using the 3D planner where it is available.** On real sites with 3D flight
limits (section 5), the simulator takes the check's time and its chance of
getting within sight of the spot from the planner's routes for that cell
(`flight_limits.npz`), and skips cells where no route exists. Elsewhere it
charges a default: descent, about 100 m under the crowns at 2 m/s, and the
climb.

**Up close.** The detection chance is 0.9 (trunks and undergrowth can still
hide someone), and false alarms (animals, warm stumps) are easy to dismiss.

**Metrics:** `under_checks`, `under_finds`, `under_blocked`.

**Fix (2026-11-26).** The check used to start only if the low look itself
re-detected the alert, because that look marked the alert as resolved first.
So checks almost never happened (0.2 per mission). It now starts from any
alert still open when the drone arrives.

## 5. Flying through real obstacles in 3D (`swarmsearch/flight3d.py`)

The search runs on a coarse grid; `flight3d` turns each move into the path a
drone would fly, through obstacles at 1 m.

### What blocks the drone

- **Terrain:** bare-earth lidar elevation.
- **Trees:** crown top and bottom per square metre, from the lidar. There are
  7,051 individual trees in the 1.2 km core, with trunks under crown peaks
  (trunk radius 0.3 m).
- **Undergrowth and low branches at flying height:** lidar returns between
  1 and 5 m above the ground.
- **Buildings:** footprints with lidar roof heights; 1 m extra is kept for
  eaves.
- **Wires and towers** (`fetch_hazards.py` → `hazards.json`), each a band of
  blocked height above the ground:
  - FAA-registered obstacles (Digital Obstacle File) with their heights.
    Guyed towers block the whole cone of their guy wires, out to 0.8 of
    their height.
  - High-voltage lines (HIFLD): a corridor 15 m each side, blocked from 8 m
    to 45 m.
  - **Assumed** local power lines along every road, at 5–13 m within 6 m of
    the road. Rural distribution lines mostly follow roads, but they are not
    in public data.
  - **Assumed** service wires from the nearest road to each house, at
    3.5–8 m.

The hamlet site has 1 registered tower (61 m), a 138 kV line, 107 road
pieces and 185 service wires. The park site has 9 registered obstacles, the
tallest 118 m.

### Drone size and season

- **Size.** Clearances are edge to edge. A SAR quad is about 1.0 m across
  with propellers and 0.4 m tall; a small under-canopy drone is 0.4 m
  across. On top of the drone's size, 1 m is kept from trunks, buildings,
  crowns and wires.
- **Season.** The lidar was flown leaf-off. For a summer search (leaf-on,
  the default) the planner assumes:
  - any lidar return at flying height in a 3 × 3 m patch blocks it (instead
    of three or more);
  - crowns reach 2 m lower;
  - openings close by 1 m.

### How the drones move

- **Cruise:** at the search height, following the terrain. The drone stays
  at least 5 m (from its underside) above any crown, roof or wire band
  within 3 m, climbing early for what is ahead. Over the 138 kV line it
  climbs to 50 m.
- **Under the canopy:** 3 m off the ground at 2 m/s, within the free space
  above. The drone enters and leaves through an opening its half-span plus
  2 m (3 m in summer) wide, with nothing above it. The opening is chosen to
  make the whole trip quickest. Where the spot is deep in undergrowth, the
  planner reports how close it could get; more than 15 m away counts as
  "could not get close".

### What this means for the search grid (`flight_limits.py`)

- **Safe height per cell.** For every search cell: the lowest search height
  that clears its tallest crown, roof, wire or tower by 5 m. The simulator
  will not let a drone enter or descend below it, and makes the drone climb
  first. The rule never targets below it.
- **Hamlet site:** the tallest obstacle per 50 m cell has a median of 28 m
  (90% of cells under 37 m). So 15% of cells need the 70 m height (tall
  trees, the high-voltage line); none are no-fly.
- **Park site:** trees are taller (median 33 m per 181 m cell), so 40% of
  cells need 70 m.
- **Under-canopy checks** in the hamlet's forest cells (263 cells, 3 spots
  each) get within sight of the spot 25% of the time and take 44 s on
  average.
- **On sampled spots under the trees:** 60% are reachable leaf-off and 35%
  in summer. The limit is undergrowth and low branches, not trunk spacing:
  a small drone does no better than the SAR quad.

### Demo

`fly_mission.py` and `render_flight3d.py` produce
results/figures/flight3d_demo.mp4: wild_mix seed 103 on the hamlet site at
09:00 (a lost hunter, 3 drones, the rule).

- **The mission.** Drone 1 checks an alert from under the trees 10 s in: it
  descends through a gap, flies 21 m under the crowns and gets within 0.2 m
  of the hunter. The planned trip takes 53 s; the simulator charged 45 s.
  The hunter is confirmed at step 19.
- **Planner test after the mission (labelled in the video).** A route under
  the canopy to a spot near houses. Edge to edge, it passes 5.9 m from a
  house, 2.1 m from trunks, 5.2 m below crowns and 1.8 m from a service
  wire's band.
- **In the video,** wires are drawn as black lines at the middle of their
  assumed height band.

## What the changes do to the results

`scripts/eval_realism.py`: the rule, 100 missions per row, the same missions
in every row (wild_mix scenarios, including the season, low cloud, wind and
failure draws of section 9). On the real sites the terrain, trees, trails
and lakes are real, with leaf-on or leaf-off cover by season; the hamlet
site also has the 3D flight limits and the executor's under-canopy
outcomes. All rows have drones always in contact. ± is one standard error.

| World | Variant | Found | Mean time to first find (min) | Under-canopy checks / mission | Finds from them | False alarms / mission | Drones lost: crashed / emergency landing / bird, per 100 missions |
|---|---|---|---|---|---|---|---|
| generated wilderness | old sensing | 32% ± 5% | 20 | 0.0 | 0 | 20.7 | 6 / 0 / 2 |
| generated wilderness | new sensing | 30% ± 4% | 20 | 0.0 | 0 | 20.2 | 9 / 0 / 1 |
| generated wilderness | new sensing + under-canopy checks | 30% ± 4% | 20 | 2.5 | 1 | 19.8 | 6 / 0 / 1 |
| real site (Catoctin, lidar) | old sensing | 14% ± 3% | 26 | 0.0 | 0 | 3.4 | 31 / 0 / 4 |
| real site (Catoctin, lidar) | new sensing | 8% ± 3% | 16 | 0.0 | 0 | 3.4 | 24 / 1 / 2 |
| real site (Catoctin, lidar) | new sensing + under-canopy checks | 10% ± 3% | 23 | 0.6 | 0 | 3.5 | 21 / 1 / 1 |
| real site (hamlet, lidar + 3D flight limits) | old sensing | 34% ± 5% | 18 | 0.0 | 0 | 105.8 | 4 / 6 / 6 |
| real site (hamlet, lidar + 3D flight limits) | new sensing | 28% ± 4% | 19 | 0.0 | 0 | 100.0 | 2 / 10 / 4 |
| real site (hamlet, lidar + 3D flight limits) | new sensing + under-canopy checks | 27% ± 4% | 15 | 23.2 | 0 | 86.8 | 3 / 11 / 7 |

- **Dense real forest is much harder.** In the park (86% leaf-on cover) the
  calibrated sensing finds 8–10% of people within the 90-minute mission,
  against 14% with the old model.
- **Drones lost.** About 20–30 per 100 missions in the park, against 5–10
  elsewhere. In 60 park missions, 8 of the 12 losses were failures or bird
  strikes with no open spot within reach (the same events at the hamlet
  are mostly emergency landings). 4 were empty batteries within two cells
  of the base on the way home: the return reserve was too thin. It is now
  12% of a battery (was 8%), not yet re-measured; the cluster job will. The
  failure (0–3% per drone-hour) and bird (1% per hour) rates are
  assumptions, and they drive this.
- **Under-canopy checks now happen, but do not find more people.** With the
  trigger fixed, drones make 3.4 checks per mission in generated wilderness,
  0.5 in the park and 26 at the hamlet. The hamlet has about 100 alerts per
  mission, mostly false alarms near houses and roads. They cut false alarms
  a little (105 → 92 per mission at the hamlet) but leave the share found
  unchanged: 2 finds from them in 300 missions. Each check costs about 45 s
  for a low chance of a real person. A cheaper filter (operator review
  first, a check only when the image is ambiguous) would likely help more
  than more checks.
- **Earlier results no longer apply.** The learned policies were trained
  under the old sensing and radio, so the rule (which needs no retraining)
  is used here; the cluster job retrains.

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

### Poor GPS under the canopy

Under dense crowns, consumer GPS loses satellites and suffers multipath.
`--gps degraded` (the default) switches the simulated receiver, while the
drone is below the crowns, to:

- 3 m of noise;
- 6 satellites;
- a slowly wandering offset of up to ±6 m (`SIM_GPS1_NOISE`, `_NUMSATS`,
  `_GLTCH_X/Y`; the glitch offsets are in degrees).

The copter flies on its own estimate. Clearances are computed from the TRUE
track (`SIMSTATE`), which is what hits a tree. `--gps good` stands for
navigating by camera there (visual-inertial odometry, as real under-canopy
drones do).

Demo mission and planner test, 5 m/s wind, SAR quad, leaf-on:

| Under the canopy | GPS only | Camera-based navigation |
|---|---|---|
| True track off the route, worst | 5.9 m | 0.6 m |
| GPS error, worst | 6.9 m | 0.5 m |
| Closest to trunks / crown above / wire band (edge to edge) | 1.7 / 2.9 / −1.0 m (inside the band) | 2.5 / 5.2 / 1.8 m |

On a second mission (seed 0), GPS only came within 0.2 m of a trunk and
went 1.8 m into a crown. With camera navigation the closest was 3.7 m.
**GPS-only flight under the canopy is not safe; it needs camera-based
navigation.** The cluster batch measures this over 100 missions.

## 7. Running at scale on Zaratan

Three independent jobs (scripts/cluster/realism_*.sbatch, settings in
realism.env):

| Job | What | Time on one 128-core node |
|---|---|---|
| realism_eval | the comparison table above with 500 missions per row | about 1–2 h |
| realism_sitl | 100 hamlet missions × winds 0, 5, 10 m/s × GPS-only and camera navigation under the canopy, in ArduPilot, each with a random under-canopy check | about 2–4 h |
| realism_surprises | map vs real world: 16 real-world draws × 100 under-canopy checks × 8 variants (sensing, navigation, wind, drone, snow) | about 1–2 h |
| realism_targets | retrain the target-choosing policy under the new defaults (400 iterations), then evaluate it against the rule on 5 conditions incl. both real sites | about 8–12 h |

On a login node, once (after `setup.sh`):

    bash scripts/cluster/setup_realism.sh           # pymavlink, laspy, scipy; ArduPilot SITL; full-density lidar
    bash scripts/cluster/submit_realism.sh -A msml642-class

Results: `runs/realism/eval/results.md`, `runs/realism/sitl/summary.md`,
`runs/realism/targets/results/results_targets.md`. Re-submitting a job skips
or resumes finished work.

## 8. The map is not the world: surprises and onboard sensing

So far the drones knew the map exactly, and everything real was on the
map. Real flights meet what the map lacks. `swarmsearch/world3d.py` builds
two 3D worlds at 1 m.

**The map** (what the planner knows):

- **Trees:** foliage from the lidar points themselves, layered rather than
  filled solid. A 1 m slice is occupied where the laser hit within
  3 × 3 m; gaps under 4 m inside a column are closed; in summer the
  foliage is grown out 1 m.
- **Tree shapes:** trees found by watershed segmentation of the crown
  surface, each with a trunk sized from its height (DBH ≈ 2 × (height −
  1.3 m) cm, a generic hardwood fit) up to its crown bottom.
- **Buildings:** with their real roof surface from the lidar, so pitched
  roofs, not boxes.
- **Power lines, pole by pole:** 10 m poles every 45 m on one side of each
  road. Three conductors at 9.5, 8 and 6 m (the lowest is telephone/cable),
  sagging 1 m mid-span. A service wire from the nearest pole to each house.
- **High-voltage lines:** 30 m pylons every 300 m, conductors at 24 m
  sagging 6 m, plus an earth wire.
- **Registered towers:** a mast with guy wires to three anchors.

**The real world** (`make_truth`) adds what the map does not know. Rates
are assumptions where no data exists:

- **Growth:** five years since the 2021 lidar, about 0.3 m a year at crown
  tops and edges.
- **Misregistration:** each footprint off by about 1 m; each trunk 0.7 m
  from the crown peak it was placed under.
- **Wires:** local lines on the other side of the road on half the roads,
  unmapped spans across fields, and wires to sheds.
- **Small structures:** sheds and carports at 35% of houses, flagpoles at 5%.
- **Forest clutter at flying height:** dead branches (25 per ha) and
  saplings (40 per ha).
- **Storm damage:** felled trees leaning on their neighbours (0.3 per ha).
- **New buildings** since the footprints were made.
- **Winter snow:** branches weighed down by 1 m.

`swarmsearch/fly_exec.py` plans each under-canopy check on the map and flies
it in the real world. Every 0.1 s:

- **Navigation.** The drone flies its own estimate: GPS (degraded under the
  crowns, as in section 6) or camera-based navigation (about 1% drift).
  Wind adds gusty drift from the wind field (section 9).
- **Sensing.** A forward depth camera (90° wide), or all-round sensing.
  Trunks, buildings, poles and structures are seen to 15 m in the open and
  8 m in the dark under the crowns. Foliage and branches are seen to 6 m
  with a 0.6 chance per frame. Wires only within 4 m (2.5 m under the
  crowns) with a 0.25 chance per frame, since depth cameras barely see thin
  wires.
- **Avoiding.**
  - If what it has seen blocks the route within stopping distance plus 3 m,
    it stops and re-plans: around the obstacle under the crowns, through
    another gap, or it gives up and retraces its own track out.
  - A local safety layer, relative to the drone and so unaffected by GPS
    jumps, never lets it move towards something it has seen; stuck for
    15 s, it turns back.
- **Collisions.** If its body touches anything real, it has collided.
- **Also:** vehicles where a low route crosses a road, and birds of prey
  above the trees (rates assumed).

Results (`scripts/real_site/surprise_eval.py`): 2 real-world draws × 30
spots under the trees on the hamlet site, the same spots for every variant,
5 m/s wind, SAR quad, summer unless noted.

| Variant | Checks | Collided | With what | Turned back | Reached the spot | Extra time (s) | Re-plans / check | Seen: solid / foliage / wire |
|---|---|---|---|---|---|---|---|---|
| no sensing, camera nav | 60 | 8% ± 4% | vehicle 3, foliage 2 | 0% | 27% | +8 | 0.0 | 0 / 0 / 0.0 |
| forward sensing, camera nav | 60 | 0% ± 0% |  | 3% | 28% | +8 | 0.1 | 15 / 116 / 2.1 |
| all-round sensing, camera nav | 60 | 0% ± 0% |  | 2% | 28% | +12 | 2.9 | 20 / 266 / 2.9 |
| forward sensing, GPS only | 60 | 18% ± 5% | foliage 11 | 13% | 18% | +8 | 3.4 | 17 / 144 / 2.3 |
| all-round sensing, GPS only | 60 | 8% ± 4% | foliage 5 | 33% | 20% | +31 | 66.8 | 67 / 362 / 3.1 |
| forward sensing, camera nav, 10 m/s wind | 60 | 0% ± 0% |  | 3% | 27% | +9 | 0.1 | 18 / 141 / 2.2 |
| forward sensing, camera nav, small drone | 60 | 0% ± 0% |  | 5% | 32% | +17 | 5.8 | 42 / 146 / 2.1 |
| forward sensing, camera nav, winter snow | 60 | 0% ± 0% |  | 5% | 37% | +8 | 0.2 | 26 / 159 / 0.1 |

- **Without onboard sensing, 8% of checks collide.** The causes are
  vehicles at road crossings and foliage the map did not have.
- **With a forward depth camera and camera navigation: no collisions in 60
  checks** (under about 5% at 95% confidence). About 3% turn back. Wind,
  drone size and winter snow do not change this.
- **GPS-only navigation under the crowns is not safe even with sensing:**
  18% collide with a forward camera, 8% with all-round sensing (which
  turns back a third of the time instead). The position error moves the
  drone sideways into foliage it is not looking at. This repeats the
  ArduPilot result in section 6.
- **Wires are the hardest to see:** about 2 wire detections per check, at
  most 4 m away.
- `results/figures/surprise_example.png` shows one case. The planned
  descent gap is crossed by a real service wire, because the line is on
  the other side of the road from the map's. The drone sees it 3.8 m away
  on the way down, stops, climbs, and goes through another gap to reach
  the spot.

**Fed back.** The default variant (forward camera, camera navigation) sets,
in `exec_stats.json`, each under-canopy check's chance of colliding or
turning back and its extra time in the field simulator. On the cluster,
`realism_surprises` runs 16 world draws × 100 spots.

## 9. Weather, season and the unexpected

- **Wind shaped by the terrain** (`swarmsearch/windfield.py`):
  - over hills: speed-up at crests and on upwind slopes, slower and gustier
    in the lee (from the ground's height above the upwind terrain within
    500 m, the usual hill speed-up rule of thumb);
  - with height: a log profile over each surface, all matching at a 100 m
    blending height (over forest: displaced and rougher), decaying under
    the canopy to about 1 m/s;
  - gusts: turbulence intensity 0.2, 0.35 in the lee.
  - In the field simulator the wind at each search cell (a multiplier per
    cell for 8 directions, in `flight_limits.npz`) drives battery use and
    drift. The return-home reserve uses the wind measured where the drone
    is, with a 25% margin, and keeps 12% of a battery spare.
  - Hamlet site: 0.7–1.3 × the open-ground wind at search height; park:
    0.6–1.5 ×.
- **Season.** A third of missions are in the leaf-off months
  (`season="winter"`). Real sites then use the leaf-off lidar cover (park
  70% instead of 86%) and generated worlds half their cover, so people are
  easier to see from above.
- **Low cloud or fog.** 15% of missions have a cloud base 60–150 m up; no
  search height within 10 m of it, and cells needing more become no-fly.
- **Birds of prey:** 0.01 drone losses per flight hour (assumed: about
  0.02 attacks per hour, half evaded).
- **Emergency landing.** An in-flight failure (not an empty battery) leaves
  about 150 m of controlled descent. With an open spot in reach (open
  ground, under 15°, dry; `flight_limits.npz`), it is an emergency landing;
  otherwise the drone comes down in the trees. They are counted
  separately. On the hamlet site 34% of search cells have an open spot and
  the farthest point is 780 m from one; on the park site, 36% and 2 km.
- **Other aircraft.** A helicopter working the search closes airspace for
  a while: the simulator's temporary no-fly zones (`p_tfr`, D8) cover it.

## Limits

- The lidar is from 2021 and leaf-off; summer cover is estimated from crown
  area.
- Trees are modelled as crowns on single trunks; small stems and saplings
  appear only as clutter at flying height.
- Under-canopy flight assumes the drone senses and avoids what the map
  misses (thin branches, wires not along roads). Small drones that do this
  exist, but this simulator does not model it.
- Local power lines and service wires are assumed (along roads, to each
  house), not mapped; real ones also cross fields and run along property
  lines. Wire heights are typical values.
- The search grid still charges one step per cell, diagonals and sprints
  included; flown at real speeds the same moves take 13–39% longer
  (section 6).
- ArduPilot SITL has no obstacles; flown tracks are checked against them
  afterwards. There is no onboard detect-and-avoid, and wind under the
  crowns is not reduced.
- The person's detection under trees is calibrated to published bounds, not
  to measured trials of this camera at these heights.
