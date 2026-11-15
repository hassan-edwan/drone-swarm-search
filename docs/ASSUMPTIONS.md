# Assumptions register

Every simulation assumes things about the world. Each assumption below is a
place where the simulation could flatter a search method that would then
fail in the field. For each one this register records what we assume, the
evidence, what the simulator does now, what it should do, and whether a
search strategy may rely on it.

**Guiding rule: assume as little as possible about the person, and where we
must assume, sample the whole realistic range, worst case included.**
A strategy must still work when every optional cue is absent; cues such as a
survivor waving or a phone ping may only ever help.

Columns:
- **Now**: what the code does today (milestones M1–M7, see `docs/REPORT.md`).
- **Target**: how it should be modelled, with the range to sample.
- **Rely?** *No*: the strategy must work without it (randomised, including
  its absence). *Hint*: it may inform the map, but the world must sometimes
  contradict it. *Yes*: a safe fact a real team would have.
- **Field check**: what a real test can measure to calibrate it.

Status: **open** = not modelled or modelled wrongly; **partial** = modelled
in some form; **done** = modelled as targeted. Sources are numbered
[1]–[16] at the end.

---

## P. The missing person

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| P1 | They want to be found and will signal (wave, shout, move into the open) when a drone is near | Dementia subjects usually do not answer searchers and can watch a team pass [3][12]; children aged 7–12 often hide from searchers [13]; some subjects are evasive [13], unconscious, or dead | Passive: survivors neither signal nor hide | Per-person responsiveness: *unresponsive* (most common), *signals* when a drone is close and they are conscious and in the open (raises detection), *evasive* (moves under cover or away when a drone is near). Mix drawn per mission, including all-unresponsive | No | open |
| P2 | They carry a working phone or radio | Often no coverage, a dead battery, or no device; we found no reliable rate | No devices | Optional, rare: a single coarse "phone ping" (an area at a past time), off in most missions | No | open |
| P3 | They move in one of a few known patterns | Lost-person behaviour is statistical over many cases [1][2]; behaviours include trail following, drainage following (downhill), view enhancing (climbing), direction travel, backtracking, staying put [1][13]; terrain drives movement [4] | 4 kinds (still, wander, drift, to safety); M7: per-person mix, one dominant | Terrain-driven behaviours from [1][4] weighted by subject category, plus a share that follows none of them (random but terrain-feasible) | Hint (map hypotheses) | partial |
| P4 | Their behaviour stays the same over the mission | People tire, stop at night, change their minds | Constant; M7 adds stop-and-go | Mobility falls with time since lost; stop at dusk; injury makes some immobile; occasional behaviour switches | Hint | partial |
| P5 | They are within a known distance of where they were last seen | Planners use distance rings from the last-seen point by subject category from ISRID [1][2] | Uniform over a 200 m area, or in zones | Distance rings by category from ISRID, sampled with wide spread; the actual ring values need the book [1] (not verified online) | Hint | open |
| P6 | We know how many people are missing | Wilderness: usually one subject, sometimes a group; disasters: unknown counts | 1–20 survivors, placed independently. **Leak:** the M7 evidence feature uses the true count | Count unknown to the drones (use an estimate with error); groups of people placed together | No | open (leak to fix) |
| P7 | Every person is equally visible | Posture, clothing contrast and partial cover change detection a lot [10][11][16] | M7: per-person visibility factor (log-normal) | Posture (standing, sitting, lying), clothing contrast, partial cover, per person | No | partial |
| P8 | The person is alive and warm | Survival falls after 24 h [3]; hypothermic or deceased subjects have low thermal contrast | Always detectable at the modelled rate | Share of low-contrast subjects (thermal much weaker), rising with time since lost | No | open |

## K. What the team knows before and during the search

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| K1 | The last-seen point is accurate | Often inferred (a car at a trailhead, a vague report) | Prior zones with accuracy 0.5–1.0 | Last-seen point with a sampled error radius; sometimes plainly wrong | Hint | partial |
| K2 | The person is inside the search area | Planners keep probability for "rest of world" [6][7] | Always inside the grid | Map keeps "outside the area" mass; some missions have the person outside (scored separately; the right action may be to report low success and widen the area) | No | open |
| K3 | The search starts when the person goes missing | Drones often launch hours later | Starts at time 0 | Launch delay sampled; the person (and the map) has moved during it | No | open |
| K4 | The prior map is right | It is a planner's guess | Zones sometimes wrong (zone accuracy) | Prior sometimes misleading by design, as well as the movement profile | Hint | partial |
| K5 | The terrain or building map is right | Disaster changes the ground; open map data has gaps (missing trails, old canopy) | M7: collapsed buildings unknown until seen | Wilderness: missing or extra trails, outdated canopy, wrong water levels | Hint | partial |
| K6 | The operator's movement profile is the truth | It is a guess | Told the true profile in 50% of missions | When "told", the profile describes only the dominant behaviour (M7), and is sometimes wrong | Hint | partial |

## S. Sensing and finding

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| S1 | We know the sensor's detection rate | Measured detection varies by terrain and visibility [8][9]; trials are small and from few groups [10][11]; a plan's believed detection chance can differ from the true one [14] | M7: true rate is 0.4–1.1× the modelled one; false alarms 0.5–3× | Calibrate the nominal rate to sweep-width experiments [8][9] for the drone's camera; keep the mismatch range | No | partial |
| S2 | Each look is an independent chance | Under canopy, single thermal frames show people as fragments that change with viewpoint; combining views from different angles works far better than repeating one [11] | Independent per pass | Per person, a persistent occlusion state per viewing direction and height: a repeated look from the same spot mostly repeats the miss; a new angle is a new chance. The map should model this too | No | open |
| S3 | Canopy and thermal factors are fixed numbers | 86–90% detection in dense forest needs a special multi-view method [10]; plain thermal is far worse | Canopy ×0.3 (RGB) / ×0.5 (thermal) | Continuous canopy density; thermal contrast by air temperature and time of day; treat [10] as an upper bound for advanced processing | No | partial |
| S4 | False alarms come at a known rate | Wildlife, sun-heated rocks, debris; manual screening gives many false positives [15] | M6/M7 rates by cover and light, ×0.5–3 | Environment-specific sources (animals in forests, hot rocks at midday) | No | partial |
| S5 | A detection is a find | A person reviews drone video; screening is slow, tiring and error-prone [15][16]; field trials found the human side to be the bottleneck [5] | A low pass confirms instantly | Operator review queue: delay per item, miss rate rising with workload and time on task; confirmation possible from mid-altitude video; optionally ground-team confirmation | No | open |
| S6 | Only buildings block the view | Ridges and slopes hide whole valleys | Building heights block sight lines (M6) | Line of sight over terrain elevation | Yes (from elevation data) | open |

## D. Drones and flight

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| D1 | Battery is a clock and missions are short | Search quadcopters fly about 40–55 min per battery (maker specifications); searches last hours with battery swaps | Linear drain; 100–200 steps of 1 s | 35–55 min per battery, energy depends on speed, wind and climbing; return to base and swap; multi-sortie missions | Yes (battery is measured) | open |
| D2 | Altitude changes are instant | Climb rates are about 3–5 m/s | One step per altitude level | Climb and descent take time; terrain following at a set height above ground; legal ceiling about 120 m above ground | Yes | open |
| D3 | One speed | Motion blur lowers detection at speed; transit is faster than scanning | One cell per step | Choice of scan or transit speed; detection falls with speed | Yes | open |
| D4 | Drones are where they think they are | Wind and GPS error, worse near tall structures | M7: wind slips, GPS off by a cell sometimes | Keep; add canyon and ridge effects in wilderness | No | done (M7) |
| D5 | Drones never collide | Real drones use onboard sensing and height separation | Shield on reported positions; M7 onboard detect-and-avoid on true positions (idealised) | Keep idealised; report near misses | Yes | partial |
| D6 | Drones are reliable | Link loss, weather aborts, emergency landings | 25% of missions lose one drone at random | Link-loss return-to-home, weather aborts, launch failures | No | partial |
| D7 | The camera points straight down | Real gimbals tilt forward: wider, slanted view, sees under canopy edges | Square footprint at nadir | Optional forward tilt with a longer, slanted footprint | Yes | open |
| D8 | Anywhere in the area can be flown | Height limits, line-of-sight rules, no-fly zones (e.g. rescue helicopters) | No limits | No-fly zones (some appearing mid-mission), height ceiling | Yes | open |

## C. Communication

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| C1 | The team shares one map instantly | Radios need line of sight; ridges and buildings block them | M7: connected groups share a map, merge on reconnection | Terrain line of sight in wilderness; optional relay or tethered drone above the base | No | partial |
| C2 | Discoveries reach everyone instantly | Small messages may travel over long-range telemetry, but not always | M7: collapse discoveries shared instantly (simplification) | Discoveries travel with the radio groups like the map | No | partial |
| C3 | Merging two maps is exact | Searches from two groups on one cell compound | M7: merge keeps the lower value per cell (approximation) | Exact merge from per-group search logs, if the approximation proves to matter | — | partial |

## E. Environment

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| E1 | The area is small and the mission short | Wilderness search areas span km from the last-seen point [1]; missions last hours | 200 m × 200 m, 100–200 s | Wilderness: about 1.6 km (50 m cells), 5 s steps, hours with swaps | Yes | open |
| E2 | The ground is flat | Terrain drives both movement [4] and sight lines | Flat plus buildings | Elevation with ridges, valleys, drainages, trails, water, cliffs; real maps for the test site | Yes | open |
| E3 | The world does not change | Fire and smoke spread, water rises, aftershocks | Static (except collapses being discovered) | Later: spreading smoke, rising water, new collapses | No | open |
| E4 | Weather and daylight are constant | Searches cross dusk and weather | Lighting fixed per mission | Light changes during a mission; wind and rain events | Yes (forecast) | open |

## O. Operations and the goal

| ID | Assumption | Evidence | Now | Target | Rely? | Status |
|---|---|---|---|---|---|---|
| O1 | After a find, the drone carries on | Drones often stay over the person, mark the location, or drop supplies until a ground team arrives | Drone carries on at once | A find occupies that drone for a sampled time | Yes | open |
| O2 | There are no ground teams | Real searches coordinate drones with ground teams [5] | None | Later: ground teams that clear areas and confirm finds | — | open |
| O3 | The goal is the mean time to find | What matters is the chance of a live rescue; survival falls after 24 h [3]; search theory maximises the probability of success over time [6][7] | Mean find time (unfound = budget) | Also report the probability of success by time and time to rescue | — | partial |

## M. Method and evaluation

| ID | Assumption | Risk | Now | Target |
|---|---|---|---|---|
| M1 | Our simulated scenario distribution covers reality | A policy tuned to our generator may fail elsewhere | One generator per mode | Several scenario families; train on some, test on a held-out family; test on real maps of the field site |
| M2 | The critic may see the truth | Fine: used only in training; the policy never sees it | Critic sees true positions, sensor scale, collapses | Keep |
| M3 | Baselines get the same information as the policy | Otherwise comparisons are unfair | Rule and sweep use each drone's own (possibly wrong) map | Keep; add an adaptive rule that estimates sensor quality from the same evidence |
| M4 | Single training seed per model | Seed-to-seed variance unknown | One seed (4 CPU cores) | Several seeds for headline results when compute allows |

---

## Hard test cases

Every scenario family is evaluated separately on these, and the report states
how each method does. A method that helps on average but fails here is not
ready for the field.

1. **Unresponsive person**: lying still, no signalling, low visibility.
2. **Evasive person**: moves under cover or away when a drone is near.
3. **Wrong last-seen point**: prior centred in the wrong place.
4. **Person outside the planned area**: success means recognising that the area has been covered and reporting it.
5. **Sensor much worse than assumed**: detection at the low end of the range.
6. **Late start**: drones launch hours after the person went missing.
7. **Radio dead zones**: most of the area out of contact with the base.
8. **Behaviour that matches no hypothesis**.

## Field checks for a production test

What a real test can measure to calibrate this register (with volunteers as
subjects, never real emergencies):

- **Detection**: for your drone, camera and terrain, run sweep-width style
  trials [8][9]: volunteers placed in known spots (standing, sitting, lying,
  under canopy, in the open) and count detections against distance and
  height. This sets S1, S3 and P7.
- **Repeated looks** (S2): fly the same spot several times from the same and
  from different angles over a hidden volunteer.
- **Operator review** (S5): time from detection to confirmation, and how many
  planted subjects operators miss over a long session.
- **Battery** (D1): flight time with your payload in your conditions; swap
  time.
- **Radio** (C1): link range from the base across your terrain.
- **GPS** (D4): position error near cliffs, in valleys and under trees.
- **Behaviour** (P1–P5) cannot be tested on real lost people. Use scripted
  volunteers, including the unresponsive and evasive cases, and rely on
  ISRID [1][2] for real-world statistics.

Field tests need the usual permits and safety measures (in the US, Part 107
and a waiver for flight beyond visual line of sight), and a human operator
who can take over at any time.

---

## Sources

1. Koester, R. J. *Lost Person Behavior* (dbS Productions, 2008) and the ISRID database. Overview: https://www.dbs-sar.com/LPB/2ISRID.pdf
2. Koester, ISRID Iceland seminar (2023): https://liu.se/dfsmedia/dd35e243dfb7406993c1815aaf88a675/67768-source/options/download/carer-semnarium-robert-koester-isridiceland-2023-01-19
3. Koester, "The Lost Alzheimer's and Related Disorders Search Subject": https://www.dbs-sar.com/SAR_Research/ALZ.pdf
4. Lin, L. & Goodrich, M. A. (2010). A Bayesian approach to modeling lost person behaviors based on terrain features in wilderness search and rescue. *Computational and Mathematical Organization Theory* 16(3). https://ideas.repec.org/a/spr/comaot/v16y2010i3d10.1007_s10588-010-9066-2.html
5. Goodrich, M. A. et al. (2008). Supporting wilderness search and rescue using a camera-equipped mini UAV. *Journal of Field Robotics*. Technical report: https://faculty.cs.byu.edu/~mike/mikeg/papers/WiSARTechnicalReport2007-1.pdf
6. Land search methods review (DTIC): https://apps.dtic.mil/sti/pdfs/ADA511658.pdf
7. SAROPS, the US Coast Guard's Bayesian search planning system (Metron): https://www.metsci.com/wp-content/uploads/2019/08/Search-and-Rescue-Optimal-Planning-System.pdf
8. Chiacchia, K. B. et al. (2014). Use of the visual range of detection to estimate effective sweep width for land search and rescue based on 10 detection experiments in North America. *Wilderness & Environmental Medicine*. https://plaudit.pub/endorsements/doi:10.1016/J.WEM.2013.09.016
9. Robe, R. Q. & Frost, J. R., sweep width estimation for ground search (DTIC): https://apps.dtic.mil/sti/pdfs/ADA513088.pdf
10. Johannes Kepler University Linz, drone that sees through forest canopy (AAAS summary): https://www.aaas.org/news/rescue-drone-sees-through-thick-forest-track-people
11. Thermal multi-view person detection under canopy (arXiv 2009.08835): https://pdf.arxiv.org/pdf/2009.08835
12. Lost dementia patients study, *BMC Geriatrics*: https://pmc.ncbi.nlm.nih.gov/articles/PMC3141319
13. Predicting lost person behavior (IKAR): https://www.alpine-rescue.org/system/production/article/documents/file/004/113/90ecc0736c02b837fa37891a4ffd75669b3133ee41c347273bb4b0af950d5326/18-Predicting-Lost-Person-Behavior.pdf
14. True versus subjective detection probability of a uniformly optimal search plan (arXiv): https://arxiv.org/html/2601.00350v1
15. Thesis on manual aerial screening for SAR: https://core.ac.uk/download/428350237.pdf
16. Psych-Occlusion: visual psychophysics for aerial detection of occluded persons (WACV 2025): https://arxiv.org/html/2412.05553v1
