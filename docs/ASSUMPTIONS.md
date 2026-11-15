# Assumptions register

Every simulation assumes things about the world. Each assumption below is a
place where the simulation could flatter a search method that would then
fail in the field. For each one this register records what we assume, the
evidence, whether a search strategy may rely on it, and how the field
environment (M8, `swarmsearch/field/`) models it.

**Guiding rule: assume as little as possible about the person, and where we
must assume, sample the whole realistic range, worst case included.**
A strategy must still work when every optional cue is absent; cues such as a
survivor waving or a phone ping may only ever help.

Columns:
- **Rely?** *No*: the strategy must work without it (randomised, including
  its absence). *Hint*: it may inform the map, but the world sometimes
  contradicts it. *Yes*: a safe fact a real team would have.
- **M8 model**: what the field environment does. The grid environments
  (M1–M7) keep their simpler models so earlier results reproduce.
- **Status**: **done** = modelled as targeted; **calibrate** = modelled,
  but the numbers are illustrative and need the cited data or field
  measurements; **kept** = a known simplification we keep on purpose;
  **open** = not yet done.
- **Code**: where it lives (`f/` = `swarmsearch/field/`).

Sources are numbered [1]–[17] at the end. The **hard test cases** and the
**field checks** for a production test follow the tables.

---

## P. The missing person

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| P1 | They want to be found and will signal when a drone is near | Dementia subjects usually do not answer searchers and can watch a team pass [3][12]; children 7–12 often hide [13]; some subjects are evasive [13], unconscious or dead | No | Each person is *unresponsive*, *signals* (when a drone is within ~150 m and low: stands, waves, moves into the open; detection up, occlusion down) or *evasive* (moves under cover, hides; occlusion up). Shares by subject category (dementia 85% unresponsive, children 35% evasive, ...); dead people are unresponsive | done (shares: calibrate) | `f/people.py` People, `f/params.py` CATEGORIES |
| P2 | They carry a working phone or radio | Often no coverage, a dead battery or no device | No | 10% of missions give a rough phone-ping area (300–1000 m radius, around the position at launch, with error); it only reweights the prior | done | `f/people.py` prior |
| P3 | They move in one of a few known patterns | Behaviour is statistical over many cases [1][2]; trail and drainage following, view enhancing, direction travel, backtracking, staying put [1][13]; terrain drives movement [4] | Hint | Terrain-driven behaviours (trail, downhill along drainages, uphill to high points, straight line, backtrack, shelter, drift in water, stay, wander) mixed by category; 5–30% of people follow *none* (a persistent random walk); the map's hypotheses use the same behaviours on its own (imperfect) map | done (mixes: calibrate) | `f/people.py` |
| P4 | Their behaviour stays the same | People tire, stop at night, change their minds | Hint | Mobility halves every 3–12 h since lost; mostly stopped in the dark; rests and resumes; behaviour changes about hourly; 5–40% injured and immobile; walking speed by Tobler's hiking function on slopes [17] | done | `f/people.py` step |
| P5 | They are within a known distance of where they were last seen | Planners use distance rings by category from ISRID [1][2] | Hint | Log-normal distance from the last-seen point, median by category (1–1.8 km), weighted by terrain features; the true spread differs from the map's (×0.7–1.4) | calibrate (ISRID ring values needed [1]) | `f/people.py` ring_density |
| P6 | We know how many people are missing | Wilderness: usually one subject; disasters: unknown counts | No | Wilderness: 1 person (15%: a group of 2–3), count known; urban and flood: 3–15 people in groups of 1–3, the team gets an estimate with log-sd 0.4. The M7 leak (true count in the evidence feature) is fixed | done | `f/params.py`, `env.py` n_est |
| P7 | Every person is equally visible | Posture, clothing contrast and partial cover change detection [10][11][16] | No | Clothing contrast (log-normal, ordinary camera), posture (standing 1.0, sitting 0.8, lying 0.65, from activity and injury), cover from the ground they are on | done (factors: calibrate) | `f/env.py` sn_posture, `f/people.py` |
| P8 | The person is alive and warm | Survival falls after 24 h [3]; hypothermic or dead people show little thermal contrast | No | 0–30% hypothermic (thermal ×0.5); a small share dead (rises with hours lost; thermal ×0.15, immobile, unresponsive) | done (shares: calibrate) | `f/people.py` |

## K. What the team knows

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| K1 | The last-seen point is accurate | Often inferred | Hint | Reported point off by 30–500 m; 10% plainly wrong (0.6–1.5 km) | done | `f/people.py` prior |
| K2 | The person is inside the search area | Planners keep a "rest of world" share [6][7] | No | Map edges are open: people can walk off and stay off; some start outside (prior tail plus 0–15%); the map tracks the mass outside per movement hypothesis and the policy sees the chance the person is outside | done | `f/env.py` p_outside, `f/people.py` |
| K3 | The search starts when the person goes missing | Drones launch hours later | No | 0.5–12 h since lost (wilderness): sets fatigue, time of day, survival; the prior is the found-distance ring for that elapsed time | done | `f/params.py` hours_lost |
| K4 | The prior map is right | It is a planner's guess | Hint | True distance spread, behaviour mix and last-seen point all differ from the map's | done | `f/people.py` |
| K5 | The terrain or building map is right | Disasters change ground; open data has gaps | Hint | Wilderness map misses 0–40% of trail stretches and has a noisy canopy map; urban collapses unknown until seen (aftershocks add more) | done | `f/terrain.py` |
| K6 | The operator's movement profile is the truth | It is a guess | Hint | 30% of missions an operator names a behaviour: right 70% of the time, and it only shifts the map's hypothesis weights | done | `f/people.py` hypothesis_bank |

## S. Sensing and finding

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| S1 | We know the sensor's detection rate | Detection varies with terrain and visibility [8][9]; trials are small [10][11]; believed and true detection differ [14] | No | True detection is the modelled one ×0.4–1.1; false alarms ×0.5–3. Optional adaptive map estimates the real rate from finds against expectations (M3) | calibrate (nominal rates: sweep-width trials with your drone [8][9]) | `f/env.py` _search |
| S2 | Each look is an independent chance | Single thermal frames under canopy show people as fragments that change with viewpoint; combining views works far better [11] | No | Per person and viewing direction (8 sides + straight down, per height) the person is hidden or not, drawn once (5% in the open to ~60% under dense canopy); moving to a new spot redraws it. The map counts looks per direction and updates with the same correlated model | done | `f/sensing.py`, `f/env.py` _search |
| S3 | Canopy and thermal factors are fixed | 86–90% in dense forest needs special processing [10] | No | Continuous canopy density; dual camera (better of ordinary and thermal); thermal contrast from air temperature, weaker under midday sun and rain; ordinary camera needs light (clock-driven) | done (factors: calibrate) | `f/sensing.py` conditions |
| S4 | False alarms come at a known rate | Animals, hot rocks, debris; manual screening gives many false positives [15] | No | Rate per visible cell rises with canopy (animals), midday sun on open wilderness ground (hot rocks) and rubble | done (rates: calibrate) | `f/sensing.py` false_alarm_rate |
| S5 | A detection is a find | A person reviews the video; screening is slow and error-prone [15][16]; the human side was the bottleneck in field trials [5] | No | Detections stream to 1–2 reviewers (8–40 s each) only while the drone is in radio contact (otherwise they wait onboard); misses rise with hours on task and backlog; high-altitude images are often not clear enough (closer look needed); 85% of false alarms dismissed; repeat sightings at the same spot are grouped; a find counts when confirmed | done (rates: calibrate) | `f/sensing.py` OperatorQueue, `f/env.py` _review |
| S6 | Only buildings block the view | Ridges hide valleys | Yes | Line of sight over elevation plus buildings, drone at its height above the ground beneath it | done | `f/sensing.py` geo_table |

## D. Drones and flight

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| D1 | Battery is a clock and missions are short | Search quadcopters fly about 40–55 min per battery; searches last hours with swaps | Yes | 35–55 min per battery; drain grows with sprinting, climbing, wind, cold and rain; the autopilot returns a drone when the energy home (with margin) runs short; 1.5–4 min swap; an empty battery is a crash | done | `f/env.py` _update_modes |
| D2 | Altitude changes are instant | Climb rates 3–5 m/s | Yes | Climb 4 m/s, descent 3 m/s while moving; heights above the ground beneath (terrain following); ≤ 120 m | done | `f/env.py` step |
| D3 | One speed | Blur lowers detection at speed | Yes | Hover (full detection), cruise one cell per step (×0.7), sprint two cells (×0.15, more battery) | done | `f/env.py` |
| D4 | Drones are where they think they are | Wind, GPS error near tall structures | No | Wind pushes drones a cell (more in small cells); GPS fixes off a cell, three times as often next to anything taller than the drone; sensing where the drone is, map updates where it thinks it is | done | `f/env.py` _fly |
| D5 | Drones never collide | Onboard sensing and height separation | Yes | Shield on reported cells; detect-and-avoid on true positions, horizontally and vertically (10 m); collisions and near misses reported | done (idealised sensing: kept) | `f/env.py` _fly |
| D6 | Drones are reliable | Link loss, weather, failures | No | Return on link loss after 30 s (half of missions); gust spells ground everyone; 2% launch failures; 0–3% per flight hour in-flight failures | done | `f/env.py` _update_modes |
| D7 | The camera points straight down | Gimbals tilt forward | Yes | 30% of missions: forward-tilted camera looks a cell ahead at mid and high, slant ×0.85 | done (approximation) | `f/env.py` _search |
| D8 | Anywhere can be flown | Ceilings, no-fly zones | Yes | 0–2 static no-fly zones and a pop-up zone (e.g. a helicopter) in 0–40% of missions for 10–20 min; drones caught inside are flown out | done | `f/env.py` |

## C. Communication

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| C1 | The team shares one map instantly | Radios need line of sight | No | Links over terrain and buildings: range 80–300 m urban, 300–2500 m open, a quarter when blocked, random drops; a tethered relay drone over the base in 20% of missions; planners stay inside radio coverage when drones return on link loss | done | `f/env.py` _sync, _link_mask |
| C2 | Discoveries reach everyone instantly | Messages travel over radio | No | Each radio group has its own copy of the ground map; collapse discoveries spread when groups merge | done | `f/env.py` _merge, _reveal_in |
| C3 | Merging two maps is exact | Searches from two groups compound | — | Merge keeps the lower value per cell and the higher look count: slightly underconfident where both groups searched. An exact merge needs per-group search logs replayed through movement; kept until it is shown to matter | kept | `f/env.py` _merge |

## E. Environment

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| E1 | The area is small and the mission short | Wilderness areas span km [1]; missions last hours | Yes | Wilderness 3.6 km (48 × 75 m), 60 min; urban 200 m (20 × 10 m), 30 min; flood 800 m (32 × 25 m), 30 min | done | `f/params.py` |
| E2 | The ground is flat | Terrain drives movement [4] and sight lines | Yes | Generated ridges and valleys, drainages (D8 flow routing), trails along gentle slopes, forest with a treeline, cliffs; or a real site from elevation, canopy, trail and water layers | done | `f/terrain.py`, `scripts/import_site.py` |
| E3 | The world does not change | Smoke, water, aftershocks | No | Urban smoke plumes downwind of fires; aftershocks collapse more buildings mid-mission; flood water rises up to 2 m | done | `f/env.py` |
| E4 | Weather and daylight are constant | Searches cross dusk and weather | Yes | Clock-driven light and thermal contrast; rain spells; gust spells | done | `f/sensing.py`, `f/env.py` |

## O. Operations and the goal

| ID | Assumption | Evidence | Rely? | M8 model | Status | Code |
|---|---|---|---|---|---|---|
| O1 | After a find, the drone carries on | Drones often stay over the person | Yes | 30–100% of finds: the nearest drone holds over the person until a ground team arrives (≤ 15 min) | done | `f/env.py` _dispatch |
| O2 | There are no ground teams | Drones work with ground teams [5] | — | 1–3 teams walk from the base (1 m/s, Tobler on slopes) to confirmed finds; idle teams search trails and streets; a team at the person's cell rescues them | done (simple tactics) | `f/env.py` _teams_step |
| O3 | The goal is the mean time to find | What matters is a live rescue [3]; search theory maximises success over time [6][7] | — | Reported: mean find time, found by half time, found inside the area, rescued share and rescue time | done | `f/env.py` metrics |

## M. Method and evaluation

| ID | Assumption | Risk | M8 model | Status |
|---|---|---|---|---|
| M1 | Our scenario distribution covers reality | A policy tuned to our generator may fail elsewhere | Three families; train on wilderness and urban, test on flood (held out); test on real site maps | done (held-out run pending) |
| M2 | The critic may see the truth | Fine: training only | Critic sees pooled true maps and hidden parameters; the policy never does | kept |
| M3 | Baselines get the same information as the policy | Unfair comparisons otherwise | Rule and sweep plan on each drone's own map and radio group, within radio coverage; adaptive-map option for an adaptive rule | done |
| M4 | One training seed per model | Seed-to-seed variance unknown | `--seed` supported; several seeds for headline results when compute allows | open (compute) |

---

## Hard test cases

Evaluated separately (`scripts/evaluate.py --field`, wilderness missions):

1. **Unresponsive person** (`wild_hard_unresponsive`)
2. **Evasive person** (`wild_hard_evasive`)
3. **Wrong last-seen point** (`wild_hard_wrong_ipp`)
4. **Person outside the area** (`wild_hard_outside`): nobody can find them; what matters is that the team does not waste the mission and the map's chance "outside" rises
5. **Sensor much worse than assumed** (`wild_hard_weak_sensor`, ×0.4)
6. **Late start** (`wild_hard_late_start`, 12 h)
7. **Radio dead zones** (`wild_hard_radio`, 300 m links)
8. **Behaviour that matches no hypothesis** (`wild_hard_off_model`)

Plus `wild_mix`, `urban_mix`, `flood_mix_heldout` and `wild_mix_adaptive_map`.

## Field checks for a production test

What a real test can measure to calibrate the rows marked *calibrate* (with
volunteers as subjects, never real emergencies):

- **Detection** (S1, S3, P7): for your drone, camera and terrain, run
  sweep-width style trials [8][9]: volunteers in known spots (standing,
  sitting, lying, under canopy, in the open), detections against distance,
  height, time of day and air temperature.
- **Repeated looks** (S2): the same hidden volunteer seen from the same and
  from different directions.
- **Operator review** (S5): time from detection to confirmation, and how
  many planted subjects operators miss over a long session.
- **False alarms** (S4): alerts per flight hour over your terrain.
- **Battery** (D1): flight time with your payload in your conditions; swap time.
- **Radio** (C1): link range from the base across your terrain.
- **GPS** (D4): position error near cliffs, in valleys and under trees.
- **Behaviour** (P1–P5) cannot be tested on real lost people. Use scripted
  volunteers, including the unresponsive and evasive cases, and replace the
  illustrative category numbers with ISRID statistics [1][2].
- **Your site** (E2): `scripts/import_site.py` turns elevation, tree canopy,
  trail and water layers into a wilderness world, so the policy can be flown
  in simulation on the test area first.

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
17. Tobler, W. (1993). Three presentations on geographical analysis and modeling (Tobler's hiking function). NCGIA Technical Report 93-1.
