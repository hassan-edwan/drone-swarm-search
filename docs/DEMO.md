# Demo on a real site

Two videos of one simulated search mission flown over real terrain:
Catoctin Mountain Park, Maryland (an 11.6 km square around the park's centre).

- `results/figures/real_site_demo.mp4`: a 3D view of the site (real elevation
  with the real aerial photo draped on it), orbiting slowly, with the drones'
  tracks, the last-seen point and the hiker's true position; each drone's
  camera and the team's probability map on the right.
- `results/figures/real_site_cameras.mp4`: each drone's own camera feed, large,
  turned so the direction of flight is up, with a head-up display (height,
  heading, battery, radio link, current action). When a drone detects the
  hiker the feed shows a detection box and a 4x zoom.

## What is real and what is simulated

Real:
- Elevation: AWS Terrain Tiles (from USGS 3DEP), about 15 m.
- Aerial photo: USDA NAIP through the USGS National Map, about 0.35 m per
  pixel in the camera views. Each camera view is the photo of exactly the
  ground the drone's camera covers at its height (84 degree field of view).
- Tree cover and water in the simulator's world are estimated from the
  photo's colour (rough; not a land-cover product). No trails: the trail
  source (OpenStreetMap) was not reachable from the build machine.

Simulated:
- The drones, their flight and batteries, the missing hiker and where they
  went, and every detection and false alarm.
- The hiker is drawn into the photo at true size (about 1.7 m), fainter
  where the simulator says trees hide them.
- The drones decide from the simulator's sensor model (detection rates by
  height, tree cover and light). No detector looks at these pictures.
  The pictures show what the cameras would see; they do not drive the search.

The mission: wild_mix scenario seed 4 (hiker, 4 drones) on the real site,
flown by the learned target policy (`runs/p3/rl/best.pt`, greedy). The hiker
is found at step 258 of 299 (about 77 minutes in). The video runs at about
54 times real time (one 18 s simulator step every 8 frames at 24 fps).

## The same site flown by the rule

- `results/figures/rule_demo.mp4`: the 3D view, as above, flown by the rule
  (BeliefGreedy) instead of the learned policy.
- `results/figures/rule_cameras.mp4`: the four camera feeds for that mission.

The mission: wild_mix scenario seed 7 (a missing hunter, 4 drones). The rule
finds the hunter at step 63 (about 19 minutes in); the mission ends at step 223.
On seed 4, the learned policy's demo mission, the rule does not find the hiker in
the 299 steps (about 90 minutes). Of seeds 0-12 the rule finds the person in two
(7 and 10). Captions name the mission's person (hiker, hunter, ...).

    python scripts/real_site/run_mission.py --site sites/catoctin --seed 7 --policy rule
    ~/venv3d/bin/python scripts/real_site/render_cameras.py --seed 7 --out results/figures/rule_cameras.mp4
    ~/venv3d/bin/python scripts/real_site/render_demo.py --seed 7 --out results/figures/rule_demo.mp4

## Making it again

    python scripts/real_site/fetch_site.py --lat 39.6385 --lon -77.4730 --out sites/catoctin
    python scripts/import_site.py --dem sites/catoctin/dem.npy --canopy sites/catoctin/canopy.npy \
        --water sites/catoctin/water.npy --cell-m 30 --ipp 193.5 193.5 --out sites/catoctin/site.npz
    python scripts/real_site/run_mission.py --site sites/catoctin --seed 4 --policy runs/p3/rl/best.pt
    ~/venv3d/bin/python scripts/real_site/render_cameras.py --out results/figures/real_site_cameras.mp4
    ~/venv3d/bin/python scripts/real_site/render_demo.py --out results/figures/real_site_demo.mp4

The water mask came from a short snippet (blue, dark pixels of the photo, more
than half of each 30 m pixel); it is saved in `sites/catoctin/water.npy`. The
renderers need PyBullet (requirements-3d.txt) and ffmpeg. Any US site works:
change --lat/--lon.
