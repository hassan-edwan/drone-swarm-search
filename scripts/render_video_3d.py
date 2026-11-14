"""Video of one 3D mission: Crazyflies in PyBullet (left) and the team's map (right).

    ~/venv3d/bin/python scripts/render_video_3d.py --condition m4_to_safety_unknown --seed 0 \\
        --out results/figures/flight_3d.mp4

The left view is PyBullet's own renderer: buildings at their real height,
the drones flown by PID, with a coloured marker on each drone (a Crazyflie
is 10 cm across, too small to see over a 20 m area) and survivors on the
ground (red = not found, green = found). The right panel is the 2D mission
layer at the same moment: belief heat, drones, survivors. Needs PyBullet
(requirements-3d.txt) and ffmpeg.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.baselines import Lawnmower  # noqa: E402
from swarmsearch.env import GridSearchEnv  # noqa: E402
from swarmsearch.evaluate import ALL_CONDITIONS as CONDITIONS  # noqa: E402
from swarmsearch.mappo import LearnedPolicy, load  # noqa: E402
from swarmsearch.sim3d import CELL_M, Swarm3D  # noqa: E402

DRONE_RGB = [(0.16, 0.47, 0.84), (0.11, 0.69, 0.48), (0.38, 0.31, 0.84), (0.91, 0.48, 0.64),
             (0.93, 0.63, 0.0), (0.0, 0.51, 0.0), (0.89, 0.29, 0.28), (0.32, 0.32, 0.31)]
W, H = 960, 720
FONT = ImageFont.load_default(size=17)
EVERY = 24  # control steps (48 Hz) between frames: 0.5 s of flight per frame


def map_panel(env, size) -> Image.Image:
    g = env.grid_size
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100, facecolor="#fcfcfb")
    ax.imshow(env.belief * env.n_free, cmap="Oranges", vmin=0, vmax=3)
    oy, ox = np.nonzero(env.obstacles)
    ax.scatter(ox, oy, marker="s", s=90, c="#333333")
    s = env.survivors
    ax.scatter(s[:, 0], s[:, 1], s=55, facecolors=["#15233B" if f else "none" for f in env.found],
               edgecolors="#15233B", linewidths=1.6)
    for i, a in enumerate(env.possible_agents[: env.scenario.n_drones]):
        if env.alive.get(a):
            ax.scatter(*env.pos[a], marker="s", s=110, color=DRONE_RGB[i], edgecolors="white", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, g - 0.5); ax.set_ylim(g - 0.5, -0.5)
    ax.set_title("Team's map: heat = where survivors may be", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#fcfcfb")
    plt.close(fig)
    return Image.open(buf).convert("RGB").resize((size, size))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="m4_to_safety_unknown", choices=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="models/m2_2m.pt", help="omit with --lawnmower")
    p.add_argument("--lawnmower", action="store_true")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (quick test)")
    p.add_argument("--out", default="results/figures/flight_3d.mp4")
    a = p.parse_args()

    sc = CONDITIONS[a.condition](a.seed)
    policy, name = (Lawnmower(), "Lawnmower") if a.lawnmower else (LearnedPolicy(load(a.model)[0], temperature=0), "MAPPO")
    env = GridSearchEnv(sc)
    sim = Swarm3D(env, backend="pybullet", physical_obstacles=not sc.corner_cutting)
    sim.reset(sc, a.seed)
    policy.reset(env)
    pb, fl = sim.flight.p, sim.flight
    cid = fl.env.CLIENT
    g = env.grid_size

    def sphere(r, rgb):
        vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=r, rgbaColor=[*rgb, 1], physicsClientId=cid)
        return pb.createMultiBody(0, -1, vis, [0, 0, -5], physicsClientId=cid)

    markers = [sphere(0.3, DRONE_RGB[i]) for i in range(len(sim.agents_all))]
    people = [sphere(0.22, (0.85, 0.2, 0.2)) for _ in range(len(env.survivors))]
    centre = [g * CELL_M / 2, g * CELL_M / 2, 0]
    view = pb.computeViewMatrixFromYawPitchRoll(centre, distance=g * 1.85, yaw=30, pitch=-64, roll=0,
                                                upAxisIndex=2, physicsClientId=cid)
    proj = pb.computeProjectionMatrixFOV(fov=50, aspect=W / H, nearVal=0.1, farVal=100, physicsClientId=cid)

    def place_people():
        for b, (x, y), f in zip(people, env.survivors, env.found):
            pb.resetBasePositionAndOrientation(b, [(x + 0.5) * CELL_M, (g - 1 - y + 0.5) * CELL_M, 0.15],
                                               [0, 0, 0, 1], physicsClientId=cid)
            pb.changeVisualShape(b, -1, rgbaColor=[0.1, 0.7, 0.3, 1] if f else [0.85, 0.2, 0.2, 1],
                                 physicsClientId=cid)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    side = H
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W + side}x{H}", "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", "-crf", "23", a.out], stdin=subprocess.PIPE)
    state = {"k": 0, "panel": map_panel(env, side)}

    def frame():
        for m, pos in zip(markers, fl.pos):
            pb.resetBasePositionAndOrientation(m, list(pos), [0, 0, 0, 1], physicsClientId=cid)
        img = pb.getCameraImage(W, H, view, proj, renderer=pb.ER_TINY_RENDERER, physicsClientId=cid)[2]
        left = Image.fromarray(np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8))
        d = ImageDraw.Draw(left)
        d.rectangle([0, 0, W, 34], fill=(252, 252, 251))
        d.text((12, 8), f"{name} flown by Crazyflies in PyBullet (PID)   ·   step {env.t}   ·   "
                        f"found {int(env.found.sum())}/{len(env.found)}", fill=(11, 11, 11), font=FONT)
        out = Image.new("RGB", (W + side, H), (252, 252, 251))
        out.paste(left, (0, 0))
        out.paste(state["panel"], (W, 0))
        ff.stdin.write(out.tobytes())

    step0 = fl.step

    def step_and_film(ref_pos, ref_vel):
        pos = step0(ref_pos, ref_vel)
        state["k"] += 1
        if state["k"] % EVERY == 0:
            frame()
        return pos

    fl.step = step_and_film
    place_people()
    while env.agents and (a.max_steps is None or env.t < a.max_steps):
        sim.step(policy.act(env))
        place_people()
        state["panel"] = map_panel(env, side)
    for _ in range(2 * a.fps):
        frame()
    ff.stdin.close()
    ff.wait()
    m = sim.metrics()
    print(f"{name}: found {m['found_frac']:.0%}, mean find time {m['find_time']:.0f} steps, "
          f"late moves {m['late_frac']:.0%}, building hits {m['obstacle_hits']}, crashed {m['crashed']}")
    print("wrote", a.out)
    sim.close()


if __name__ == "__main__":
    main()
