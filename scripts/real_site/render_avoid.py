"""Video of one drone avoiding obstacles under the trees, close up.

    ~/venv3d/bin/python scripts/real_site/render_avoid.py --check 114 --out results/figures/avoid_demo.mp4

One under-canopy check (swarmsearch/fly_exec.py): the route is planned on the
drone's MAP (lidar, footprints, assumed power lines) and flown in the REAL
world, which has things the map lacks (saplings, dead branches, sheds, wires,
fallen trees...; world3d.make_truth). The drone's forward depth camera sees
them as it flies; when the route ahead is no longer clear it stops and
re-plans through the space it now knows.

Left: a camera following the drone along its own flown path. The world is
drawn as the simulator holds it, in 1 m voxels: green and brown = on the map,
orange (foliage) and red (anything else) = only in the real world.
Top right: the drone's forward camera (90 x 70 deg), with what the depth
camera has just detected marked. Bottom right: from above at under-canopy
flying height (1-6 m): the route planned on the map (blue, dashed), the
current plan after re-planning (blue), the route flown (orange) and the
detections (yellow).

--check picks the n-th sampled check of the search that
scripts/real_site/plot_surprise.py also uses (spots under the trees, starts
80 m away, random generator seed 3); 114, 283 and 299 on real world 1 each
stop, re-plan and reach the spot.
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "real_site"))

from swarmsearch import flight3d as f3  # noqa: E402
from swarmsearch import fly_exec as fx  # noqa: E402
from swarmsearch import world3d as w3  # noqa: E402

W, H = 1920, 1080
TOP, BOT = 76, 64
LW, LH = 1280, H - TOP - BOT
RW = W - LW
FW, FH = 616, 462  # forward camera panel
FPS = 24
MODE_TXT = {"cruise": "cruising over the trees", "descend": "descending through a gap in the canopy",
            "under": "flying under the canopy", "look": "looking at the alert spot",
            "climb": "climbing back out through the gap", "hover": "hovering"}
GREENS = ([0.24, 0.45, 0.19], [0.30, 0.52, 0.22], [0.19, 0.38, 0.16])
COL = {"trunk": [0.40, 0.29, 0.19], "building": [0.80, 0.76, 0.70], "pole": [0.55, 0.55, 0.55],
       "wire": [0.06, 0.06, 0.06], "new_veg": [0.98, 0.58, 0.12], "new_solid": [0.90, 0.16, 0.12],
       "new_wire": [0.80, 0.02, 0.02]}
SHADE = {"top": 1.0, "ns": 0.80, "ew": 0.66, "bottom": 0.5}
DRONE_RGB = [0.10, 0.85, 1.0]


def font(n, bold=False):
    f = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    return ImageFont.truetype(f, n) if os.path.exists(f) else ImageFont.load_default()


# ------------------------------------------------------------------ the check
class Recorder(fx.Executor):
    """The executor, also recording what it detects (per step) and each new plan."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.nlocal, self.plans = [], []

    def _sense(self, *a, **k):
        new = super()._sense(*a, **k)
        self.nlocal.append(len(self.local))
        return new

    def _replan(self, *a, **k):
        out = super()._replan(*a, **k)
        self.plans.append((len(self.nlocal) - 1, out[0]))
        return out


def worlds(site, fine_from, shift, world):
    """The map and the real world (cached next to the site: world_*.npz is not committed)."""
    pm, pt = site / "world_map.npz", site / f"world_truth_{world}.npz"
    if pm.exists() and pt.exists():
        return w3.load(pm), w3.load(pt)
    M = w3.build_map(fine_from, Path(fine_from).with_name("fine_points.npz"), site / "buildings.json",
                     site / "hazards.json", shift=tuple(shift))
    T = w3.make_truth(M, json.load(open(site / "hazards.json")), json.load(open(site / "buildings.json")), seed=world)
    w3.save(M, pm)
    w3.save(T, pt)
    return M, T


def fly(a, M, T):
    from swarmsearch.windfield import WindField
    S = Path(a.site)
    outer = f3.CoarseTerrain(S / "lidar10.npz", hazards=S / "hazards.json")
    L = np.load(S / "lidar10.npz")
    wf = WindField(L["dtm"], L["chm"], 10.0, 5.0, 270.0)
    base = f3.VoxPlanner(M, outer)
    factory = lambda extra: base if extra is None else f3.VoxPlanner(M, outer, known_extra=extra)  # noqa: E731
    rng = np.random.default_rng(3)
    forest = np.argwhere(M.top() > 10)
    for n in range(a.check + 1):  # the same draws as plot_surprise.py / the search
        tx, ty = M.xy(*forest[rng.integers(len(forest))])
        if not M.inside(tx, ty, 100):
            continue
        ang = rng.uniform(0, 2 * np.pi)
        start = (tx + 80 * np.cos(ang), ty + 80 * np.sin(ang))
    plan, _ = base.under_check(start, (tx, ty), 40.0)
    ex = Recorder(M, T, factory, wind=wf, nav="vio", sensing=True, seed=a.check, wind_ij=outer.ij)
    r = ex.fly_check(start, (tx, ty), 40.0)
    plans = [(i, [(x, y, h + M.ground(x, y), m) for x, y, h, m in p]) for i, p in ex.plans if p]
    return dict(trace=r.trace, events=r.events, outcome=r.outcome, reached=r.reached, replans=r.replans,
                planned_s=r.planned_s, time_s=r.time_s, detected=r.detected, plan0=[q[1:] for q in plan],
                plans=plans, local=np.array(ex.local, float).reshape(-1, 3), nlocal=ex.nlocal,
                target=(tx, ty), start=start, check=a.check, world=a.world)


# ------------------------------------------------------------------ scene
def box_of(rec, M, margin=18.0):
    P = np.array([q[1:3] for q in rec["trace"]] + [q[:2] for q in rec["plan0"]])
    x0, y0 = P.min(0) - margin
    x1, y1 = P.max(0) + margin
    i0, j0 = M.ij(x0, y1)
    i1, j1 = M.ij(x1, y0)
    return i0, i1 + 1, j0, j1 + 1


def classify(M, T, box):
    """Real-world voxels by colour class. 'New' = an object the map lacks: no voxel of its kind on the map
    within 2 m (so crown growth and small trunk or building offsets keep their usual colours)."""
    from scipy import ndimage
    i0, i1, j0, j1 = box
    occ_t, occ_m = T.occ[i0:i1, j0:j1], M.occ[i0:i1, j0:j1]
    near = lambda m: ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), bool), iterations=2)  # noqa: E731
    has = lambda o, k: (o & k) > 0  # noqa: E731
    wire, wire_m = has(occ_t, w3.WIRE), has(occ_m, w3.WIRE)
    sol_k = w3.TRUNK | w3.BLDG | w3.POLE | w3.STRUCT
    sol, sol_m = has(occ_t, sol_k) & ~wire, has(occ_m, sol_k)
    veg, veg_m = (occ_t > 0) & ~wire & ~sol, has(occ_m, w3.VEG)
    C = {"new_wire": wire & ~near(wire_m), "new_solid": sol & ~near(sol_m), "new_veg": veg & ~near(veg_m)}
    C["wire"] = wire & ~C["new_wire"]
    rest = sol & ~C["new_solid"]
    C["building"] = rest & has(occ_t, w3.BLDG)
    C["pole"] = rest & has(occ_t, w3.POLE | w3.STRUCT) & ~C["building"]
    C["trunk"] = rest & ~C["building"] & ~C["pole"]
    old = veg & ~C["new_veg"]
    ii, jj, kk = np.indices(veg.shape)
    hsh = (ii * 7 + jj * 13 + kk * 3) % 3
    for s_ in range(3):
        C[f"veg{s_}"] = old & (hsh == s_)
    return C, occ_t > 0


SLAB = 3  # metres: the follow view hides the slabs above the drone
CUT_M = 1.5  # ... from this far above the drone


def voxel_meshes(M, T, box, z0):
    """Exposed faces of the real world's voxels, grouped by colour, face direction and height slab."""
    i0, i1, j0, j1 = box
    g = T.dtm[i0:i1, j0:j1] - z0
    cat, solid = classify(M, T, box)
    pad = np.pad(solid, 1)
    nb = {  # neighbour occupied in each direction: +k up, -k down, -i north, +i south, +j east, -j west
        "up": pad[1:-1, 1:-1, 2:], "down": pad[1:-1, 1:-1, :-2], "north": pad[:-2, 1:-1, 1:-1],
        "south": pad[2:, 1:-1, 1:-1], "east": pad[1:-1, 2:, 1:-1], "west": pad[1:-1, :-2, 1:-1]}
    xw = T.east0 - T.n / 2 + j0  # west edge of the box
    yn = T.north0 + T.n / 2 - i0  # north edge
    out = []
    for name, mask in cat.items():
        base = GREENS[int(name[-1])] if name.startswith("veg") else COL[name]
        for d, sh in (("up", "top"), ("down", "bottom"), ("north", "ns"), ("south", "ns"), ("east", "ew"), ("west", "ew")):
            f = mask & ~nb[d]
            fi, fj, fk = np.nonzero(f)
            if not len(fi):
                continue
            x, y, z = xw + fj, yn - fi, g[fi, fj] + fk
            if d == "up":
                q = [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y - 1, z + 1), (x, y - 1, z + 1)]
            elif d == "down":
                q = [(x, y, z), (x, y - 1, z), (x + 1, y - 1, z), (x + 1, y, z)]
            elif d == "north":
                q = [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
            elif d == "south":
                q = [(x, y - 1, z), (x, y - 1, z + 1), (x + 1, y - 1, z + 1), (x + 1, y - 1, z)]
            elif d == "east":
                q = [(x + 1, y, z), (x + 1, y - 1, z), (x + 1, y - 1, z + 1), (x + 1, y, z + 1)]
            else:
                q = [(x, y, z), (x, y, z + 1), (x, y - 1, z + 1), (x, y - 1, z)]
            Vall = np.stack([np.stack([a, b, c], -1).astype(float) for a, b, c in q], 1)  # (n, 4, 3)
            rgba = [c * SHADE[sh] for c in base] + [1.0]
            for slab in np.unique(fk // SLAB):
                sel = fk // SLAB == slab
                V = Vall[sel].reshape(-1, 3)
                b4 = np.arange(sel.sum())[:, None] * 4
                F = np.concatenate([b4 + [0, 1, 2], b4 + [0, 2, 3], b4 + [0, 2, 1], b4 + [0, 3, 2]], 1).reshape(-1)
                out.append((int(slab), V, F, rgba))
    return out


def push(p, V, F, rgba, chunk=60000):
    F = F.reshape(-1, 3)
    ids = []
    start = 0
    while start < len(F):
        tri = F[start:start + chunk // 3]
        used, inv = np.unique(tri, return_inverse=True)
        vs = p.createVisualShape(p.GEOM_MESH, vertices=V[used].tolist(), indices=inv.ravel().astype(int).tolist(),
                                 rgbaColor=rgba)
        ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs))
        start += chunk // 3
    return ids


def ground_texture(site_dir, M, box, canopy):
    """The aerial photo under the box, darkened to forest floor where the canopy covers it."""
    from render_demo import BLOCK_M, Site, block_path, blocks_for, crop, fetch_block
    site = Site(Path(site_dir), dict(cell_m=1.0, g=1))
    i0, i1, j0, j1 = box
    x0, y1 = M.east0 - M.n / 2 + j0, M.north0 + M.n / 2 - i0
    wdt, hgt = j1 - j0, i1 - i0
    half = max(wdt, hgt) / 2
    cx, cy = x0 + wdt / 2, y1 - hgt / 2
    todo = [b for b in blocks_for(cx, cy, half + BLOCK_M * 0.1) if not block_path(site, *b).exists()]
    for b in todo:
        fetch_block((str(site_dir), site.meta, *b))
    px = 1024
    im = crop(site, cx, cy, half, px=px)
    s = px / (2 * half)
    im = im.crop((int((half - wdt / 2) * s), int((half - hgt / 2) * s), int((half + wdt / 2) * s),
                  int((half + hgt / 2) * s))).resize((wdt * 8, hgt * 8), Image.BICUBIC)
    a = np.asarray(im, float)
    cov = np.asarray(Image.fromarray((np.clip(canopy / 4.0, 0, 1) * 255).astype(np.uint8)).resize(
        (wdt * 8, hgt * 8), Image.BILINEAR), float)[..., None] / 255
    floor = np.array([92, 78, 54], float) + np.random.default_rng(0).normal(0, 9, a.shape[:2])[..., None]
    a = a * (1 - 0.8 * cov) + floor * 0.8 * cov
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def build_scene(site_dir, M, T, box, z0):
    import pybullet as p
    p.connect(p.DIRECT)
    i0, i1, j0, j1 = box
    dem = T.dtm[i0:i1, j0:j1] - z0
    m_i, m_j = dem.shape
    jj, ii = np.meshgrid(np.arange(m_j + 1), np.arange(m_i + 1))
    dp = np.pad(dem, ((0, 1), (0, 1)), mode="edge")
    xw, yn = T.east0 - T.n / 2 + j0, T.north0 + T.n / 2 - i0
    verts = np.stack([xw + jj, yn - ii, dp], -1).reshape(-1, 3)
    uv = np.stack([jj / m_j, 1 - ii / m_i], -1).reshape(-1, 2)
    idx = []
    W_ = m_j + 1
    for i in range(m_i):
        for j in range(m_j):
            a_, b_, c_, d_ = i * W_ + j, i * W_ + j + 1, (i + 1) * W_ + j, (i + 1) * W_ + j + 1
            idx += [a_, c_, b_, b_, c_, d_]
    vs = p.createVisualShape(p.GEOM_MESH, vertices=verts.tolist(), indices=idx, uvs=uv.tolist(),
                             normals=[[0, 0, 1]] * len(verts))
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
    canopy = (T.occ[i0:i1, j0:j1, 3:] & w3.VEG).any(axis=2).astype(float) * 4
    tex = Path("/tmp") / f"avoid_tex_{os.getpid()}.png"
    ground_texture(site_dir, M, box, canopy).save(tex)
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(str(tex)), rgbaColor=[1, 1, 1, 1])
    slabs = {}
    for slab, V, F, rgba in voxel_meshes(M, T, box, z0):
        slabs.setdefault(slab, []).extend(push(p, V, F, rgba))
    # the drone: a body, four arms and rotors (span 1 m, the SAR quad)
    parts = [(p.GEOM_BOX, dict(halfExtents=[0.16, 0.16, 0.06]), [0, 0, 0], DRONE_RGB + [1])]
    for ang in (45, 135, 225, 315):
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        parts.append((p.GEOM_BOX, dict(halfExtents=[0.18, 0.025, 0.02]), [0.18 * c, 0.18 * s, 0], [0.1, 0.1, 0.1, 1]))
        parts.append((p.GEOM_CYLINDER, dict(radius=0.14, length=0.015), [0.36 * c, 0.36 * s, 0.05], [0.75, 0.75, 0.78, 1]))
    shapes = []
    for kind, kw, pos, rgba in parts:
        shapes.append(p.createVisualShape(kind, rgbaColor=rgba, visualFramePosition=pos, **kw))
    drone = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shapes[0],
                              linkMasses=[0] * (len(shapes) - 1), linkCollisionShapeIndices=[-1] * (len(shapes) - 1),
                              linkVisualShapeIndices=shapes[1:], linkPositions=[[0, 0, 0]] * (len(shapes) - 1),
                              linkOrientations=[[0, 0, 0, 1]] * (len(shapes) - 1),
                              linkInertialFramePositions=[[0, 0, 0]] * (len(shapes) - 1),
                              linkInertialFrameOrientations=[[0, 0, 0, 1]] * (len(shapes) - 1),
                              linkParentIndices=[0] * (len(shapes) - 1),
                              linkJointTypes=[p.JOINT_FIXED] * (len(shapes) - 1),
                              linkJointAxis=[[0, 0, 1]] * (len(shapes) - 1))
    return p, drone, slabs


def render(p, eye, target, w, h, fov, up=(0, 0, 1), near=0.3):
    V = p.computeViewMatrix(list(eye), list(target), list(up))
    P = p.computeProjectionMatrixFOV(fov=fov, aspect=w / h, nearVal=near, farVal=800.0)
    _, _, rgb, depth, _ = p.getCameraImage(w, h, V, P, renderer=p.ER_TINY_RENDERER, shadow=0,
                                           lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
    rgb = np.reshape(rgb, (h, w, 4))[..., :3].astype(np.uint8).copy()
    depth = np.reshape(depth, (h, w))
    sky = depth >= 0.99999
    t = (np.arange(h) / h)[:, None, None]
    grad = (np.array([112, 148, 196]) * (1 - t) + np.array([208, 220, 232]) * t).astype(np.uint8)
    rgb[sky] = np.broadcast_to(grad, (h, w, 3))[sky]
    M = np.array(P).reshape(4, 4).T @ np.array(V).reshape(4, 4).T
    return rgb, depth, M


def project(M, pts, w, h):
    pts = np.atleast_2d(np.asarray(pts, float))
    c = np.c_[pts, np.ones(len(pts))] @ M.T
    ndc = c[:, :3] / c[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h, ndc[:, 2] * 0.5 + 0.5], c[:, 3] > 0.05


def draw_path(dr, M, depth, pts, w, h, color, width, dash=False, occl=True):
    """A 3D polyline; segments hidden behind something are left out (depth test)."""
    if len(pts) < 2:
        return
    q, ok = project(M, pts, w, h)
    for k in range(len(q) - 1):
        if not (ok[k] and ok[k + 1]):
            continue
        if dash and (k // 3) % 2:
            continue
        a, b = q[k], q[k + 1]
        vis = True
        if occl:
            for u in (0.0, 0.5, 1.0):
                x, y, z = a + u * (b - a)
                xi, yi = int(x), int(y)
                if 0 <= xi < w and 0 <= yi < h and z > depth[yi, xi] + 2e-4:
                    vis = False
        if vis and -w < a[0] < 2 * w and -h < a[1] < 2 * h:
            dr.line([tuple(a[:2]), tuple(b[:2])], fill=color, width=width)


# ------------------------------------------------------------------ frames
_W: dict = {}


def _init(site_dir, rec_path):
    rec = pickle.load(open(rec_path, "rb"))
    M = w3.load(Path(site_dir) / "world_map.npz")
    T = w3.load(Path(site_dir) / f"world_truth_{rec['world']}.npz")
    p, drone, slabs = build_scene(site_dir, M, T, rec["box"], rec["z0"])
    _W.update(rec=rec, M=M, T=T, p=p, drone=drone, slabs=slabs, hidden=set())


def cut_above(k_cut):
    """Hide the voxel slabs from k_cut metres above the ground up (None: show all)."""
    p, slabs, hidden = _W["p"], _W["slabs"], _W["hidden"]
    want = {s for s in slabs if k_cut is not None and s * SLAB >= k_cut}
    for s in want ^ hidden:
        for b in slabs[s]:
            p.resetBasePositionAndOrientation(b, [0, 0, -2000] if s in want else [0, 0, 0], [0, 0, 0, 1])
    _W["hidden"] = want


def state(rec, t):
    """Drone position (local z), mode and trace index at time t."""
    tt = rec["tt"]
    k = int(np.clip(np.searchsorted(tt, t, side="right") - 1, 0, len(tt) - 1))
    k2 = min(k + 1, len(tt) - 1)
    u = 0.0 if k2 == k else float(np.clip((t - tt[k]) / max(tt[k2] - tt[k], 1e-9), 0, 1))
    pos = rec["P"][k] * (1 - u) + rec["P"][k2] * u
    return pos, rec["modes"][k], k


def heading_at(rec, k):
    P = rec["P"]
    for d in range(1, 40):
        a, b = P[max(0, k - d)], P[min(len(P) - 1, k + d)]
        v = b - a
        if math.hypot(v[0], v[1]) > 0.3:
            return math.atan2(v[1], v[0]), v
    return 0.0, np.array([1.0, 0, 0])


def make(job):
    fi, t, eye, look, banner = job
    rec, M, T, p, drone = _W["rec"], _W["M"], _W["T"], _W["p"], _W["drone"]
    z0 = rec["z0"]
    pos, mode, k = state(rec, t)
    yaw, v = heading_at(rec, k)
    p.resetBasePositionAndOrientation(drone, pos.tolist(), p.getQuaternionFromEuler([0, 0, yaw]))
    canvas = Image.new("RGB", (W, H), (14, 15, 18))
    dc = ImageDraw.Draw(canvas)
    # --- follow camera
    d = np.asarray(look) - np.asarray(eye)
    up = (0, 0, 1)
    if math.hypot(d[0], d[1]) < 0.35 * np.linalg.norm(d):
        up = (math.cos(yaw), math.sin(yaw), 0)
    agl = pos[2] + z0 - T.ground(pos[0], pos[1])
    cut_above(agl + CUT_M if agl < 25 else None)
    rgb, depth, Mx = render(p, eye, look, LW, LH, 58, up=up)
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    plan = rec["plan0_local"] if not rec["plans"] or rec["plans"][0][0] > k else \
        [pl for i, pl in rec["plans_local"] if i <= k][-1]
    draw_path(dr, Mx, depth, rec["plan0_local"], LW, LH, (90, 140, 230), 2, dash=True)
    if plan is not rec["plan0_local"]:
        draw_path(dr, Mx, depth, plan, LW, LH, (60, 120, 255), 4)
    draw_path(dr, Mx, depth, rec["P"][max(0, k - 600):k + 1:2], LW, LH, (255, 150, 30), 4)
    # detections in the last 2 s: small yellow squares
    n1 = rec["nlocal"][min(k, len(rec["nlocal"]) - 1)] if rec["nlocal"] else 0
    n0 = rec["nlocal"][max(0, min(k, len(rec["nlocal"]) - 1) - 20)] if rec["nlocal"] else 0
    det = rec["local_abs"][n0:n1]
    if len(det):
        q, ok = project(Mx, det, LW, LH)
        for (x, y, z), o in zip(q, ok):
            if o and 0 <= x < LW and 0 <= y < LH and z <= depth[int(y), int(x)] + 5e-4:
                dr.rectangle([x - 3, y - 3, x + 3, y + 3], outline=(255, 235, 60), width=2)
    canvas.paste(img, (0, TOP))
    # --- forward camera (the drone's own view)
    hdir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    vz = 0.0
    if mode == "descend":
        vz = -0.9
    elif mode == "climb":
        vz = 0.9
    fwd = hdir + np.array([0, 0, vz])
    p.resetBasePositionAndOrientation(drone, [0, 0, -500], [0, 0, 0, 1])
    cut_above(None)
    e2 = pos + np.array([0, 0, 0.05]) + 0.25 * hdir
    rgb2, depth2, M2 = render(p, e2, e2 + fwd, FW, FH, 70, near=0.1)
    img2 = Image.fromarray(rgb2)
    d2 = ImageDraw.Draw(img2)
    if len(det):
        q, ok = project(M2, det, FW, FH)
        for (x, y, z), o in zip(q, ok):
            if o and 0 <= x < FW and 0 <= y < FH and z <= depth2[int(y), int(x)] + 1e-3:
                d2.rectangle([x - 4, y - 4, x + 4, y + 4], outline=(255, 235, 60), width=2)
    cx, cy = FW // 2, FH // 2
    for a_, b_ in (((cx - 30, cy), (cx - 10, cy)), ((cx + 10, cy), (cx + 30, cy)), ((cx, cy - 30), (cx, cy - 10)),
                   ((cx, cy + 10), (cx, cy + 30))):
        d2.line([a_, b_], fill=(120, 255, 140), width=2)
    d2.rectangle([0, 0, FW, 30], fill=(0, 0, 0))
    d2.text((10, 6), f"forward camera   {agl:4.1f} m up   heading {(90 - math.degrees(yaw)) % 360:3.0f}°",
            font=font(16, True), fill=(120, 255, 140))
    if banner:
        d2.rectangle([0, FH - 34, FW, FH], fill=(150, 20, 15))
        d2.text((10, FH - 28), "OBSTACLE AHEAD: STOP AND RE-PLAN", font=font(18, True), fill=(255, 255, 255))
    canvas.paste(img2, (LW + 12, TOP + 8))
    dc.rectangle([LW + 11, TOP + 7, LW + 12 + FW, TOP + 8 + FH], outline=(80, 80, 90))
    # --- from above
    mp = rec["topmap"].copy()
    dm = ImageDraw.Draw(mp)
    s, (mx0, my1) = rec["top_scale"], rec["top_origin"]
    to = lambda x, y: ((x - mx0) * s, (my1 - y) * s)  # noqa: E731

    def line(pts, col, wdt, dash=False):
        q = [to(x, y) for x, y in pts]
        for a in range(len(q) - 1):
            if not dash or (a // 3) % 2 == 0:
                dm.line([q[a], q[a + 1]], fill=col, width=wdt)
    line([(x, y) for x, y, *_ in rec["plan0"]], (90, 140, 230), 2, dash=True)
    if plan is not rec["plan0_local"]:
        line([(x, y) for x, y, *_ in [pl for i, pl in rec["plans"] if i <= k][-1]], (60, 120, 255), 3)
    line([(q[1], q[2]) for q in rec["trace"][:k + 1:3]], (255, 150, 30), 3)
    for x, y, _ in rec["local"][:n1:3]:
        u_, v_ = to(x, y)
        dm.point((u_, v_), fill=(255, 235, 60))
    tx, ty = to(*rec["target"])
    dm.ellipse([tx - 7, ty - 7, tx + 7, ty + 7], outline=(60, 230, 90), width=3)
    ux, uy = to(pos[0], pos[1])
    dm.ellipse([ux - 6, uy - 6, ux + 6, uy + 6], fill=tuple(int(255 * c) for c in DRONE_RGB), outline=(0, 0, 0))
    dm.line([(ux, uy), (ux + 16 * math.cos(yaw), uy - 16 * math.sin(yaw))], fill=(0, 0, 0), width=3)
    oy = TOP + FH + 22
    canvas.paste(mp, (LW + 12, oy + 24))
    dc.text((LW + 12, oy), "From above, at under-canopy flying height (1-6 m)", font=font(16, True), fill=(230, 230, 230))
    lx = LW + 12 + mp.size[0] + 12
    items = [((90, 140, 230), "planned on the map"), ((60, 120, 255), "re-planned"), ((255, 150, 30), "flown"),
             ((255, 235, 60), "seen by the camera"), ((60, 230, 90), "alert spot"), ((150, 150, 150), "on the map"),
             ((235, 110, 40), "not on the map"), ((0, 0, 0), "wire"), ((200, 10, 10), "wire not on the map")]
    for i, (c, txt) in enumerate(items):
        y = oy + 30 + 24 * i
        dc.rectangle([lx, y + 3, lx + 14, y + 15], fill=c, outline=(200, 200, 200))
        dc.text((lx + 20, y), txt, font=font(14), fill=(200, 200, 200))
    # --- captions
    dc.text((16, 8), f"Detect and avoid under the trees  ·  t = {t:4.1f} s", font=font(26, True), fill=(255, 255, 255))
    sub = f"{MODE_TXT.get(mode, mode)}  ·  {agl:.1f} m above the ground"
    dc.text((16, 44), sub, font=font(18), fill=(255, 205, 120) if banner else (210, 210, 210))
    if banner:
        dc.rectangle([0, TOP, LW, TOP + 46], fill=(150, 20, 15))
        dc.text((16, TOP + 10), banner, font=font(20, True), fill=(255, 255, 255))
    lg = "Follow camera, trees above the drone cut away · 1 m voxels (wires as 1 m bands) · orange/red: objects " \
         "not on the drone's map · yellow: just detected"
    dc.text((16, TOP + LH - 26), lg, font=font(15), fill=(240, 240, 240))
    dc.rectangle([0, H - BOT, W, H], fill=(0, 0, 0))
    dc.text((16, H - BOT + 8), "Real: ground, trees and buildings from USGS 3DEP lidar (2021) and building footprints; "
            "aerial photo USDA NAIP (Catoctin, Maryland).", font=font(15), fill=(170, 230, 170))
    dc.text((16, H - BOT + 32), "Simulated: what the map lacks (drawn at random: saplings, dead branches, sheds, wires, "
            "fallen trees), the drone, its depth camera, navigation and wind.", font=font(15), fill=(255, 200, 140))
    return fi, np.asarray(canvas)


# ------------------------------------------------------------------ main
def topmap(rec, M, T, box, px=440):
    C, solid = classify(M, T, box)
    band = slice(1, 7)
    tb = solid[:, :, band].any(2)
    newb = (C["new_veg"] | C["new_solid"])[:, :, band].any(2)
    wire = (C["wire"] | C["new_wire"])[:, :, band].any(2)
    img = np.zeros(tb.shape + (3,), float) + [38, 44, 36]
    img[tb] = [150, 150, 150]
    img[newb] = [235, 110, 40]
    img[wire] = [0, 0, 0]
    img[C["new_wire"][:, :, band].any(2)] = [200, 10, 10]
    im = Image.fromarray(img.astype(np.uint8))
    sc = px / max(tb.shape)
    im = im.resize((int(tb.shape[1] * sc), int(tb.shape[0] * sc)), Image.NEAREST)
    i0, _, j0, _ = box
    return im, sc, (M.east0 - M.n / 2 + j0, M.north0 + M.n / 2 - i0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0))
    ap.add_argument("--world", type=int, default=1, help="which draw of the real world (what the map lacks)")
    ap.add_argument("--check", type=int, default=114)
    ap.add_argument("--slow", type=float, default=1.0, help="video seconds per mission second under the trees")
    ap.add_argument("--fast", type=float, default=0.33, help="video seconds per mission second above the trees")
    ap.add_argument("--out", default="results/figures/avoid_demo.mp4")
    ap.add_argument("--still", default="")
    ap.add_argument("--at", type=float, default=-1.0, help="render only the frame at this mission time")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    S = Path(a.site)
    M, T = worlds(S, a.fine_from, a.shift, a.world)
    t0 = time.time()
    rec = fly(a, M, T)
    print(f"check {a.check}: {rec['outcome']}, reached {rec['reached']}, {rec['replans']} re-plans, "
          f"{rec['time_s']:.0f} s (planned {rec['planned_s']:.0f} s), detected {rec['detected']}, "
          f"events {[(e[0], round(e[1], 1)) for e in rec['events']]} ({time.time() - t0:.0f}s)", flush=True)
    box = box_of(rec, M)
    i0, i1, j0, j1 = box
    z0 = float(T.dtm[i0:i1, j0:j1].min())
    rec.update(box=box, z0=z0)
    rec["P"] = np.array([(q[1], q[2], q[3] - z0) for q in rec["trace"]])
    rec["tt"] = np.array([q[0] for q in rec["trace"]])
    rec["modes"] = [q[4] for q in rec["trace"]]
    rec["plan0_local"] = np.array([(x, y, z - z0) for x, y, z, *_ in rec["plan0"]])
    rec["plans_local"] = [(i, np.array([(x, y, z - z0) for x, y, z, _ in pl])) for i, pl in rec["plans"]]
    L_ = rec["local"]
    rec["local_abs"] = np.array([(x, y, h + T.ground(x, y) - z0) for x, y, h in L_]).reshape(-1, 3)
    rec["topmap"], rec["top_scale"], rec["top_origin"] = topmap(rec, M, T, box)
    # what made it stop: the class of the voxels detected in the second before each stop
    C, _ = classify(M, T, box)
    say = {"new_veg": "a sapling or branch that is not on the map", "new_solid": "an object that is not on the map",
           "new_wire": "a wire that is not on the map", "wire": "a wire", "trunk": "a trunk (off its mapped spot)",
           "building": "a building", "pole": "a pole", "veg": "foliage lower or wider than mapped (growth)"}
    banners = []
    for e in rec["events"]:
        if e[0] != "stop and re-plan":
            continue
        k = int(np.searchsorted(rec["tt"], e[1]))
        n0, n1 = rec["nlocal"][max(0, k - 10)], rec["nlocal"][min(k, len(rec["nlocal"]) - 1)]
        kinds = {}
        for x, y, h in L_[n0:n1]:
            i, j = T.ij(x, y)
            kk = int(min(h, w3.NZ - 1))
            for nm, m in C.items():
                if 0 <= i - i0 < m.shape[0] and 0 <= j - j0 < m.shape[1] and m[i - i0, j - j0, kk]:
                    nm = "veg" if nm.startswith("veg") else nm
                    kinds[nm] = kinds.get(nm, 0) + (5 if nm.startswith("new") else 1)
        what = say[max(kinds, key=kinds.get)] if kinds else "an obstacle"
        banners.append((e[1], f"Saw {what} on the route ahead: stop, re-plan"))
        print(f"stop at {e[1]:.1f} s: {kinds}")
    # frame times: slow under the trees, fast above; a pause at each stop
    times, flags = [], []
    low_t = [q[0] for q in rec["trace"] if q[4] != "cruise"]
    t = max(0.0, low_t[0] - 4.0) if low_t else 0.0
    climb = [q[0] for q in rec["trace"] if q[4] == "climb"]
    tend = min(rec["tt"][-1], climb[-1] + 2.0) if climb else rec["tt"][-1]
    bi = 0
    while t <= tend:
        _, mode, k = state(rec, t)
        low = mode in ("under", "look", "descend", "climb") or rec["P"][k][2] + z0 - T.ground(*rec["P"][k][:2]) < 25
        if bi < len(banners) and t >= banners[bi][0]:
            times += [t] * int(2.5 * FPS)
            flags += [banners[bi][1]] * int(2.5 * FPS)
            bi += 1
        times.append(t)
        flags.append(next((b for tb, b in banners if tb <= t < tb + 4.0), ""))
        rate = a.slow if low else a.fast
        if mode == "look" and not flags[-1]:
            rate *= 0.4
        t += 1.0 / (FPS * rate)
    # follow camera: behind the drone (against its heading) and above; the canopy above the drone is cut
    # away in that view while it is low. The offset is smoothed; the camera is lifted until nothing still
    # drawn (or the ground) is between it and the drone.
    solid = T.occ > 0

    def clear(e, pos, kv):
        for u in np.linspace(0.0, 1.0, 50):
            q = e + u * (pos - e)
            if np.linalg.norm(q - pos) < 1.5:
                break
            i, j = T.ij(q[0], q[1])
            kq = int(math.floor(q[2] - (T.dtm[i, j] - z0)))
            if kq < 0 or (kq < kv and solid[i, j, min(kq, w3.NZ - 1)]):
                return False
        return True

    eyes, looks = [], []
    off_s, lift_s = None, 0.0
    for t in times:
        pos, mode, k = state(rec, t)
        agl = pos[2] + z0 - T.ground(pos[0], pos[1])
        yaw, _ = heading_at(rec, k)
        back, upm = (7.0, 5.0) if agl < 25 else (18.0, 9.0)
        off = np.array([-back * math.cos(yaw), -back * math.sin(yaw), upm])
        off_s = off if off_s is None else 0.93 * off_s + 0.07 * off
        eye = pos + off_s
        kv = int(math.ceil((agl + CUT_M) / SLAB) * SLAB) if agl < 25 else w3.NZ
        lift = 0.0
        while lift < 25 and not clear(eye + [0, 0, lift], pos, kv):
            lift += 1.0
        lift_s = lift if lift >= lift_s else 0.95 * lift_s + 0.05 * lift
        if not clear(eye + [0, 0, lift_s], pos, kv):
            lift_s = lift
        eyes.append(eye + [0, 0, lift_s])
        looks.append(pos + np.array([1.0 * math.cos(yaw), 1.0 * math.sin(yaw), -0.5]))
    jobs = [(i, times[i], eyes[i], looks[i], flags[i]) for i in range(len(times))]
    if a.at >= 0:
        i = int(np.argmin(np.abs(np.array(times) - a.at)))
        jobs = [jobs[i]]
    elif a.frames:
        jobs = jobs[:a.frames]
    rp = Path("/tmp") / f"avoid_rec_{os.getpid()}.pkl"
    pickle.dump(rec, open(rp, "wb"))
    print(f"{len(jobs)} frames ({len(jobs) / FPS:.0f} s of video)", flush=True)
    if a.still or a.at >= 0:
        _init(str(S), str(rp))
        _, fr = make(jobs[0])
        Image.fromarray(fr).save(a.still or "avoid_still.png")
        print("wrote", a.still)
        if a.at >= 0:
            return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    t1 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(S), str(rp))) as ex:
        for n, (fi, fr) in enumerate(ex.map(make, jobs, chunksize=4)):
            ff.stdin.write(fr.tobytes())
            if n % 100 == 0:
                print(f"frame {n}/{len(jobs)} {time.time() - t1:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {len(jobs)} frames, {len(jobs) / FPS:.0f} s, {time.time() - t1:.0f}s to render")


if __name__ == "__main__":
    main()
