"""Video of drones flying through a real site's obstacles: over, around and under.

    ~/venv3d/bin/python scripts/real_site/render_flight3d.py --site sites/catoctin_west --seed 103 \\
        --out results/figures/flight3d_demo.mp4

Left: 3D view (PyBullet's software renderer) of the area mapped by lidar at
full density: the aerial photo on the bare-earth terrain; every tree the
lidar shows, as a crown (from its measured crown bottom to its top) on a
trunk; every building at its measured height. Drone tracks are drawn solid
where the drone is in view and dashed where trees hide it (under the canopy).
Top right: a side cut through the real lidar points along the drone's route
under the trees. Bottom right: the whole site from above with the obstacles.
Ends with a planner test (labelled as such): a route under the canopy that
has to pass between houses and trees.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swarmsearch import flight3d as f3  # noqa: E402

W, H = 1920, 1080
LW = 1240
FPS = 24
COLS = [(255, 196, 0), (0, 200, 255), (255, 90, 200), (140, 255, 90)]
MODE_TXT = {"cruise": "cruising over the trees", "descend": "descending through a gap in the canopy",
            "under": "flying under the canopy, between trunks", "look": "looking at the alert spot (thermal, 3 m up)",
            "climb": "climbing back out through the gap", "hover": "hovering and looking"}


def font(n, bold=False):
    f = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    return ImageFont.truetype(f, n) if os.path.exists(f) else ImageFont.load_default()


# ------------------------------------------------------------------ scene
def ellipsoid(c, r, rz, nu=8, nv=6):
    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(-np.pi / 2, np.pi / 2, nv)
    V = [(c[0] + r * math.cos(v) * math.cos(u), c[1] + r * math.cos(v) * math.sin(u), c[2] + rz * math.sin(v))
         for v in vs for u in us]
    F = []
    for i in range(nv - 1):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            cc, d = (i + 1) * nu + j, (i + 1) * nu + (j + 1) % nu
            F += [a, b, cc, b, d, cc]
    return V, F


def prism(poly, z0, z1):
    """Extruded polygon (walls and a fan roof)."""
    n = len(poly)
    V = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [i, j, n + i, j, n + j, n + i]
    for i in range(1, n - 1):
        F += [n, n + i, n + i + 1]
    return V, F


class Mesh:
    def __init__(self):
        self.V, self.F = [], []

    def add(self, V, F):
        o = len(self.V)
        self.V += V
        self.F += [o + i for i in F]


def push(p, mesh, rgba, chunk=30000):
    """Add a mesh to PyBullet in chunks (it limits vertices per shape)."""
    V, F = np.asarray(mesh.V, float), np.asarray(mesh.F, int).reshape(-1, 3)
    if not len(F):
        return
    start = 0
    while start < len(F):
        tri = F[start:start + chunk // 3]
        used, inv = np.unique(tri, return_inverse=True)
        vs = p.createVisualShape(p.GEOM_MESH, vertices=V[used].tolist(), indices=inv.ravel().astype(int).tolist(),
                                 rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
        start += chunk // 3


def build_scene(s3, buildings, photo, z0, area, step=6):
    import pybullet as p
    p.connect(p.DIRECT)
    # terrain with the photo
    e0, n0, half = s3.east0, s3.north0, area / 2
    i0, j0 = s3.ij(e0 - half, n0 + half)
    n = int(area)
    dem = s3.dtm[i0:i0 + n:step, j0:j0 + n:step] - z0
    m = dem.shape[0]
    jj, ii = np.meshgrid(np.arange(m), np.arange(m))
    X = e0 - half + jj * step
    Y = n0 + half - ii * step
    verts = np.stack([X, Y, dem], -1).reshape(-1, 3)
    uv = np.stack([jj / (m - 1), 1 - ii / (m - 1)], -1).reshape(-1, 2)
    gy, gx = np.gradient(dem, step)
    nrm = np.stack([-gx, gy, np.ones_like(dem)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    idx = []
    for i in range(m - 1):
        for j in range(m - 1):
            a, b, c, d = i * m + j, i * m + j + 1, (i + 1) * m + j, (i + 1) * m + j + 1
            idx += [a, c, b, b, c, d]
    vs = p.createVisualShape(p.GEOM_MESH, vertices=verts.tolist(), indices=idx, uvs=uv.tolist(),
                             normals=nrm.reshape(-1, 3).tolist())
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
    tex = Path("/tmp") / f"flight3d_tex_{os.getpid()}.png"
    photo.save(tex)
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(str(tex)), rgbaColor=[1, 1, 1, 1])
    # trees
    crowns = [Mesh() for _ in range(3)]
    trunks = Mesh()
    rng = np.random.default_rng(0)
    for x, y, h, r, b in s3.trees:
        if abs(x - e0) > half or abs(y - n0) > half:
            continue
        g = s3.ground(x, y) - z0
        b = float(np.clip(b, 2.0, h - 1.5))
        V, F = ellipsoid((x, y, g + (b + h) / 2), r, (h - b) / 2)
        crowns[int(rng.integers(3))].add(V, F)
        V, F = prism([(x + 0.3 * math.cos(a), y + 0.3 * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)],
                     g, g + b + 0.5)
        trunks.add(V, F)
    for mesh, col in zip(crowns, ([0.22, 0.42, 0.18, 1], [0.28, 0.48, 0.2, 1], [0.18, 0.36, 0.16, 1])):
        push(p, mesh, col)
    push(p, trunks, [0.36, 0.27, 0.18, 1])
    # buildings
    walls, roofs = Mesh(), Mesh()
    for poly, hgt in buildings:
        cx, cy = np.mean(poly, 0)
        if abs(cx - e0) > half or abs(cy - n0) > half:
            continue
        g = min(s3.ground(x, y) for x, y in poly) - z0
        V, F = prism([tuple(q) for q in poly[:-1]] if np.allclose(poly[0], poly[-1]) else [tuple(q) for q in poly],
                     g, g + hgt)
        walls.add(V, F)
    push(p, walls, [0.78, 0.74, 0.68, 1])
    return p


def camera_matrices(p, eye, target, w, h, fov=45):
    V = p.computeViewMatrix(eye, target, [0, 0, 1])
    P = p.computeProjectionMatrixFOV(fov=fov, aspect=w / h, nearVal=2.0, farVal=6000.0)
    return V, P


def render(p, eye, target, w=LW, h=H - 70, fov=45):
    V, P = camera_matrices(p, eye, target, w, h, fov)
    _, _, rgb, depth, _ = p.getCameraImage(w, h, V, P, renderer=p.ER_TINY_RENDERER, lightDirection=[-0.5, -0.4, 1.0],
                                           shadow=0, lightAmbientCoeff=0.6, lightDiffuseCoeff=0.5, lightSpecularCoeff=0.0)
    rgb = np.reshape(rgb, (h, w, 4))[..., :3].astype(np.uint8).copy()
    depth = np.reshape(depth, (h, w))
    sky = depth >= 0.99999
    t = (np.arange(h) / h)[:, None, None]
    grad = (np.array([120, 152, 192]) * (1 - t) + np.array([205, 216, 228]) * t).astype(np.uint8)
    rgb[sky] = np.broadcast_to(grad, (h, w, 3))[sky]
    M = np.array(P).reshape(4, 4).T @ np.array(V).reshape(4, 4).T
    return rgb, depth, M


def project(M, pts, w, h):
    pts = np.atleast_2d(np.asarray(pts, float))
    c = np.c_[pts, np.ones(len(pts))] @ M.T
    ndc = c[:, :3] / c[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h, ndc[:, 2] * 0.5 + 0.5], c[:, 3] > 0


# ------------------------------------------------------------------ flight data
def sample(traj, t):
    """Position and mode at time t (linear between points)."""
    T = traj["t"]
    if t <= T[0]:
        return traj["p"][0], traj["m"][0]
    if t >= T[-1]:
        return traj["p"][-1], traj["m"][-1]
    k = int(np.searchsorted(T, t)) - 1
    u = (t - T[k]) / max(T[k + 1] - T[k], 1e-9)
    return traj["p"][k] * (1 - u) + traj["p"][k + 1] * u, traj["m"][k + 1]


def load_traj(lst, z0):
    a = np.array([q[:4] for q in lst], float)
    o = np.argsort(a[:, 0], kind="stable")
    return dict(t=a[o, 0], p=np.c_[a[o, 1:3], a[o, 3] - z0], m=[lst[i][4] for i in o])


# ------------------------------------------------------------------ panels
def section_panel(pts_xyz, pts_cls, path_xyz, s3, z0, w, h, title):
    """Side cut along a polyline through the lidar points within 4 m of it."""
    P = np.asarray(path_xyz, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    L = seg[-1]
    # project points onto the path
    xy = pts_xyz[:, :2]
    best_d = np.full(len(xy), np.inf)
    best_s = np.zeros(len(xy))
    for k in range(len(P) - 1):
        a, b = P[k, :2], P[k + 1, :2]
        ab = b - a
        ll = max(np.dot(ab, ab), 1e-9)
        u = np.clip(((xy - a) @ ab) / ll, 0, 1)
        q = a + u[:, None] * ab
        d = np.hypot(*(xy - q).T)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = seg[k] + u[better] * math.sqrt(ll)
    sel = best_d < 4.0
    S, Z, C = best_s[sel], pts_xyz[sel, 2] - z0, pts_cls[sel]
    zmin = float(min(Z.min(), P[:, 2].min()) - 2) if len(Z) else 0.0
    zmax = float(max(Z.max() if len(Z) else 0, P[:, 2].max()) + 4)
    pad_l, pad_b, pad_t = 50, 34, 34
    sx = (w - pad_l - 14) / max(L, 1.0)
    sz = (h - pad_b - pad_t) / max(zmax - zmin, 1.0)
    img = Image.new("RGB", (w, h), (18, 20, 24))
    d = ImageDraw.Draw(img)
    col = np.where(C[:, None] == 2, np.array([150, 120, 80]), np.array([70, 170, 70]))
    u = (pad_l + S * sx).astype(int)
    v = (h - pad_b - (Z - zmin) * sz).astype(int)
    arr = np.asarray(img).copy()
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    for du, dv in ((0, 0), (1, 0), (0, 1)):
        uu, vv = np.clip(u[ok] + du, 0, w - 1), np.clip(v[ok] + dv, 0, h - 1)
        arr[vv, uu] = col[ok]
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.text((10, 6), title, font=font(16, True), fill=(235, 235, 235))
    for zz in range(int(math.ceil(zmin / 10) * 10), int(zmax), 10):
        vv = h - pad_b - (zz - zmin) * sz
        d.line([(pad_l - 4, vv), (pad_l, vv)], fill=(160, 160, 160))
        d.text((6, vv - 7), f"{zz - (P[:, 2].min() - 3.0):.0f} m", font=font(11), fill=(150, 150, 150))
    d.text((8, h - pad_b + 6), f"0 m", font=font(12), fill=(170, 170, 170))
    d.text((w - 70, h - pad_b + 6), f"{L:.0f} m", font=font(12), fill=(170, 170, 170))
    d.text((pad_l + 6, h - 20), "lidar points within 4 m of the route:  green = trees   brown = ground",
           font=font(12), fill=(170, 170, 170))
    tf = dict(sx=sx, sz=sz, zmin=zmin, pad_l=pad_l, pad_b=pad_b, h=h, P=P, seg=seg)
    return img, tf


def section_xy(tf, x, y, z):
    P, seg = tf["P"], tf["seg"]
    best = (np.inf, 0.0)
    for k in range(len(P) - 1):
        a, b = P[k, :2], P[k + 1, :2]
        ab = b - a
        ll = max(np.dot(ab, ab), 1e-9)
        u = float(np.clip(np.dot(np.array([x, y]) - a, ab) / ll, 0, 1))
        q = a + u * ab
        dd = math.hypot(x - q[0], y - q[1])
        if dd < best[0]:
            best = (dd, seg[k] + u * math.sqrt(ll))
    return tf["pad_l"] + best[1] * tf["sx"], tf["h"] - tf["pad_b"] - (z - tf["zmin"]) * tf["sz"], best[0]


def map_panel(photo_full, site_size, s3, buildings, w):
    img = photo_full.resize((w, w)).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    k = w / site_size

    def px(x, y):
        return (x + site_size / 2) * k, (site_size / 2 - y) * k

    for poly, _ in buildings:
        d.polygon([px(x, y) for x, y in poly], outline=(255, 255, 255, 230), fill=(255, 255, 255, 90))
    e, n, half = s3.east0, s3.north0, s3.n / 2
    d.rectangle([px(e - half, n + half), px(e + half, n - half)], outline=(255, 230, 120, 220), width=2)
    return img, px


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin_west")
    ap.add_argument("--seed", type=int, default=103)
    ap.add_argument("--fine-from", default="sites/catoctin/fine1.npz")
    ap.add_argument("--shift", type=float, nargs=2, default=(2000.0, 0.0))
    ap.add_argument("--speed", type=float, default=4.0, help="video seconds per mission second, inverted")
    ap.add_argument("--out", default="results/figures/flight3d_demo.mp4")
    ap.add_argument("--still", default="")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--start-t", type=float, default=0.0, help="start at this mission time (s)")
    a = ap.parse_args()
    site = Path(a.site)
    D = json.load(open(site / f"flight_seed{a.seed}.json"))
    info = D["info"]
    s3 = f3.Site3D.load(a.fine_from, shift=tuple(a.shift))
    meta = json.load(open(site / "meta.json"))
    z0 = float(s3.dtm.min())
    area = float(s3.n)
    # buildings with lidar roof heights
    polys = json.load(open(site / "buildings.json"))
    buildings = []
    for pg in polys:
        pg = np.asarray(pg, float)
        cx, cy = pg.mean(0)
        hgt = 6.0
        if s3.inside(cx, cy):
            i, j = s3.ij(cx, cy)
            win = s3.bldg_h[max(0, i - 4):i + 5, max(0, j - 4):j + 5]
            if (win > 0).any():
                hgt = float(np.clip(win[win > 0].max(), 3.0, 25.0))
        buildings.append((pg, hgt))
    # photo of the detailed area
    photo_full = Image.open(site / "photo.jpg")
    pxm = photo_full.size[0] / meta["size_m"]
    c0 = photo_full.size[0] / 2
    box = (c0 + (s3.east0 - area / 2) * pxm, c0 - (s3.north0 + area / 2) * pxm,
           c0 + (s3.east0 + area / 2) * pxm, c0 - (s3.north0 - area / 2) * pxm)
    photo = photo_full.crop(tuple(int(v) for v in box)).resize((4096, 4096), Image.LANCZOS)
    t0 = time.time()
    p = build_scene(s3, buildings, photo, z0, area)
    print(f"scene built in {time.time() - t0:.0f}s: {len(s3.trees)} trees, {len(buildings)} buildings", flush=True)
    trajs = {d: load_traj(tr, z0) for d, tr in D["trajs"].items()}
    names = sorted(d for d in trajs if d != "test")
    test = info.get("test") if "test" in trajs else None
    ev = next((e for e in info["events"] if e["kind"] == "under"), None)
    T_end = max(tr["t"][-1] for tr in trajs.values()) + 3.0
    pts = np.load(Path(a.fine_from).with_name("fine_points.npz"))
    pxyz = pts["xyz"].astype(float)
    pxyz[:, 0] += a.shift[0]
    pxyz[:, 1] += a.shift[1]
    pcls = pts["cls"]
    # cameras: wide view, then close to the under-canopy check
    wide_t = (np.array(info["base"]), )
    bx, by = info["base"]
    shots = []
    eye_w = [bx - 520, by - 620, s3.ground(bx, by) - z0 + 380]
    tgt_w = [bx + 40, by + 40, s3.ground(bx, by) - z0]
    shots.append(("wide", eye_w, tgt_w, 45))
    if ev:
        gx, gy = ev["gap"]
        lx, ly, lz = ev["look"]
        mx, my = (gx + lx) / 2, (gy + ly) / 2
        gz = s3.ground(mx, my) - z0
        shots.append(("close", [mx - 70, my - 95, gz + 55], [mx, my, gz + 4], 50))
    if test:
        gx, gy = test["gap"]
        lx, ly, _ = test["look"]
        mx, my = (gx + lx) / 2, (gy + ly) / 2
        gz = s3.ground(mx, my) - z0
        shots.append(("test", [mx - 75, my - 100, gz + 60], [mx, my, gz + 4], 55))
    renders = {}
    for name, eye, tgt, fov in shots:
        renders[name] = render(p, eye, tgt, fov=fov)
        print(f"rendered {name} view", flush=True)
    # side cut along the under-canopy leg of the check
    def make_section(tr, title, margin=20.0):
        legs = [i for i, m in enumerate(tr["m"]) if m in ("under", "look")]
        leg = tr["p"][legs[0]:legs[-1] + 1]
        far = int(np.argmax([np.hypot(*(q[:2] - leg[0][:2])) for q in leg]))
        out_leg = leg[:far + 1]
        keep = [out_leg[0]]
        for q in out_leg[1:]:
            if np.hypot(*(q[:2] - keep[-1][:2])) > 2.0:
                keep.append(q)
        if len(keep) < 2:
            keep = [out_leg[0], out_leg[-1]]
        pp = np.array([[q[0], q[1], q[2] + z0] for q in keep])
        # extend the cut past both ends to show the gap and what lies beyond
        d0 = pp[1, :2] - pp[0, :2]
        d1 = pp[-1, :2] - pp[-2, :2]
        d0 /= max(np.linalg.norm(d0), 1e-9)
        d1 /= max(np.linalg.norm(d1), 1e-9)
        pp = np.vstack([[*(pp[0, :2] - margin * d0), pp[0, 2]], pp, [*(pp[-1, :2] + margin * d1), pp[-1, 2]]])
        near = (np.abs(pxyz[:, 0] - pp[:, 0].mean()) < 250) & (np.abs(pxyz[:, 1] - pp[:, 1].mean()) < 250)
        return section_panel(pxyz[near], pcls[near], pp, s3, 0.0, W - LW, 420, title)

    sec, tf = (make_section(trajs[ev["drone"]], "Side cut through the real lidar along D1's route under the trees")
               if ev else (None, None))
    sec_t, tf_t = (make_section(trajs["test"], "Side cut along the planner test route (passes a house)")
                   if test else (None, None))
    mp, mpx = map_panel(photo_full, meta["size_m"], s3, buildings, 500)
    # frames
    n = int(T_end * FPS / a.speed)
    if a.frames:
        n = min(n, a.frames)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                           "-preset", "medium", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    under_t = (ev["t"], ev["t"] + ev["slot_s"]) if ev else (1e9, 1e9)
    found_t = info["found_step"] * info["step_s"] if info.get("found_step") is not None else None
    n0 = int(a.start_t * FPS / a.speed)
    for fi in range(n0, n0 + n if a.frames else n):
        t = fi * a.speed / FPS
        in_test = test is not None and t >= test["t"] - 2
        shot = "test" if in_test else ("close" if ev and under_t[0] - 4 <= t <= under_t[1] + 4 else "wide")
        rgb, depth, M = renders[shot]
        hh, ww = depth.shape
        view = Image.fromarray(rgb).convert("RGBA")
        ov = Image.new("RGBA", view.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        # the person (true position)
        for (px_, py_) in info["people"]:
            g = s3.ground(px_, py_) - z0
            q1, ok = project(M, [[px_, py_, g + 0.5]], ww, hh)
            u, v, zz = q1[0]
            if ok[0]:
                fnd = found_t is not None and t >= found_t
                c = (60, 230, 90) if fnd else (235, 50, 50)
                dr.ellipse([u - 6, v - 6, u + 6, v + 6], fill=c + (255,), outline=(0, 0, 0, 255))
                lab = "hunter FOUND" if fnd else "missing hunter"
                dr.text((u + 9, v - 8), lab, font=font(15, True), fill=(255, 255, 255, 255), stroke_width=2,
                        stroke_fill=(0, 0, 0, 255))
        if ev:
            gx, gy = ev["gap"]
            g = s3.ground(gx, gy) - z0
            q1, ok = project(M, [[gx, gy, g + 0.3]], ww, hh)
            u, v, _ = q1[0]
            if ok[0] and t >= under_t[0] - 4:
                dr.ellipse([u - 10, v - 10, u + 10, v + 10], outline=(255, 255, 255, 230), width=2)
                dr.text((u + 12, v - 6), "gap in the canopy", font=font(14), fill=(255, 255, 255, 255),
                        stroke_width=2, stroke_fill=(0, 0, 0, 255))
        draw = [(k, d, COLS[k % len(COLS)], f"D{k + 1}") for k, d in enumerate(names)]
        if in_test:
            draw = [(9, "test", (255, 255, 255), "test drone")]
        for k, d, col, lab in draw:
            tr = trajs[d]
            m = tr["t"] <= t
            hist = tr["p"][m][-400:]
            pos, mode = sample(tr, t)
            P = np.vstack([hist, pos]) if len(hist) else pos[None]
            q, okk = project(M, P, ww, hh)
            for i in range(1, len(q)):
                if not (okk[i] and okk[i - 1]):
                    continue
                (u0, v0, z0p), (u1, v1, z1p) = q[i - 1], q[i]
                if not (-50 < u1 < ww + 50 and -50 < v1 < hh + 50):
                    continue
                iu, iv = int(np.clip(u1, 0, ww - 1)), int(np.clip(v1, 0, hh - 1))
                hidden = z1p > depth[iv, iu] + 2e-5
                if hidden and i % 2:
                    continue
                dr.line([(u0, v0), (u1, v1)], fill=col + ((130,) if hidden else (240,)), width=2 if hidden else 3)
            (u, v, zz) = q[-1]
            if okk[-1] and 0 <= u < ww and 0 <= v < hh:
                hidden = zz > depth[int(v), int(u)] + 2e-5
                r = 8
                dr.ellipse([u - r, v - r, u + r, v + r], fill=col + (120 if hidden else 255,), outline=(0, 0, 0, 255),
                           width=2)
                dr.text((u + 11, v - 20), lab + (" (under the trees)" if hidden and mode in ("under", "look") else ""),
                        font=font(17, True), fill=col + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        view = Image.alpha_composite(view, ov).convert("RGB")
        canvas = Image.new("RGB", (W, H), (14, 15, 18))
        canvas.paste(view, (0, 70))
        dc = ImageDraw.Draw(canvas)
        head = {"wide": f"Mission time {t:4.0f} s  ·  lidar-mapped core, 1.2 km",
                "close": f"Mission time {t:4.0f} s  ·  D1 checks an alert under the trees",
                "test": "Planner test (not from the mission): under the trees, past a house"}[shot]
        dc.text((16, 8), head, font=font(24, True), fill=(255, 255, 255) if shot != "test" else (255, 230, 150))
        dc.text((16, 42), f"{info['n_drones']} drones, {info['policy']}  ·  every tree and building from USGS lidar and "
                "Microsoft footprints  ·  drone markers not to scale", font=font(15), fill=(200, 200, 200))
        # status lines
        y0 = 80
        for k, d in (enumerate(names) if not in_test else [(0, "test")]):
            pos, mode = sample(trajs[d], t)
            agl = pos[2] + z0 - (s3.ground(pos[0], pos[1]) if s3.inside(pos[0], pos[1]) else pos[2] + z0 - 40)
            who = "test drone" if d == "test" else f"D{k + 1}"
            txt = f"{who}: {MODE_TXT.get(mode, mode)}, {agl:.0f} m above ground"
            dc.rectangle([10, y0 + 26 * k, 20 + dc.textlength(txt, font=font(16)), y0 + 26 * k + 22], fill=(0, 0, 0))
            dc.text((15, y0 + 2 + 26 * k), txt, font=font(16), fill=COLS[k % len(COLS)])
        # right: side cut
        cur = (sec_t, tf_t, "test", test["t"] if test else 0, 1e9, (255, 255, 255), "test") if in_test and sec_t is not None \
            else ((sec, tf, ev["drone"], under_t[0], under_t[1], COLS[0], "D1") if sec is not None else None)
        if cur is not None:
            secimg, tf, dname, ta, tb, ccol, clab = cur
            sc = secimg.copy()
            ds = ImageDraw.Draw(sc)
            tr = trajs[dname]
            m = (tr["t"] <= t) & (tr["t"] >= ta)
            for q in tr["p"][m][::2]:
                u, v, dd = section_xy(tf, q[0], q[1], q[2] + z0)
                if dd < 6:
                    ds.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=ccol)
            pos, mode = sample(tr, t)
            u, v, dd = section_xy(tf, pos[0], pos[1], pos[2] + z0)
            if dd < 6 and ta <= t <= tb:
                ds.ellipse([u - 6, v - 6, u + 6, v + 6], fill=ccol, outline=(0, 0, 0))
                ds.text((u + 8, v - 18), clab, font=font(14, True), fill=ccol)
            canvas.paste(sc, (LW, 0))
        # right: map
        mm = mp.copy()
        dm = ImageDraw.Draw(mm)
        for k, d in enumerate(names):
            tr = trajs[d]
            m = tr["t"] <= t
            pts_ = [mpx(q[0], q[1]) for q in tr["p"][m][::3]]
            if len(pts_) > 1:
                dm.line(pts_, fill=COLS[k % len(COLS)], width=2)
            pos, _ = sample(tr, t)
            u, v = mpx(pos[0], pos[1])
            dm.ellipse([u - 5, v - 5, u + 5, v + 5], fill=COLS[k % len(COLS)], outline=(0, 0, 0))
        canvas.paste(mm, (LW + 10, 440))
        dc.text((LW + 10 + mm.size[0] + 10, 450), "Whole site,\n3.2 km\n\nwhite:\nbuildings\n\nyellow box:\nlidar core",
                font=font(14), fill=(200, 200, 200))
        dc.rectangle([0, H - 30, W, H], fill=(0, 0, 0))
        dc.text((16, H - 26), "Real: terrain, trees, buildings (USGS 3DEP lidar 2021, leaf-off; Microsoft footprints), photo "
                "(USDA NAIP).  Simulated: drones, the hunter, detections.", font=font(15), fill=(200, 220, 200))
        if fi == n0 and a.still:
            canvas.save(a.still)
        ff.stdin.write(np.asarray(canvas).tobytes())
        if fi % 200 == 0:
            print(f"frame {fi}/{n} {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {n} frames ({n / FPS:.0f} s)")


if __name__ == "__main__":
    main()
