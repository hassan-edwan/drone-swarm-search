"""Render a recorded mission on a real site: real terrain and aerial photo,
simulated drones and person.

    ~/venv3d/bin/python scripts/real_site/render_demo.py --site sites/catoctin --seed 4 \\
        --out results/figures/real_site_demo.mp4

Left: a 3D view of the site (USGS elevation with the USDA NAIP aerial photo
draped on it, drawn by PyBullet's software renderer), orbiting slowly, with the
drones (markers, not to scale), their recent tracks, the last-seen point, the
command post, and the missing person's true position (which the drones do not
know). Right: each drone's downward camera, cut from the real aerial photo at
its true height and field of view (84 deg), with the simulated person drawn in
at true size when inside the image; and the team's probability map.

What is real and what is simulated is written on every frame. The search
policy decides from the simulator's sensor model, not from these pictures.
"""

import argparse
import io
import json
import math
import os
import pickle
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
LW = 1280  # 3D view width
CAM = 320  # drone camera tile
HALF_FOV = math.radians(42.0)
SUB = 2  # frames per simulator step
FPS = 24
COLS = [(255, 196, 0), (0, 200, 255), (255, 90, 200), (140, 255, 90), (255, 140, 40), (180, 140, 255)]
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
M_PER_DEG_LAT = 110_540.0
BLOCK_M = 400.0
BLOCK_PX = 1024
REAL = "Real: terrain (USGS 3DEP) and aerial photo (USDA NAIP), Catoctin Mountain Park, Maryland"
SIM = "Simulated: drones, missing hiker, detections. Drones decide from the simulator's sensor model, not these images."


def font(n, bold=False):
    for f in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",):
        if os.path.exists(f):
            return ImageFont.truetype(f, n)
    return ImageFont.load_default()


# ----------------------------------------------------------------- geometry
class Site:
    def __init__(self, d: Path, mission: dict):
        self.d = d
        self.meta = json.load(open(d / "meta.json"))
        self.dem = np.load(d / "dem.npy").astype(float)
        self.n = self.dem.shape[0]
        self.size = self.meta["size_m"]
        self.cell = mission["cell_m"]
        self.g = mission["g"]
        self.z0 = float(self.dem.min())

    def cell_to_xy(self, c):
        """Simulator cell coordinates (x, y; row 0 north; centres at integers) to metres east/north of centre."""
        return (c[0] + 0.5 - self.g / 2) * self.cell, (self.g / 2 - c[1] - 0.5) * self.cell

    def elev(self, x, y):
        j = (x / self.size + 0.5) * (self.n - 1)
        i = (0.5 - y / self.size) * (self.n - 1)
        j, i = np.clip(j, 0, self.n - 1.001), np.clip(i, 0, self.n - 1.001)
        j0, i0 = int(j), int(i)
        fx, fy = j - j0, i - i0
        e = self.dem
        return float((e[i0, j0] * (1 - fx) + e[i0, j0 + 1] * fx) * (1 - fy)
                     + (e[i0 + 1, j0] * (1 - fx) + e[i0 + 1, j0 + 1] * fx) * fy) - self.z0

    def lonlat(self, x, y):
        lat0, lon0 = self.meta["lat"], self.meta["lon"]
        return lon0 + x / (111_320.0 * math.cos(math.radians(lat0))), lat0 + y / M_PER_DEG_LAT


# ----------------------------------------------------------------- the photo, close up
def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def block_path(site: Site, bi, bj) -> Path:
    return site.d / "blocks" / f"{bi}_{bj}.jpg"


def fetch_block(args):
    site_dir, meta, bi, bj = args
    p = Path(site_dir) / "blocks" / f"{bi}_{bj}.jpg"
    if p.exists():
        return
    lat0, lon0 = meta["lat"], meta["lon"]
    x0, y0 = bj * BLOCK_M, bi * BLOCK_M
    k = 111_320.0 * math.cos(math.radians(lat0))
    b = (lon0 + x0 / k, lat0 + y0 / M_PER_DEG_LAT, lon0 + (x0 + BLOCK_M) / k, lat0 + (y0 + BLOCK_M) / M_PER_DEG_LAT)
    h = max(1, round(BLOCK_PX * (b[3] - b[1]) / (b[2] - b[0])))
    url = (f"{NAIP}?bbox={b[0]},{b[1]},{b[2]},{b[3]}&bboxSR=4326&imageSR=4326&size={BLOCK_PX},{h}"
           f"&format=jpg&compressionQuality=90&f=image")
    im = Image.open(io.BytesIO(get(url))).convert("RGB").resize((BLOCK_PX, BLOCK_PX), Image.BICUBIC)
    p.parent.mkdir(exist_ok=True)
    tmp = p.with_suffix(".tmp.jpg")
    im.save(tmp, quality=92)
    os.replace(tmp, p)


def blocks_for(x, y, half):
    """Blocks (row index from the south, column from the west, in metres from the centre) under a square."""
    out = set()
    for bx in range(math.floor((x - half) / BLOCK_M), math.floor((x + half) / BLOCK_M) + 1):
        for by in range(math.floor((y - half) / BLOCK_M), math.floor((y + half) / BLOCK_M) + 1):
            out.add((by, bx))
    return out


_cache: dict = {}


def crop(site: Site, x, y, half, px=CAM) -> Image.Image:
    """North-up photo of the square of half-width `half` metres centred at (x, y)."""
    bs = blocks_for(x, y, half)
    by0, by1 = min(b[0] for b in bs), max(b[0] for b in bs)
    bx0, bx1 = min(b[1] for b in bs), max(b[1] for b in bs)
    mos = Image.new("RGB", ((bx1 - bx0 + 1) * BLOCK_PX, (by1 - by0 + 1) * BLOCK_PX))
    for by, bx in bs:
        key = (by, bx)
        if key not in _cache:
            if len(_cache) > 64:
                _cache.clear()
            _cache[key] = Image.open(block_path(site, by, bx)).copy()
        mos.paste(_cache[key], ((bx - bx0) * BLOCK_PX, (by1 - by) * BLOCK_PX))
    s = BLOCK_PX / BLOCK_M  # px per metre
    left = (x - half - bx0 * BLOCK_M) * s
    top = ((by1 + 1) * BLOCK_M - (y + half)) * s
    return mos.resize((px, px), Image.BICUBIC, box=(left, top, left + 2 * half * s, top + 2 * half * s))


# ----------------------------------------------------------------- 3D view
def build_scene(site: Site, tex_px=4096, step=2):
    import pybullet as p
    p.connect(p.DIRECT)
    e = site.dem[::step, ::step] - site.z0
    n = e.shape[0]
    js, is_ = np.meshgrid(np.arange(n), np.arange(n))
    X = (js / (n - 1) - 0.5) * site.size
    Y = (0.5 - is_ / (n - 1)) * site.size
    verts = np.stack([X, Y, e], -1).reshape(-1, 3)
    gy, gx = np.gradient(e, site.size / (n - 1))
    nrm = np.stack([-gx, gy, np.ones_like(e)], -1)  # rows go south, so d/dy(north) = -d/drow
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    uv = np.stack([js / (n - 1), 1 - is_ / (n - 1)], -1).reshape(-1, 2)
    idx = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            idx += [a, c, b, b, c, d]
    vs = p.createVisualShape(p.GEOM_MESH, vertices=verts.tolist(), indices=idx, uvs=uv.tolist(),
                             normals=nrm.reshape(-1, 3).tolist(), meshScale=[1, 1, 1])
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
    tex = site.d / f"texture_{tex_px}.png"
    if not tex.exists():
        Image.open(site.d / "photo.jpg").resize((tex_px, tex_px), Image.LANCZOS).save(tex)
    t = p.loadTexture(str(tex))
    p.changeVisualShape(body, -1, textureUniqueId=t, rgbaColor=[1, 1, 1, 1])
    return p


def camera(frame_i, n_frames, centre, site):
    ang = math.radians(200 + 70 * frame_i / max(1, n_frames - 1))  # from the south-south-west, turning
    dist, pitch = 6200.0, math.radians(34)
    tx, ty = centre
    tz = site.elev(tx, ty)
    eye = [tx + dist * math.cos(pitch) * math.sin(ang), ty + dist * math.cos(pitch) * math.cos(ang),
           tz + dist * math.sin(pitch)]
    return eye, [tx, ty, tz]


def render_3d(p, eye, target, w=LW, h=H):
    V = p.computeViewMatrix(eye, target, [0, 0, 1])
    P = p.computeProjectionMatrixFOV(fov=40, aspect=w / h, nearVal=50.0, farVal=40000.0)
    _, _, rgb, depth, _ = p.getCameraImage(w, h, V, P, renderer=p.ER_TINY_RENDERER,
                                           lightDirection=[-0.4, -0.5, 1.0], shadow=0,
                                           lightAmbientCoeff=0.55, lightDiffuseCoeff=0.55, lightSpecularCoeff=0.0)
    rgb = np.reshape(rgb, (h, w, 4))[..., :3].astype(np.uint8).copy()
    depth = np.reshape(depth, (h, w))
    sky = depth >= 0.99999
    if sky.any():  # hazy sky gradient behind the hills
        t = (np.arange(h) / h)[:, None, None]
        grad = (np.array([118, 150, 190]) * (1 - t) + np.array([200, 214, 226]) * t).astype(np.uint8)
        rgb[sky] = np.broadcast_to(grad, (h, w, 3))[sky]
    M = np.array(P).reshape(4, 4).T @ np.array(V).reshape(4, 4).T
    return rgb, depth, M


def project(M, pts, w=LW, h=H):
    pts = np.atleast_2d(pts)
    c = np.c_[pts, np.ones(len(pts))] @ M.T
    ndc = c[:, :3] / c[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * w, (1 - ndc[:, 1]) / 2 * h, ndc[:, 2] * 0.5 + 0.5], c[:, 3] > 0


# ----------------------------------------------------------------- frames
def lerp(a, b, f):
    return np.asarray(a, float) * (1 - f) + np.asarray(b, float) * f


def hhmm(hour):
    h = int(hour) % 24
    return f"{h:02d}:{int((hour - int(hour)) * 60):02d}"


def person_offset(seed):
    return np.random.default_rng(seed + 7).uniform(-0.1, 0.1, 2)


def draw_person(im, cx, cy, m_per_px, alpha):
    """A person lying or sitting, seen from above: about 1.7 m by 0.5 m, red jacket."""
    L = 1.7 / m_per_px
    Wd = 0.55 / m_per_px
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    a = int(255 * alpha)
    sh = 0.25 / m_per_px  # soft shadow, sun from the south-east in the morning photo
    d.ellipse([cx - Wd / 2 - sh, cy - L / 2 + sh, cx + Wd / 2 - sh, cy + L / 2 + sh], fill=(0, 0, 0, int(a * 0.35)))
    arm = 0.6 / m_per_px
    d.line([(cx - Wd / 2, cy - L * 0.3), (cx - Wd / 2 - arm * 0.5, cy + L * 0.05)], fill=(170, 25, 25, a),
           width=max(1, int(0.13 / m_per_px)))
    d.line([(cx + Wd / 2, cy - L * 0.3), (cx + Wd / 2 + arm * 0.4, cy - L * 0.45)], fill=(170, 25, 25, a),
           width=max(1, int(0.13 / m_per_px)))
    d.ellipse([cx - Wd / 2, cy - L / 2, cx + Wd / 2, cy + L * 0.05], fill=(200, 30, 30, a))  # jacket
    d.ellipse([cx - Wd / 2.4, cy + L * 0.05, cx + Wd / 2.4, cy + L / 2], fill=(40, 50, 80, a))  # legs
    d.ellipse([cx - Wd / 3, cy - L / 2 - Wd / 1.6, cx + Wd / 3, cy - L / 2 + Wd / 10], fill=(220, 180, 150, a))
    im.alpha_composite(lay) if im.mode == "RGBA" else im.paste(Image.alpha_composite(im.convert("RGBA"), lay))
    return im


_W: dict = {}


def _worker_init(site_dir, seed):
    _W["mission"] = pickle.load(open(Path(site_dir) / f"mission_seed{seed}.pkl", "rb"))
    _W["site"] = Site(Path(site_dir), _W["mission"])


def make_frame(job):
    """One output frame from the 3D render and the mission state."""
    (fi, si, f, rgb, depth, M) = job
    mission, site = _W["mission"], _W["site"]
    F = mission["frames"]
    a0, a1 = F[si], F[min(si + 1, len(F) - 1)]
    canvas = Image.new("RGB", (W, H), (16, 18, 22))
    view = Image.fromarray(rgb).convert("RGBA")
    ov = Image.new("RGBA", view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fs, fb, fl = font(18), font(20, True), font(15)

    def ground(cell):
        x, y = site.cell_to_xy(cell)
        return np.array([x, y, site.elev(x, y)])

    def pin(cell, colour, label, h=260.0):
        g = ground(cell)
        (u0, v0, _), (u1, v1, _) = project(M, [g, g + [0, 0, h]])[0]
        d.line([(u0, v0), (u1, v1)], fill=colour + (255,), width=3)
        d.ellipse([u1 - 7, v1 - 7, u1 + 7, v1 + 7], fill=colour + (255,), outline=(0, 0, 0, 255), width=2)
        tw = d.textlength(label, font=fs)
        d.rectangle([u1 + 10, v1 - 13, u1 + 16 + tw, v1 + 13], fill=(0, 0, 0, 170))
        d.text((u1 + 13, v1 - 11), label, font=fs, fill=(255, 255, 255, 255))

    # last-seen point, command post
    pin(mission["ipp"], (255, 255, 255), "last seen here")
    if tuple(mission["base"]) != tuple(mission["ipp"]):
        pin(mission["base"], (200, 200, 200), "command post")
    # the person: true position (the drones do not know it)
    found = bool(a1["found"].any()) if f > 0.5 else bool(a0["found"].any())
    pc = np.asarray(a0["people"][0], float) + person_offset(mission["seed"])
    pin(pc, (60, 230, 90) if found else (230, 40, 40),
        "hiker FOUND" if found else "missing hiker (drones do not know)", h=180.0)
    # drones: recent tracks and markers
    names = sorted(a0["drones"])
    for k, a in enumerate(names):
        col = COLS[k % len(COLS)]
        tr = []
        for s in range(max(0, si - 30), si + 1):
            dd = F[s]["drones"].get(a)
            if dd is None:
                continue
            x, y = site.cell_to_xy(dd["true"])
            tr.append([x, y, site.elev(x, y) + dd["alt_m"]])
        dn = a1["drones"].get(a, a0["drones"][a])
        pos = lerp(a0["drones"][a]["true"], dn["true"], f)
        alt = float(lerp(a0["drones"][a]["alt_m"], dn["alt_m"], f))
        x, y = site.cell_to_xy(pos)
        here = np.array([x, y, site.elev(x, y) + alt])
        tr.append(here.tolist())
        if len(tr) > 1:
            q, ok = project(M, np.array(tr))
            pts = [(u, v) for (u, v, _), o in zip(q, ok) if o]
            for i in range(1, len(pts)):
                fade = int(60 + 160 * i / len(pts))
                d.line([pts[i - 1], pts[i]], fill=col + (fade,), width=3)
        (u, v, z) = project(M, here)[0][0]
        (ug, vg, _) = project(M, here - [0, 0, alt])[0][0]
        hidden = 0 <= int(v) < H and 0 <= int(u) < LW and z > depth[int(v), int(u)] + 1e-4
        d.line([(ug, vg), (u, v)], fill=(255, 255, 255, 120), width=1)
        r = 9
        d.ellipse([u - r, v - r, u + r, v + r], fill=col + (140 if hidden else 255,), outline=(0, 0, 0, 255), width=2)
        d.text((u + 12, v - 22), f"D{k + 1}", font=fb, fill=col + (255,), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    view = Image.alpha_composite(view, ov)
    # captions on the 3D view
    cap = Image.new("RGBA", view.size, (0, 0, 0, 0))
    dc = ImageDraw.Draw(cap)
    dc.rectangle([0, 0, LW, 70], fill=(0, 0, 0, 150))
    hour = lerp(a0["hour"], a1["hour"], f)
    mins = (si + f) * mission["step_s"] / 60
    dc.text((16, 8), f"Drone swarm search for a lost hiker  ·  mission time {int(mins):d} min  ·  local time {hhmm(float(hour))}",
            font=font(24, True), fill=(255, 255, 255, 255))
    dc.text((16, 42), f"{mission['n_drones']} drones, {mission['policy']}  ·  area 11.6 x 11.6 km  ·  markers not to scale",
            font=fs, fill=(210, 210, 210, 255))
    dc.rectangle([0, H - 62, LW, H], fill=(0, 0, 0, 160))
    dc.text((16, H - 56), REAL, font=fs, fill=(170, 230, 170, 255))
    dc.text((16, H - 30), SIM, font=fs, fill=(255, 200, 140, 255))
    view = Image.alpha_composite(view, cap)
    canvas.paste(view.convert("RGB"), (0, 0))

    # drone cameras
    dc = ImageDraw.Draw(canvas)
    off = person_offset(mission["seed"])
    for k, a in enumerate(names[:4]):
        col = COLS[k % len(COLS)]
        ox, oy = LW + (k % 2) * CAM, (k // 2) * (CAM + 34)
        dd0 = a0["drones"][a]
        dn = a1["drones"].get(a, dd0)
        pos = lerp(dd0["true"], dn["true"], f)
        alt = float(lerp(dd0["alt_m"], dn["alt_m"], f))
        mode = dd0["mode"]
        if mode == "landed" or alt < 5:
            img = Image.new("RGB", (CAM, CAM), (30, 30, 34))
            ImageDraw.Draw(img).text((20, CAM // 2 - 10), "landed: battery swap", font=fs, fill=(200, 200, 200))
        else:
            half = alt * math.tan(HALF_FOV)
            x, y = site.cell_to_xy(pos)
            img = crop(site, x, y, half).convert("RGBA")
            # the simulated person, at true size, if inside the image
            for i, pp in enumerate(a0["people"]):
                px_, py_ = site.cell_to_xy(np.asarray(pp, float) + off)
                u = (px_ - (x - half)) / (2 * half) * CAM
                v = ((y + half) - py_) / (2 * half) * CAM
                if -10 < u < CAM + 10 and -10 < v < CAM + 10:
                    occ = dd0["occ"][i] if dd0["occ"] and dd0["occ"][i] is not None else 0.0
                    img = draw_person(img, u, v, 2 * half / CAM, max(0.15, 1 - float(occ)))
            img = img.convert("RGB")
        canvas.paste(img, (ox, oy + 34))
        dc.rectangle([ox, oy, ox + CAM - 1, oy + 33], fill=(10, 10, 12))
        dc.text((ox + 8, oy + 7), f"D{k + 1} camera", font=font(16, True), fill=col)
        info = "landed" if mode == "landed" else f"{alt:.0f} m up · {2 * alt * math.tan(HALF_FOV):.0f} m wide"
        dc.text((ox + 120, oy + 8), info, font=fl, fill=(220, 220, 220))
        # alert flashes
        for (t, cell, true) in mission["alerts"].get(a, []):
            if t <= si < t + 6:
                txt = "ALERT: possible person - sent to operator" if not true else "PERSON DETECTED"
                c = (255, 220, 0) if not true else (60, 230, 90)
                dc.rectangle([ox + 6, oy + CAM - 4, ox + CAM - 6, oy + CAM + 26], fill=(0, 0, 0))
                dc.text((ox + 12, oy + CAM), txt, font=font(14, True), fill=c)
        dc.rectangle([ox, oy, ox + CAM - 1, oy + CAM + 33], outline=(60, 60, 66), width=1)
    # team map
    mx, my = LW, 2 * (CAM + 34)
    ms = H - my - 30
    dc.rectangle([mx, my, W, H], fill=(10, 10, 12))
    linked = [a for a in names if a0["drones"][a]["linked"]]
    who = linked[0] if linked else names[0]
    b = np.asarray(a0["drones"][who]["belief"], float)
    b = b / max(b.max(), 1e-12)
    small = Image.open(site.d / "texture_4096.png") if (site.d / "texture_4096.png").exists() else None
    base_im = (small.resize((ms, ms)) if small else Image.new("RGB", (ms, ms))).convert("RGB")
    heat = np.zeros((b.shape[0], b.shape[1], 4), np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = (200 * (1 - b)).astype(np.uint8)
    heat[..., 3] = (np.clip(b * 1.4, 0, 1) * 200).astype(np.uint8)
    hm = Image.fromarray(heat, "RGBA").resize((ms, ms), Image.BILINEAR)
    # the map covers the simulator grid (64 x 181 m); the photo covers the same square
    mp = Image.alpha_composite(base_im.convert("RGBA"), hm)
    dm = ImageDraw.Draw(mp)
    sc = ms / mission["g"]
    for k, a in enumerate(names):
        dd = a0["drones"][a]
        u, v = (np.asarray(dd["true"], float) + 0.5) * sc
        dm.ellipse([u - 5, v - 5, u + 5, v + 5], fill=COLS[k % len(COLS)], outline=(0, 0, 0))
    canvas.paste(mp.convert("RGB"), (mx + 8, my + 28))
    dc.text((mx + 8, my + 4), "Team map: where the hiker probably is", font=font(16, True), fill=(230, 230, 230))
    tx = mx + ms + 18
    lines = ["red = likely", "clear = searched", "or unlikely", "", "dots = drones", "",
             f"{len(linked)}/{len(names)} drones", "in radio contact"]
    for i, s in enumerate(lines):
        dc.text((tx, my + 34 + 22 * i), s, font=fl, fill=(200, 200, 200))
    return fi, canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="results/figures/real_site_demo.mp4")
    ap.add_argument("--frames", type=int, default=0, help="render only this many frames (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="first simulator step")
    ap.add_argument("--still", default="", help="also save the first frame as this image")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sd = Path(a.site)
    mission = pickle.load(open(sd / f"mission_seed{a.seed}.pkl", "rb"))
    site = Site(sd, mission)
    F = mission["frames"]
    # prefetch the close-up photo blocks under every camera
    need = set()
    for fr in F:
        for dd in fr["drones"].values():
            if dd["mode"] == "landed":
                continue
            x, y = site.cell_to_xy(dd["true"])
            need |= blocks_for(x, y, dd["alt_m"] * math.tan(HALF_FOV) + BLOCK_M * 0.6)
    need = sorted(need)
    todo = [b for b in need if not block_path(site, *b).exists()]
    print(f"photo blocks: {len(need)} needed, {len(todo)} to download", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(fetch_block, [(str(sd), site.meta, *b) for b in todo]))
    # camera path: the swarm's centre, smoothed
    cen = np.array([np.mean([site.cell_to_xy(dd["true"]) for dd in fr["drones"].values()], 0) for fr in F])
    k = 40
    pad = np.r_[np.repeat(cen[:1], k, 0), cen, np.repeat(cen[-1:], k, 0)]
    cen = np.array([pad[i:i + 2 * k + 1].mean(0) for i in range(len(F))])
    steps = range(a.start, len(F))
    jobs = [(si, f) for si in steps for f in np.arange(SUB) / SUB]
    if a.frames:
        jobs = jobs[: a.frames]
    n = len(jobs)
    p = build_scene(site)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(str(sd), a.seed)) as ex:
        pending = {}
        nxt = 0
        for fi, (si, f) in enumerate(jobs):
            c = lerp(cen[si], cen[min(si + 1, len(F) - 1)], f)
            eye, tgt = camera(fi, n, c, site)
            rgb, depth, M = render_3d(p, eye, tgt)
            pending[fi] = ex.submit(make_frame, (fi, si, f, rgb, depth.astype(np.float32), M))
            while nxt in pending and (pending[nxt].done() or len(pending) > 3 * a.workers):
                _, im = pending.pop(nxt).result()
                if nxt == 0 and a.still:
                    im.save(a.still)
                ff.stdin.write(np.asarray(im).tobytes())
                nxt += 1
            if fi % 50 == 0:
                print(f"frame {fi}/{n}  {time.time() - t0:.0f}s", flush=True)
        while nxt in pending:
            _, im = pending.pop(nxt).result()
            if nxt == 0 and a.still:
                im.save(a.still)
            ff.stdin.write(np.asarray(im).tobytes())
            nxt += 1
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {n} frames, {n / FPS:.0f} s, {time.time() - t0:.0f}s to render")


if __name__ == "__main__":
    main()
