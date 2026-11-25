"""Each drone's own camera feed, flying over the real site.

    python scripts/real_site/render_cameras.py --site sites/catoctin --seed 4 \\
        --out results/figures/real_site_cameras.mp4

One large panel per drone: the downward camera's view, cut from the real
aerial photo (USDA NAIP) at the drone's true height and 84 deg field of view,
turned so the direction of flight is up, as a pilot's feed would be. Between
simulator steps (18 s each) the view glides along the drone's path. A head-up
display gives height, heading, battery, radio link and what the drone is
doing. The missing person (simulated) is drawn at true size when inside the
image, fainter where trees hide them. Alerts and the find are flagged.

The pictures are real; the drones' choices come from the simulator's sensor
model, which does not look at these pictures.
"""

import argparse
import math
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_demo import (COLS, HALF_FOV, REAL, Site, crop, draw_person, font, hhmm, lerp,  # noqa: E402
                         person_offset)

W, H = 1920, 1080
FPS = 24
DIRS = ["north-west", "north", "north-east", "west", "east", "south-west", "south", "south-east"]  # swarmsearch.env.MOVES order
MODE = {"rtb": "low battery: flying home", "landed": "landed: battery swap", "link": "lost radio: flying back",
        "weather": "gusts: flying home", "overwatch": "holding over the find", "nofly_exit": "leaving a no-fly zone"}

_W: dict = {}


def _init(site_dir, seed, sub):
    _W["m"] = pickle.load(open(Path(site_dir) / f"mission_seed{seed}.pkl", "rb"))
    _W["site"] = Site(Path(site_dir), _W["m"])
    _W["sub"] = sub
    _W["head"] = headings(_W["m"])


def headings(m):
    """Smoothed heading (radians, 0 = north, clockwise) per drone per step."""
    F = m["frames"]
    out = {}
    for a in F[0]["drones"]:
        h, prev = [], 0.0
        for si in range(len(F)):
            d0 = F[si]["drones"].get(a)
            d1 = F[min(si + 1, len(F) - 1)]["drones"].get(a)
            if d0 is not None and d1 is not None:
                dx = float(d1["true"][0] - d0["true"][0])
                dy = -float(d1["true"][1] - d0["true"][1])  # rows go south
                if abs(dx) + abs(dy) > 1e-6:
                    prev = math.atan2(dx, dy)
            h.append(prev)
        out[a] = h
    return out


def act_name(act):
    from_sim = {None: "hover and look"}
    if act in from_sim:
        return from_sim[act]
    try:
        act = int(act)
    except (TypeError, ValueError):
        return str(act)
    if act < 8:
        return "fly " + DIRS[act]
    names = {8: "hover and look", 9: "climb", 10: "descend for a closer look"}
    if act in names:
        return names[act]
    if 11 <= act < 19:
        return "sprint " + DIRS[act - 11]
    return f"action {act}"


def ang_lerp(a, b, f):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * f


def feed(site, m, a, si, f, k, px):
    F = m["frames"]
    d0 = F[si]["drones"].get(a)
    if d0 is None:
        return None, None
    d1 = F[min(si + 1, len(F) - 1)]["drones"].get(a, d0)
    pos = lerp(d0["true"], d1["true"], f)
    alt = float(lerp(d0["alt_m"], d1["alt_m"], f))
    hd = _W["head"][a]
    head = ang_lerp(hd[max(0, si - 1)], hd[si], min(1.0, 0.5 + f))
    info = dict(alt=alt, head=head, d0=d0, pos=pos, act=d1.get("act"))
    if d0["mode"] == "landed" or alt < 5:
        return None, info
    half = alt * math.tan(HALF_FOV)
    x, y = site.cell_to_xy(pos)
    big = 1.45  # crop wider, turn, then cut the square back out
    n = int(px * big)
    img = crop(site, x, y, half * big, px=n).convert("RGBA")
    off = person_offset(m["seed"])
    for i, pp in enumerate(F[si]["people"]):
        qx, qy = site.cell_to_xy(np.asarray(pp, float) + off)
        u = (qx - (x - half * big)) / (2 * half * big) * n
        v = ((y + half * big) - qy) / (2 * half * big) * n
        if -20 < u < n + 20 and -20 < v < n + 20:
            occ = d0["occ"][i] if d0["occ"] and d0["occ"][i] is not None else 0.0
            img = draw_person(img, u, v, 2 * half * big / n, max(0.15, 1 - float(occ)))
            info["person_raw"] = (u, v)
    img = img.rotate(math.degrees(head), resample=Image.BICUBIC)  # heading up
    c = (n - px) // 2
    if "person_raw" in info:  # where the person lands after turning (PIL turns anticlockwise)
        dx, dy = info["person_raw"][0] - n / 2, info["person_raw"][1] - n / 2
        th = head
        u = dx * math.cos(th) + dy * math.sin(th) + n / 2 - c
        v = -dx * math.sin(th) + dy * math.cos(th) + n / 2 - c
        if 0 <= u < px and 0 <= v < px:
            info["person_px"] = (u, v)
    return img.crop((c, c, c + px, c + px)).convert("RGB"), info


def hud(im, k, a, info, m, si, label_w):
    d = ImageDraw.Draw(im, "RGBA")
    s = im.size[0]
    col = COLS[k % len(COLS)]
    g = (120, 255, 140, 230)
    # crosshair
    c = s // 2
    for (x0, y0, x1, y1) in [(c - 40, c, c - 12, c), (c + 12, c, c + 40, c), (c, c - 40, c, c - 12), (c, c + 12, c, c + 40)]:
        d.line([(x0, y0), (x1, y1)], fill=g, width=2)
    # corner brackets
    L = 30
    for (x, y, dx, dy) in [(8, 8, 1, 1), (s - 9, 8, -1, 1), (8, s - 9, 1, -1), (s - 9, s - 9, -1, -1)]:
        d.line([(x, y), (x + dx * L, y)], fill=g, width=2)
        d.line([(x, y), (x, y + dy * L)], fill=g, width=2)
    d0 = info["d0"]
    deg = (math.degrees(info["head"]) + 360) % 360
    fh = font(17, True)
    d.rectangle([0, 0, s, 30], fill=(0, 0, 0, 150))
    d.text((10, 5), f"D{k + 1}", font=font(19, True), fill=col)
    d.text((52, 6), f"ALT {info['alt']:.0f} m   HDG {deg:03.0f}°   BAT {100 * float(d0['bat']):.0f}%", font=fh, fill=g)
    link = "LINK" if d0["linked"] else "NO LINK"
    d.text((s - 90, 6), link, font=fh, fill=g if d0["linked"] else (255, 120, 80, 255))
    mode = d0["mode"]
    doing = MODE.get(mode, act_name(info["act"]))
    d.rectangle([0, s - 30, s, s], fill=(0, 0, 0, 150))
    d.text((10, s - 26), doing, font=font(16), fill=(240, 240, 240))
    d.text((s - 150, s - 26), f"{2 * info['alt'] * math.tan(HALF_FOV):.0f} m across", font=font(16), fill=(200, 200, 200))
    # north arrow
    ax, ay, r = s - 34, 64, 18
    t = -info["head"]
    tip = (ax + r * math.sin(t), ay - r * math.cos(t))
    d.ellipse([ax - r - 4, ay - r - 4, ax + r + 4, ay + r + 4], fill=(0, 0, 0, 120))
    d.line([(ax, ay), tip], fill=(255, 80, 80), width=3)
    d.text((tip[0] - 5, tip[1] - 9), "N", font=font(13, True), fill=(255, 255, 255))
    # alerts
    detected = d0["mode"] == "overwatch"
    for (t0, cell, true) in m["alerts"].get(a, []):
        if t0 <= si < t0 + 5:
            txt = "PERSON DETECTED" if true else "ALERT: possible person (a false alarm)"
            colr = (60, 230, 90) if true else (255, 220, 0)
            d.rectangle([0, 34, s, 66], fill=(0, 0, 0, 190))
            d.text((10, 39), txt, font=font(17, True), fill=colr)
            detected |= bool(true)
    if detected and "person_px" in info:
        u, v = info["person_px"]
        b = 16
        z, zr = 4, 22  # zoomed inset of the area around the person
        box = (int(u - zr), int(v - zr), int(u + zr), int(v + zr))
        inset = im.crop(box).resize((2 * zr * z, 2 * zr * z), Image.BICUBIC)
        d.rectangle([u - b, v - b, u + b, v + b], outline=(60, 230, 90), width=2)
        ix, iy = s - 2 * zr * z - 14, s - 2 * zr * z - 44
        im.paste(inset, (ix, iy))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([ix - 2, iy - 2, ix + 2 * zr * z + 1, iy + 2 * zr * z + 1], outline=(60, 230, 90), width=2)
        d.rectangle([ix, iy, ix + 70, iy + 22], fill=(0, 0, 0, 170))
        d.text((ix + 6, iy + 2), "4x zoom", font=font(14, True), fill=(60, 230, 90))
        d.line([(u + b, v + b), (ix, iy)], fill=(60, 230, 90, 160), width=1)
    return im


def make(job):
    fi, si, f = job
    m, site, sub = _W["m"], _W["site"], _W["sub"]
    F = m["frames"]
    names = sorted(F[0]["drones"])
    n = len(names)
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    top, bot = 56, 52
    px = min((W - 20 * (cols + 1)) // cols if cols > 2 else 1e9, (H - top - bot - 12 * (rows + 1)) // rows)
    px = int(px)
    canvas = Image.new("RGB", (W, H), (14, 15, 18))
    dc = ImageDraw.Draw(canvas)
    hour = float(lerp(F[si]["hour"], F[min(si + 1, len(F) - 1)]["hour"], f))
    mins = (si + f) * m["step_s"] / 60
    found = F[si]["found"].any()
    dc.text((20, 12), f"Drone camera feeds  ·  mission time {int(mins)} min  ·  local time {hhmm(hour)}"
            + ("  ·  HIKER FOUND" if found else ""), font=font(26, True), fill=(255, 255, 255) if not found else (90, 240, 120))
    grid_w = cols * px + (cols - 1) * 16
    x0 = 20
    for k, a in enumerate(names):
        r, c = divmod(k, cols)
        ox, oy = x0 + c * (px + 16), top + 6 + r * (px + 12)
        img, info = feed(site, m, a, si, f, k, px)
        if img is None:
            img = Image.new("RGB", (px, px), (28, 28, 32))
            ImageDraw.Draw(img).text((30, px // 2 - 12), f"D{k + 1}: landed, battery swap", font=font(22), fill=(200, 200, 200))
        else:
            img = hud(img, k, a, info, m, si, px)
        canvas.paste(img, (ox, oy))
        dc.rectangle([ox - 1, oy - 1, ox + px, oy + px], outline=COLS[k % len(COLS)], width=2)
    # side panel: where each drone is
    sx = x0 + grid_w + 24
    sw = W - sx - 20
    if sw > 200:
        tex = site.d / "texture_4096.png"
        ms = min(sw, H - top - bot - 330)
        mp = Image.open(tex).resize((ms, ms)).convert("RGB") if tex.exists() else Image.new("RGB", (ms, ms))
        dm = ImageDraw.Draw(mp)
        sc = ms / m["g"]
        ipp = np.asarray(m["ipp"], float)
        dm.rectangle([(ipp[0]) * sc, (ipp[1]) * sc, (ipp[0] + 1) * sc, (ipp[1] + 1) * sc], outline=(255, 255, 255), width=2)
        for k, a in enumerate(names):
            pts = []
            for s in range(max(0, si - 60), si + 1):
                dd = F[s]["drones"].get(a)
                if dd is not None:
                    pts.append(tuple((np.asarray(dd["true"], float) + 0.5) * sc))
            if len(pts) > 1:
                dm.line(pts, fill=COLS[k % len(COLS)], width=2)
            if pts:
                u, v = pts[-1]
                dm.ellipse([u - 6, v - 6, u + 6, v + 6], fill=COLS[k % len(COLS)], outline=(0, 0, 0))
        canvas.paste(mp, (sx, top + 6))
        ty = top + 6 + ms + 14
        dc.text((sx, ty), "Where each drone is (11.6 km square)", font=font(17, True), fill=(230, 230, 230))
        dc.text((sx, ty + 26), "white square = last seen", font=font(15), fill=(200, 200, 200))
        notes = ["Each feed: real aerial photo at the", "drone's true height and 84° view,", "turned so flight direction is up.",
                 "", "Between 18 s simulator steps the", "view glides along the path:", f"video runs about {m['step_s'] * FPS / sub:.0f}x real time.",
                 "", "The hiker (red jacket) is drawn in", "at true size: about 1.7 m. Fainter", "= partly hidden by trees."]
        for i, s in enumerate(notes):
            dc.text((sx, ty + 62 + 21 * i), s, font=font(15), fill=(190, 190, 190))
    dc.rectangle([0, H - bot, W, H], fill=(0, 0, 0))
    dc.text((20, H - bot + 6), REAL, font=font(16), fill=(170, 230, 170))
    dc.text((20, H - bot + 28), "Simulated: drones, missing hiker, detections. The drones decide from the simulator's sensor model, not these images.",
            font=font(16), fill=(255, 200, 140))
    return fi, canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="sites/catoctin")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--sub", type=int, default=8, help="frames per simulator step")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--still", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/figures/real_site_cameras.mp4")
    a = ap.parse_args()
    m = pickle.load(open(Path(a.site) / f"mission_seed{a.seed}.pkl", "rb"))
    end = a.end or len(m["frames"]) - 1
    jobs = [(si, i / a.sub) for si in range(a.start, end) for i in range(a.sub)]
    if a.frames:
        jobs = jobs[: a.frames]
    jobs = [(i, si, f) for i, (si, f) in enumerate(jobs)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.site, a.seed, a.sub)) as ex:
        for fi, im in ex.map(make, jobs, chunksize=4):
            if fi == 0 and a.still:
                im.save(a.still)
            ff.stdin.write(np.asarray(im).tobytes())
            if fi % 200 == 0:
                print(f"frame {fi}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {len(jobs)} frames, {len(jobs) / FPS:.0f} s video, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
