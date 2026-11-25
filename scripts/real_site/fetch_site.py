"""Download a real site for the photoreal demo: elevation and an aerial photo.

    python scripts/real_site/fetch_site.py --lat 39.6385 --lon -77.4730 --size-m 11600 --out sites/catoctin

Sources (public, no key):
- Elevation: AWS Terrain Tiles (terrarium PNG, from USGS 3DEP in the US), zoom 13 (about 15 m).
- Photo: USDA NAIP aerial imagery through the USGS National Map image service.

Writes, in --out:
- dem.npy: elevation in metres on a north-up grid of --dem-m pixels (row 0 = north).
- photo.jpg: the whole area as one north-up image (--photo-px square).
- canopy.npy: tree cover 0-1 on the DEM grid, estimated from the photo's colour.
- meta.json: centre, size, pixel sizes.
The geometry is a local flat approximation around the centre (fine over ~10 km).
"""

import argparse
import io
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

M_PER_DEG_LAT = 110_540.0
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"retry {url[:80]}...: {e}", file=sys.stderr)
            time.sleep(2 ** i)


def bbox(lat0, lon0, half_m):
    dlat = half_m / M_PER_DEG_LAT
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat0)))
    return lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat


def naip(b, px: int) -> Image.Image:
    """North-up image of bbox b (lon/lat), px x px, linear in lon/lat. The
    service keeps pixels square in degrees, so ask for the matching aspect
    and stretch back to a square (pixels square in metres for a square bbox)."""
    h = max(1, round(px * (b[3] - b[1]) / (b[2] - b[0])))
    url = (f"{NAIP}?bbox={b[0]},{b[1]},{b[2]},{b[3]}&bboxSR=4326&imageSR=4326&size={px},{h}"
           f"&format=jpg&compressionQuality=90&f=image")
    im = Image.open(io.BytesIO(get(url))).convert("RGB")
    return im.resize((px, px), Image.BICUBIC)


def naip_tiled(b, px: int, tile: int = 2048) -> Image.Image:
    n = max(1, math.ceil(px / tile))
    out = Image.new("RGB", (n * tile, n * tile))
    jobs = []
    for i in range(n):  # row, from north
        for j in range(n):
            sub = (b[0] + (b[2] - b[0]) * j / n, b[3] - (b[3] - b[1]) * (i + 1) / n,
                   b[0] + (b[2] - b[0]) * (j + 1) / n, b[3] - (b[3] - b[1]) * i / n)
            jobs.append((i, j, sub))
    with ThreadPoolExecutor(4) as ex:
        for (i, j, _), im in zip(jobs, ex.map(lambda q: naip(q[2], tile), jobs)):
            out.paste(im, (j * tile, i * tile))
    return out.resize((px, px), Image.LANCZOS)


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def dem(b, n_px: int, z: int = 13) -> np.ndarray:
    lats = np.linspace(b[3], b[1], n_px)  # north to south
    lons = np.linspace(b[0], b[2], n_px)
    LON, LAT = np.meshgrid(lons, lats)
    n = 2 ** z
    X = (LON + 180.0) / 360.0 * n
    Y = (1 - np.arcsinh(np.tan(np.radians(LAT))) / np.pi) / 2 * n
    xs = range(int(X.min()), int(X.max()) + 1)
    ys = range(int(Y.min()), int(Y.max()) + 1)
    tiles = [(x, y) for x in xs for y in ys]

    def fetch(t):
        im = np.asarray(Image.open(io.BytesIO(get(TERRARIUM.format(z=z, x=t[0], y=t[1])))).convert("RGB"), float)
        return t, im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768

    with ThreadPoolExecutor(8) as ex:
        got = dict(ex.map(fetch, tiles))
    x0, y0 = xs[0], ys[0]
    mosaic = np.zeros((len(ys) * 256, len(xs) * 256))
    for (x, y), e in got.items():
        mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = e
    px = (X - x0) * 256 - 0.5
    py = (Y - y0) * 256 - 0.5
    i0, j0 = np.floor(py).astype(int), np.floor(px).astype(int)
    fy, fx = py - i0, px - j0
    i0, j0 = np.clip(i0, 0, mosaic.shape[0] - 2), np.clip(j0, 0, mosaic.shape[1] - 2)
    m = mosaic
    return ((m[i0, j0] * (1 - fx) + m[i0, j0 + 1] * fx) * (1 - fy)
            + (m[i0 + 1, j0] * (1 - fx) + m[i0 + 1, j0 + 1] * fx) * fy)


def canopy_from_photo(photo: Image.Image, n_px: int) -> np.ndarray:
    """Tree cover per DEM pixel: share of photo pixels that are dark and green
    (summer leaf-on NAIP). A rough estimate, not a land-cover product."""
    a = np.asarray(photo, float)
    r, g, bl = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * bl
    tree = (g - r > 8) & (lum < 75)
    k = a.shape[0] // n_px
    t = tree[: k * n_px, : k * n_px].reshape(n_px, k, n_px, k).mean((1, 3))
    return np.clip(t, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=39.6385)
    p.add_argument("--lon", type=float, default=-77.4730)
    p.add_argument("--size-m", type=float, default=11600.0)
    p.add_argument("--dem-m", type=float, default=30.0)
    p.add_argument("--photo-px", type=int, default=6144)
    p.add_argument("--out", default="sites/catoctin")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = bbox(a.lat, a.lon, a.size_m / 2)
    n = int(round(a.size_m / a.dem_m))
    e = dem(b, n)
    np.save(out / "dem.npy", e.astype(np.float32))
    print(f"elevation {e.min():.0f}-{e.max():.0f} m on {n}x{n}")
    photo = naip_tiled(b, a.photo_px)
    photo.save(out / "photo.jpg", quality=92)
    c = canopy_from_photo(photo, n)
    np.save(out / "canopy.npy", c.astype(np.float32))
    print(f"photo {photo.size}, tree cover {c.mean():.0%}")
    json.dump(dict(lat=a.lat, lon=a.lon, size_m=a.size_m, dem_m=a.dem_m, bbox=b, photo_px=a.photo_px,
                   sources=dict(elevation="AWS Terrain Tiles (USGS 3DEP)", photo="USDA NAIP via USGS National Map")),
              open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
