"""Turn real map layers for a field-test site into a wilderness world (M8).

    python scripts/import_site.py --dem dem.tif --canopy canopy.tif --trails trails.geojson \
        --water water.geojson --ipp 512 300 --out sites/my_site.npz
    python scripts/train.py ...  /  FieldScenario(family="wilderness", site="sites/my_site.npz")

Inputs (all in the same projected coordinate system, metres):
- --dem: elevation raster (GeoTIFF with rasterio installed, or a .npy array with --cell-m).
- --canopy: tree cover raster, percent (0-100) or fraction (optional; e.g. a national tree-canopy layer).
- --trails / --water: GeoJSON lines or polygons (needs rasterio), or .npy masks on the DEM grid (optional).
- --ipp: last-seen point as DEM pixel column and row; --base: command post (defaults to the IPP).

The layers are cropped to a square centred on the IPP and resampled to the
wilderness family's grid (48 cells of 75 m). Public sources: USGS 3DEP
elevation, NLCD tree canopy, OpenStreetMap trails and water.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmsearch.field.params import FAMILY  # noqa: E402


def read_raster(path: str, cell_m: float | None):
    if path.endswith(".npy"):
        if cell_m is None:
            raise SystemExit("--cell-m is needed with .npy inputs")
        return np.load(path).astype(float), None, cell_m
    import rasterio  # optional dependency
    with rasterio.open(path) as src:
        return src.read(1).astype(float), src, abs(src.transform.a)


def rasterize(path: str, like, shape) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(bool)
    import json
    from rasterio.features import rasterize as rio_rasterize
    feats = json.load(open(path))["features"]
    return rio_rasterize([(f["geometry"], 1) for f in feats], out_shape=shape, transform=like.transform,
                         all_touched=True).astype(bool)


def resample(a: np.ndarray, src_cell: float, centre, g: int, cell_m: float, how: str) -> np.ndarray:
    """Square of g x g cells of cell_m around centre (col, row), by block mean or block max."""
    k = cell_m / src_cell
    out = np.zeros((g, g))
    c0, r0 = centre[0] - g / 2 * k, centre[1] - g / 2 * k
    for i in range(g):
        for j in range(g):
            r1, r2 = int(r0 + i * k), int(r0 + (i + 1) * k)
            q1, q2 = int(c0 + j * k), int(c0 + (j + 1) * k)
            block = a[max(0, r1):max(r1 + 1, r2), max(0, q1):max(q1 + 1, q2)]
            if block.size == 0:
                raise SystemExit("the site is smaller than the search area: give a larger DEM")
            out[i, j] = block.max() if how == "max" else block.mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--canopy")
    p.add_argument("--trails")
    p.add_argument("--water")
    p.add_argument("--cell-m", type=float, help="DEM pixel size for .npy inputs")
    p.add_argument("--ipp", type=float, nargs=2, required=True, metavar=("COL", "ROW"))
    p.add_argument("--base", type=float, nargs=2, metavar=("COL", "ROW"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    fam = FAMILY["wilderness"]
    dem, src, cell = read_raster(a.dem, a.cell_m)
    centre = a.ipp
    elev = resample(dem, cell, centre, fam.grid, fam.cell_m, "mean")
    canopy = np.zeros_like(elev)
    if a.canopy:
        c, _, _ = read_raster(a.canopy, cell)
        c = c / 100.0 if c.max() > 1.0 else c
        canopy = np.clip(resample(c, cell, centre, fam.grid, fam.cell_m, "mean"), 0, 1)
    trail = np.zeros_like(elev, dtype=bool)
    if a.trails:
        trail = resample(rasterize(a.trails, src, dem.shape).astype(float), cell, centre, fam.grid, fam.cell_m, "max") > 0
    water = np.zeros_like(elev, dtype=bool)
    if a.water:
        water = resample(rasterize(a.water, src, dem.shape).astype(float), cell, centre, fam.grid, fam.cell_m, "mean") > 0.5
    k = fam.cell_m / cell
    to_cell = lambda c: (int(np.clip((c[0] - centre[0]) / k + fam.grid / 2, 0, fam.grid - 1)),  # noqa: E731
                         int(np.clip((c[1] - centre[1]) / k + fam.grid / 2, 0, fam.grid - 1)))
    ipp = to_cell(a.ipp)
    base = to_cell(a.base) if a.base else ipp
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, elev=elev, canopy=canopy, trail=trail, water=water, base=np.array(base), ipp=np.array(ipp),
             cell_m=fam.cell_m)
    print(f"wrote {a.out}: {fam.grid}x{fam.grid} cells of {fam.cell_m:.0f} m, elevation {elev.min():.0f}-{elev.max():.0f} m, "
          f"canopy {canopy.mean():.0%}, {trail.sum()} trail cells, {water.sum()} water cells, base {base}, IPP {ipp}")


if __name__ == "__main__":
    main()
