"""Tree canopy height (m) from the Meta/WRI v2 global 1 m canopy height map (public S3, EPSG:3857 COGs,
tiled by level-10 Bing quadkeys)."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..arrays import Floats
from ..grid import Grid
from ..net import cached
from .dem import LonLatBounds

BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm"
LEVEL = 10


def _tile_xy(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lr)) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, z: int) -> str:
    s = ""
    for i in range(z, 0, -1):
        d, m = 0, 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        s += str(d)
    return s


def tile_urls(lonlat_bounds: LonLatBounds) -> list[str]:
    minx, miny, maxx, maxy = lonlat_bounds
    x0, y0 = _tile_xy(minx, maxy, LEVEL)
    x1, y1 = _tile_xy(maxx, miny, LEVEL)
    return [f"{BASE}/{quadkey(x, y, LEVEL)}.tif" for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def fetch_canopy(grid: Grid) -> Floats:
    """Canopy height (m) on the grid; 0 where no tile covers it."""

    def build() -> Floats:
        out = np.full(grid.shape, np.nan, dtype="float32")

        def one(u: str) -> Floats | None:
            try:
                return grid.read_raster("/vsicurl/" + u, src_nodata=255)
            except Exception:  # a missing tile: treated as open ground
                return None

        with ThreadPoolExecutor(8) as ex:
            for arr in ex.map(one, tile_urls(grid.lonlat_bounds(pad_m=50))):
                if arr is not None:
                    fill = np.isnan(out) & ~np.isnan(arr)
                    out[fill] = arr[fill]
        filled: Floats = np.nan_to_num(out, nan=0.0)
        return filled

    chm: Floats = cached("canopy", grid.to_dict() | {"v": 2}, build)
    return chm
