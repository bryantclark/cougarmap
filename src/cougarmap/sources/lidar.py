"""Bare-earth elevation at its native 1 m from USGS 3DEP lidar, for the worn-trail detector (worn.py).

One window over the whole analysis area (never point queries), read from the same 1 m tiles as dem.py and cached
in the download cache (lidar1m/) as a compressed GeoTIFF. Areas without 1 m lidar get coverage 0."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry.base import BaseGeometry

from ..arrays import Floats
from ..grid import Grid
from ..net import cache_lock, cache_path
from .dem import LidarTile, lidar_tiles

VERSION = 1  # bump when what is cached changes


class Lidar1m(NamedTuple):
    grid: Grid  # 1 m cells, on whole metres, in the analysis CRS
    z: Floats  # elevation (m); NaN where there is no 1 m lidar
    coverage: float  # share of the grid with 1 m lidar
    sources: list[str]  # tile titles used, newest first


def lidar_grid(geom_lonlat: BaseGeometry, epsg: int, pad_m: float) -> Grid:
    """The 1 m grid over an area (its bounds plus pad_m), in the analysis CRS."""
    return Grid.around_geometry(geom_lonlat, 1.0, pad_m=pad_m, epsg=epsg)


def sub_grid(grid: Grid, bounds: tuple[float, float, float, float]) -> tuple[int, int, Grid] | None:
    """The cells of grid inside bounds (west, south, east, north in grid's CRS): (row, col offset, sub-grid), None
    when they miss the grid."""
    w, s, e, n = bounds
    gw, gs, ge, gn = grid.bounds
    c0, c1 = max(0, int((max(w, gw) - gw) // grid.res)), min(grid.width, int(-(-(min(e, ge) - gw) // grid.res)))
    r0, r1 = max(0, int((gn - min(n, gn)) // grid.res)), min(grid.height, int(-(-(gn - max(s, gs)) // grid.res)))
    if c1 <= c0 or r1 <= r0:
        return None
    return r0, c0, Grid(grid.epsg, grid.x0 + c0 * grid.res, grid.y0 - r0 * grid.res, grid.res, c1 - c0, r1 - r0)


def _read(grid: Grid, tiles: list[LidarTile]) -> tuple[Floats, list[str]]:
    """The tiles merged onto grid, newest first, nearest neighbour (tiles and grid are both on whole metres, so
    the values are the tiles' own); each tile is read only where it overlaps the grid."""
    z = np.full(grid.shape, np.nan, np.float32)

    def one(t: LidarTile) -> tuple[LidarTile, tuple[int, int, Floats] | None]:
        url = "/vsicurl/" + t["url"] if t["url"].startswith("http") else t["url"]  # (a local file as is)
        try:
            with rasterio.open(url) as ds:
                part = sub_grid(grid, transform_bounds(ds.crs, grid.crs, *ds.bounds))
            if part is None:
                return t, None
            r0, c0, sub = part
            return t, (r0, c0, sub.read_raster(url, resampling=Resampling.nearest))
        except Exception:  # a missing or unreadable tile leaves a gap (reported as coverage)
            return t, None

    used = []
    with ThreadPoolExecutor(8) as ex:
        for t, got in ex.map(one, tiles):  # newest first
            if got is None:
                continue
            r0, c0, arr = got
            win = z[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]]
            fill = np.isnan(win) & ~np.isnan(arr)
            if fill.any():
                win[fill] = arr[fill]
                used.append(t["title"])
    return z, used


def _write(path: str, grid: Grid, z: Floats, sources: list[str]) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as ds:
        ds.write(z, 1)
        ds.update_tags(sources=json.dumps(sources))


def fetch_lidar_1m(grid: Grid) -> Lidar1m:
    """1 m elevation on grid (res 1), from the disk cache or the 3DEP 1 m tiles."""
    path = cache_path("lidar1m", grid.to_dict() | {"v": VERSION}, ".tif")
    with cache_lock(path):
        if path.exists():
            try:
                with rasterio.open(path) as ds:
                    z = ds.read(1, out_dtype=np.float32)
                    sources: list[str] = json.loads(ds.tags().get("sources", "[]"))
                return Lidar1m(grid, z, float(np.isfinite(z).mean()), sources)
            except Exception:  # a damaged cache file is fetched again
                path.unlink(missing_ok=True)
        tiles = lidar_tiles(grid.lonlat_bounds())
        z, used = _read(grid, tiles) if tiles else (np.full(grid.shape, np.nan, np.float32), [])
        tmp = path.with_name(path.stem + ".part.tif")
        _write(str(tmp), grid, z, used)
        tmp.replace(path)
    return Lidar1m(grid, z, float(np.isfinite(z).mean()), used)
