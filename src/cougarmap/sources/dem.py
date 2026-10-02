"""Elevation from USGS 3DEP: 1 m lidar tiles where they exist, 1/3 arc-second (~10 m) seamless otherwise."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

import numpy as np
from scipy import ndimage

from ..arrays import Floats
from ..grid import Grid
from ..net import cached, get_json

LonLatBounds = tuple[float, float, float, float]  # west, south, east, north

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
S3 = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"


class DemInfo(TypedDict):
    sources: list[str]  # tiles and products used
    lidar_fraction: float  # share of the grid covered by 1 m lidar


class LidarTile(TypedDict):
    title: str
    url: str
    date: str


def lidar_tiles(lonlat_bounds: LonLatBounds) -> list[LidarTile]:
    """1 m DEM tiles (newest first) intersecting the bounds."""
    bbox = ",".join(f"{v:.5f}" for v in lonlat_bounds)

    def fetch() -> list[LidarTile]:
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            j = get_json(
                TNM,
                params=dict(
                    datasets="Digital Elevation Model (DEM) 1 meter",
                    bbox=bbox,
                    max=100,
                    offset=offset,
                    outputFormat="JSON",
                ),
                timeout=90,
            )
            got = j.get("items", [])
            items += got
            offset += len(got)
            if not got or offset >= j.get("total", 0):
                break
        return [
            LidarTile(title=i["title"], url=i["downloadURL"], date=i.get("publicationDate", ""))
            for i in items
            if i.get("downloadURL", "").endswith(".tif")
        ]

    tiles: list[LidarTile] = cached("tnm1m", bbox, fetch)
    return sorted(tiles, key=lambda t: t["date"], reverse=True)


def _seamless_tiles(lonlat_bounds: LonLatBounds, product: str) -> list[str]:
    minx, miny, maxx, maxy = lonlat_bounds
    urls = []
    for lat in range(math.floor(miny) + 1, math.floor(maxy) + 2):
        for lon in range(math.floor(minx), math.floor(maxx) + 1):
            name = f"n{lat:02d}w{-lon:03d}" if lon < 0 else f"n{lat:02d}e{lon:03d}"
            urls.append(f"{S3}/{product}/TIFF/current/{name}/USGS_{product}_{name}.tif")
    return urls


def _merge_lidar(grid: Grid, tiles: list[LidarTile], dem: Floats) -> list[str]:
    """Fill dem's gaps from the 1 m tiles (read in parallel, merged newest first); the titles used."""

    def one(t: LidarTile) -> tuple[LidarTile, Floats | None]:
        try:
            return t, grid.read_raster("/vsicurl/" + t["url"])
        except Exception:  # a missing or unreadable tile: the seamless DEM fills in
            return t, None

    used = []
    with ThreadPoolExecutor(8) as ex:
        for t, arr in ex.map(one, tiles):  # tiles are sorted newest first
            if arr is None:
                continue
            fill = np.isnan(dem) & ~np.isnan(arr)
            if fill.any():
                dem[fill] = arr[fill]
                used.append(t["title"])
    return used


def _fill_holes(dem: Floats) -> Floats:
    """Small holes get the nearest valid value (scipy, not the parallel numba EDT: downloads run on worker
    threads, and numba's fallback threading layer must not run parallel kernels concurrently)."""
    bad = np.isnan(dem)
    if not bad.any():
        return dem
    if bad.all():
        raise RuntimeError(
            "no elevation data for this area: CougarMap only covers the US (check the coordinates; "
            "west longitudes are negative, e.g. 47.37,-116.10)"
        )
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    out: Floats = dem[tuple(idx)]
    return out


def fetch_dem(grid: Grid, allow_lidar: bool = True) -> tuple[Floats, DemInfo]:
    """Elevation (m) on the grid, plus a note on which sources were used."""

    def build() -> tuple[Floats, DemInfo]:
        lb = grid.lonlat_bounds(pad_m=50)
        dem = np.full(grid.shape, np.nan, dtype="float32")
        used = _merge_lidar(grid, lidar_tiles(lb), dem) if allow_lidar and grid.res < 8 else []
        lidar_frac = float(1 - np.isnan(dem).mean())
        if np.isnan(dem).any():
            for u in _seamless_tiles(lb, "13" if grid.res < 25 else "1"):
                try:
                    grid.read_raster("/vsicurl/" + u, dst=dem)
                    used.append(u.replace("\\", "/").rsplit("/", 1)[-1])
                except Exception:  # tiles that don't exist (ocean, outside the US) are skipped
                    continue
        return _fill_holes(dem), DemInfo(sources=used, lidar_fraction=round(lidar_frac, 3))

    out: tuple[Floats, DemInfo] = cached("dem", grid.to_dict() | {"lidar": allow_lidar, "v": 1}, build)
    return out
