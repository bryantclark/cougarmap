"""A square-cell raster grid in a local UTM zone, plus the conversions everything else needs.

Conventions: arrays are indexed [row, col] with row 0 at the north edge. Vector fields are stored as
(east, north) components in metres.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import shapely
from affine import Affine
from numpy.typing import ArrayLike
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize as _rasterize
from rasterio.warp import reproject
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from .arrays import Floats, Mask
from .config import THREADS

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
# a stalled remote read (a connection that stops sending) fails after a minute instead of hanging the run; the
# missing tile is then filled from the seamless DEM like any unreadable one
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "60")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "1024")


def _coords(g: BaseGeometry) -> list[list[float]]:
    out: list[list[float]] = shapely.get_coordinates(g).tolist()
    return out


def geojson(g: BaseGeometry) -> dict[str, Any]:
    """GeoJSON-like mapping of a 2-D geometry, built from numpy coordinate arrays (shapely's mapping() makes a
    Python tuple per vertex, which dominates rasterizing big road and land layers)."""
    t = g.geom_type
    if isinstance(g, LineString) and t == "LineString":  # not a LinearRing
        return {"type": t, "coordinates": _coords(g)}
    if isinstance(g, MultiLineString):
        return {"type": t, "coordinates": [_coords(p) for p in g.geoms]}
    if isinstance(g, Polygon):
        return {"type": t, "coordinates": [_coords(g.exterior), *(_coords(r) for r in g.interiors)]}
    if isinstance(g, MultiPolygon):
        return {
            "type": t,
            "coordinates": [[_coords(p.exterior), *(_coords(r) for r in p.interiors)] for p in g.geoms],
        }
    return dict(mapping(g))


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class Grid:
    epsg: int
    x0: float  # west edge
    y0: float  # north edge
    res: float
    width: int
    height: int

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def around_geometry(
        cls, geom_lonlat: BaseGeometry, res: float, pad_m: float = 0.0, epsg: int | None = None
    ) -> Grid:
        c = geom_lonlat.centroid
        epsg = epsg or utm_epsg(c.x, c.y)
        fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
        g = shp_transform(fwd, geom_lonlat)
        minx, miny, maxx, maxy = g.bounds
        minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
        x0 = math.floor(minx / res) * res
        y0 = math.ceil(maxy / res) * res
        w = math.ceil((maxx - x0) / res)
        h = math.ceil((y0 - miny) / res)
        return cls(epsg, x0, y0, res, w, h)

    def coarsen(self, res: float) -> Grid:
        """Same footprint, bigger cells (res should be >= self.res)."""
        w = max(1, math.ceil(self.width * self.res / res))
        h = max(1, math.ceil(self.height * self.res / res))
        return Grid(self.epsg, self.x0, self.y0, res, w, h)

    # ---- basic properties ---------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    @property
    def area_km2(self) -> float:
        return self.width * self.height * self.res**2 / 1e6

    # Pickle only the defining fields: the cached transformers are rebuilt on demand (and states saved before
    # this held a full-size coordinate mesh in here too).
    def __getstate__(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, state[f.name])

    @cached_property
    def _to_ll(self) -> Transformer:
        return Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @cached_property
    def _from_ll(self) -> Transformer:
        return Transformer.from_crs(4326, self.epsg, always_xy=True)

    def lonlat_bounds(self, pad_m: float = 0.0) -> tuple[float, float, float, float]:
        minx, miny, maxx, maxy = self.bounds
        g = shp_transform(self._to_ll.transform, box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m))
        west, south, east, north = g.bounds
        return float(west), float(south), float(east), float(north)

    # ---- coordinates --------------------------------------------------------------------------
    # Scalar in, scalar out (numpy scalars); arrays in, arrays out.
    def xy(self, row: ArrayLike, col: ArrayLike) -> tuple[Any, Any]:
        """Cell-centre coordinates (x east, y north) of row/col."""
        return self.x0 + (np.asarray(col) + 0.5) * self.res, self.y0 - (np.asarray(row) + 0.5) * self.res

    def rowcol(self, x: ArrayLike, y: ArrayLike) -> tuple[Any, Any]:
        """Row/col of the cell holding x/y (may be outside the grid: check contains_rc)."""
        col = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        row = np.floor((self.y0 - np.asarray(y)) / self.res).astype(int)
        return row, col

    def to_lonlat(self, x: Any, y: Any) -> tuple[Any, Any]:
        return self._to_ll.transform(x, y)

    def from_lonlat(self, lon: Any, lat: Any) -> tuple[Any, Any]:
        return self._from_ll.transform(lon, lat)

    def contains_rc(self, row: ArrayLike, col: ArrayLike) -> Any:
        """True where row/col is a cell of the grid (a bool for scalars, a mask for arrays)."""
        row, col = np.asarray(row), np.asarray(col)
        return (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)

    def cell(self, lon: float, lat: float) -> tuple[int, int] | None:
        """(row, col) of the cell holding a lon/lat point, or None outside the grid."""
        r, c = self.rowcol(*self.from_lonlat(lon, lat))
        return (int(r), int(c)) if self.contains_rc(r, c) else None

    def cell_centers(self) -> tuple[Floats, Floats]:
        """Cell-centre x of every column (width,) and y of every row (height,); broadcast them as
        xs[None, :] and ys[:, None] instead of building full-size meshes."""
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.res
        ys = self.y0 - (np.arange(self.height) + 0.5) * self.res
        return xs, ys

    def _fwd(self, xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x, y = self._from_ll.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    def project(self, geom_lonlat: BaseGeometry) -> BaseGeometry:
        """lon/lat geometry -> this grid's CRS (all coordinates in one vectorized call)."""
        out: BaseGeometry = shapely.transform(geom_lonlat, self._fwd)
        return out

    def project_all(self, geoms_lonlat: Sequence[BaseGeometry]) -> list[BaseGeometry]:
        """Many lon/lat geometries -> this grid's CRS in one vectorized call (much faster than one at a time)."""
        if not len(geoms_lonlat):
            return []
        arr = np.empty(len(geoms_lonlat), dtype=object)
        arr[:] = list(geoms_lonlat)
        return list(shapely.transform(arr, self._fwd))

    def unproject(self, geom_xy: BaseGeometry) -> BaseGeometry:
        out: BaseGeometry = shp_transform(self._to_ll.transform, geom_xy)
        return out

    # ---- rasterization / resampling -----------------------------------------------------------
    def rasterize(
        self,
        geoms_xy: Iterable[BaseGeometry],
        values: float | Sequence[float] = 1,
        fill: float = 0,
        dtype: npt.DTypeLike = "float32",
        all_touched: bool = True,
    ) -> npt.NDArray[Any]:
        """Burn geometries (in this grid's CRS) into a new array: one value for all, or one per geometry
        (later geometries win where they overlap)."""
        geoms = list(geoms_xy)
        if isinstance(values, (int, float, np.number)):
            shapes = [(geojson(g), values) for g in geoms if not g.is_empty]
        else:
            shapes = [(geojson(g), v) for g, v in zip(geoms, values, strict=True) if not g.is_empty]
        if not shapes:
            return np.full(self.shape, fill, dtype=dtype)
        out: npt.NDArray[Any] = _rasterize(
            shapes, out_shape=self.shape, transform=self.transform, fill=fill, all_touched=all_touched, dtype=dtype
        )
        return out

    def mask(self, geoms_xy: Iterable[BaseGeometry], all_touched: bool = True) -> Mask:
        """Cells the geometries touch (all_touched) or whose centres they cover."""
        return self.rasterize(geoms_xy, 1, dtype="uint8", all_touched=all_touched).astype(bool)

    def resample_from(
        self, src: np.ndarray, src_grid: Grid, resampling: Resampling = Resampling.bilinear, nodata: float = np.nan
    ) -> Floats:
        """src (on src_grid) warped onto this grid, as float32; nodata outside src."""
        dst = np.full(self.shape, nodata, dtype="float32")
        reproject(
            src.astype("float32"),
            dst,
            src_transform=src_grid.transform,
            src_crs=src_grid.crs,
            dst_transform=self.transform,
            dst_crs=self.crs,
            resampling=resampling,
            src_nodata=nodata,
            dst_nodata=nodata,
            num_threads=THREADS,
        )
        return dst

    def resample_lonlat(
        self, src: np.ndarray, transform: Affine, resampling: Resampling = Resampling.bilinear
    ) -> Floats:
        """A lon/lat (EPSG:4326) raster with the given transform warped onto this grid, as float32 (NaN outside it
        and where it is NaN)."""
        dst = np.full(self.shape, np.nan, dtype="float32")
        reproject(
            src.astype("float32"),
            dst,
            src_transform=transform,
            src_crs="EPSG:4326",
            dst_transform=self.transform,
            dst_crs=self.crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            num_threads=THREADS,
        )
        return dst

    def read_raster(
        self,
        path: str,
        resampling: Resampling = Resampling.bilinear,
        band: int = 1,
        src_nodata: float | None = None,
        dst: Floats | None = None,
    ) -> Floats:
        """Warp (part of) a local or remote raster onto this grid. Reads only the needed window,
        at a decimated resolution when the source is much finer (uses COG overviews)."""
        if dst is None:
            dst = np.full(self.shape, np.nan, dtype="float32")
        with rasterio.open(path) as ds:
            nodata = src_nodata if src_nodata is not None else ds.nodata
            to_src = Transformer.from_crs(self.epsg, ds.crs, always_xy=True)
            minx, miny, maxx, maxy = self.bounds
            pad = 4 * self.res
            gb = shp_transform(to_src.transform, box(minx - pad, miny - pad, maxx + pad, maxy + pad)).bounds
            sb = ds.bounds
            ib = (max(gb[0], sb.left), max(gb[1], sb.bottom), min(gb[2], sb.right), min(gb[3], sb.top))
            if ib[0] >= ib[2] or ib[1] >= ib[3]:
                return dst
            win = window_from_bounds(*ib, transform=ds.transform).round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0:
                return dst
            # decimate if the source is much finer than the target
            src_res = abs(ds.transform.a)
            if ds.crs.is_geographic:
                src_res *= 111_000 * math.cos(math.radians((gb[1] + gb[3]) / 2))
            factor = max(1.0, (self.res / src_res) / 1.1)
            out_h = max(1, int(win.height / factor))
            out_w = max(1, int(win.width / factor))
            arr = ds.read(
                band,
                window=win,
                out_shape=(out_h, out_w),
                resampling=Resampling.average if factor > 1 else Resampling.nearest,
            )
            arr = arr.astype("float32")
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr[arr < -1e5] = np.nan
            wt = ds.window_transform(win) @ Affine.scale(win.width / out_w, win.height / out_h)
            tmp = np.full(self.shape, np.nan, dtype="float32")
            reproject(
                arr,
                tmp,
                src_transform=wt,
                src_crs=ds.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=resampling,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
        fill = np.isnan(dst) & ~np.isnan(tmp)
        dst[fill] = tmp[fill]
        return dst

    def write_tif(self, path: str | Path, arr: np.ndarray, nodata: float = np.nan) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=1,
            dtype=arr.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=nodata,
            compress="deflate",
        ) as ds:
            ds.write(arr, 1)

    def to_dict(self) -> dict[str, float]:
        return dict(epsg=self.epsg, x0=self.x0, y0=self.y0, res=self.res, width=self.width, height=self.height)
