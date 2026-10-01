"""Small hand-made model states and contexts for unit tests (no downloads, no full analysis)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

import synthetic
from cougarmap import terrain as T
from cougarmap.aoi import AOI
from cougarmap.arrays import Floats
from cougarmap.config import Options
from cougarmap.context import Context
from cougarmap.grid import Grid
from cougarmap.sources import vector as vec
from cougarmap.sources.dem import DemInfo
from cougarmap.sources.vector import Feature, Water
from cougarmap.state import SAVED, Layers, ModelState

N, RES = 300, 5.0
X0, Y0 = 400_000.0, 5_300_000.0


def recorder(calls: list[tuple[Any, ...]], result: Any) -> Callable[..., Any]:
    """A stand-in function that records its positional arguments (keyword ones under a trailing dict)."""

    def fake(*a: Any, **k: Any) -> Any:
        calls.append((*a, k) if k else a)
        return result

    return fake


def grid(n: int = N, res: float = RES, width: int | None = None) -> Grid:
    return Grid(synthetic.EPSG, X0, Y0, res, width or n, n)


def feature(g: Grid, geom_xy: BaseGeometry, **props: Any) -> Feature:
    """A source feature (lon/lat) from a geometry in g's coordinates."""
    return Feature(g.unproject(geom_xy), props)


def blank_layers(fine: Grid, mid: Grid) -> Layers:
    """Every saved layer, neutral: nothing scores, everything is public and reachable, no roads near."""
    raw: dict[str, Any] = {}
    for name in SAVED:
        g = mid if name.endswith("_mid") else fine
        if name.endswith("pred_mid"):
            raw[name] = np.full(g.height * g.width, -1, np.int64)
        else:
            raw[name] = np.zeros(g.shape, "float32")
    raw.update(
        cliff=np.zeros(fine.shape, bool),
        meadow=np.zeros(fine.shape, bool),
        lake=np.zeros(fine.shape, bool),
        usable=np.ones(fine.shape, bool),
        public=np.ones(fine.shape, bool),
        closed_track_mid=np.zeros(mid.shape, bool),
        n_on=np.zeros(fine.shape, np.uint8),
        land_id=np.zeros(fine.shape, np.int32),
        meadow_label=np.zeros(fine.shape, np.int32),
        meadow_ha=np.zeros(1, np.float32),
        water_kind=np.full(fine.shape, -1, np.int8),
        landform_mid=np.full(mid.shape, T.SLOPE, np.int8),
        land_names=["Test National Forest"],
        land_access=["OA"],
        water_labels=[],
        saddle_points=[],
        road_dist=np.full(fine.shape, 500.0, "float32"),
        paved_dist=np.full(fine.shape, 2000.0, "float32"),
        rec_dist=np.full(fine.shape, 2000.0, "float32"),
        context=np.ones(fine.shape, "float32"),
        season=np.ones(fine.shape, "float32"),
        winter_range_mid=np.zeros(mid.shape, bool),
        trail_kind=np.zeros(fine.shape, np.int8),
    )
    for k in ("walk_m", "walk_s", "walk_any_m", "walk_any_s"):
        raw[k] = np.full(fine.shape, 300.0, "float32")
    raw["score"] = np.full(fine.shape, 50.0, "float32")
    return cast("Layers", raw)


def state(n: int = N, res: float = RES, **opts: Any) -> ModelState:
    """A flat n x n state (fine and mid grid the same) with blank_layers; the AOI covers the whole grid."""
    g = grid(n, res)
    west, south, east, north = g.lonlat_bounds()
    return ModelState(
        aoi=AOI("toy", box(west, south, east, north)),
        opts=Options(month=10, **opts),
        month=10,
        fine=g,
        mid=g,
        wind=synthetic.wind(),
        dem_info=DemInfo(sources=["toy"], lidar_fraction=1.0),
        z=np.zeros(g.shape, "float32"),
        chm=np.zeros(g.shape, "float32"),
        aoi_mask=np.ones(g.shape, bool),
        layers=blank_layers(g, g),
    )


def context(
    z: Floats,
    chm: Floats | None = None,
    res: float = RES,
    osm: list[Feature] | None = None,
    water: Water | None = None,
    **opts: Any,
) -> Context:
    """A Context on one grid (fine == mid, see grid()) over z, for running single factors. Vector features are in
    lon/lat, as the sources give them (feature() makes one from grid coordinates)."""
    g = grid(z.shape[0], res, width=z.shape[1])
    west, south, east, north = g.lonlat_bounds()
    water = water or Water(points=[], flowlines=[], waterbodies=[])
    vec.project_features([*(osm or []), *water["flowlines"], *water["waterbodies"]], g)
    return Context(
        aoi=AOI("toy", box(west, south, east, north)),
        opts=Options(month=10, **opts),
        month=10,
        fine=g,
        mid=g,
        wind=synthetic.wind(),
        dem_info=DemInfo(sources=["toy"], lidar_fraction=1.0),
        z=z.astype("float32"),
        z_mid=z.astype("float32"),
        chm=np.zeros(z.shape, "float32") if chm is None else chm.astype("float32"),
        aoi_mask=np.ones(z.shape, bool),
        water=water,
        osm=osm or [],
        mvum=[],
        land=[],
    )
