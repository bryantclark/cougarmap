"""Everything an analysis run needs, fetched once: grids, rasters, vector layers, wind climate."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine

from .aoi import AOI
from .arrays import Floats
from .config import MILE_M, WINTER, WORN, Options
from .grid import Grid
from .sources import buildings as bld_src
from .sources import canopy as canopy_src
from .sources import dem as dem_src
from .sources import lidar as lidar_src
from .sources import snow as snow_src
from .sources import vector as vec
from .sources import weather
from .sources.vector import Feature, Water
from .state import ModelState

Log = Callable[[str], object]


@dataclass(kw_only=True)
class Context(ModelState):
    """A run in progress: the model state plus the downloaded inputs the factors read (vector features are
    projected onto the grids' CRS, see Feature.xy)."""

    z_mid: Floats  # 10 m DEM
    water: Water
    osm: list[Feature]  # roads, trails, fences, rail, pipelines
    mvum: list[Feature]  # Forest Service roads with their open seasons
    land: list[Feature]  # PAD-US public land
    buildings: list[bld_src.Building] = field(default_factory=list)  # (lon, lat, footprint m2)
    rec_points: list[Feature] = field(default_factory=list)  # trailheads, campgrounds, parking (OpenStreetMap)
    snow_cm: tuple[Floats, Affine] | None = None  # winter months: typical snow depth (cm, lon/lat grid; SNODAS)
    phs: list[Feature] = field(default_factory=list)  # deer/elk winter range (WDFW PHS), in its months
    lidar1m: lidar_src.Lidar1m | None = None  # Options.worn_trails: the area's 1 m lidar (None: off or unavailable)
    cache: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict, repr=False)  # masks shared by factors

    def memo(self, key: tuple[Any, ...], fn: Callable[[], np.ndarray]) -> np.ndarray:
        """fn() computed once per run (read-only, since several factors share it)."""
        if key not in self.cache:
            self.cache[key] = fn()
            self.cache[key].flags.writeable = False
        return self.cache[key]


def choose_res(area_km2: float, opts: Options) -> float:
    r = math.sqrt(area_km2 * 1e6 / opts.max_cells)
    r = max(opts.min_res_m, r)
    return float(math.ceil(r * 2) / 2)


def grids(aoi: AOI, opts: Options) -> tuple[Grid, Grid]:
    """The fine analysis grid and the 10 m terrain/walking grid (padded by the walk limit), in one UTM zone."""
    res = choose_res(aoi.area_km2(), opts)
    fine = Grid.around_geometry(aoi.geom, res, pad_m=max(150.0, 20 * res))
    walk_pad = opts.max_walk_miles * MILE_M * 1.05 + 200
    mid = Grid.around_geometry(aoi.geom, max(10.0, res), pad_m=walk_pad, epsg=fine.epsg)
    return fine, mid


def fetch(aoi: AOI, opts: Options, fine: Grid, mid: Grid, log: Log = print) -> dict[str, Any]:
    """Download (or read from the disk cache) every input, all sources at once."""
    month = opts.month or dt.date.today().month
    lb = mid.lonlat_bounds()
    lat, lon = aoi.centroid

    def optional(fn: Callable[[], list[Any]], what: str, without: str) -> list[Any]:
        try:
            return fn()
        except Exception as e:  # these services are occasionally down; the run goes on without them
            log(f"  {what} unavailable ({e}); {without}")
            return []

    jobs: dict[str, Callable[[], Any]] = dict(
        dem=lambda: dem_src.fetch_dem(fine),
        dem_mid=lambda: dem_src.fetch_dem(mid, allow_lidar=False),
        chm=lambda: canopy_src.fetch_canopy(fine),
        water=lambda: vec.fetch_water(lb),
        osm=lambda: vec.fetch_osm(lb),
        mvum=lambda: optional(lambda: vec.fetch_mvum(lb), "MVUM", "using OpenStreetMap roads only"),
        land=lambda: vec.fetch_public_land(lb),
        buildings=lambda: optional(
            lambda: bld_src.fetch_buildings(fine.lonlat_bounds(pad_m=opts.houses_radius_m + 100)),
            "building footprints",
            "no populated-area penalty",
        ),
        rec=lambda: optional(
            lambda: vec.fetch_osm_points(fine.lonlat_bounds(pad_m=opts.rec_reach_m + 100)),
            "recreation sites",
            "no trailhead/campground penalty",
        ),
        wind=lambda: weather.prevailing(lat, lon, month, opts.wind_from_deg),
    )
    if opts.worn_trails and aoi.area_km2() <= WORN.max_km2:

        def lidar1m() -> lidar_src.Lidar1m | None:
            try:
                return lidar_src.fetch_lidar_1m(lidar_src.lidar_grid(aoi.geom, fine.epsg, WORN.pad_m))
            except Exception as e:  # TNM or the tile store is occasionally down; the run goes on without the layer
                log(f"  1 m lidar unavailable ({e}); no worn-trail layer")
                return None

        jobs["lidar1m"] = lidar1m
    if month in WINTER.months:  # the winter module's inputs: nothing new is downloaded the rest of the year

        def snow() -> tuple[Floats, Affine] | None:
            try:
                return snow_src.fetch_snow_depth(lb, month)
            except Exception as e:  # NSIDC is occasionally down; the winter module then ignores snow
                log(f"  snow depth (SNODAS) unavailable ({e}); winter habitat ignores snow")
                return None

        jobs["snow"] = snow
        if month in WINTER.phs_months:
            jobs["phs"] = lambda: optional(
                lambda: vec.fetch_phs_winter_range(lb), "deer/elk winter range (WDFW)", "no winter-range prior"
            )
    log(
        "elevation, canopy height, water (NHD), roads/trails/fences (OpenStreetMap), forest road seasons (MVUM), "
        "public land (PAD-US), buildings, recreation sites, wind climate"
        + (", snow depth" if month in WINTER.months else "")
        + (", 1 m lidar" if "lidar1m" in jobs else "")
        + "..."
    )
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(fn) for k, fn in jobs.items()}
        return {k: f.result() for k, f in futs.items()}


def prefetch(aoi: AOI, opts: Options, log: Log = print) -> None:
    """Warm the download cache for an area (find_hotspots fetches its next blocks while analyzing one)."""
    fetch(aoi, opts, *grids(aoi, opts), log=log)


def build(aoi: AOI, opts: Options, log: Log = print) -> Context:
    month = opts.month or dt.date.today().month
    fine, mid = grids(aoi, opts)
    log(
        f"area {aoi.area_km2():.1f} km2 -> analysis at {fine.res:g} m ({fine.width}x{fine.height}), "
        f"terrain/walking at {mid.res:g} m ({mid.width}x{mid.height})"
    )
    d = fetch(aoi, opts, fine, mid, log=log)
    (z, info), (z_mid, _) = d["dem"], d["dem_mid"]
    log(f"  lidar coverage {info['lidar_fraction']:.0%}")
    notes = (
        ["No snow-depth data (SNODAS) for this run: the winter habitat ignores snow."]
        if month in WINTER.months and d.get("snow") is None
        else []
    )
    notes += _worn_notes(aoi, opts, d.get("lidar1m"))
    water: Water = d["water"]
    vec.project_features(
        [*d["osm"], *d["mvum"], *d["land"], *water["flowlines"], *water["waterbodies"], *d["rec"], *d.get("phs", [])],
        fine,
    )
    return Context(
        aoi=aoi,
        opts=opts,
        month=month,
        fine=fine,
        mid=mid,
        z=z,
        z_mid=z_mid,
        chm=d["chm"],
        dem_info=info,
        wind=d["wind"],
        water=water,
        osm=d["osm"],
        mvum=d["mvum"],
        land=d["land"],
        aoi_mask=fine.mask([fine.project(aoi.geom)], all_touched=False),
        buildings=d["buildings"],
        rec_points=d["rec"],
        snow_cm=d.get("snow"),
        phs=d.get("phs", []),
        lidar1m=d.get("lidar1m") if d.get("lidar1m") is not None and d["lidar1m"].coverage > 0 else None,
        notes=notes,
    )


def _worn_notes(aoi: AOI, opts: Options, got: lidar_src.Lidar1m | None) -> list[str]:
    """One line on why the worn-trail layer is missing or partial (Options.worn_trails only)."""
    if not opts.worn_trails:
        return []
    if aoi.area_km2() > WORN.max_km2:
        return [
            f"No worn-trail layer: the area is over {WORN.max_km2:g} km2 (1 m lidar is analyzed for smaller areas)."
        ]
    if got is None:
        return ["No worn-trail layer: the 1 m lidar could not be downloaded this time."]
    if got.coverage == 0:
        return ["No worn-trail layer: there is no 1 m lidar here."]
    if got.coverage < 0.95:
        return [f"Worn trails only where there is 1 m lidar ({got.coverage:.0%} of the area)."]
    return []
