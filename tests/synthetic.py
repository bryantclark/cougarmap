"""A synthetic landscape served in place of the real data sources, so the whole model runs offline in tests.

The area is a 1.2 km square (local coordinates u east, v north, metres from its centre) in UTM zone 11:
- terrain: a north-south ridge at u = -350 with saddles along it, a valley at u = +350, a 20 m cliff band at
  u = 100, a gentle tilt to the north;
- canopy: 18 m timber with a 2.25 ha meadow, a 20 m glade (too small to count), a 6 m powerline cut (too narrow),
  and open grass on the valley floor;
- water: a spring, a perennial creek down the valley, a seasonal draw, a 1.4 ha lake and a small marsh;
- roads: a paved highway south of the area (with a trailhead where the forest road leaves it), a forest road (on
  the Motor Vehicle Use Map, open in October), a gated forest road (not on it), a trail, a fence, a private
  driveway and a paved residential street;
- winter: shallow snow in the west half, deep in the east, and deer winter range over the valley (u > 250);
- land: national forest (open access) west of u = 250, restricted state land to the north-east, private
  elsewhere; a homestead of four buildings and one cabin;
- 1 m lidar: the same terrain with lidar-like noise and a worn trail benched into the east-facing slope along
  u = TRAIL_U (v from -350 to 350), crossing the mapped path near its east end.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from cougarmap.arrays import Floats
from cougarmap.grid import Grid
from cougarmap.sources import buildings, canopy, dem, lidar, snow, vector, weather
from cougarmap.sources.dem import DemInfo
from cougarmap.sources.vector import Feature, Water
from cougarmap.sources.weather import GroundWind, PeriodWind, Wind

EPSG = 32611
LON, LAT = -117.6, 47.9  # the centre of the synthetic area (an arbitrary spot in UTM zone 11)
HALF = 600.0  # the area is 2 x HALF on a side
_FWD = Transformer.from_crs(4326, EPSG, always_xy=True)
_INV = Transformer.from_crs(EPSG, 4326, always_xy=True)
CX, CY = _FWD.transform(LON, LAT)


def lonlat(u: float, v: float) -> tuple[float, float]:
    """Local metres from the centre -> (lon, lat)."""
    lon, lat = _INV.transform(CX + u, CY + v)
    return float(lon), float(lat)


def geom(g: BaseGeometry) -> BaseGeometry:
    """A geometry in local metres -> lon/lat."""

    def inv(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        lon, lat = _INV.transform(CX + np.asarray(x), CY + np.asarray(y))
        return lon, lat

    return shp_transform(inv, g)


def bbox() -> list[float]:
    """[west, south, east, north] of the area."""
    (w, s), (e, n) = lonlat(-HALF, -HALF), lonlat(HALF, HALF)
    return [w, s, e, n]


# ---- rasters -------------------------------------------------------------------------------------------------


def _uv(grid: Grid) -> tuple[Floats, Floats]:
    xs, ys = grid.cell_centers()
    return np.broadcast_to(xs[None, :] - CX, grid.shape), np.broadcast_to(ys[:, None] - CY, grid.shape)


def elevation(u: Floats, v: Floats) -> Floats:
    z = 1200 + 70 * np.cos(2 * math.pi * (u + 350) / 1400) + 25 * np.sin(2 * math.pi * v / 1000) + 0.02 * v
    out: Floats = (z + 20 * np.clip((u - 100) / 6, 0, 1)).astype("float32")
    return out


def canopy_height(u: Floats, v: Floats) -> Floats:
    chm = np.full(u.shape, 18.0, "float32")
    chm[(u > -250) & (u < -100) & (v > -150) & (v < 0)] = 0  # meadow
    chm[(np.abs(u + 450) < 10) & (np.abs(v - 400) < 10)] = 0  # small glade
    chm[np.abs(v - 300) < 3] = 0  # powerline cut
    chm[u > 450] = 0.5  # valley-floor grass
    return chm


def fetch_dem(grid: Grid, allow_lidar: bool = True) -> tuple[Floats, DemInfo]:
    return elevation(*_uv(grid)), DemInfo(sources=["synthetic"], lidar_fraction=1.0 if allow_lidar else 0.0)


def fetch_canopy(grid: Grid) -> Floats:
    return canopy_height(*_uv(grid))


TRAIL_U = -90.0
TRAIL_V = (-350.0, 350.0)


def benched(u: Floats, v: Floats) -> Floats:
    """elevation() with a worn track benched into the slope (which falls to the east here) at TRAIL_U: a 1.5 m
    cut, a 3 m flat tread, a 1.5 m fill."""
    z = elevation(u, v).astype(np.float64)
    on = (v >= TRAIL_V[0]) & (v <= TRAIL_V[1])
    d = u - TRAIL_U
    up, tread, down = (elevation(np.full_like(u, TRAIL_U + k), v).astype(np.float64) for k in (-1.5, 1.5, 4.5))
    z = np.where(on & (d >= -1.5) & (d < 0), up + (tread - up) * (d + 1.5) / 1.5, z)
    z = np.where(on & (d >= 0) & (d <= 3), tread, z)
    z = np.where(on & (d > 3) & (d <= 4.5), tread + (down - tread) * (d - 3) / 1.5, z)
    return z.astype(np.float32)


def fetch_lidar_1m(grid: Grid) -> lidar.Lidar1m:
    """The benched terrain at 1 m with 2 cm of lidar-like noise."""
    rng = np.random.default_rng(abs(hash(tuple(grid.to_dict().values()))) % 2**32)
    u, v = _uv(grid)
    z = benched(u, v) + rng.normal(0, 0.02, grid.shape).astype(np.float32)
    return lidar.Lidar1m(grid, z, 1.0, ["synthetic 1 m"])


# ---- vectors -------------------------------------------------------------------------------------------------


def _line(*pts: tuple[float, float]) -> BaseGeometry:
    return geom(LineString(pts))


def fetch_water(_lb: Any) -> Water:
    return Water(
        points=[Feature(geom(Point(-200, 100)), {"ftype": 458, "fcode": 45800, "gnis_name": None})],
        flowlines=[
            Feature(_line((350, -2500), (350, 2500)), {"fcode": 46006, "ftype": 460}),
            Feature(_line((-350, -400), (350, -400)), {"fcode": 46003, "ftype": 460}),
        ],
        waterbodies=[
            Feature(geom(box(380, -300, 500, -180)), {"ftype": 390, "fcode": 39004}),
            Feature(geom(box(-500, -500, -470, -470)), {"ftype": 466, "fcode": 46600}),
        ],
    )


FOREST_ROAD = ((-500, -700), (-500, 500))


def fetch_osm(_lb: Any) -> list[Feature]:
    return [
        Feature(_line((-3000, -700), (3000, -700)), {"highway": "secondary"}),
        Feature(_line(*FOREST_ROAD), {"highway": "track"}),
        Feature(_line((-100, 200), (300, 600)), {"highway": "track"}),  # gated: not on the MVUM
        Feature(_line((-500, 0), (0, -50)), {"highway": "path"}),
        Feature(_line((-50, 200), (-50, 500)), {"barrier": "fence"}),
        Feature(_line((550, -700), (560, -560)), {"highway": "service", "service": "driveway"}),
        Feature(_line((600, -700), (600, -500)), {"highway": "residential", "surface": "asphalt"}),
    ]


def fetch_osm_points(_lb: Any) -> list[Feature]:
    return [Feature(geom(Point(-500, -680)), {"highway": "trailhead"})]  # where the forest road leaves the highway


def fetch_mvum(_lb: Any) -> list[Feature]:
    props = {"passengervehicle": "open", "passengervehicle_datesopen": "05/01-11/30", "name": "FR 100"}
    return [Feature(_line(*FOREST_ROAD), props)]


def fetch_public_land(_lb: Any) -> list[Feature]:
    forest = {"unit_nm": "Test National Forest", "mngnm_desc": "Forest Service", "pub_access": "OA"}
    state = {"unit_nm": "Test State Land", "mngnm_desc": "State Department of Natural Resources", "pub_access": "RA"}
    return [
        Feature(geom(box(-3000, -3000, 250, 3000)), forest),
        Feature(geom(box(250, 300, 3000, 3000)), state),
    ]


HOMESTEAD = [(560.0, -600.0), (575.0, -610.0), (580.0, -590.0), (565.0, -585.0)]


def fetch_buildings(_lb: Any) -> list[tuple[float, float, float]]:
    return [(*lonlat(u, v), 150.0) for u, v in [*HOMESTEAD, (-550.0, 550.0)]]


def _period(from_deg: float, hours: int = 200) -> PeriodWind:
    return PeriodWind(
        from_deg=from_deg,
        from_compass=weather.compass(from_deg),
        consistency=0.6,
        speed_ms=3.0,
        hours=hours,
        high_pressure_share=0.5,
        rose=[1 / 16] * 16,
    )


def wind(from_deg: float = 270.0, source: str = "synthetic") -> Wind:
    return Wind(
        from_deg=from_deg,
        from_compass=weather.compass(from_deg),
        consistency=0.6,
        most_common_from=None if source == "set by user" else weather.compass(from_deg),
        source=source,
        all_days_from_compass="WSW",
        all_days_consistency=0.4,
        speed_ms=3.0,
        day_from_deg=250.0,
        day_from_compass="WSW",
        day_consistency=0.5,
        dawn=_period(from_deg),
        dusk=_period(from_deg),
        day=_period(250.0),
        ground=GroundWind(
            from_compass="N",
            from_deg=0.0,
            consistency=0.3,
            most_common_from="NNE",
            dawn_from_compass="N",
            dusk_from_compass="NNW",
            source=weather.GROUND_SOURCE,
        ),
    )


def prevailing(_lat: float, _lon: float, _month: int, override_from_deg: float | None = None) -> Wind:
    if override_from_deg is not None:
        return wind(override_from_deg % 360, "set by user")
    return wind()


def fetch_snow_depth(lb: Any, _month: int) -> tuple[Floats, Affine]:
    """Winter snow (cm) on a 0.005 deg lon/lat grid over the bounds: 10 cm in the west, 70 cm in the east."""
    w, _s, e, n = lb
    cols = max(2, math.ceil((e - w) / 0.005))
    a = np.full((max(2, math.ceil((n - _s) / 0.005)), cols), 10.0, "float32")
    a[:, cols // 2 :] = 70.0
    return a, Affine(0.005, 0, w, 0, -0.005, n)


def fetch_phs_winter_range(_lb: Any) -> list[Feature]:
    return [Feature(geom(box(250, -3000, 3000, 3000)), {"occurrence_name": "Mule Deer", "notes": "WINTER RANGE"})]


def install(mp: pytest.MonkeyPatch) -> None:
    """Serve every data source from the synthetic landscape."""
    mp.setattr(dem, "fetch_dem", fetch_dem)
    mp.setattr(canopy, "fetch_canopy", fetch_canopy)
    mp.setattr(lidar, "fetch_lidar_1m", fetch_lidar_1m)
    mp.setattr(vector, "fetch_water", fetch_water)
    mp.setattr(vector, "fetch_osm", fetch_osm)
    mp.setattr(vector, "fetch_osm_points", fetch_osm_points)
    mp.setattr(vector, "fetch_mvum", fetch_mvum)
    mp.setattr(vector, "fetch_public_land", fetch_public_land)
    mp.setattr(buildings, "fetch_buildings", fetch_buildings)
    mp.setattr(weather, "prevailing", prevailing)
    mp.setattr(snow, "fetch_snow_depth", fetch_snow_depth)
    mp.setattr(vector, "fetch_phs_winter_range", fetch_phs_winter_range)


@contextmanager
def offline() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as mp:
        install(mp)
        yield mp
