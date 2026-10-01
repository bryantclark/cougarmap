"""The data sources and the HTTP layer, with the network faked: retries, the disk cache, and parsing of every
service's answers (Open-Meteo, ArcGIS (NHD, PAD-US, MVUM), Overpass, USGS 3DEP, Meta canopy, Microsoft
buildings, Nominatim, iNaturalist)."""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import math
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import requests
from shapely.geometry import LineString, Point, Polygon, box

import synthetic
import toys
from cougarmap import aoi, context, net, scout
from cougarmap.config import Options
from cougarmap.grid import Grid
from cougarmap.sources import buildings, canopy, dem, snow, vector, weather

Handler = Callable[[str, dict[str, Any]], Any]  # (url, params or form data) -> JSON, text, bytes, or an int status


class FakeResponse:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.status_code = status
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> Any:
        return json.loads(self._body) if isinstance(self._body, str) else self._body

    @property
    def text(self) -> str:
        return str(self._body)

    @property
    def content(self) -> bytes:
        return bytes(self._body)


class FakeSession:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[str] = []

    def _answer(self, url: str, args: dict[str, Any] | None) -> FakeResponse:
        self.calls.append(url)
        body = self.handler(url, args or {})
        return FakeResponse(None, body) if isinstance(body, int) else FakeResponse(body)

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float = 0) -> FakeResponse:
        return self._answer(url, params)

    def post(self, url: str, data: dict[str, Any] | None = None, timeout: float = 0) -> FakeResponse:
        return self._answer(url, data)


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[Handler], FakeSession]:
    """Serve HTTP from a handler, with an empty cache and no retry sleeps."""
    monkeypatch.setattr(net, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    def install(handler: Handler) -> FakeSession:
        s = FakeSession(handler)
        monkeypatch.setattr(net, "session", lambda: s)
        return s

    return install


# ---- net ------------------------------------------------------------------------------------------------------


def test_get_retries_then_gives_up(web: Callable[[Handler], FakeSession]) -> None:
    answers = iter([503, 429, {"ok": 1}])
    s = web(lambda url, p: next(answers))
    assert net.get_json("https://x.test/a", params={"q": 1}) == {"ok": 1} and len(s.calls) == 3
    web(lambda url, p: 500)
    with pytest.raises(RuntimeError, match="request failed after 2 tries"):
        net.get("https://x.test/b", retries=2)
    web(lambda url, p: 404)
    with pytest.raises(RuntimeError, match="404"):
        net.get("https://x.test/c", data={"a": 1}, retries=1)


def test_session_is_per_thread() -> None:
    from concurrent.futures import ThreadPoolExecutor

    main = net.session()
    assert net.session() is main and str(main.headers["User-Agent"]).startswith("cougarmap/")
    with ThreadPoolExecutor(1) as ex:
        assert ex.submit(net.session).result() is not main


def test_cache_stores_reloads_and_rebuilds_corrupt_entries(web: Callable[[Handler], FakeSession]) -> None:
    built: list[int] = []

    def build() -> dict[str, int]:
        built.append(1)
        return {"v": len(built)}

    assert net.cached("ns", {"a": 1}, build) == {"v": 1}
    assert net.cached("ns", {"a": 1}, build) == {"v": 1} and len(built) == 1
    net._key("ns", {"a": 1}).write_bytes(b"garbage")
    assert net.cached("ns", {"a": 1}, build) == {"v": 2}


# ---- weather --------------------------------------------------------------------------------------------------


def _hourly(year: int = 2025) -> dict[str, Any]:
    """A year of hourly weather: wind from the west at dawn and dusk on high-pressure days, from the south by
    day; rain every fifth day."""
    t0 = dt.datetime(year, 1, 1)
    hours = [t0 + dt.timedelta(hours=i) for i in range(365 * 24)]
    days = sorted({h.date() for h in hours})
    wd, p, rain = [], [], []
    for i, h in enumerate(hours):
        dawn_dusk = h.hour in (6, 7, 8, 17, 18, 19)
        wd.append(270.0 + (i % 7) - 3 if dawn_dusk else 180.0)
        p.append(1020.0 if h.timetuple().tm_yday % 2 else 1005.0)
        rain.append(1.0 if h.timetuple().tm_yday % 5 == 0 else 0.0)
    return dict(
        hourly=dict(
            time=[h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            wind_direction_850hPa=wd,
            wind_direction_10m=[(d + 90) % 360 for d in wd],
            wind_speed_850hPa=[8.0] * len(hours),
            wind_speed_10m=[2.0] * len(hours),
            pressure_msl=p,
            precipitation=rain,
        ),
        daily=dict(
            sunrise=[f"{d}T07:00" for d in days],
            sunset=[f"{d}T18:00" for d in days],
        ),
    )


def test_prevailing_high_pressure_dawn_dusk_wind(web: Callable[[Handler], FakeSession]) -> None:
    year, asked = _hourly(), set()

    def answer(_url: str, p: dict[str, Any]) -> dict[str, Any]:
        asked.add(p["hourly"].split(",")[0])
        return year

    s = web(answer)
    w = weather.prevailing(47.9, -117.6, 10)
    assert w["from_compass"] == "W" and abs(w["from_deg"] - 270) < 2 and w["consistency"] > 0.9
    assert w["day_from_compass"] == "S" and w["most_common_from"] == "W" and "HRRR 850 hPa" in w["source"]
    g = w["ground"]  # the 10 m wind: 90 deg on from the 850 hPa one in this fake
    assert g["from_compass"] == g["most_common_from"] == g["dawn_from_compass"] == "N" and g["consistency"] > 0.9
    assert w["dawn"]["hours"] > 0 and 0 < w["dawn"]["high_pressure_share"] < 1 and len(w["dawn"]["rose"]) == 16
    assert {u.split("/v1/")[0] for u in s.calls} == {"https://historical-forecast-api.open-meteo.com"}
    assert asked == {"wind_direction_850hPa", "wind_direction_10m"}
    over = weather.prevailing(47.9, -117.6, 10, override_from_deg=-45)
    assert over["from_deg"] == 315 and over["source"] == "set by user" and over["consistency"] == 0.7
    assert over["most_common_from"] is None and weather.label(over) == "NW (set by user)"
    assert len(s.calls) == 2  # the second call came from the cache


def test_wind_labels_say_how_far_to_trust_the_direction() -> None:
    w = synthetic.wind(306.0, weather.UPPER_SOURCE)
    w["consistency"], w["most_common_from"] = 0.26, "NE"
    assert weather.label(w) == "NW (HRRR 850 hPa; consistency R 0.26, most common NE)"
    d = weather.describe(w)
    assert d["prevailing_from"] == "NW" and d["most_common_from"] == "NE" and d["ground_level"]["from_compass"] == "N"
    old = cast("dict[str, Any]", synthetic.wind())  # a state saved before the ground-level wind
    del old["ground"], old["most_common_from"]
    old["surface_dawn_from_compass"] = "SSW"
    d = weather.describe(cast("weather.Wind", old))
    assert d["valley_dawn_from"] == "SSW" and "ground_level" not in d and d["most_common_from"] is None
    assert weather.most_common([0.0] * 16) is None


def test_circular_mean_without_data() -> None:
    d, r = weather._circ(np.array([]))
    assert math.isnan(d) and r == 0


# ---- ArcGIS services, Overpass ------------------------------------------------------------------------------------


def _geojson_feature(geom: dict[str, Any], **props: Any) -> dict[str, Any]:
    return dict(type="Feature", geometry=geom, properties=props)


def test_arcgis_query_pages_and_skips_bad_geometries(web: Callable[[Handler], FakeSession]) -> None:
    pt = dict(type="Point", coordinates=[-117.6, 47.9])
    pages = {
        0: [_geojson_feature(pt, FCode=1), _geojson_feature(pt, FCode=2)],
        2: [_geojson_feature(pt, FCode=3), _geojson_feature(dict(type="Polygon", coordinates=[[[0, 0]]]), FCode=4)],
        4: [],
    }
    web(lambda url, p: dict(type="FeatureCollection", features=pages[p["resultOffset"]]))
    feats = vector.arcgis_query("https://arc.test/0", (-118.0, 47.0, -117.0, 48.0), page=2)
    assert [f.props["fcode"] for f in feats] == [1, 2, 3] and isinstance(feats[0].geom, Point)
    web(lambda url, p: dict(error=dict(code=400, message="bad query")))
    with pytest.raises(RuntimeError, match="bad query"):
        vector.arcgis_query("https://arc.test/0", (0.0, 0.0, 1.0, 1.0))


def test_tiled_query_splits_failing_tiles_and_dedupes(
    web: Callable[[Handler], FakeSession], capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(url: str, p: dict[str, Any]) -> Any:
        w, s, e, n = map(float, p["geometry"].split(","))
        if e - w > 0.05:
            return 500  # big tiles time out; their quarters answer
        line = dict(type="LineString", coordinates=[[w, s], [e, n]])
        return dict(features=[_geojson_feature(line, permanent_identifier="same"), _geojson_feature(line, x=w)])

    web(handler)
    feats = vector.arcgis_query_tiled(
        "https://arc.test/6", (0.0, 0.0, 0.08, 0.08), "fcode", 0.08, "permanent_identifier"
    )
    assert len({f.props.get("permanent_identifier") for f in feats}) == 2 and len(feats) == 5  # 1 shared + 4 tiles
    web(lambda url, p: 500)
    assert vector.arcgis_query_tiled("https://arc.test/6", (0.0, 0.0, 0.01, 0.01), "fcode", 0.05) == []
    assert "could not be fetched" in capsys.readouterr().err


def test_water_land_and_forest_roads(web: Callable[[Handler], FakeSession]) -> None:
    poly = dict(type="Polygon", coordinates=[[[-117.61, 47.89], [-117.59, 47.89], [-117.59, 47.91], [-117.61, 47.89]]])

    def handler(url: str, p: dict[str, Any]) -> Any:
        if "PADUS" in url:
            return dict(features=[_geojson_feature(poly, Unit_Nm="Test NF", Pub_Access="OA")])
        if "MVUM" in url:
            line = dict(type="LineString", coordinates=[[-117.6, 47.9], [-117.59, 47.91]])
            return dict(features=[_geojson_feature(line, passengervehicle="open")])
        layer = url.rsplit("/", 2)[-2]
        point = dict(type="Point", coordinates=[-117.6, 47.9])
        return dict(features=[_geojson_feature(point if layer == "0" else poly, fcode=int(layer))])

    web(handler)
    lb = (-117.62, 47.88, -117.58, 47.92)
    w = vector.fetch_water(lb)
    assert set(w) == {"points", "flowlines", "waterbodies"} and w["points"][0].props["fcode"] == 0
    land = vector.fetch_public_land(lb)
    assert land[0].props == {"unit_nm": "Test NF", "pub_access": "OA"} and isinstance(land[0].geom, Polygon)
    assert vector.fetch_mvum(lb)[0].props["passengervehicle"] == "open"


def test_overpass_falls_back_to_the_mirror_and_parses_ways(web: Callable[[Handler], FakeSession]) -> None:
    way = dict(
        type="way", tags={"highway": "track"}, geometry=[dict(lat=47.9, lon=-117.6), dict(lat=47.91, lon=-117.6)]
    )
    stub = dict(type="way", tags={}, geometry=[dict(lat=47.9, lon=-117.6)])  # one node: not a line
    s = web(lambda url, p: 504 if "overpass-api.de" in url else dict(elements=[way, stub]))
    feats = vector.fetch_osm((-117.62, 47.88, -117.58, 47.92))
    assert len(feats) == 1 and isinstance(feats[0].geom, LineString) and feats[0].props == {"highway": "track"}
    assert any("kumi" in u for u in s.calls)
    web(lambda url, p: 500)
    with pytest.raises(RuntimeError, match="overpass failed"):
        vector.overpass_json("[out:json];")


def test_recreation_points_from_nodes_and_way_centres(web: Callable[[Handler], FakeSession]) -> None:
    node = dict(type="node", lat=47.9, lon=-117.6, tags={"highway": "trailhead"})
    way = dict(type="way", center=dict(lat=47.91, lon=-117.61), tags={"tourism": "camp_site"})
    bare = dict(type="relation", tags={"amenity": "parking"})  # no centre given: skipped
    queries: list[str] = []

    def handler(url: str, p: dict[str, Any]) -> Any:
        queries.append(p["data"])
        return dict(elements=[node, way, bare])

    web(handler)
    pts = vector.fetch_osm_points((-117.62, 47.88, -117.58, 47.92))
    assert [p.geom.coords[0] for p in pts] == [(-117.6, 47.9), (-117.61, 47.91)]
    assert pts[1].props == {"tourism": "camp_site"}
    assert all(t in queries[0] for t in ("trailhead", "camp_site", "parking", "out tags center"))


def _snodas_tar(grid: np.ndarray) -> bytes:
    """A tiny SNODAS-like archive: the depth grid (gzipped, int16 big-endian) and another member."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tf:
        for name, raw in (
            ("us_ssmv11034tS__T0001TTNATS2025011505HP001.dat.gz", b"swe"),
            ("us_ssmv11036tS__T0001TTNATS2025011505HP001.dat.gz", grid.astype(">i2").tobytes()),
        ):
            data = gzip.compress(raw)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return bio.getvalue()


def test_snodas_member_decode_and_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snow, "NROWS", 6)
    monkeypatch.setattr(snow, "NCOLS", 8)
    grid = np.arange(48, dtype=np.int16).reshape(6, 8) * 10
    grid[0, 0] = -9999
    a = snow.decode(snow.depth_member(_snodas_tar(grid)))
    assert np.array_equal(a, grid)
    r = snow.RES_DEG
    # a box over the north-west corner, partly off the grid: clipped there
    lb = (snow.ULX - 3 * r, snow.ULY - 2.5 * r, snow.ULX + 1.5 * r, snow.ULY + 1 * r)
    cm, t = snow.crop(a, lb, pad_deg=0.0)
    assert cm.shape == (3, 2) and np.isnan(cm[0, 0]) and cm[1, 1] == pytest.approx(9.0)  # mm -> cm
    assert t.c == pytest.approx(snow.ULX) and t.f == pytest.approx(snow.ULY)
    off, _ = snow.crop(a, (0.0, 0.0, 1.0, 1.0))  # far outside the grid
    assert off.size == 0
    with pytest.raises(RuntimeError, match="no snow-depth grid"):
        snow.depth_member(_snodas_tar_without_depth())


def _snodas_tar_without_depth() -> bytes:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w"):
        pass
    return bio.getvalue()


def test_snow_depth_median_over_the_years(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snow, "NROWS", 6)
    monkeypatch.setattr(snow, "NCOLS", 8)
    days = snow.dates(1)
    assert len(days) == 5 and all(d.month == 1 and d.day == 15 for d in days)
    assert snow.dates(1, today=dt.date(2026, 10, 1))[0] == dt.date(2026, 1, 15)
    assert snow.dates(11, today=dt.date(2026, 10, 1))[0] == dt.date(2025, 11, 15)
    depth = {
        d: gzip.compress((np.full((6, 8), 100 * (i + 1), np.int16)).astype(">i2").tobytes()) for i, d in enumerate(days)
    }

    def day(d: dt.date) -> bytes:
        if d == days[0]:
            raise RuntimeError("a gap in the archive")
        return depth[d]

    monkeypatch.setattr(snow, "_day", day)
    r = snow.RES_DEG
    lb = (snow.ULX + r, snow.ULY - 3 * r, snow.ULX + 3 * r, snow.ULY - r)
    med, _ = snow.fetch_snow_depth(lb, 1)
    assert np.allclose(med, 35.0)  # median of 20, 30, 40, 50 cm: the missing year is skipped
    monkeypatch.setattr(snow, "_day", lambda d: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError, match="no SNODAS snow depth"):
        snow.fetch_snow_depth(lb, 1)
    monkeypatch.setattr(snow, "_day", lambda d: depth[days[1]])
    with pytest.raises(RuntimeError, match="outside the SNODAS grid"):
        snow.fetch_snow_depth((0.0, 0.0, 1.0, 1.0), 1)


def test_snodas_day_downloads_once(web: Callable[[Handler], FakeSession]) -> None:
    tar = _snodas_tar(np.zeros((2, 2), np.int16))
    s = web(lambda url, p: tar)
    d = dt.date(2025, 1, 15)
    assert snow._day(d) == snow._day(d)
    assert s.calls == ["https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2025/01_Jan/SNODAS_20250115.tar"]


def test_winter_range_keeps_deer_and_elk_winter_polygons(web: Callable[[Handler], FakeSession]) -> None:
    poly = {"type": "Polygon", "coordinates": [[[-117.6, 47.9], [-117.59, 47.9], [-117.59, 47.91], [-117.6, 47.9]]]}

    def feat(**props: Any) -> dict[str, Any]:
        return dict(type="Feature", geometry=poly, properties=props)

    rows = [
        feat(Occurrence_Name="Mule Deer", Notes="WINTER RANGE"),
        feat(Occurrence_Name="Rocky Mountain Elk", PriorityArea_Desc="Regular Concentration"),
        feat(Occurrence_Name="Bald Eagle", Notes="WINTER ROOST"),
        feat(Occurrence_Name="White-tailed Deer", Notes="fawning area"),
    ]
    web(lambda url, p: dict(type="FeatureCollection", features=rows))
    got = vector.fetch_phs_winter_range((-117.62, 47.88, -117.58, 47.92))
    assert [f.props["occurrence_name"] for f in got] == ["Mule Deer", "Rocky Mountain Elk"]


@pytest.mark.parametrize(
    ("month", "snow_wanted", "phs_wanted"), [(10, False, False), (4, True, False), (1, True, True)]
)
def test_winter_inputs_are_fetched_only_in_winter(month: int, snow_wanted: bool, phs_wanted: bool) -> None:
    calls: list[str] = []

    def snow_depth(lb: Any, m: int) -> Any:
        calls.append("snow")
        return synthetic.fetch_snow_depth(lb, m)

    def phs(_lb: Any) -> list[Any]:
        calls.append("phs")
        return []

    with synthetic.offline() as mp:
        mp.setattr(snow, "fetch_snow_depth", snow_depth)
        mp.setattr(vector, "fetch_phs_winter_range", phs)
        area = aoi.AOI("syn", box(*_syn_bounds()))
        opts = Options(month=month)
        d = context.fetch(area, opts, *context.grids(area, opts), log=lambda *_: None)
    assert ("snow" in calls) == snow_wanted and ("phs" in calls) == phs_wanted
    assert ("snow" in d) == snow_wanted


def _syn_bounds() -> tuple[float, float, float, float]:
    w, s, e, n = synthetic.bbox()
    return w, s, e, n


def test_a_failed_snow_download_is_noted(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(_lb: Any, _m: int) -> Any:
        raise RuntimeError("NSIDC down")

    with synthetic.offline() as mp:
        mp.setattr(snow, "fetch_snow_depth", down)
        ctx = context.build(aoi.AOI("syn", box(*_syn_bounds())), Options(month=1), log=lambda *_: None)
    assert ctx.snow_cm is None and any("SNODAS" in n for n in ctx.notes)


def test_road_classification_and_mvum_seasons() -> None:
    for hw in vector.DRIVABLE:
        assert vector.osm_is_drivable({"highway": hw}) == "yes"
    assert vector.osm_is_drivable({"highway": "service", "service": "driveway"}) is None
    assert vector.osm_is_drivable({"highway": "track", "motor_vehicle": "no"}) is None
    assert vector.mvum_open_in_month({"highclearancevehicle": "open"}, 2)  # no dates: open all year
    assert vector.mvum_open_in_month({"passengervehicle": "open", "passengervehicle_datesopen": "garbage"}, 2)
    assert not vector.mvum_open_in_month({"passengervehicle": "open", "passengervehicle_datesopen": "06/01-09/30"}, 2)


# ---- rasters: 3DEP elevation, canopy height ----------------------------------------------------------------------


def _tif(tmp_path: Path, name: str, grid: Grid, values: np.ndarray, nodata: float = np.nan) -> str:
    p = tmp_path / name
    grid.write_tif(p, values, nodata=nodata)
    return str(p)


@pytest.fixture
def local_rasters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid.read_raster reads local files for "/vsicurl/<path>" URLs."""
    real = Grid.read_raster

    def read(self: Grid, path: str, *a: Any, **k: Any) -> Any:
        return real(self, path.removeprefix("/vsicurl/"), *a, **k)

    monkeypatch.setattr(Grid, "read_raster", read)


def test_dem_merges_lidar_then_fills_with_the_seamless_dem(
    web: Callable[[Handler], FakeSession], local_rasters: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = toys.grid(60, 3.0)
    src = Grid(g.epsg, g.x0 - 30, g.y0 + 30, 1.5, 160, 160)  # 1.5 m source tiles covering the grid
    lidar = np.full(src.shape, 100.0, "float32")
    lidar[:, 80:] = np.nan  # east half not flown
    newer = _tif(tmp_path, "new.tif", src, lidar)
    older = _tif(tmp_path, "old.tif", src, np.full(src.shape, 50.0, "float32"))
    coarse = Grid(g.epsg, g.x0 - 100, g.y0 + 100, 10.0, 60, 60)
    seamless = _tif(tmp_path, "seamless.tif", coarse, np.full(coarse.shape, 7.0, "float32"))
    items = [
        dict(title="old", downloadURL=older, publicationDate="2019-01-01"),
        dict(title="new", downloadURL=newer, publicationDate="2023-01-01"),
        dict(title="laz", downloadURL="x.laz"),
        dict(title="missing", downloadURL=str(tmp_path / "missing.tif"), publicationDate="2024-01-01"),
    ]
    web(lambda url, p: dict(items=items[p["offset"] :][:2], total=len(items)))
    assert [t["title"] for t in dem.lidar_tiles(g.lonlat_bounds())] == ["missing", "new", "old"]
    z, info = dem.fetch_dem(g)
    assert z[30, 5] == 100 and z[30, 55] == 50  # newest tile first, the older one where it has no data
    assert info["lidar_fraction"] == 1.0 and info["sources"] == ["new", "old"]

    monkeypatch.setattr(dem, "_seamless_tiles", lambda lb, product: [str(tmp_path / "nope.tif"), seamless])
    z10, info10 = dem.fetch_dem(toys.grid(30, 10.0))  # 10 m: no lidar, the 1/3 arc-second DEM
    assert (z10 == 7).all() and info10 == dict(sources=["seamless.tif"], lidar_fraction=0.0)


def test_dem_fills_small_holes_and_fails_without_data(
    web: Callable[[Handler], FakeSession], local_rasters: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seamless_tiles = dem._seamless_tiles
    g = toys.grid(20, 10.0)
    vals = np.full((30, 30), 3.0, "float32")
    vals[10:20, 10:20] = np.nan
    src = Grid(g.epsg, g.x0 - 50, g.y0 + 50, 10.0, 30, 30)
    t = _tif(tmp_path, "holes.tif", src, vals)
    monkeypatch.setattr(dem, "_seamless_tiles", lambda lb, product: [t])
    z, _ = dem.fetch_dem(g, allow_lidar=False)
    assert np.isfinite(z).all() and (z == 3).all()
    monkeypatch.setattr(dem, "_seamless_tiles", lambda lb, product: [])
    with pytest.raises(RuntimeError, match="no elevation data"):
        dem.fetch_dem(toys.grid(21, 10.0), allow_lidar=False)
    names = seamless_tiles((-117.7, 47.8, -117.5, 48.1), "13")
    assert names[0].endswith("USGS_13_n48w118.tif") and len(names) == 2
    assert seamless_tiles((10.2, 47.1, 10.3, 47.2), "1")[0].endswith("n48e010.tif")


def test_canopy_merges_tiles_and_treats_gaps_as_open(
    web: Callable[[Handler], FakeSession], local_rasters: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile_urls = canopy.tile_urls
    g = toys.grid(40, 5.0)
    src = Grid(g.epsg, g.x0 - 20, g.y0 + 20, 1.0, 120, 240)
    vals = np.full(src.shape, 12, "uint8")
    vals[:, 120:] = 255  # nodata
    t = _tif(tmp_path, "chm.tif", src, vals, nodata=255)
    monkeypatch.setattr(canopy, "tile_urls", lambda lb: [t, str(tmp_path / "missing.tif")])
    chm = canopy.fetch_canopy(g)
    assert chm[20, 5] == pytest.approx(12) and chm[20, 35] == 0
    assert tile_urls((-117.8, 48.85, -117.79, 48.86))[0].endswith("/0212310000.tif")


def test_building_centroids_and_footprints_inside_the_bounds(web: Callable[[Handler], FakeSession]) -> None:
    lines = [
        json.dumps(dict(geometry=dict(coordinates=[[[-117.600, 47.900], [-117.599, 47.900], [-117.599, 47.901]]]))),
        json.dumps(dict(geometry=dict(coordinates=[[[-110.0, 40.0], [-110.0, 40.001], [-110.001, 40.0]]]))),
    ]
    tile = gzip.compress("\n".join(lines).encode())
    lb = (-117.61, 47.89, -117.59, 47.91)
    x, y = canopy._tile_xy(-117.6, 47.9, buildings.LEVEL)
    qk = canopy.quadkey(x, y, buildings.LEVEL)
    web(lambda url, p: f"QuadKey,Url\n{qk},https://b.test/t.gz\n" if url == buildings.INDEX else tile)
    pts = buildings.fetch_buildings(lb)
    assert len(pts) == 1 and pts[0][:2] == pytest.approx((-117.59933, 47.90033), abs=1e-4)
    assert pts[0][2] == pytest.approx(0.5 * 74.6 * 110.5, rel=0.01)  # a right triangle 0.001 deg on a side, in m2


# ---- places, sightings --------------------------------------------------------------------------------------------


def test_geocode_and_parse_location(web: Callable[[Handler], FakeSession]) -> None:
    found = [dict(lat="47.9", lon="-117.6", display_name="Testville, WA")]
    web(lambda url, p: found if p["q"] == "Testville" else [])
    assert aoi.parse_location("Testville") == (47.9, -117.6, "Testville, WA")
    with pytest.raises(ValueError, match="could not find"):
        aoi.geocode("nowhere")


def test_inat_sightings_and_scout_roads(web: Callable[[Handler], FakeSession]) -> None:
    obs = [dict(geojson=dict(coordinates=[-117.6, 47.9])), dict(geojson=None)]
    way = dict(tags={"highway": "track"}, geometry=[dict(lat=47.9, lon=-117.6), dict(lat=47.91, lon=-117.6)])
    web(lambda url, p: dict(results=obs) if "inaturalist" in url else dict(elements=[way]))
    assert scout._inat_cougars((-117.7, 47.8, -117.5, 48.0)) == [(47.9, -117.6)]
    roads = scout._osm_roads((-117.7, 47.8, -117.2, 48.0))  # two 0.25 deg tiles
    assert len(roads) == 2 and roads[0].props == {"highway": "track"}
    web(lambda url, p: 500)
    assert scout._inat_cougars((-117.0, 47.0, -116.9, 47.1)) == []
    assert scout._osm_roads((-117.0, 47.0, -116.9, 47.1)) == []
