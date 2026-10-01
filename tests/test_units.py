from pathlib import Path

import numpy as np
import pytest

from cougarmap import aoi
from cougarmap.corridor import funnel_score
from cougarmap.sources import canopy, vector
from cougarmap.sources.weather import _circ, compass

KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>t</name>
<Folder><name>Areas</name>
<Placemark><name>Test Area</name><Polygon><outerBoundaryIs><LinearRing><coordinates>
-117.62,47.90,0 -117.60,47.90,0 -117.60,47.92,0 -117.62,47.92,0 -117.62,47.90,0
</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
<Placemark><name>Cam01</name><description>base of a cliff</description>
<Point><coordinates>-117.61,47.91,0</coordinates></Point></Placemark>
<Placemark><name>Seasonal Water</name><Point><coordinates>-117.611,47.911,0</coordinates></Point></Placemark>
<Placemark><name>Scrape</name><Point><coordinates>-117.612,47.912,0</coordinates></Point></Placemark>
</Folder></Document></kml>"""


def test_kml_areas_and_points(tmp_path: Path) -> None:
    p = tmp_path / "a.kml"
    p.write_text(KML)
    a = aoi.from_kml(p, "test")
    assert a.name == "Test Area"
    assert 3 < a.area_km2() < 4
    kinds = {q["name"]: q["kind"] for q in a.user_points}
    assert kinds == {"Cam01": "camera", "Seasonal Water": "seasonal_water", "Scrape": "sign"}


def test_circle_area() -> None:
    a = aoi.circle(47.9, -117.6, 2.0)
    assert abs(a.area_km2() - np.pi * 4) < 0.1


def test_parse_location_coords() -> None:
    lat, lon, _ = aoi.parse_location("47.92, -117.58")
    assert (lat, lon) == (47.92, -117.58)
    assert aoi.parse_location("47.3712 -116.1029")[2] == "47.3712,-116.1029"
    assert aoi.parse_coords("47.37N, 116.10W") == (47.37, -116.1, None)
    assert aoi.parse_coords("-33.9, 151.2") == (-33.9, 151.2, None)  # southern hemisphere: left alone
    assert aoi.parse_coords("Missoula, MT") is None
    lat, lon, note = aoi.parse_coords("47.3712, 116.1029") or (0, 0, None)  # minus sign lost: made west
    assert (lat, lon) == (47.3712, -116.1029) and note and "west" in note


def test_circular_mean_wraps_north() -> None:
    d, R = _circ(np.array([350.0, 10.0]))
    assert min(d, 360 - d) < 1e-6 and R > 0.98
    assert compass(270) == "W" and compass(359) == "N" and compass(22) == "NNE"


def test_mvum_season_parsing() -> None:
    p = dict(passengervehicle="open", passengervehicle_datesopen="04/01-11/30")
    assert vector.mvum_open_in_month(p, 6)
    assert not vector.mvum_open_in_month(p, 1)
    winter = dict(passengervehicle="open", passengervehicle_datesopen="12/01-03/31")
    assert vector.mvum_open_in_month(winter, 1) and not vector.mvum_open_in_month(winter, 7)
    assert not vector.mvum_open_in_month(dict(passengervehicle="closed"), 6)


def test_osm_drivable() -> None:
    assert vector.osm_is_drivable({"highway": "tertiary"}) == "yes"
    assert vector.osm_is_drivable({"highway": "track"}) == "maybe"
    assert vector.osm_is_drivable({"highway": "track", "access": "private"}) is None
    assert vector.osm_is_drivable({"highway": "path"}) is None


def test_quadkey_matches_bucket_tile() -> None:
    x, y = canopy._tile_xy(-117.8, 48.85, 10)
    assert canopy.quadkey(x, y, 10) == "0212310000"


def test_funnel_finds_gap_in_wall() -> None:
    R = np.ones((60, 60))
    R[30, :] = 500.0
    R[30, 28:32] = 1.0  # gap
    f = funnel_score(R, 30.0, local_m=900)
    assert f[30, 29] > 0.5
    assert f[10, 10] < 0.2


def test_kmz_paths_and_errors(tmp_path: Path) -> None:
    import zipfile

    ring = " ".join(f"{-117.6 + 0.01 * np.cos(a)},{47.9 + 0.01 * np.sin(a)},0" for a in np.linspace(0, 6.25, 30))
    kml = KML.replace(
        "</Folder>",
        f"<Placemark><name>Drawn Path</name><LineString><coordinates>{ring}"
        "</coordinates></LineString></Placemark><Placemark><name>Road</name><LineString><coordinates>"
        "-117.6,47.9,0 -117.5,47.95,0</coordinates></LineString></Placemark></Folder>",
    )
    p = tmp_path / "a.kmz"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("doc.kml", kml)
    names = [a.name for a in aoi.areas_in_kml(p)]
    assert names == ["Test Area", "Drawn Path"]  # a nearly closed path is an area, an open one is not
    assert {pm.name for pm in aoi.read_kml(p)} >= {"Road", "Cam01"}
    both = aoi.from_kml(p)
    assert both.name == "a" and both.area_km2() > 3  # every area, named after the file
    with pytest.raises(ValueError, match="no area named"):
        aoi.from_kml(p, "Elsewhere")
    empty = tmp_path / "e.kml"
    empty.write_text(KML.split("<Folder>", maxsplit=1)[0] + "</Document></kml>")
    with pytest.raises(ValueError, match="no named polygons"):
        aoi.from_kml(empty)


def test_point_kinds_and_default_names() -> None:
    kinds = {n: aoi.point_kind(n) for n in ("Seasonal water", "Spring", "Kill site", "Cam 12", "Trailhead")}
    assert kinds == {
        "Seasonal water": "seasonal_water",
        "Spring": "water",
        "Kill site": "sign",
        "Cam 12": "camera",
        "Trailhead": None,
    }
    assert aoi.circle(47.9, -117.6, 1.5).name == "47.9000,-117.6000 r1.5km"
    assert aoi.bbox(-117.61, 47.89, -117.59, 47.91).name == "bbox -117.610,47.890,-117.590,47.910"
    assert aoi.AOI("x", aoi.bbox(-117.61, 47.89, -117.59, 47.91).geom).centroid == pytest.approx((47.9, -117.6))
