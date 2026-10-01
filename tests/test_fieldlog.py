"""The field log (cougarmap.fieldlog): the version-2 records, migrating version-1 camera results, camera effort,
and reading tracks and routes from GPX/KML/KMZ files."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from cougarmap import fieldlog as fl

V1 = dict(
    logged="2026-09-29T13:00:29",
    name="Cam01",
    lat=48.1,
    lon=-117.9,
    lion_seen=True,
    start="2026-09-26",
    end="2026-09-27",
    detections=2,
    times=[],
    notes="two nights in a row",
)


def _write(path: Path, *recs: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


# ---- migration --------------------------------------------------------------------------------------------------


def _plain(recs: list[fl.Record]) -> list[dict[str, Any]]:
    return [dict(r) for r in recs]


def test_v1_result_becomes_an_unpaired_deployment() -> None:
    taken: set[str] = set()
    dep, check, *events = _plain(fl.migrate(dict(V1), taken))
    assert dep["type"] == "deployment" and dep["v"] == fl.SCHEMA_VERSION
    assert dep["id"] == "Cam01" and dep["arm"] == "unpaired" and dep["zone"] is None
    # a version-1 end is "result through", not a removal: still out, with a check the morning after
    assert (dep["start"], dep["end"]) == ("2026-09-26", None)
    assert check["type"] == "check" and check["date"] == "2026-09-28" and not check["removed"]
    assert [e["species"] for e in events] == ["cougar", "cougar"] and all(e["datetime"] is None for e in events)
    # the same camera logged again is a second deployment; a result with no lion has no events
    again = _plain(fl.migrate(dict(V1, lion_seen=False, detections=0), taken))
    assert [r["type"] for r in again] == ["deployment", "check"] and again[0]["id"] == "Cam01#2"
    timed = _plain(fl.migrate(dict(V1, name=None, times=["2026-09-26T05:10"], detections=None), taken))
    assert timed[0]["id"] == "obs-2026-09-29" and [e["datetime"] for e in timed[2:]] == ["2026-09-26T05:10"]
    undated = _plain(fl.migrate(dict(V1, name="X", start=None, end=None), taken))
    assert [r["type"] for r in undated] == ["deployment", "event", "event"]  # no dates: no known effort
    assert _plain(fl.migrate(dict(timed[0]), taken)) == [timed[0]]  # version 2 passes through
    with pytest.raises(ValueError, match="unknown field-log record version"):
        fl.migrate(dict(type="deployment", v=99), taken)


def test_loading_migrates_in_memory_and_appending_rewrites_once(tmp_path: Path) -> None:
    log = _write(tmp_path / "observations.jsonl", V1)
    before = log.read_text()
    recs = fl.load(log)
    assert [r["type"] for r in recs] == ["deployment", "check", "event", "event"]
    assert log.read_text() == before  # reading never changes the file
    dep = fl.new_deployment(recs, 48.2, -117.8, name="Cam01", arm="model", zone="z1", start="2026-10-01")
    assert dep["id"] == "Cam01#2"
    assert fl.append([dep], log) == 5
    assert (tmp_path / "observations.v1.bak").read_text() == before
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(x["v"] == fl.SCHEMA_VERSION for x in lines) and len(lines) == 5
    assert fl.append([], log) == 5  # already version 2: nothing rewritten
    assert fl.load(tmp_path / "missing.jsonl") == []


def test_v2_ids_win_over_migrated_ones(tmp_path: Path) -> None:
    v2 = fl.new_deployment([], 48.0, -117.0, name="Cam01", start="2026-10-01")
    log = _write(tmp_path / "obs.jsonl", V1, dict(v2))
    ids = [r["id"] for r in fl.load(log) if r["type"] == "deployment"]
    assert ids == ["Cam01#2", "Cam01"]


# ---- deployments, checks, effort --------------------------------------------------------------------------------


def test_deployment_updates_and_validation() -> None:
    d = fl.new_deployment([], 48.0, -117.0, name="A", arm="Control", trail_type="game trail", start="2026-10-01")
    assert d["arm"] == "control" and d["trail_type"] == "game-trail" and d["lure"] is False
    upd = fl.new_deployment([d], None, None, end="2026-12-01", deployment="A")
    merged = fl.deployments([d, upd])["A"]
    assert merged["end"] == "2026-12-01" and merged["lat"] == 48.0 and merged["arm"] == "control"
    with pytest.raises(ValueError, match="arm must be one of"):
        fl.new_deployment([], 48.0, -117.0, arm="random")
    with pytest.raises(ValueError, match="needs lat and lon"):
        fl.new_deployment([], None, None, name="B")
    with pytest.raises(ValueError, match="no camera deployment"):
        fl.new_deployment([d], None, None, deployment="nope")
    with pytest.raises(ValueError):
        fl.new_deployment([], 48.0, -117.0, start="1 Oct")


def test_camera_nights_downtime_and_independent_detections() -> None:
    d = fl.new_deployment([], 48.0, -117.0, name="A", start="2026-11-20", downtime_nights=1)
    recs: list[fl.Record] = [d]
    recs += fl.new_check(
        recs,
        "A",
        "2026-12-10",
        downtime_nights=2,
        events=[
            dict(datetime="2026-11-25T05:00", species="cougar"),
            dict(datetime="2026-11-25T05:20", species="cougar", count=2),  # same visit (< 30 min)
            dict(datetime="2026-11-27T22:00", species="cougar"),
            dict(time="2026-11-26T07:00", species="Deer", count=3),
        ],
    )
    (cam,) = fl.cameras(recs, 30.0)
    assert cam.nights == pytest.approx(20 - 3)
    assert cam.nights_by_month[11] == pytest.approx(11 * 17 / 20) and set(cam.nights_by_month) == {11, 12}
    assert cam.detections == dict(cougar=2, deer=1, elk=0, other=0)
    assert cam.last_check == "2026-12-10" and len(cam.times) == 3
    recs += fl.new_check(recs, "A", "2026-12-20", removed=True)
    assert fl.cameras(recs, 30.0)[0].nights == pytest.approx(30 - 3)
    with pytest.raises(ValueError, match="came down on 2026-12-20"):  # it would add detections with no effort
        fl.new_check(recs, "A", "2026-12-30")
    unchecked = fl.new_deployment([], 48.0, -117.0, name="B")
    assert fl.cameras([unchecked], 30.0)[0].nights == 0
    with pytest.raises(ValueError, match="species must be one of"):
        fl.new_check(recs, "A", events=[dict(species="bear")])
    with pytest.raises(ValueError, match="no camera deployment"):
        fl.new_check(recs, "Z")
    for bad in ("2025-11-25T05:00", "2026-12-21T05:00"):  # before it went out, after the check
        with pytest.raises(ValueError, match="outside camera 'A'"):
            fl.new_check(recs, "A", "2026-12-20", events=[dict(datetime=bad)])
    assert fl.independent([None, None, "2026-01-01T00:00", "2026-01-01T00:40"], 30) == 4


def test_event_times_with_and_without_a_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        d = fl.new_deployment([], 48.0, -117.0, name="A", start="2026-10-01")
        recs: list[fl.Record] = [d]
        recs += fl.new_check(
            recs,
            "A",
            "2026-10-05",
            events=[
                dict(datetime="2026-10-02T05:40"),  # typed by hand: local
                dict(datetime="2026-10-03T05:40-07:00"),  # a phone export with an offset
                dict(datetime="2026-10-03T12:50:30Z"),  # UTC: 05:50:30 local, the same visit as 05:40
            ],
        )
        times = [dict(r)["datetime"] for r in recs[2:]]
        assert times == ["2026-10-02T05:40", "2026-10-03T05:40", "2026-10-03T05:50:30"]
        (cam,) = fl.cameras(recs, 30.0)
        assert cam.detections["cougar"] == 2
        # a log written before times were normalized still loads, mixed formats and all
        old = [*recs[:2], dict(recs[2], datetime="2026-10-03T12:40:00+00:00"), recs[3]]
        assert fl.cameras(cast(list[fl.Record], old), 30.0)[0].detections["cougar"] == 1
        assert fl.normalize_time("2026-10-02") == "2026-10-02"
        with pytest.raises(ValueError):
            fl.new_check(recs, "A", "2026-10-05", events=[dict(datetime="Oct 3 5:40")])
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_checks_after_the_end_are_refused_and_a_deployment_can_reopen() -> None:
    v1 = fl.migrate(dict(V1), set())  # Cam01: result through 2026-09-27, still out
    recs: list[fl.Record] = list(v1)
    assert fl.cameras(recs, 30.0)[0].nights == 2  # "two nights in a row"
    recs += fl.new_check(recs, "Cam01", "2026-10-01", events=[dict(datetime="2026-09-30T22:00")])
    (cam,) = fl.cameras(recs, 30.0)
    assert cam.nights == 5 and cam.detections["cougar"] == 3
    recs.append(fl.new_deployment(recs, None, None, end="2026-10-01", deployment="Cam01"))
    with pytest.raises(ValueError, match="reopen it first"):
        fl.new_check(recs, "Cam01", "2026-10-08")
    recs.append(fl.new_deployment(recs, None, None, end="", deployment="Cam01"))  # still out after all
    assert fl.deployments(recs)["Cam01"]["end"] is None
    recs += fl.new_check(recs, "Cam01", "2026-10-08")
    assert fl.cameras(recs, 30.0)[0].nights == 12
    assert fl.new_deployment([], 48.0, -117.0, name="B", end="")["end"] is None


# ---- tracks, transects, GPS files -------------------------------------------------------------------------------

GPX = """<?xml version="1.0"?>
<gpx version="1.1" creator="phone" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="48.0005" lon="-117.0005"><name>Lion 1</name></wpt>
  <wpt lat="48.0" lon="-117.0"><name>start</name></wpt>
  <wpt lat="48.0002" lon="-117.0"/>
  <trk><trkseg>
    <trkpt lat="48.0000" lon="-117.0000"><time>2026-12-03T08:15:00Z</time></trkpt>
    <trkpt lat="48.0010" lon="-117.0000"><time>2026-12-03T08:20:00Z</time></trkpt>
    <trkpt lat="48.0010" lon="-117.0010"><time>2026-12-03T08:25:00Z</time></trkpt>
  </trkseg></trk>
  <rte><rtept lat="48.1" lon="-117.1"/><rtept lat="48.1" lon="-117.2"/></rte>
</gpx>"""

KML = """<?xml version="1.0"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>
  <Placemark><name>route</name><TimeStamp><when>2026-12-04T09:00:00Z</when></TimeStamp>
    <LineString><coordinates>-117.0,48.0,0 -117.0,48.002,0</coordinates></LineString></Placemark>
  <Placemark><name>lion crossed</name><Point><coordinates>-117.0,48.001,0</coordinates></Point></Placemark>
  <Placemark><name>Parking</name><MultiGeometry><Point><coordinates>-117.0,48.0,0</coordinates></Point>
  </MultiGeometry></Placemark>
  <Placemark><gx:Track><when>2026-12-05T07:00:00Z</when><when>2026-12-05T07:01:00Z</when>
    <gx:coord>-117.1 48.1 900</gx:coord><gx:coord>-117.1 48.101 905</gx:coord></gx:Track></Placemark>
</Document></kml>"""


def test_read_gpx_kml_kmz(tmp_path: Path) -> None:
    g = fl.read_geo(_text(tmp_path / "t.gpx", GPX))
    assert len(g.lines) == 2 and g.lines[0][0] == [48.0, -117.0] and g.points[0] == [48.0005, -117.0005]
    assert g.point_names == ["Lion 1", "start", None]
    assert g.first_time == "2026-12-03T08:15:00Z"
    k = fl.read_geo(_text(tmp_path / "t.kml", KML))
    assert k.lines == [[[48.0, -117.0], [48.002, -117.0]], [[48.1, -117.1], [48.101, -117.1]]]
    assert k.points == [[48.001, -117.0], [48.0, -117.0]] and k.point_names == ["lion crossed", "Parking"]
    assert k.first_time == "2026-12-04T09:00:00Z"
    kmz = tmp_path / "t.kmz"
    with zipfile.ZipFile(kmz, "w") as z:
        z.writestr("doc.kml", KML)
    assert fl.read_geo(kmz).lines == k.lines
    with zipfile.ZipFile(tmp_path / "empty.kmz", "w") as z:
        z.writestr("readme.txt", "nothing")
    with pytest.raises(ValueError, match=r"no \.kml inside"):
        fl.read_geo(tmp_path / "empty.kmz")
    with pytest.raises(ValueError, match="not a GPX or KML"):
        fl.read_geo(_text(tmp_path / "x.xml", "<html/>"))
    with pytest.raises(FileNotFoundError):
        fl.read_geo(tmp_path / "missing.gpx")


def _text(p: Path, s: str) -> Path:
    p.write_text(s)
    return p


def test_tracks(tmp_path: Path) -> None:
    t = fl.new_track([], file=str(_text(tmp_path / "t.gpx", GPX)), snow_age_h=12, confidence="Certain")
    assert t["date"] == "2026-12-03" and t["confidence"] == "certain" and len(t["lines"]) == 2
    assert t["id"] == "track-2026-12-03" and t["source"] == "t.gpx"
    t2 = fl.new_track([t], points=[[48.0, -117.0], [48.001, -117.0]], date="2026-12-03")
    assert t2["id"] == "track-2026-12-03#2"
    wpts = _text(tmp_path / "w.gpx", '<gpx><wpt lat="48" lon="-117"/><wpt lat="48.001" lon="-117"/></gpx>')
    assert fl.new_track([], file=str(wpts))["lines"] == [[[48.0, -117.0], [48.001, -117.0]]]
    with pytest.raises(ValueError, match="a track needs a line"):
        fl.new_track([])
    with pytest.raises(ValueError, match="confidence must be one of"):
        fl.new_track([], points=[[48.0, -117.0], [48.001, -117.0]], confidence="sure")
    assert fl.line_length_m([[48.0, -117.0], [48.001, -117.0]]) == pytest.approx(111.3, abs=0.5)


def test_transects(tmp_path: Path) -> None:
    s1, ignored = fl.new_transect([], "Ridge road", file=str(_text(tmp_path / "r.kml", KML)), surface="Snow")
    assert s1["date"] == "2026-12-04" and s1["crossings"] == [[48.001, -117.0]] and len(s1["lines"]) == 2
    assert ignored == ["Parking"]  # only waypoints named as a lion crossing count
    # a repeat survey of the same route, nothing crossed: the route's line is reused
    s2, _ = fl.new_transect([s1], "Ridge road", date="2026-12-11")
    assert s2["lines"] == s1["lines"] and s2["crossings"] == [] and s2["id"] == "Ridge road 2026-12-11"
    s3, _ = fl.new_transect([], "Mud road", line=[[48.0, -117.0], [48.01, -117.0]], crossings=[[48.005, -117.0]])
    assert s3["crossings"] == [[48.005, -117.0]]
    gpx, ignored = fl.new_transect([], "G", file=str(_text(tmp_path / "r.gpx", GPX)))
    assert gpx["crossings"] == [[48.0005, -117.0005]] and ignored == ["start", "(no name)"]
    with pytest.raises(ValueError, match="has no line yet"):
        fl.new_transect([], "New road")
    with pytest.raises(ValueError, match="surface must be one of"):
        fl.new_transect([s1], "Ridge road", surface="ice")


def test_the_human_arm_reads_its_earlier_name() -> None:
    """ "expert" was the human arm's name: old records and old inputs both come back as "human"."""
    dep = fl.new_deployment([], 48.0, -117.0, name="H1", arm="Expert", start="2026-10-01")
    assert dep["arm"] == "human"
    old: dict[str, Any] = {**dep, "arm": "expert"}
    (rec,) = fl.migrate(old, set())
    assert rec["type"] == "deployment" and rec["arm"] == "human"
    with pytest.raises(ValueError, match="arm must be one of"):
        fl.new_deployment([], 48.0, -117.0, name="H2", arm="tracker", start="2026-10-01")
