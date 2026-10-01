"""The agent-facing operations end to end, on the synthetic area (conftest.analyzed), with every data source
served offline: analyze, outputs, re-pick, explain, validate, observations, KML import, hotspots."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import synthetic
import toys
from cougarmap import analyze, api, context, export, statefile
from cougarmap import aoi as aoi_mod
from cougarmap.analyze import WEAK_AREA_SCORE
from cougarmap.config import OBSERVATIONS_FILE, OUT_DIR, PRIVATE_DIR
from cougarmap.sources import canopy, vector
from cougarmap.sources.vector import Water
from cougarmap.state import SAVED, STATE_VERSION, load_state

KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>t</name>
<Folder><name>Areas</name>
<Placemark><name>Synthetic Area</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates>
</LinearRing></outerBoundaryIs></Polygon></Placemark>
<Placemark><name>Cam01</name><Point><coordinates>{cam},0</coordinates></Point></Placemark>
<Placemark><name>Guzzler water</name><Point><coordinates>{water},0</coordinates></Point></Placemark>
</Folder></Document></kml>"""


def _kml(tmp_path: Path, cam: tuple[float, float]) -> Path:
    w, s, e, n = synthetic.bbox()
    ring = " ".join(f"{x},{y},0" for x, y in [(w, s), (e, s), (e, n), (w, n), (w, s)])
    p = tmp_path / "areas.kml"
    p.write_text(KML.format(ring=ring, cam=f"{cam[1]},{cam[0]}", water=",".join(map(str, synthetic.lonlat(-100, 300)))))
    return p


# ---- analyze_area and its outputs ---------------------------------------------------------------------------


def test_analysis_finds_spots_with_reasons(analyzed: dict[str, Any]) -> None:
    s, cands, priv = analyzed["summary"], analyzed["candidates"], analyzed["private_candidates"]
    assert s["resolution_m"] == 3.0 and s["month"] == 10 and s["options"]["public_only"]
    assert s["wind"]["prevailing_from"] == "W" and s["lidar_fraction"] == 1.0
    assert 0.5 < s["coverage"]["public_fraction"] < 0.9
    assert cands and priv and s["n_candidates"] == len(cands) and s["private_land"]["n_spots"] == len(priv)
    assert all(c["public"] and c["land"] == "Test National Forest (Forest Service)" for c in cands)
    assert not any(p["public"] for p in priv)
    for c in cands + priv:
        assert "row" not in c and c["reasons"] and 0 < c["score"] <= 100 and c["walk_miles"] is not None
        assert set(c["factors"]) == {"wind", "edges", "pinch", "water", "travel", "habitat"}
    assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    text = " ".join(r for c in cands + priv for r in c["reasons"])
    assert "opening" in text and "paved road" in text


def test_outputs_kmz_geojson_summary_state(analyzed: dict[str, Any]) -> None:
    d: Path = analyzed["dir"]
    out = analyzed["summary"]["outputs"]
    assert {Path(v).name for v in out.values()} == {"cougarmap.kmz", "summary.json", "candidates.geojson", "state.pkl"}
    with zipfile.ZipFile(d / "cougarmap.kmz") as z:
        doc = z.read("doc.kml").decode()
        assert "consistency R 0.60" in doc and "Ground-level (HRRR 10 m) dawn/dusk wind for reference: from N" in doc
        pngs = [n for n in z.namelist() if n.endswith(".png")]
        from xml.etree import ElementTree as ET

        root = ET.fromstring(doc)
        ns = {"k": "http://www.opengis.net/kml/2.2"}
        overlays = root.findall(".//k:GroundOverlay", ns)
        assert len(overlays) == len(pngs) >= 12  # one PNG per layer
        names = [o.findtext("k:name", namespaces=ns) for o in overlays]
        assert (
            "Lion score on private land (top 30%)" in names
            and "Travel lines (drainage bottoms; ridge spines at crossings)" in names
        )
        from io import BytesIO

        from PIL import Image

        assert Image.open(BytesIO(z.read(pngs[0]))).mode == "RGBA"
    n_spots = len(analyzed["candidates"]) + len(analyzed["private_candidates"])
    assert doc.count("<Point>") >= n_spots and "Private land spots" in doc and "Walking routes from road" in doc
    alts = [c for c in analyzed["candidates"] if c["trail_alternate"]]  # the forest road, trail and gated road
    assert alts and "Alternate spots on a trail/two-track (optional, not the pick)" in doc
    assert all(c["trail_alternate"]["distance_m"] <= 150 for c in alts) and "trail alt (" in doc
    gj = json.loads((d / "candidates.geojson").read_text())
    assert len(gj["features"]) == n_spots and "row" not in gj["features"][0]["properties"]
    assert json.loads((d / "summary.json").read_text())["candidates"][0]["name"] == "#1"
    st = load_state(d / "state.pkl")
    assert statefile.load(d / "state.pkl")["slim"]["version"] == STATE_VERSION
    assert set(st.layers) == set(SAVED) and not st.unmodeled


def test_summary_runtime_covers_the_whole_run(analyzed: dict[str, Any]) -> None:
    saved = json.loads((analyzed["dir"] / "summary.json").read_text())["summary"]
    assert saved["runtime_s"] == analyzed["summary"]["runtime_s"] > 0


def test_dry_solid_timber_still_gets_spots(tmp_path: Path) -> None:
    """No mapped water and unbroken timber: every score is low, but the best spots are still returned, with a
    note saying they are weak."""
    with synthetic.offline() as mp:
        mp.setattr(analyze, "OUT_DIR", tmp_path)  # not over the shared analyzed area (same bbox)
        mp.setattr(vector, "fetch_water", lambda _lb: Water(points=[], flowlines=[], waterbodies=[]))
        mp.setattr(canopy, "fetch_canopy", lambda g: np.full(g.shape, 18.0, "float32"))
        r = api.analyze_area(bbox=synthetic.bbox(), month=10, public_only=False, log=lambda *_: None)
    s = r["summary"]
    assert r["candidates"] and s["best_score"] < WEAK_AREA_SCORE
    assert any("scores low" in n for n in s["notes"])


def test_a_winter_run_uses_the_winter_module(tmp_path: Path) -> None:
    """January: the season multiplier is on (shallow snow west, deep east, deer winter range over the valley)."""
    with synthetic.offline() as mp:
        mp.setattr(analyze, "OUT_DIR", tmp_path)
        r = api.analyze_area(bbox=synthetic.bbox(), month=1, public_only=False, log=lambda *_: None)
    st = load_state(r["summary"]["outputs"]["state"])
    season = st.layers["season"]
    assert 0.6 <= season.min() < season.max() <= 1 and st.layers["winter_range_mid"].any()
    assert all("season" in c["factors"] for c in r["candidates"])
    assert not any("SNODAS" in n for n in r["summary"]["notes"])
    with zipfile.ZipFile(r["summary"]["outputs"]["kmz"]) as z:
        assert "Winter ground (low, sun-facing, shallow snow" in z.read("doc.kml").decode()


def test_validation_runs_can_leave_out_the_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[aoi_mod.AOI] = []

    def fake_run(a: aoi_mod.AOI, *_: Any, **__: Any) -> dict[str, Any]:
        seen.append(a)
        return dict(summary={}, candidates=[], private_candidates=[])

    monkeypatch.setattr(api, "run", fake_run)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    kml = PRIVATE_DIR / "pins.kml"
    kml.write_text(_kml(tmp_path, synthetic.lonlat(0, 0)[::-1]).read_text())
    try:
        api.analyze_area(kml=str(kml), log=lambda *_: None)
        api.analyze_area(kml=str(kml), log=lambda *_: None, user_pins=False)
    finally:
        kml.unlink()
    assert [p["kind"] for p in seen[0].user_points] == ["water"] and seen[1].user_points == []


def test_kml_text_is_escaped(analyzed: dict[str, Any]) -> None:
    st = load_state(analyzed["dir"] / "state.pkl")
    from cougarmap.analyze import Result, apply_masks, pick_all, summarize

    apply_masks(st)
    cands, priv = pick_all(st)
    cands[0]["reasons"] = ["<b>bold</b> & co"]
    st.aoi.name = "A & B <area>"
    result = Result(summary=summarize(st, cands, 0, priv), candidates=cands, private_candidates=priv, state=st)
    doc = export.kml_doc(result, {})
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in doc and "CougarMap - A &amp; B &lt;area&gt;" in doc


# ---- follow-ups on a saved area -----------------------------------------------------------------------------


def test_explain_point(analyzed: dict[str, Any]) -> None:
    c = analyzed["candidates"][0]
    area = analyzed["dir"].name
    e = api.explain_point(area, c["lat"], c["lon"])
    best, at = e["best_nearby"], e["exact_point"]
    assert best["raw_score"] >= at["raw_score"] and best["reasons"] and "row" not in best
    assert best["usable"] and not best["on_private_layer"] and 50 < best["percentile_in_area"] <= 100
    p = analyzed["private_candidates"][0]
    assert api.explain_point(area, p["lat"], p["lon"], search_m=0)["best_nearby"]["on_private_layer"]
    assert api.explain_point(area, 10.0, 10.0) == dict(error="point is outside the analyzed area")


def test_repick_switches_land_rules_without_rewriting_arrays(area: str) -> None:
    path = OUT_DIR / area / "state.pkl"
    with path.open("rb") as f:
        _, index_at = statefile._read_index(f)
    arrays_before = path.read_bytes()[:index_at]
    kmz_before = (OUT_DIR / area / "cougarmap.kmz").stat().st_mtime_ns
    r = api.repick(area, n_candidates=3, per_zone=1, public_only=False)
    assert len(r["candidates"]) <= 3 and r["private_candidates"] == [] and not r["summary"]["options"]["public_only"]
    assert any(not c["public"] for c in api.repick(area, n_candidates=30, per_zone=30, public_only=False)["candidates"])
    assert path.read_bytes()[:index_at] == arrays_before  # only the options were rewritten
    assert (OUT_DIR / area / "cougarmap.kmz").stat().st_mtime_ns >= kmz_before
    st = load_state(path)
    assert st.opts.n_candidates == 30 and not st.opts.public_only
    short = api.repick(area, max_walk_miles=0.05, public_only=True)
    assert all(c["walk_miles"] <= 0.05 for c in short["candidates"])


def test_repick_at_the_analysis_options_gives_the_same_spots(analyzed: dict[str, Any], area: str) -> None:
    """Same spots, same numbers (canopy, slope, paved-road distance): the analysis describes its spots from the
    values its saved state keeps."""
    r = api.repick(area)
    assert r["candidates"] == analyzed["candidates"]
    assert r["private_candidates"] == analyzed["private_candidates"]


def test_a_failed_repick_keeps_the_saved_options(area: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(api, "write_outputs", broken)
    with pytest.raises(OSError, match="disk full"):
        api.repick(area, n_candidates=3)
    with api._STATES.use(api._state_for(area)) as st:
        assert st.opts.n_candidates == 15


def test_repick_rewrites_an_old_format_state_once(area: str) -> None:
    path = OUT_DIR / area / "state.pkl"
    import gzip
    import pickle

    tree = statefile.load(path)
    with gzip.open(path, "wb") as f:
        pickle.dump(tree, f)
    assert not statefile.is_v2(path)
    api.repick(area)
    assert statefile.is_v2(path)


def test_state_lookup(analyzed: dict[str, Any], tmp_path: Path) -> None:
    d: Path = analyzed["dir"]
    assert api._state_for(d.name) == OUT_DIR / d.name / "state.pkl"
    assert api._state_for(str(d)) == d / "state.pkl"
    assert api._state_for(str(d / "state.pkl")) == d / "state.pkl"
    assert api._state_for(d.name[:12]) == OUT_DIR / d.name / "state.pkl"  # a unique prefix of the name
    outside = tmp_path / "state.pkl"
    outside.write_bytes(b"not a state")
    with pytest.raises(ValueError, match="outside the results folder"):
        api._state_for(str(outside))
    with pytest.raises(FileNotFoundError, match="no saved analysis"):
        api._state_for("nowhere near")


def test_validate_and_field_log(analyzed: dict[str, Any], tmp_path: Path) -> None:
    """Log a season of field work on the synthetic area (paired cameras, a snow track, two transect surveys) and
    test the area against it."""
    best = analyzed["candidates"][0]
    r = api.log_result(best["lat"], best["lon"], True, name="Cam01", detections=2, times=["2026-10-01T06:10"])
    assert r["saved"] and r["total"] == 3 and Path(r["file"]) == OBSERVATIONS_FILE
    assert r["record"]["arm"] == "unpaired" and len(r["events"]) == 2
    assert api.read_observations()[0]["type"] == "deployment"
    far = [synthetic.lonlat(-450, -450)[::-1], synthetic.lonlat(450, -450)[::-1]]
    for z, (lat, lon) in enumerate(far):
        m = api.log_camera(best["lat"], best["lon"], name=f"M{z}", arm="model", zone=f"z{z}", start="2026-07-01")
        c = api.log_camera(lat, lon, name=f"C{z}", arm="control", zone=f"z{z}", start="2026-07-01", lure=False)
        assert m["deployment"]["id"] == f"M{z}" and not m["updated"]
        ev = [dict(datetime=f"2026-08-0{d}T05:00", species="cougar") for d in (1, 3, 5)]
        k = api.log_check(f"M{z}", "2026-09-28", events=[*ev, dict(datetime="2026-08-02T05:00", species="deer")])
        assert k["events_logged"] == 4 and k["camera_nights"] == 89 and k["detections"]["cougar"] == 3
        api.log_check(c["deployment"]["id"], "2026-09-28", downtime_nights=9)
    t0 = api.log_camera(best["lat"], best["lon"], name="T0", arm="on-feature", zone="z0", start="2026-07-01")
    api.log_check(t0["deployment"]["id"], "2026-09-28", events=[dict(datetime="2026-08-01T12:00:00Z")])
    upd = api.log_camera(deployment="C1", end="2026-09-28", notes="pulled")
    assert upd["updated"] and upd["deployment"]["end"] == "2026-09-28" and upd["deployment"]["arm"] == "control"
    t = api.log_track(points=[list(synthetic.lonlat(u, 0)[::-1]) for u in (-60, 0, 60)], date="2026-12-02")
    assert t["saved"] and 0.1 < t["km"] < 0.14
    route = [list(synthetic.lonlat(u, best_v)[::-1]) for u, best_v in ((-500, 0), (500, 0))]
    s1 = api.log_transect("Ridge road", line=route, crossings=[list(synthetic.lonlat(0, 0)[::-1])], date="2026-12-02")
    s2 = api.log_transect("Ridge road", date="2026-12-09")
    assert s1["crossings"] == 1 and s2["crossings"] == 0 and s2["km"] == s1["km"]
    assert "ignored_waypoints" not in s1
    (lat0, lon0), (lat1, lon1) = route
    gpx = tmp_path / "survey.gpx"
    gpx.write_text(
        f'<gpx><wpt lat="{lat0}" lon="{lon0}"><name>Parking</name></wpt>'
        f'<wpt lat="{(lat0 + lat1) / 2}" lon="{(lon0 + lon1) / 2}"><name>lion</name></wpt>'
        f'<trk><trkseg><trkpt lat="{lat0}" lon="{lon0}"/><trkpt lat="{lat1}" lon="{lon1}"/></trkseg></trk></gpx>'
    )
    s3 = api.log_transect("Ridge road", file=str(gpx), date="2026-12-16")
    assert s3["crossings"] == 1 and s3["ignored_waypoints"] == ["Parking"] and "Parking" in s3["note"]
    log = api.field_log()
    assert [c["id"] for c in log["cameras"]] == ["Cam01", "M0", "C0", "M1", "C1", "T0"]
    assert len(log["tracks"]) == 1 and [x["crossings"] for x in log["transects"]] == [1, 0, 1]

    v = api.validate(analyzed["dir"].name, kml=str(_kml(tmp_path, (best["lat"], best["lon"]))))
    cams = v["cameras"]
    pair, trail = cams["paired"]
    assert pair["zones"] == 2 and pair["zones_better"] == 2 and pair["rate_ratio"] > 5
    assert (trail["treatment"], trail["control"], trail["zones"]) == ("on-feature", "model", 1)
    assert cams["by_arm"]["model"]["camera_nights"] == 178 and cams["by_arm"]["control"]["camera_nights"] == 160
    assert cams["by_arm"]["model"]["vs_base"] > 1 and cams["sample_size"]["paired_zones"] == 2
    assert {c["id"] for c in cams["cameras"]} == {"Cam01", "M0", "C0", "M1", "C1", "T0"}
    assert cams["by_arm"]["unpaired"]["vs_base"] is None and cams["by_arm"]["unpaired"]["too_few_nights"]
    assert all(0 < c["model_rank"] <= 1 for c in cams["cameras"])
    (track,) = v["snow_tracks"]["tracks"]
    assert 0 <= track["percentile"] <= 1 and v["snow_tracks"]["sign_test"]["n"] == 1
    split = v["snow_tracks"]["by_start"]
    assert track["start_road_m"] is not None
    assert (split["near_road"] or split["away_from_road"])["n"] == 1  # the one track lands in one group
    assert (split["near_road"] is None) == (not track["found_near_road"])
    tx = v["crossing_transects"]
    assert tx["surveys"] == 3 and tx["crossings"] == 2 and tx["routes"] == ["Ridge road"] and 0 <= tx["auc"] <= 1
    assert v["human_picks"]["picks"] == 1 and v["human_picks"]["by_pick"]["Cam01"] <= 0.5
    assert any("model vs control" in line for line in v["summary"])
    assert any("too few nights" in line for line in v["summary"])  # Cam01's single logged night
    assert api.validate(analyzed["dir"].name)["human_picks"] is None  # no camera pins in the private folder
    OBSERVATIONS_FILE.unlink()
    empty = api.validate(analyzed["dir"].name)
    assert empty["summary"] == ["no lion truth logged in this area yet: see docs/FIELD_PROTOCOL.md for what to record"]
    assert any(line.startswith("human camera picks (1)") for line in v["summary"])


# ---- areas, KML import, places, wind, opening files ---------------------------------------------------------


def test_list_and_import_kml(tmp_path: Path) -> None:
    p = _kml(tmp_path, synthetic.lonlat(0, 0)[::-1])
    areas = api.list_areas(str(p))
    assert [a["name"] for a in areas["areas"]] == ["Synthetic Area"] and 1.3 < areas["areas"][0]["area_km2"] < 1.6
    assert {pt["kind"] for pt in areas["points"]} == {"camera", "water"}
    r = api.import_kml(str(p))
    assert r["imported"] and Path(r["path"]).parent == PRIVATE_DIR and r["areas"]
    assert api.import_kml(str(tmp_path / "missing.kml"))["imported"] is False
    (PRIVATE_DIR / "broken.kml").write_text("<kml")
    logged: list[str] = []
    pins = api._known_points(logged.append)
    assert {q["kind"] for q in pins} == {"camera", "water"} and "broken.kml" in logged[0]
    for f in PRIVATE_DIR.glob("*.kml"):
        f.unlink()


def test_resolve_area(tmp_path: Path) -> None:
    lat, lon = synthetic.LAT, synthetic.LON
    a = api.resolve_area(location=f"{lat},{lon}", radius_km=1.0)
    assert a.name == f"{lat} r1km" and 3.0 < a.area_km2() < 3.3
    assert api.resolve_area(bbox=synthetic.bbox()).name.startswith("bbox ")
    assert api.resolve_area(kml=str(_kml(tmp_path, (lat, lon))), area_name="synthetic").name == "Synthetic Area"
    with pytest.raises(ValueError, match="give a location"):
        api.resolve_area()


def test_wind_summary() -> None:
    with synthetic.offline():
        w = api.wind_summary(f"{synthetic.LAT},{synthetic.LON}", month=10)
    assert w["prevailing_from"] == "W" and w["dawn"]["from_compass"] == "W" and w["month"] == 10
    assert w["most_common_from"] == "W" and w["dawn"]["most_common_from"] == "N"  # the fake's flat rose
    assert w["ground_level"]["from_compass"] == "N" and "consistency" in w["how_to_read"]


def test_open_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **_: opened.append(args))
    f = tmp_path / "x.kmz"
    f.write_bytes(b"")
    assert api.open_file(str(f)) == dict(opened=True, path=str(f)) and opened[0][-1] == str(f)
    assert api.open_file(str(tmp_path / "nope"))["opened"] is False


# ---- find_hotspots ------------------------------------------------------------------------------------------


def test_find_hotspots_small_radius_analyzes_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(api, "analyze_area", toys.recorder(calls, dict(summary={})))
    r = api.find_hotspots("47.9,-117.6", radius_km=2)
    assert r["how"] == "analyzed the whole area in detail" and calls[0][:2] == ("47.9,-117.6", 2)
    with pytest.raises(ValueError, match="give a location"):
        api.find_hotspots()


def test_find_hotspots_scouts_then_merges_blocks(analyzed: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    blocks: list[dict[str, Any]] = [
        dict(rank=i, score=80.0 - i, land=f"Forest {i}", why=[f"why {i}"], bbox=synthetic.bbox(), center={})
        for i in (1, 2)
    ]
    monkeypatch.setattr(
        api, "scout_region", lambda *a, **k: dict(location="Testville, WA", month=10, wind={}, blocks=blocks)
    )
    monkeypatch.setattr(context, "prefetch", lambda *a, **k: None)
    seen: list[dict[str, Any]] = []

    def fake_analyze(**k: Any) -> dict[str, Any]:
        seen.append(k)
        return {key: analyzed[key] for key in ("summary", "candidates", "private_candidates")}

    monkeypatch.setattr(api, "analyze_area", fake_analyze)
    r = api.find_hotspots("Testville", radius_km=20, blocks=2, log=lambda *_: None)
    assert [k["area_name"] for k in seen] == ["Testville block 1", "Testville block 2"]
    assert [b["block"] for b in r["blocks"]] == [1, 2] and r["top_spots"][0]["name"].startswith("Block ")
    assert [(b["land"], b["why"]) for b in r["blocks"]] == [("Forest 1", ["why 1"]), ("Forest 2", ["why 2"])]
    scores = [s["score"] for s in r["top_spots"]]
    assert scores == sorted(scores, reverse=True) and len(r["private_spots"]) <= 5
    with zipfile.ZipFile(r["kmz"]) as z:
        names = z.namelist()
        assert "doc.kml" in names and "b1/doc.kml" in names and "b2/doc.kml" in names
        assert z.read("doc.kml").decode().count("<NetworkLink>") == 2
    assert json.loads((Path(r["kmz"]).parent / "summary.json").read_text())["radius_km"] == 20

    monkeypatch.setattr(api, "scout_region", lambda *a, **k: dict(location="Nowhere", blocks=[]))
    assert "no promising" in api.find_hotspots("Nowhere", radius_km=20)["error"]


def test_scout_region_labels_the_place(monkeypatch: pytest.MonkeyPatch) -> None:
    from cougarmap import scout

    monkeypatch.setattr(scout, "scout", lambda *a, **k: dict(blocks=[], args=a))
    r = api.scout_region("47.9,-117.6", 10)
    assert r["location"] == "47.9,-117.6" and r["args"][:3] == (47.9, -117.6, 10)


def test_layers_are_float32_on_load(analyzed: dict[str, Any]) -> None:
    st = load_state(analyzed["dir"] / "state.pkl")
    assert all(v.dtype != np.float16 for v in st.layers.values() if isinstance(v, np.ndarray))
