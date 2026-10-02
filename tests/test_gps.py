"""The GPS falsification harness (cougarmap.gps) offline: the collar readers, night fixes, the iSSF-style strata,
the tiles, the used-vs-available ranks, the Boyce index and the shifted-map null, and the whole check on the
analyzed synthetic area, where a planted signal is recovered and the null lands near 0.5."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import synthetic
from cougarmap import gps, net
from cougarmap.config import GPS, PROJECT_ROOT, Gps
from cougarmap.evaluate import production_score
from cougarmap.state import load_state

CFG = Gps(shift_m=600.0, min_strata=10)  # the synthetic area is only ~1.5 km across


def _t(s: str) -> float:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC).timestamp()


# ---- data -------------------------------------------------------------------------------------------------------


def test_sun_elevation() -> None:
    t = np.array([_t("2020-06-21T12:00"), _t("2020-06-21T00:00"), _t("2020-03-20T12:00")])
    e = gps.sun_elevation(t, np.zeros(3), np.array([45.0, 45.0, 0.0]))
    assert e[0] == pytest.approx(68.4, abs=0.5)  # noon at the solstice: 90 - 45 + 23.4
    assert e[1] == pytest.approx(-21.6, abs=0.5)
    assert e[2] == pytest.approx(88.0, abs=2.5)  # the equinox sun near the zenith at the equator
    # local time follows longitude: noon UTC is about midnight on the date line
    assert gps.sun_elevation(t[:1], np.array([180.0]), np.array([45.0]))[0] < -15


def test_readers(tmp_path: Path) -> None:
    head = (
        "event-id,visible,timestamp,location-long,location-lat,manually-marked-outlier,sensor-type,"
        "individual-taxon-canonical-name,tag-local-identifier,individual-local-identifier,study-name\n"
    )
    rows = [
        '1,true,2020-01-01 02:00:00.000,-111.5,39.5,,"gps","Puma concolor","t1","P1","s"',
        '2,true,2020-01-01 00:00:00.000,-111.4,39.4,,"gps","Puma concolor","t1","P1","s"',
        '3,true,2020-01-01 00:00:00.000,-111.4,39.4,,"gps","Puma concolor","t1","P1","s"',  # duplicate
        '4,false,2020-01-01 04:00:00.000,-111.0,39.0,,"gps","Puma concolor","t1","P1","s"',  # not visible
        '5,true,2020-01-01 06:00:00.000,-111.0,39.0,true,"gps","Puma concolor","t1","P1","s"',  # outlier
        '6,true,2020-01-01 08:00:00.000,,,,"gps","Puma concolor","t1","P1","s"',  # no fix
        '7,true,2020-01-01 08:00:00.000,-111.0,39.0,,"gps","Canis latrans","t2","C1","s"',
    ]
    p = tmp_path / "x.csv"
    p.write_text(head + "\n".join(rows) + "\n")
    tracks = gps.read_movebank_csv(p)
    assert list(tracks) == ["P1"]
    tr = tracks["P1"]
    assert tr.t.tolist() == [_t("2020-01-01T00:00"), _t("2020-01-01T02:00")] and tr.lon[0] == -111.4
    j = tmp_path / "x.json"
    loc = dict(timestamp=1.6e12, location_long=-112.0, location_lat=38.3)
    j.write_text(
        json.dumps(
            dict(
                individuals=[
                    dict(
                        individual_local_identifier="F1",
                        individual_taxon_canonical_name="Puma concolor",
                        locations=[loc],
                    ),
                    dict(
                        individual_local_identifier="C1",
                        individual_taxon_canonical_name="Canis latrans",
                        locations=[loc],
                    ),
                    dict(
                        individual_local_identifier="F2",
                        individual_taxon_canonical_name="Puma concolor",
                        locations=[dict(timestamp=1.6e12, location_long=None, location_lat=None)],
                    ),
                ]
            )
        )
    )
    assert list(gps.read_movebank_json(j)) == ["F1"] and gps.read_movebank_json(j)["F1"].t[0] == 1.6e9


def test_fetch_downloads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class R:
        content = b'{"individuals": []}'

    def get(url: str, timeout: float) -> R:
        calls.append(url)
        return R()

    monkeypatch.setattr(net, "get", get)
    p = gps.fetch("fishlake", log=lambda *_: None)
    assert p == gps.data_path("fishlake") and gps.fetch("fishlake") == p and len(calls) == 1
    assert gps.load("fishlake") == {}
    p.unlink()


# ---- strata -----------------------------------------------------------------------------------------------------


def _track(n: int = 400, every_h: float = 2.0, seed: int = 0, straight: bool = False) -> gps.Track:
    """A puma-ish random walk in central Utah, a fix every `every_h` hours from midnight UTC (with one gap)."""
    rng = np.random.default_rng(seed)
    t = _t("2020-07-01T00:00") + np.arange(n) * every_h * 3600
    t[n // 2 :] += 5 * 3600  # one irregular gap
    heading = np.full(n, 0.3) if straight else np.cumsum(rng.normal(0, 0.8, n))
    step = rng.gamma(2, 200, n)
    x, y = np.cumsum(step * np.sin(heading)), np.cumsum(step * np.cos(heading))
    lon, lat = gps._lonlat(x, y, 39.0)
    return gps.Track("P1", t, lon - 111.5, lat + 39.0)


def test_strata_follow_the_animals_own_steps() -> None:
    tr = _track()
    s = gps.animal_strata(tr, "syn")
    assert s is not None and len(s) > 50
    assert s.avail.shape == (len(s), GPS.n_available, 2)
    sun = gps.sun_elevation(s.t, s.used[:, 0], s.used[:, 1])
    assert (sun < GPS.max_sun_deg).all()  # night and twilight fixes only
    # the irregular step (7 h, not the usual 2 h) never makes a stratum
    assert not np.isin(s.t, tr.t[len(tr.t) // 2]).any()
    # every available point is one of the animal's own night step lengths from the previous fix
    i = np.searchsorted(tr.t, s.t) - 1
    x0, y0 = gps._local(tr.lon[i], tr.lat[i], float(np.mean(tr.lat)))
    ax, ay = gps._local(s.avail[..., 0], s.avail[..., 1], float(np.mean(tr.lat)))
    d = np.hypot(ax - x0[:, None], ay - y0[:, None])
    assert np.isin(np.round(d), np.round(s.step_m)).mean() > 0.9
    again = gps.animal_strata(tr, "syn")
    assert again is not None and np.array_equal(again.avail, s.avail)  # seeded by the animal
    assert gps.animal_strata(_track(every_h=12), "syn") is None  # fixes too far apart
    assert gps.animal_strata(gps.Track("x", np.zeros(2), np.zeros(2), np.zeros(2)), "syn") is None


def test_straight_walker_draws_points_straight_ahead() -> None:
    tr = _track(straight=True)
    s = gps.animal_strata(tr, "syn", dataclasses.replace(GPS, max_sun_deg=90))
    assert s is not None
    i = np.searchsorted(tr.t, s.t) - 1
    lat0 = float(np.mean(tr.lat))
    x0, y0 = gps._local(tr.lon[i], tr.lat[i], lat0)
    ax, ay = gps._local(s.avail[..., 0], s.avail[..., 1], lat0)
    h = np.arctan2(ax - x0[:, None], ay - y0[:, None])
    tx, ty = gps._local(tr.lon, tr.lat, lat0)
    ahead = math.atan2(tx[1] - tx[0], ty[1] - ty[0])  # the walker's one heading (0.3 on its own plane)
    # the first step, and the first after the gap, have no heading to turn from: random directions there
    free = np.isin(s.t, [tr.t[1], tr.t[len(tr.t) // 2 + 1]])
    assert free.sum() == 2 and np.allclose(h[~free], ahead, atol=1e-6) and not np.allclose(h[free], ahead)


def test_strata_and_tiles() -> None:
    tracks = {a: dataclasses.replace(_track(seed=i), animal=a) for i, a in enumerate(["A", "B"])}
    s = gps.build_strata("syn", tracks)
    assert set(s.animal) == {"A", "B"} and len(s.take(s.animal == "A")) < len(s)
    assert len(gps.concat("syn", [], 15)) == 0 and gps.choose_tiles(gps.concat("syn", [], 15), 3) == []
    assert set(s.months) == {7, 8}  # 400 fixes 2 h apart from 1 July
    # a homebody with lots of fixes in two cells, a second animal in a third: the cap moves the second tile to it
    cfg = Gps(tile_km=1.0, tile_need=50)
    dlat = 1000 / (gps.EARTH_R * math.pi / 180)
    dlon = dlat / math.cos(math.radians(39.005))
    pts = [(0.5, 0.5)] * 60 + [(1.5, 0.5)] * 55 + [(5.5, 5.5)] * 40
    used = np.array([(-111 + i * dlon, 39 + j * dlat) for i, j in pts])
    who = np.array(["home"] * 115 + ["visitor"] * 40)
    n = len(used)
    syn = gps.Strata("syn", who, np.zeros(n), used, np.zeros((n, 15, 2)), np.zeros(n))
    tiles = gps.choose_tiles(syn, 3, cfg)
    assert [t.name for t in tiles] == ["syn-t1", "syn-t2"]  # the third cell adds nothing (home is used up)
    assert tiles[0].contains(used[:, 0], used[:, 1]).sum() == 60
    assert tiles[1].contains(used[:, 0], used[:, 1]).sum() == 40


# ---- scores -----------------------------------------------------------------------------------------------------


def test_ranks_validity_and_boyce() -> None:
    sc = gps.Scored(np.array([5.0, 0.0, np.nan]), np.array([[1.0, 5.0, 9.0, np.nan], [1.0, 2.0, 3.0, 4.0], [1] * 4]))
    assert sc.ranks()[:2].tolist() == pytest.approx([0.5, 0.0])
    assert sc.valid(Gps(min_available=3)).tolist() == [True, True, False]
    assert sc.valid(Gps(min_available=4)).tolist() == [False, True, False]
    rng = np.random.default_rng(0)
    avail = rng.uniform(0, 1, 5000)
    assert gps.boyce(rng.uniform(0, 1, 3000) ** 0.5, avail) == pytest.approx(1.0, abs=0.1)  # use density 2x
    null = [gps.boyce(rng.uniform(0, 1, 2000), avail) for _ in range(30)]
    assert abs(float(np.mean([b for b in null if b is not None]))) < 0.25
    assert gps.boyce(np.ones(3), avail) is None  # too few used
    assert gps.boyce(np.ones(50), np.zeros(500)) is None  # every available score ties
    assert gps._p_above_half(np.zeros(0), np.zeros(0)) is None
    assert gps._p_above_half(np.full(100, 0.9), np.full(100, 15.0)) < 1e-6  # type: ignore[operator]


def _planted(st: Any, n: int = 240, seed: int = 0) -> gps.Strata:
    """Strata in the synthetic area: half the used fixes drawn in proportion to score squared, half at random; available
    points at random inside the area. Half the steps are 'moving', half in month 10 and half in January."""
    rng = np.random.default_rng(seed)
    score = production_score(st)
    m = st.aoi_mask & (np.arange(st.fine.width)[None, :] > 2) & (np.arange(st.fine.height)[:, None] > 2)
    rr, cc = np.nonzero(m)
    w = score[rr, cc].astype(float) ** 2
    pick = np.concatenate([rng.choice(len(rr), n // 2, p=w / w.sum()), rng.choice(len(rr), n // 2)])
    av = rng.choice(len(rr), (n, 15))

    def ll(i: np.ndarray) -> np.ndarray:
        x, y = st.fine.xy(rr[i], cc[i])
        lon, lat = st.fine.to_lonlat(x, y)
        return np.stack([lon, lat], axis=-1)

    t = np.where(np.arange(n) % 2 == 0, _t("2020-10-05T03:00"), _t("2021-01-05T03:00"))
    return gps.Strata(
        "syn",
        np.array([f"P{(i // 2) % 4}" for i in range(n)]),
        t,
        ll(pick),
        ll(av),
        np.where(np.arange(n) % 4 < 2, 50.0, 500.0),
    )


@pytest.fixture
def gps_root(analyzed: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiles.json with the synthetic area as the one tile, analyzed for month 10 only, and planted strata."""
    tile = gps.Tile("syn", "syn-t1", tuple(synthetic.bbox()))  # type: ignore[arg-type]
    gps.write_tiles([tile], tmp_path)
    d = tile.dir(10, tmp_path)
    d.mkdir(parents=True)
    shutil.copy(analyzed["dir"] / "state.pkl", d / "state.pkl")
    planted = _planted(load_state(d / "state.pkl"))
    monkeypatch.setattr(gps, "strata_for", lambda ds, cfg=GPS: planted)
    return tmp_path


def test_gps_check_recovers_a_planted_signal(gps_root: Path) -> None:
    logs: list[str] = []
    res = gps.gps_check(root=gps_root, cfg=CFG, log=logs.append)
    assert any("month 1: not analyzed" in x for x in logs)  # the January fixes have no state to be scored on
    o = res["overall"]["pooled"]
    assert o["n"] == 120  # the October half
    assert o["rank20"] > 0.6 and o["p20"] < 0.001 and o["boyce20"] > 0.5
    assert 0.35 < o["null20"] < 0.65 and 0.3 < o["null_ctx"] < 0.7  # the shifted map: no signal
    assert o["n_moving"] == 60 and o["moving_rank20"] is not None
    a = res["overall"]["across_animals"]
    assert a["n_animals"] == 4 and a["sign_rank20"]["above_half"] == 4
    assert set(res["animals"]) == {"syn:P0", "syn:P1", "syn:P2", "syn:P3"}
    assert res["datasets"]["syn"]["pooled"]["n"] == 120
    # an all-random score function: about chance, and the planted one beats it in every animal
    rnd = np.random.default_rng(1)
    flat = gps.gps_check(lambda st: rnd.uniform(0, 1, st.fine.shape), root=gps_root, cfg=CFG, log=lambda *_: None)
    assert 0.35 < flat["overall"]["pooled"]["rank20"] < 0.65
    c = gps.compare(res, flat, cfg=CFG)
    assert c["overall"]["better"] == 4 and c["syn"]["n"] == 4 and c["overall"]["mean_diff"] > 0.1


def test_ablations_and_the_dem_shift(gps_root: Path) -> None:
    st = load_state(gps.Tile("syn", "syn-t1", (0, 0, 0, 0)).dir(10, gps_root) / "state.pkl")
    prod = production_score(st)
    rec = gps.recombine(st)
    assert np.corrcoef(prod.ravel(), rec.ravel())[0, 1] > 0.999  # the saved layers rebuild the production score
    fns = {**gps.ablations(), "dem E": gps.dem_shift("E", CFG), "dem N": gps.dem_shift("N", CFG)}
    for name, fn in fns.items():
        s = fn(st)
        assert s.shape == prod.shape and np.isfinite(s).all(), name
        if name == "no winter module":  # an October tile: the module is off
            assert np.array_equal(s, rec)
        elif name != "recombined":
            assert not np.array_equal(s, rec), name
    assert (fns["no penalties"](st) >= rec - 1e-4).all()
    res = gps.gps_check_many(
        {"recombined": gps.recombine, "dem E": fns["dem E"]}, root=gps_root, cfg=CFG, log=lambda *_: None
    )
    assert res["recombined"]["score"] == "recombined" and res["dem E"]["overall"]["pooled"]["n"] == 120


def test_empty_pool_and_season_months() -> None:
    assert gps._Pool().summary(np.zeros(0, bool), GPS) == dict(n=0)
    assert gps.season_months(1) == (12, 1, 2, 3, 4)
    assert gps.compare(
        dict(animals={"d:a": dict(n=1, dataset="d", rank20=0.6)}, datasets={"d": {}}),
        dict(animals={"d:a": dict(n=1, dataset="d", rank20=0.5)}),
    )["overall"] == dict(n=0, better=0, worse=0, mean_diff=None, p_value=None)


def test_analyze_tiles_runs_each_season_once(tmp_path: Path) -> None:
    tile = gps.Tile("syn", "syn-t1", tuple(synthetic.bbox()))  # type: ignore[arg-type]
    cfg = Gps(seasons=((10, (10,)),))
    logs: list[str] = []
    with synthetic.offline():
        gps.analyze_tiles([tile], tmp_path, cfg, log=logs.append)
    st = load_state(tile.dir(10, tmp_path) / "state.pkl")
    assert st.month == 10
    n = len(logs)
    gps.analyze_tiles([tile], tmp_path, cfg, log=logs.append)  # already analyzed: nothing to do
    assert len(logs) == n
    gps.write_tiles([tile], tmp_path)
    assert gps.read_tiles(tmp_path) == [tile]
    assert tile.aoi().name == "gps syn-t1"


@pytest.mark.slow
def test_saved_gps_tiles_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analyzed GPS tiles (scripts/gps_check.py run, out/gps) with the collar data in the real cache folder:
    the shifted-map null sits at chance and the production baseline holds (v3: all fixes 0.514 at 20 m, Fishlake
    0.543; v2 was 0.508). A tripwire for the harness and its recorded baseline, not evidence for the model
    (docs/VALIDATION.md)."""
    root = PROJECT_ROOT / "out" / "gps"
    monkeypatch.setattr(gps, "DATA_DIR", Path.home() / ".cache" / "cougarmap" / "gps")  # conftest moved the cache
    if not (root / "tiles.json").exists() or not all(
        (t.dir(m, root) / "state.pkl").exists() and gps.data_path(t.dataset).exists()
        for t in gps.read_tiles(root)
        for m, _ in GPS.seasons
    ):
        pytest.skip(f"needs the analyzed GPS tiles in {root}: uv run python scripts/gps_check.py all")
    res = gps.gps_check(production_score, root, log=lambda *_: None)
    o = res["overall"]["pooled"]
    assert 0.48 <= o["null20"] <= 0.52 and 0.48 <= o["null_ctx"] <= 0.52
    assert o["rank20"] >= 0.50 and res["datasets"]["fishlake"]["pooled"]["rank20"] >= 0.52
