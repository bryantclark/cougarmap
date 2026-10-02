"""The lion-truth tests (cougarmap.truth) on synthetic rasters: each recovers a planted signal, and gives about
0.5 (and no false alarm) on a raster with no signal."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from cougarmap import truth as tr
from cougarmap.config import Truth

RES = 5.0
N = 400  # a 2 km square
CFG = Truth(n_shifted=100, n_permutations=2000)
MASK = np.ones((N, N), bool)


def _valley() -> tuple[np.ndarray, np.ndarray]:
    """A raster high along a winding valley line, and that line (row, col)."""
    cols = np.arange(40.0, 360.0, 4.0)
    rows = 200 + 40 * np.sin(cols / 30)
    r, c = np.mgrid[0:N, 0:N]
    d = np.abs(r - (200 + 40 * np.sin(c / 30)))
    return (100 * np.exp(-(d**2) / (2 * 4.0**2))).astype("float32"), np.stack([rows, cols], axis=1)


def _noise(seed: int) -> np.ndarray:
    a = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(N, N)), 4)
    return (a - a.min()).astype("float32")


def _walk(rng: np.random.Generator, steps: int = 30) -> np.ndarray:
    """A random lion-ish path (~300 m) somewhere in the middle of the square."""
    start = rng.uniform(120, 280, 2)
    heading = np.cumsum(rng.normal(0, 0.4, steps)) + rng.uniform(0, 2 * np.pi)
    return start + np.cumsum(np.stack([np.sin(heading), np.cos(heading)], axis=1) * 2.0, axis=0)


def test_helpers() -> None:
    d = tr.densify(np.array([[0.0, 0.0], [0.0, 10.0]]), 2.5)
    assert len(d) == 5 and d[-1].tolist() == [0.0, 10.0]
    assert tr.densify(np.array([[1.0, 1.0]]), 1).shape == (1, 2)
    v, ok = tr.values_at(
        np.arange(4.0).reshape(2, 2),
        np.array([[True, False], [True, True]]),
        np.array([[0.2, 0.9], [0.5, 1.5], [5.0, 0.0], [1.0, 1.0]]),
    )
    assert ok.tolist() == [True, False, False, True] and v[0] == 0 and v[3] == 3 and np.isnan(v[1])
    assert tr.percentile_of(2, np.array([1, 2, 3])) == pytest.approx(0.5)
    assert tr.percentile_of(9, np.array([1, 2, 3])) == 1.0
    assert tr.base_rate(8) == 0.87 and tr.base_rate(1) == 0.37 and tr.base_rate(10) == pytest.approx(0.62)


# ---- snow tracks ------------------------------------------------------------------------------------------------


def test_track_along_the_valley_beats_its_shifted_copies() -> None:
    score, line = _valley()
    rng = np.random.default_rng(1)
    res = tr.path_test(score, MASK, [line], RES, rng, CFG)
    assert res is not None and res.n_null == CFG.n_shifted
    assert res.percentile > 0.95 and res.mean_score > 3 * res.null_mean and res.inside_share == 1.0
    # the same track 30 m off the valley floor no longer looks like a lion
    off = tr.path_test(score, MASK, [line + np.array([12.0, 0.0])], RES, rng, CFG)
    assert off is not None and off.percentile < res.percentile


def test_random_tracks_on_a_raster_without_signal_land_near_half() -> None:
    rng = np.random.default_rng(2)
    pcts = []
    for i in range(40):
        res = tr.path_test(_noise(i % 4), MASK, [_walk(rng)], RES, rng, CFG)
        assert res is not None
        pcts.append(res.percentile)
    assert 0.38 < float(np.mean(pcts)) < 0.62
    p = tr.sign_test(pcts)["p_value"]
    assert p is not None and p > 0.05


def test_track_outside_the_area_or_too_short() -> None:
    score, line = _valley()
    rng = np.random.default_rng(0)
    assert tr.path_test(score, MASK, [line + 1000], RES, rng, CFG) is None
    assert tr.path_test(score, MASK, [line[:1]], RES, rng, CFG) is None
    tiny = np.zeros((N, N), bool)
    tiny[:5, :5] = True  # no shifted copy fits inside
    assert tr.path_test(score, tiny, [np.array([[1.0, 1.0], [3.0, 3.0]])], RES, rng, CFG) is None


def test_sign_test_and_tracks_needed() -> None:
    st = tr.sign_test([0.9] * 10)
    assert st["above_half"] == 10 and st["p_value"] == pytest.approx(1 / 1024)
    assert tr.sign_test([0.9] * 8 + [0.1] * 2 + [0.5])["p_value"] == pytest.approx(56 / 1024)
    assert tr.sign_test([0.5])["p_value"] is None
    n9, n7 = tr.tracks_needed(0.9), tr.tracks_needed(0.7)
    assert n9 is not None and n7 is not None and 4 <= n9 < n7 <= 60
    assert tr.tracks_needed(0.5) is None


# ---- crossing transects -----------------------------------------------------------------------------------------


def _route() -> tuple[np.ndarray, np.ndarray]:
    """A straight 1.5 km route across a raster whose score rises and falls along it."""
    c = np.mgrid[0:N, 0:N][1]
    score = (50 + 50 * np.sin(c / 20.0)).astype("float32")
    return score, np.array([[200.0, 50.0], [200.0, 350.0]])


def test_crossings_at_high_scores_are_recovered() -> None:
    score, route = _route()
    cols = np.arange(50, 350)
    high = cols[np.sin(cols / 20.0) > 0.9]
    rng = np.random.default_rng(3)
    surveys = [tr.Survey([route], np.stack([[200.0] * 3, rng.choice(high, 3)], axis=1)) for _ in range(4)]
    surveys.append(tr.Survey([route], np.zeros((0, 2))))  # surveyed, nothing crossed
    out = tr.transect_test(score, MASK, surveys, RES, rng, CFG)
    assert out["surveys"] == 5 and out["with_crossings"] == 4 and out["crossings"] == 12
    assert out["auc"] > 0.9 and out["p_value"] < 0.01 and out["route_km"] == pytest.approx(5 * 1.5, abs=0.1)
    per = out["crossings_per_10km"]
    assert per["top_third"] > 0 and per["bottom_third"] == 0


def test_random_crossings_give_an_auc_near_half() -> None:
    score, route = _route()
    rng = np.random.default_rng(4)
    aucs, ps = [], []
    for _ in range(30):
        cr = np.stack([np.full(5, 200.0), rng.uniform(50, 350, 5)], axis=1)
        out = tr.transect_test(score, MASK, [tr.Survey([route], cr)], RES, rng, CFG)
        aucs.append(out["auc"])
        ps.append(out["p_value"])
    assert 0.4 < float(np.mean(aucs)) < 0.6
    assert np.mean(np.array(ps) < 0.05) <= 0.15


def test_transects_with_nothing_to_score() -> None:
    score, route = _route()
    out = tr.transect_test(score, MASK, [], RES, np.random.default_rng(0), CFG)
    assert out["surveys"] == 0 and out["auc"] is None and "crossings_per_10km" not in out
    empty = tr.transect_test(score, MASK, [tr.Survey([route], np.zeros((0, 2)))], RES, np.random.default_rng(0), CFG)
    assert empty["auc"] is None and empty["p_value"] is None and empty["crossings_per_10km"]["top_third"] == 0


# ---- cameras ------------------------------------------------------------------------------------------------------


def _zones(n: int, rate_ratio: float, seed: int, nights: float = 180, base: float = 0.87) -> list[tr.CamStat]:
    rng = np.random.default_rng(seed)
    out = []
    for z in range(n):
        site = rng.gamma(2.0, 0.5)  # zones differ (CV ~0.7); both cameras share the zone's level
        for arm, rr in (("model", rate_ratio), ("control", 1.0)):
            lam = base / 100 * nights * site * rr * rng.gamma(4.0, 0.25)
            n = int(rng.poisson(lam))
            out.append(tr.CamStat(f"{arm}{z}", arm, f"z{z}", nights, n, 3, 1, base / 100 * nights, "on-feature"))
    return out


def test_model_cameras_that_detect_three_times_as_often_are_recovered() -> None:
    cams = _zones(30, 3.0, seed=5)
    out = tr.camera_test(cams, np.random.default_rng(0), CFG)
    (p,) = out["paired"]
    assert p["treatment"] == "model" and p["zones"] == 30
    assert p["p_value"] < 0.01 and 1.8 < p["rate_ratio"] < 5 and p["zones_better"] > p["zones_worse"]
    m, c = out["by_arm"]["model"], out["by_arm"]["control"]
    assert m["base_per_100"] == pytest.approx(0.87) and m["vs_base"] > c["vs_base"]
    assert m["p_above_base"] < 0.01 and m["deer_per_100"] == pytest.approx(100 * 3 / 180, abs=1e-3)


def test_no_difference_gives_no_false_alarm() -> None:
    rng = np.random.default_rng(0)
    ps = []
    for seed in range(200):
        p = tr.paired_test(_zones(12, 1.0, seed), "model", "control", rng, CFG)  # 12 zones: exact test
        assert p is not None
        ps.append(p["p_value"])
    ps_a = np.array(ps)
    assert np.mean(ps_a < 0.05) <= 0.08 and 0.4 < float(np.mean(ps_a)) < 0.65


def test_paired_test_edges() -> None:
    rng = np.random.default_rng(0)
    assert tr.sign_flip_p(np.array([1.0, 1.0, 1.0]), rng, 100) == pytest.approx(1 / 8)
    assert tr.sign_flip_p(np.zeros(0), rng, 100) == 1.0
    assert 0 < tr.sign_flip_p(np.ones(20), rng, 500) < 0.01  # Monte Carlo past 16 zones
    lone = [tr.CamStat("a", "model", "z1", 100, 2), tr.CamStat("b", "control", "z2", 100, 0)]
    assert tr.paired_test(lone, "model", "control", rng) is None  # no zone holds both arms
    pair = [tr.CamStat("a", "model", "z1", 100, 2), tr.CamStat("b", "control", "z1", 100, 0)]
    p = tr.paired_test(pair, "model", "control", rng)
    assert p is not None and p["rate_ratio"] == pytest.approx(5.0)  # 0.5 added to each arm's count
    assert tr.arm_rates([tr.CamStat("c", "unpaired", None, 0, 0)])["unpaired"]["vs_base"] is None


def test_arm_rates_hold_back_the_comparison_below_min_nights() -> None:
    # one night with two cougars: 200 per 100 nights, but no "230x the base rate, p = 0" in the raw output
    few = tr.arm_rates([tr.CamStat("c", "unpaired", None, 1.0, 2, base_expected=0.0087, placement="on-feature")])[
        "unpaired"
    ]
    assert few["cougar_per_100"] == 200 and few["too_few_nights"]
    assert few["vs_base"] is None and few["p_above_base"] is None
    on = tr.CamStat("c", "unpaired", None, 30.0, 2, base_expected=0.26, placement="on-feature")
    enough = tr.arm_rates([on])["unpaired"]
    assert not enough["too_few_nights"] and enough["vs_base"] == pytest.approx(7.69) and enough["p_above_base"] < 0.05


def test_the_trail_alternate_camera_pairs_with_the_picks_own_camera() -> None:
    """The protocol: the pick's camera is logged as model, the trail alternate's as on-feature in the same zone.
    Both comparisons keep the zone: model vs control, and on-feature vs model."""
    cams = [
        tr.CamStat("m", "model", "z1", 100, 2),
        tr.CamStat("c", "control", "z1", 100, 0),
        tr.CamStat("t", "on-feature", "z1", 100, 4),
        tr.CamStat("e", "human", "z2", 100, 1),
        tr.CamStat("t2", "on-feature", "z2", 100, 1),
    ]
    out = tr.camera_test(cams, np.random.default_rng(0), CFG)
    pairs = {(p["treatment"], p["control"]): p for p in out["paired"]}
    assert set(pairs) == {("model", "control"), ("on-feature", "model"), ("on-feature", "human")}
    assert pairs["model", "control"]["zones"] == 1 and pairs["on-feature", "model"]["rate_ratio"] == 2.0


def test_an_off_trail_camera_is_not_judged_against_the_on_trail_base_rate() -> None:
    """1 cougar per 100 nights off a trail is not "below base": it isn't compared with on-trail cameras at all."""
    off = tr.CamStat("m", "model", None, 200, 2, base_expected=1.74, placement="off-feature")
    unk = tr.CamStat("u", "model", None, 100, 0, base_expected=0.87)  # trail_type not logged
    m = tr.arm_rates([off, unk])["model"]
    assert m["vs_base"] is None and m["p_above_base"] is None and m["compared_nights"] == 0
    assert m["not_compared"] == {"off-feature": 1, "unrecorded": 1} and m["cougar_per_100"] == pytest.approx(0.667)
    assert m["by_placement"]["off-feature"]["cougar_per_100"] == 1.0
    lines = tr._camera_lines(tr.camera_test([off, unk], np.random.default_rng(0), CFG))
    assert any("off trails" in x and "not comparable" in x for x in lines)
    assert any("no trail_type logged" in x for x in lines) and not any("x a random" in x for x in lines)
    # on-trail cameras of the same arm are still compared, on their own nights and base
    on = tr.CamStat("t", "model", None, 100, 2, base_expected=0.87, placement="on-feature")
    both = tr.arm_rates([off, on])["model"]
    assert both["compared_nights"] == 100 and both["vs_base"] == pytest.approx(2.3) and both["camera_nights"] == 300


def test_a_zone_whose_cameras_were_placed_differently_is_flagged() -> None:
    rng = np.random.default_rng(0)
    pair = [
        tr.CamStat("m", "model", "z1", 100, 3, placement="on-feature"),
        tr.CamStat("c", "control", "z1", 100, 1, placement="off-feature"),
        tr.CamStat("m2", "model", "z2", 100, 1, placement="on-feature"),
        tr.CamStat("c2", "control", "z2", 100, 1, placement="on-feature"),
        tr.CamStat("m3", "model", "z3", 100, 1),
        tr.CamStat("c3", "control", "z3", 100, 1, placement="on-feature"),
    ]
    p = tr.paired_test(pair, "model", "control", rng)
    assert p is not None and p["zones"] == 3
    assert p["zones_placed_differently"] == 1 and p["zones_placement_unrecorded"] == 1
    assert "2 of 3 zone(s)" in p["placement_warning"]
    alike = tr.paired_test(pair[2:4], "model", "control", rng)
    assert alike is not None and alike["zones_placed_differently"] == 0 and "placement_warning" not in alike
    # on-feature vs model compares placement on purpose: no warning there
    trail = [tr.CamStat("t", "on-feature", "z1", 100, 3, placement="on-feature"), pair[0]]
    t = tr.paired_test(trail, "on-feature", "model", rng)
    assert t is not None and "placement_warning" not in t and "zones_placed_differently" not in t
    lines = tr._camera_lines(tr.camera_test(pair, rng, CFG))
    assert any("don't compare like with like" in x for x in lines)


def test_camera_power_and_zones_needed() -> None:
    cfg = Truth(site_cv=0.75)
    strong = tr.camera_power(20, 3.0, cfg=cfg, sims=400, perms=200)
    none = tr.camera_power(20, 1.0, cfg=cfg, sims=400, perms=200)
    assert strong > 0.7 and none < 0.1
    z3 = tr.zones_needed(3.0, cfg=cfg, sims=300, perms=200)
    z2 = tr.zones_needed(2.0, cfg=cfg, sims=300, perms=200)
    assert z3 is not None and z2 is not None and z3 < z2
    assert tr.zones_needed(1.0, cfg=cfg, sims=200, perms=100) is None
