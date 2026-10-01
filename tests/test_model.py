"""The camera-placement model: combine (spot x habitat over the camera zone), travel lines, site penalties,
the land/access masks, picking spots and zones, and the plain-English reasons."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, cast

import numpy as np
import pytest

import toys
from cougarmap import analyze as analyze_mod
from cougarmap import factors as F
from cougarmap import terrain as T
from cougarmap.analyze import (
    FACTORS,
    apply_masks,
    combine,
    describe,
    peak_threshold,
    pick_all,
    pick_candidates,
    pick_notes,
    site_penalty,
    spread,
    summarize,
    traffic_penalty,
    wind_name,
    zones,
)
from cougarmap.arrays import Floats
from cougarmap.config import MILE_M, PLACEMENT, Habitat, Options, Weights
from cougarmap.sources import vector
from cougarmap.state import Layers, ModelState, SaddlePoint

RES = 3.0
W = Weights()


def layers(shape: tuple[int, int] = (400, 400), **vals: float) -> Layers:
    keys = (*FACTORS, "edge_meadow", "travel")
    A = {k: np.full(shape, vals.get(k, 0.0), "float32") for k in keys}
    A["season"] = np.full(shape, vals.get("season", 1.0), "float32")
    return cast("Layers", A)


def score_at(A: Layers, rc: tuple[int, int] = (200, 200), w: Weights = W) -> float:
    return float(combine(A, w, RES)[0][rc])


# ---- combine ------------------------------------------------------------------------------------------------


def test_range_dtype_and_n_on() -> None:
    rng = np.random.default_rng(0)
    A = cast(
        "Layers", {k: rng.random((300, 300)).astype("float32") for k in (*FACTORS, "edge_meadow", "travel", "season")}
    )
    A["season"] = np.ones((300, 300), "float32")
    s, n = combine(A, W, RES)
    assert s.dtype == np.float32 and s.min() >= 0 and s.max() <= 100
    assert n.dtype == np.uint8
    expected = sum((A[k] >= W.stack_threshold).astype(int) for k in ("wind", "edges", "pinch", "water"))
    assert np.array_equal(n, expected)
    assert {"edge_density", "water_density", "context"} <= set(A)


def test_season_scales_the_habitat_and_is_neutral_outside_winter() -> None:
    A = layers((200, 200), wind=0.3, edges=0.3, pinch=0.3, water=0.3, edge_meadow=0.1, travel=0.3)
    s1, _ = combine(A, W, RES)
    context = A["context"].copy()
    ctx = toys.context(np.zeros((200, 200), "float32"))  # month 10: the winter module is off
    F.compute_season(ctx, log=lambda *_: None)
    assert np.array_equal(ctx.layers["season"], np.ones((200, 200), "float32"))
    A["season"] = ctx.layers["season"]
    assert np.array_equal(combine(A, W, RES)[0], s1)  # bit for bit
    A["season"] = np.full((200, 200), 0.6, "float32")
    assert np.allclose(combine(A, W, RES)[0], 0.6 * s1, rtol=1e-5)
    assert np.array_equal(A["context"], context)  # the habitat layer itself is left without the season


@pytest.mark.parametrize("k", [*FACTORS, "travel"])
def test_monotone_in_each_term(k: str) -> None:
    base = dict(wind=0.3, edges=0.3, pinch=0.3, water=0.3, edge_meadow=0.1, travel=0.3)
    lo = layers((400, 400), **base)
    hi = layers((400, 400), **{**base, k: base[k] + 0.3})
    assert score_at(hi) > score_at(lo)


def test_all_factors_full_saturates() -> None:
    assert score_at(layers(wind=1, edges=1, pinch=1, water=1, edge_meadow=1, travel=1)) == 100.0
    assert score_at(layers()) == 0.0


def test_no_step_at_threshold() -> None:
    """Stacking ramps in: crossing 0.5 doesn't jump the score."""
    a = score_at(layers(wind=0.6, edges=0.49, pinch=0.2, water=0.2, edge_meadow=0.1))
    b = score_at(layers(wind=0.6, edges=0.51, pinch=0.2, water=0.2, edge_meadow=0.1))
    assert abs(b - a) / a < 0.03


def test_stacking_rewards_more_factors() -> None:
    one = score_at(layers(edges=1.0, edge_meadow=0.1))
    two = score_at(layers(edges=1.0, wind=1.0, edge_meadow=0.1))
    assert two > (0.30 + 0.35) / 0.30 * one  # more than additive


def test_same_spot_scores_higher_in_good_habitat() -> None:
    """Identical spot terms; the one inside a timber/opening mosaic wins over solid timber."""
    A = layers((600, 1200), wind=0.4, edges=0.6, pinch=0.2, water=0.2)
    A["edge_meadow"][:, 600:] = 0.3
    s, _ = combine(A, W, RES)
    assert s[300, 1000] > 1.5 * s[300, 200]
    assert A["context"][300, 1000] > A["context"][300, 200]


def test_zone_rewards_a_patch_over_a_lone_cell() -> None:
    A = layers(wind=0.1, edges=0.1, pinch=0.1, water=0.1, edge_meadow=0.1)
    A["edges"][100, 100] = 1.0  # one hot cell
    A["edges"][290:310, 290:310] = 1.0  # a 60 m patch of the same strength
    s, _ = combine(A, W, RES)
    assert s[300, 300] > s[100, 100]


def test_zone_zero_disables_smoothing() -> None:
    w0 = dataclasses.replace(W, habitat=Habitat(zone_m=0.0))
    A = layers(wind=0.1, edges=0.1, pinch=0.1, water=0.1, edge_meadow=0.1)
    A["edges"][100, 100] = 1.0
    s, _ = combine(A, w0, RES)
    assert s[100, 100] > s[100, 101] and s[100, 101] == pytest.approx(s[300, 300])


def _sine_dem(shape: tuple[int, int] = (300, 400)) -> Floats:
    x = np.arange(shape[1]) * 10.0
    return (100 * np.sin(2 * np.pi * x / 2000.0))[None, :].repeat(shape[0], 0)


def test_valley_bottom_beats_mid_slope() -> None:
    """Same factors everywhere: the travel line (drainage bottom) lifts the bottom over the mid-slope."""
    line, _ = T.travel_lines(_sine_dem(), 10.0)  # 10 m cells; bottoms at col 150, mid-slope at col 100
    A = layers(line.shape, wind=0.4, edges=0.4, pinch=0.2, water=0.2, edge_meadow=0.1)
    A["travel"] = line
    s, _ = combine(A, W, 10.0)
    assert s[150, 150] > 1.2 * s[150, 100]


@pytest.mark.slow
def test_combine_fast_enough() -> None:
    rng = np.random.default_rng(2)
    A = cast(
        "Layers", {k: rng.random((3000, 3000)).astype("float32") for k in (*FACTORS, "edge_meadow", "travel", "season")}
    )
    t = time.perf_counter()
    combine(A, W, 4.5)
    assert time.perf_counter() - t < 3.0


# ---- penalties ----------------------------------------------------------------------------------------------


def test_traffic_penalty_shape() -> None:
    o = Options()
    d = np.array([0, 10, 20, 100, 400, 799, 800, 2000, 1e7], "float32")
    p = traffic_penalty(d, o)
    assert p[0] == pytest.approx(0.3) and p[2] == pytest.approx(0.3)
    assert p[6] == pytest.approx(1.0) and p[7] == 1.0 and p[8] == 1.0
    assert np.all(np.diff(p) >= 0)


def test_osm_is_paved() -> None:
    assert vector.osm_is_paved({"highway": "tertiary"})
    assert vector.osm_is_paved({"highway": "primary_link"})
    assert vector.osm_is_paved({"highway": "residential", "surface": "asphalt"})
    assert not vector.osm_is_paved({"highway": "track"})
    assert not vector.osm_is_paved({"highway": "unclassified", "surface": "gravel"})
    assert not vector.osm_is_paved({"highway": "residential"})
    assert not vector.osm_is_paved({"surface": "asphalt"})  # a paved parking lot area, not a road


def test_site_penalty_combines_traffic_and_populated_areas() -> None:
    A = cast("Layers", {"score": np.ones((3, 3), "float32"), "paved_dist": np.full((3, 3), 2000.0, "float32")})
    A["rec_dist"] = np.full((3, 3), 2000.0, "float32")
    A["houses"] = np.full((3, 3), 15.0, "float32")  # rural: a homestead and some neighbours cost nothing
    assert np.all(site_penalty(A, Options()) == 1)
    A["houses"] = np.full((3, 3), (15 + 60) / 2, "float32")  # halfway to town density
    assert np.allclose(site_penalty(A, Options()), 0.65)
    A["houses"] = np.full((3, 3), 60.0, "float32")  # ~76 houses/km2: the full cut
    assert np.allclose(site_penalty(A, Options()), 0.3)
    A["houses"] = np.full((3, 3), 400.0, "float32")  # a town
    assert np.allclose(site_penalty(A, Options()), 0.3)
    A["paved_dist"] = np.zeros((3, 3), "float32")
    assert np.allclose(site_penalty(A, Options()), 0.3 * 0.3)


def test_recreation_sites_cut_the_score() -> None:
    A = cast("Layers", {"score": np.ones((1, 4), "float32"), "paved_dist": np.full((1, 4), 2000.0, "float32")})
    A["houses"] = np.zeros((1, 4), "float32")
    A["rec_dist"] = np.array([[0.0, 50.0, 225.0, 400.0]], "float32")
    assert np.allclose(site_penalty(A, Options()), [[0.7, 0.7, 0.85, 1.0]])


# ---- the land/access rules ----------------------------------------------------------------------------------

N = toys.N


def toy() -> ModelState:
    """A flat toy state with a paved road down column 10."""
    st = toys.state()
    col = np.arange(N, dtype="float32")[None, :].repeat(N, 0)
    st.layers["paved_dist"] = np.abs(col - 10) * toys.RES
    return st


def test_paved_road_cuts_the_score() -> None:
    st = toy()
    apply_masks(st)
    fin = st.layers["final"]
    assert fin[150, 12] < 0.5 * fin[150, 210]  # 10 m from pavement vs 1 km away
    assert fin[150, 210] == pytest.approx(50.0)


def test_gravel_track_is_not_traffic() -> None:
    """Road distance (any open road, gravel included) doesn't penalize; only pavement does."""
    st = toy()
    st.layers["road_dist"][:, 200:] = 10.0  # a gravel road beside the right half
    apply_masks(st)
    assert st.layers["final"][150, 250] == pytest.approx(st.layers["final"][150, 180])


def test_masks_exclude_cliffs_lakes_roadsides_and_outside() -> None:
    st = toy()
    A = st.layers
    A["cliff"][10, 10] = True
    A["lake"][20, 20] = True
    A["road_dist"][30, 30] = 5.0
    st.aoi_mask[40, 40] = False
    apply_masks(st)
    for rc in ((10, 10), (20, 20), (30, 30), (40, 40)):
        assert not A["usable"][rc] and A["final"][rc] == 0
    assert A["usable"][50, 50]


def test_walk_limit_and_private_land() -> None:
    st = toy()
    A = st.layers
    A["walk_m"][:, :100] = 2 * MILE_M  # too far on the public-access walk
    A["public"][:, 200:] = False  # private east third, reached only by the any-route walk
    apply_masks(st)
    assert not A["usable"][150, 50] and A["usable"][150, 150]
    assert not A["usable"][150, 250] and A["usable_private"][150, 250]
    assert A["final_private"][150, 150] == 0  # public ground is never on the private layer
    st.opts.public_only = False  # private land in the main list: the any-route walk decides
    apply_masks(st)
    assert A["usable"][150, 250] and A["usable"][150, 50]


def test_populated_areas_cut_the_score() -> None:
    st = toy()
    st.layers["houses"][100, 250] = 100  # town density, far from the paved road
    st.layers["houses"][100, 252] = 3  # a few rural neighbours
    apply_masks(st)
    fin = st.layers["final"]
    assert fin[100, 250] == pytest.approx(fin[100, 251] * (1 - st.opts.houses_penalty))
    assert fin[100, 252] == pytest.approx(fin[100, 251])


# ---- picking spots ------------------------------------------------------------------------------------------


def bumps(st: ModelState, centers: list[tuple[int, int, float]], sigma: float = 6.0) -> Floats:
    """A score of Gaussian bumps (row, col, height) on the toy grid."""
    r, c = np.mgrid[0:N, 0:N]
    s = sum(h * np.exp(-((r - a) ** 2 + (c - b) ** 2) / (2 * sigma**2)) for a, b, h in centers)
    return np.asarray(s, "float32")


def test_spread_and_zones() -> None:
    xy = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (5000.0, 0.0), (300.0, 0.0)]
    assert spread(xy, per_zone=2, zone_radius_m=800, n=10) == [0, 1, 3]
    assert spread(xy, per_zone=5, zone_radius_m=800, n=2) == [0, 1]
    assert zones(xy, 800) == [1, 1, 1, 2, 1]
    assert zones([], 800) == []


def test_pick_candidates_ranked_spaced_and_zoned() -> None:
    st = toy()
    A = st.layers
    A["score"] = bumps(st, [(50, 50, 90), (50, 90, 80), (60, 130, 70), (250, 250, 60), (200, 60, 50), (150, 150, 40)])
    apply_masks(st)
    A["usable"][150, 150] = False  # nothing is picked where the rules forbid it
    A["final"][~A["usable"]] = 0
    o = dataclasses.replace(st.opts, n_candidates=4, per_zone=2, zone_radius_m=400, candidate_spacing_m=100)
    cands = pick_candidates(st, A["final"], o, A["usable"])
    assert [c["rank"] for c in cands] == [1, 2, 3, 4] and [c["name"] for c in cands] == ["#1", "#2", "#3", "#4"]
    scores = [c["score"] for c in cands]
    assert scores == sorted(scores, reverse=True)
    xy = [st.fine.xy(c["row"], c["col"]) for c in cands]
    for i in range(len(xy)):
        for j in range(i):
            assert math.dist(xy[i], xy[j]) >= 100 - 1e-6
    assert all(A["usable"][c["row"], c["col"]] for c in cands)
    for i in range(len(xy)):  # each spot had fewer than per_zone better spots within zone_radius_m
        assert sum(math.dist(xy[i], xy[j]) < 400 for j in range(i)) < 2
    assert cands[0]["zone"] == 1 and len({c["zone"] for c in cands}) >= 2
    assert pick_candidates(st, A["final"], o, np.zeros_like(A["usable"])) == []


def test_a_weak_area_still_gets_its_best_spots_with_a_note() -> None:
    """Dry ground or solid timber can score low everywhere: the best of it is still offered, and the summary says
    the spots are weak (a fixed cutoff of 5 used to return no spots and no reason)."""
    st = toy()
    A = st.layers
    A["score"] = bumps(st, [(60, 60, 4.0), (220, 220, 3.0)])
    apply_masks(st)
    cands, _ = pick_all(st)
    assert len(cands) == 2 and cands[0]["score"] < 5
    notes = summarize(st, cands, 0, [])["notes"]
    assert len(notes) == 1 and f"scores low (best {cands[0]['score']:.0f} of 100)" in notes[0]
    A["score"] = bumps(st, [(60, 60, 80), (220, 220, 3.0)])  # a strong area keeps the cutoff: no weak filler
    apply_masks(st)
    cands, _ = pick_all(st)
    assert len(cands) == 1 and summarize(st, cands, 0, [])["notes"] == []
    assert pick_notes([], A["usable"]) == ["No camera spots: the usable ground scores near zero everywhere."]
    assert pick_notes([], np.zeros_like(A["usable"])) == []


def test_peak_threshold() -> None:
    assert peak_threshold(90) == 5.0 and peak_threshold(10) == 2.0 and peak_threshold(1) == 0.5


def test_pick_all_private_layer_and_summary() -> None:
    st = toy()
    A = st.layers
    A["score"] = bumps(st, [(60, 60, 80), (220, 220, 70)])
    A["public"][:, 150:] = False
    apply_masks(st)
    cands, priv = pick_all(st)
    assert cands and priv and all(c["public"] for c in cands) and not any(c["public"] for c in priv)
    assert priv[0]["name"] == "P1"
    s = summarize(st, cands, 1.23, priv)
    assert s["n_candidates"] == len(cands) and s["private_land"]["n_spots"] == len(priv)
    assert s["coverage"]["public_fraction"] == 0.5 and s["runtime_s"] == 1.2
    assert s["zones"][0]["best_rank"] == 1
    st.opts.public_only = False
    apply_masks(st)
    assert pick_all(st)[1] == [] and summarize(st, [], 0, [])["private_land"] is None


# ---- explaining a spot --------------------------------------------------------------------------------------


def test_old_state_explains_without_new_factors() -> None:
    st = dataclasses.replace(toy(), unmodeled=frozenset({"travel", "context", "paved_dist"}))
    apply_masks(st)
    d = describe(st, 150, 12)
    assert d["paved_road_distance_m"] is None and not {"travel", "habitat"} & set(d["factors"])
    assert not any("paved road" in r for r in d["reasons"])


def test_describe_new_reasons() -> None:
    st = toy()
    A = st.layers
    A["travel"][150, 150] = 0.9
    A["travel_pos_mid"][150, 150] = -0.9  # a drainage bottom
    A["edge_valley"][150, 150] = 1.0  # the old valley-bottom tie-breaker would also fire here
    A["edge_density"][150, 150] = 0.2
    A["water_density"][150, 150] = 0.25
    A["closed_track_mid"][150, 150] = True
    apply_masks(st)
    r = describe(st, 150, 150)["reasons"]
    text = " | ".join(r)
    assert any(s.startswith("drainage bottom - a natural travel line") for s in r)
    assert "valley bottom" not in r  # said once, not twice
    assert "patchwork of timber and openings" in text
    assert "water around" in text
    assert "closed/gated forest road" in text

    A["travel_pos_mid"][150, 150] = 0.9
    assert any(s.startswith("ridge spine") for s in describe(st, 150, 150)["reasons"])

    near = describe(st, 150, 12)
    assert near["paved_road_distance_m"] == 10
    assert any("from a paved road: traffic and people" in s for s in near["reasons"])
    assert {"travel", "habitat"} <= set(near["factors"])


def _reasons(st: ModelState, **cells: Any) -> list[str]:
    for k, v in cells.items():
        cast("dict[str, Any]", st.layers)[k][100, 100] = v
    apply_masks(st)
    return describe(st, 100, 100)["reasons"]


def test_wind_reasons() -> None:
    st = toy()
    A = st.layers
    A["drainx_mid"][:], A["drainn_mid"][:] = 1.0, 0.0  # drains east; the wind blows east (from 270)
    assert "they line up within 0 deg" in " ".join(_reasons(st, conv_mid=0.5))
    A["drainx_mid"][:], A["drainn_mid"][:] = 0.0, -1.0  # drains south: across the wind
    A["windx_mid"][:], A["windn_mid"][:] = 0.0, -1.0
    assert "gets channeled along this valley (toward S)" in " ".join(_reasons(st))
    st = toy()
    assert "strong cold-air drainage" in " ".join(_reasons(st, drain_mid=0.8, windward_mid=0.5))
    assert "windward side of a ridge in the dawn/dusk W high-pressure wind" in _reasons(st)
    st.wind["consistency"] = 0.26  # an unsteady wind: the reasons say so
    assert (
        "windward side of a ridge in the dawn/dusk W high-pressure wind (an unsteady wind here: consistency R 0.26)"
        in (_reasons(st))
    )
    mine = dataclasses.replace(st, opts=dataclasses.replace(st.opts, wind_from_deg=270.0))
    assert "windward side of a ridge in the W wind" in _reasons(mine)  # the user's own wind


def test_edge_reasons_of_older_states() -> None:
    st = dataclasses.replace(toy(), unmodeled=frozenset({"edge_q_drain", "edge_q_wind", "travel_gate_mid"}))
    A = st.layers
    A["meadow"][100, 110:130] = True
    A["meadow_label"][100, 110:130] = 1
    A["meadow_ha"] = np.array([0, 3.2], "float32")
    A["flowx"][:] = 1.0
    r = _reasons(st, edge_meadow=0.6, edge_downwind=0.5, edge_ridge=0.5, edge_water=0.5, edge_route=0.5)
    assert any("most downwind edge of a 3.2 ha opening" in s and "toward E" in s for s in r)
    assert {"ridgeline", "water edge", "on a trail/two-track through timber (travel route)"} <= set(r)
    assert any(s.startswith("in timber cover overlooking a 3.2 ha opening") for s in _reasons(st, edge_downwind=0))


def test_pinch_water_and_people_reasons() -> None:
    st = toy()
    A = st.layers
    x, y = st.fine.xy(100, 100)
    A["saddle_points"] = [SaddlePoint(row=0, col=0, rise_m=32.0, drop_m=20.0, ridge_axis_deg=0.0, x=x + 30, y=y)]
    A["water_labels"] = ["spring/seep (NHD)"]
    A["water_kind"][100, 100] = 0
    r = _reasons(
        st,
        pinch_saddle=0.5,
        pinch_cliffbase=0.5,
        pinch_clifftop=0.5,
        pinch_bank=0.5,
        pinch_funnel=0.5,
        pinch_fence=0.5,
        water=0.6,
        water_scarcity=0.9,
        houses=40,
        rec_dist=120,
    )
    assert "saddle - crossing here saves about 32 m of climbing (30 m away)" in r
    assert {"cliff rim", "natural travel funnel (movement concentrates here)", "along a fence/rail line"} <= set(r)
    assert "near spring/seep (NHD) - few other water sources within a mile" in r
    assert any(s.startswith("populated area: 40 houses within 500 m") for s in r)
    assert "120 m from a trailhead/campground/parking area: people (score reduced)" in r


def test_winter_reasons_and_season_factor() -> None:
    st = toy()
    assert not any("winter" in s for s in _reasons(st, winter_mid=0.9))  # October: the module is off
    st.month = 1
    r = _reasons(st, winter_mid=0.7, winter_range_mid=True, season=0.9)
    assert {
        "winter: low, sun-facing ground with shallow snow, where deer winter",
        "inside WDFW-mapped deer/elk winter range",
    } <= set(r)
    assert describe(st, 100, 100)["factors"]["season"] == pytest.approx(0.9)
    assert "season" not in describe(st, 50, 50)["factors"]


def test_trail_alternate_beside_a_quiet_road() -> None:
    st = toy()
    A = st.layers
    A["trail_kind"][:, 120:123] = 3  # a two-track running north-south, 100-110 m east of column 100
    A["score"][:] = 50.0
    A["score"][:, 120:123] = 30.0
    A["score"][100, 121] = 45.0  # the best cell beside it
    apply_masks(st)
    alt = describe(st, 100, 100)["trail_alternate"]
    assert alt is not None and alt["kind"] == "two-track" and alt["direction"] == "E"
    pen = float(site_penalty(A, st.opts)[100, 121])
    assert alt["distance_m"] == 105 and alt["score"] == pytest.approx(45.0 * pen, abs=0.05)
    assert alt["walk_miles"] is not None
    assert alt["reason"].startswith(
        f"alternate spot on the two-track through this zone, 105 m E (score {45 * pen:.0f})"
    )
    assert describe(st, 100, 121)["trail_alternate"] is None  # already beside it
    close = describe(st, 100, 116)["trail_alternate"]  # 25 m off the track
    assert close is not None and close["reason"].startswith("the two-track passes 25 m E of this spot")
    assert describe(st, 100, 50)["trail_alternate"] is None  # 350 m away: out of reach
    A["rec_dist"][:, 120:123] = 50.0  # a trailhead there: busy
    assert describe(st, 100, 100)["trail_alternate"] is None
    A["rec_dist"][:] = 2000.0
    A["score"][:, 120:123] = 10.0  # less than half the spot's score
    apply_masks(st)
    assert describe(st, 100, 100)["trail_alternate"] is None
    old = dataclasses.replace(toy(), unmodeled=frozenset({"trail_kind"}))
    apply_masks(old)
    assert describe(old, 100, 100)["trail_alternate"] is None


def test_trail_alternate_prefers_quiet_lines_and_flags_open_roads(monkeypatch: pytest.MonkeyPatch) -> None:
    st = toy()
    A = st.layers
    A["trail_kind"][:, 120:123] = 3  # a two-track 100-110 m east...
    A["trail_kind"][:, 80:83] = 4  # ...and a trail 50-60 m west
    A["score"][:] = 50.0
    A["score"][:, 120:123] = 45.0
    A["score"][:, 80:83] = 35.0
    A["road_dist"][:] = 500.0
    A["road_dist"][:, 119:124] = 10.0  # the two-track is open to vehicles: on the road you drive in on
    apply_masks(st)
    alt = describe(st, 100, 100)["trail_alternate"]
    assert alt is not None and alt["kind"] == "trail" and not alt["open_to_vehicles"]  # quiet first
    assert "quiet dirt roads and trails" in alt["reason"]
    A["trail_kind"][:, 80:83] = 0  # no quiet line left: the open road, said plainly
    apply_masks(st)
    alt = describe(st, 100, 100)["trail_alternate"]
    assert alt is not None and alt["open_to_vehicles"] and alt["kind"] == "two-track"
    assert "two-track, on or right beside a road open to vehicles this month," in alt["reason"]
    assert "theft" in alt["reason"]
    monkeypatch.setattr(analyze_mod, "PLACEMENT", PLACEMENT)  # the real floor: no alternate under 30
    A["score"][:, 120:123] = 25.0
    apply_masks(st)
    assert describe(st, 100, 100)["trail_alternate"] is None


def test_reasons_name_which_wind_they_mean() -> None:
    st = toy()
    assert wind_name(st, "dawn").startswith("the dawn/dusk ") and wind_name(st, "dawn").endswith(" high-pressure wind")
    assert wind_name(st, "day").startswith("the daytime ")
    mine = dataclasses.replace(st, opts=dataclasses.replace(st.opts, wind_from_deg=270.0))
    assert wind_name(mine, "dawn") == wind_name(mine, "day") == "the W wind"


def test_walk_fields_unreachable_and_private_route() -> None:
    st = toy()
    A = st.layers
    A["walk_m"][:] = A["walk_s"][:] = 1e7
    A["walk_any_m"][:] = 800.0
    A["public"][100, 100] = False
    apply_masks(st)
    d = describe(st, 100, 100)
    assert d["walk_miles"] == round(800 / MILE_M, 2) and not d["public"]  # private: the any-route walk
    pub = describe(st, 50, 50)
    assert pub["walk_miles"] is None and pub["walk_minutes"] is None


# ---- factor wiring ------------------------------------------------------------------------------------------


def test_compute_travel_on_a_context() -> None:
    ctx = toys.context(_sine_dem((120, 400)), res=10.0)
    ctx.layers["saddle_points"] = []
    F.compute_travel(ctx, log=lambda *_: None)
    A = ctx.layers
    assert A["travel"].dtype == np.float32 and A["travel"][60, 150] > 0.8 and A["travel"][60, 100] < 0.1
    assert A["travel_pos_mid"][60, 150] < 0 < A["travel_pos_mid"][60, 50]
    assert 0.4 < A["travel"][60, 50] <= 0.5  # a spine away from any crossing keeps half its line


def _ew_ridges(month: int) -> Layers:
    """Ridges running east-west (spines at rows 50 and 250, a bottom at row 150), flanks 0-43 deg."""
    y = np.arange(300) * 10.0
    z = (300 * np.sin(2 * np.pi * y / 2000.0))[:, None].repeat(120, 1).astype("float32")
    ctx = toys.context(z, res=10.0)
    ctx.month = month
    ctx.layers["saddle_points"] = []
    F.compute_travel(ctx, log=lambda *_: None)
    return ctx.layers


def test_winter_thermals_and_saddles_open_ridge_spines() -> None:
    oct_, jan = _ew_ridges(10), _ew_ridges(1)
    assert not oct_["travel_thermal_mid"].any() and jan["travel_thermal_mid"][50, 60] > 0.5
    assert jan["travel"][50, 60] > 1.3 * oct_["travel"][50, 60]  # winter: the ridge above the south flank
    assert jan["travel"][150, 60] == oct_["travel"][150, 60]  # bottoms count all year
    ctx = toys.context(_sine_dem((120, 400)), res=10.0)
    ctx.layers["saddle_points"] = [SaddlePoint(row=60, col=50, rise_m=30.0, drop_m=20.0, ridge_axis_deg=0.0, x=0, y=0)]
    F.compute_travel(ctx, log=lambda *_: None)
    assert ctx.layers["travel_gate_mid"][60, 50] == 1 and ctx.layers["travel"][60, 50] > 0.8


def test_travel_and_edge_reasons_of_model_v5() -> None:
    st = toy()
    spine = dict(travel=0.9, travel_pos_mid=0.9)
    assert "ridge spine at a saddle/junction - lions cross here" in _reasons(st, **spine, travel_gate_mid=0.8)
    st = toy()
    r = _reasons(st, **spine, travel_thermal_mid=0.8)
    assert "ridgeline above large south/southeast slopes - winter thermals" in r
    st = toy()
    A = st.layers
    A["meadow"][100, 95:130] = True
    A["meadow_label"][100, 95:130] = 1
    A["meadow_ha"] = np.array([0, 3.2], "float32")
    A["drainx_mid"][:], A["drainn_mid"][:] = 0.0, -1.0  # drains south
    r = _reasons(st, edge_meadow=0.6, edge_downwind=0.5, edge_q_drain=0.9, edge_q_wind=0.8)
    assert (
        "at the timber edge, in the open of a 3.2 ha opening, at its most downwind end for the evening cold-air "
        "drainage (toward S) and the daytime WSW high-pressure wind - where kills cluster"
    ) in r
    A["meadow"][100, 100] = False
    r = _reasons(st, edge_q_drain=0.2)
    assert any(
        s.startswith(
            "in timber overlooking a 3.2 ha opening, at its most downwind end for the daytime WSW high-pressure wind"
        )
        for s in r
    )
    r = _reasons(st, edge_downwind=0.0)
    assert "in timber cover overlooking a 3.2 ha opening - can watch prey without being seen" in r
