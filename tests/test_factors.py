"""Each factor on small synthetic rasters: what it should reward, and what it must not."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Point, box

import toys
from cougarmap import factors as F
from cougarmap.arrays import Floats
from cougarmap.config import MILE_M, WINTER, EdgeBand
from cougarmap.context import Context
from cougarmap.sources import vector as vec
from cougarmap.sources.vector import Feature, Water

RES = toys.RES
quiet = lambda *_: None


def flat(n: int = 200, width: int | None = None) -> Floats:
    return np.zeros((n, width or n), "float32")


def run(ctx: Context, *steps: str) -> Context:
    for s in steps:
        getattr(F, f"compute_{s}")(ctx, quiet)
    return ctx


# ---- edges --------------------------------------------------------------------------------------------------


def meadow_chm(n: int = 200) -> Floats:
    """20 m timber with a 100 x 100 m meadow in the middle, a 6 m road cut and a 15 x 15 m glade."""
    chm = np.full((n, n), 20.0, "float32")
    chm[80:100, 80:100] = 0  # 100 m meadow (5 m cells)
    chm[150:152, :] = 0  # 10 m-wide cut... at 5 m cells: 2 cells
    chm[20:23, 20:23] = 0  # 225 m2 glade
    return chm


def test_openings_keep_meadows_drop_strips_and_glades() -> None:
    chm = meadow_chm()
    op = F.openings(chm, np.zeros(chm.shape, bool), RES)
    assert op.n == 1 and op.meadow[90, 90] and not op.meadow[151, 50] and not op.meadow[21, 21]
    assert op.area_m2[0] == pytest.approx(100 * 100)
    lake = np.zeros(chm.shape, bool)
    lake[80:100, 80:100] = True
    assert F.openings(chm, lake, RES).n == 0  # open water is not a meadow


def test_hunting_edge_sits_in_timber_beside_the_meadow() -> None:
    chm = meadow_chm()
    op = F.openings(chm, np.zeros(chm.shape, bool), RES)
    hunt, _ = F.hunting_edge(chm, op.meadow, RES)
    row = hunt[90]
    assert row[90] == 0  # in the meadow
    assert row[77] > 0.5 and row[103] > 0.5  # 10-15 m back in the timber
    assert row[60] == 0  # 100 m deep in timber
    assert hunt.dtype == np.float32 and hunt.max() <= 1


def test_hunting_edge_reaches_into_the_opening() -> None:
    chm = meadow_chm()
    op = F.openings(chm, np.zeros(chm.shape, bool), RES)
    hunt, nearest = F.hunting_edge(chm, op.meadow, RES)
    row = hunt[90]
    assert row[81] > 0.9 and row[82] > 0.9 and row[98] > 0.9  # 10-15 m out from the timber
    assert row[85] == 0 and row[90] == 0  # 30 m out and the middle of the meadow
    in_cover, _ = F.hunting_edge(chm, op.meadow, RES, EdgeBand(open_weight=0.0))
    assert np.array_equal(hunt[~op.meadow], in_cover[~op.meadow]) and not in_cover[op.meadow].any()
    assert nearest[0][90, 85] == 90 and nearest[1][90, 85] == 85  # an opening cell's nearest opening is itself
    grass = np.zeros(chm.shape, "float32")
    assert not F.hunting_edge(grass, np.ones(chm.shape, bool), RES)[0].any()  # open flats: no cover, no edge


def test_downwind_end_of_the_edge_scores_highest() -> None:
    chm = meadow_chm()
    op = F.openings(chm, np.zeros(chm.shape, bool), RES)
    hunt, nearest = F.hunting_edge(chm, op.meadow, RES)
    east = np.ones(chm.shape, "float32")  # air moving east over the meadow
    q = F.downwind_position(hunt, op, nearest, east, np.zeros_like(east), toys.grid(200))
    west, east = q[90, 70:80], q[90, 100:110]  # the band on either side of the meadow
    assert east.min() > 0.75 and east.max() > 0.95 and west.max() < 0.25
    none = F.openings(np.full(chm.shape, 20.0, "float32"), np.zeros(chm.shape, bool), RES)
    assert not F.downwind_position(hunt, none, nearest, east, east, toys.grid(200)).any()


def edge_ctx(drain_east: bool = True, day_from_deg: float = 270.0) -> Context:
    """meadow_chm on flat ground, a trail along row 40, cold air draining east (or west) and a daytime wind."""
    g = toys.grid(200)
    x0, y0 = g.xy(0, 0)
    track = toys.feature(g, LineString([(x0, y0 - 40 * RES), (x0 + 199 * RES, y0 - 40 * RES)]), highway="track")
    ctx = toys.context(flat(), meadow_chm(), osm=[track])
    ctx.wind["day_from_deg"] = day_from_deg
    A = ctx.layers
    A["drainx_mid"] = np.full(g.shape, 1.0 if drain_east else -1.0, "float32")
    A["drainn_mid"] = np.zeros(g.shape, "float32")
    A["landform_mid"] = np.full(g.shape, 6, np.int8)
    A["acc_mid"] = np.ones(g.shape)
    return run(ctx, "edges")


def test_compute_edges_on_a_context() -> None:
    A = edge_ctx().layers
    assert A["edge_downwind"][90, 103] > A["edge_downwind"][90, 77]
    assert A["edge_route"][40, 50] > 0.3 and A["edge_route"][60, 50] == 0  # along the trail through timber
    assert A["meadow_ha"][1] == pytest.approx(1.0, abs=0.02) and A["meadow_label"][90, 90] == 1
    assert A["edges"].min() >= 0 and A["edges"].max() <= 1


def test_downwind_ends_per_air_current() -> None:
    both = edge_ctx(drain_east=True, day_from_deg=270.0).layers  # drainage east, wind blowing east
    assert both["edge_q_drain"][90, 103:110].max() > 0.95 and both["edge_q_wind"][90, 103:110].max() > 0.95
    assert both["edge_q"][90, 103] > 0.7 and both["edge_q"][90, 77] < 0.1
    split = edge_ctx(drain_east=True, day_from_deg=90.0).layers  # drainage east, wind blowing west
    q = split["edge_q"][90]
    end = int(np.argmax(split["edge_q_drain"][90]))  # the band's east end (mirrored about the meadow at 179 - c)
    assert q[end] == pytest.approx(0.5, abs=0.05) and q[179 - end] == pytest.approx(0.5, abs=0.05)
    assert q[103] == pytest.approx(q[76], abs=0.02)  # the two ends mirror each other
    assert split["edge_meadow"][90, 103] == pytest.approx(split["edge_meadow"][90, 76], rel=0.05)
    assert both["edge_meadow"][90, 103] > split["edge_meadow"][90, 103] > both["edge_meadow"][90, 76]


def test_daytime_wind_direction() -> None:
    w = toys.state().wind
    assert F.daytime_wind_deg(w, None) == 250.0 and F.daytime_wind_deg(w, 400.0) == 40.0  # the user's wind wins
    w["day_from_deg"] = math.nan  # no daytime climatology: the dawn/dusk wind
    assert F.daytime_wind_deg(w, None) == w["from_deg"]


# ---- wind ---------------------------------------------------------------------------------------------------


def valley(n: int = 160) -> Floats:
    """A V valley running north-south (axis at the middle column), draining south."""
    r, c = np.mgrid[0:n, 0:n]
    z: Floats = (3.0 * np.abs(c - n // 2) + 0.8 * (n - r)).astype("float32") + 1000
    return z


def test_drainage_and_wind_agree_on_the_valley_floor() -> None:
    ctx = toys.context(valley(), res=10.0)
    run(ctx, "terrain", "wind")
    A = ctx.layers
    assert -1.0 <= A["drainn_mid"][120, 80] < -0.5  # cold air drains south
    with_wind = A["conv_mid"][120, 80]
    ctx2 = toys.context(valley(), res=10.0)
    ctx2.wind = {**ctx2.wind, "from_deg": 180.0}  # from the south: against the drainage
    run(ctx2, "terrain", "wind")
    ctx.wind = {**ctx.wind, "from_deg": 0.0}
    run(ctx, "wind")
    assert ctx.layers["conv_mid"][120, 80] > ctx2.layers["conv_mid"][120, 80]
    assert with_wind >= 0
    for c in (ctx, ctx2):
        w = c.layers["wind"]
        assert np.isfinite(w).all() and w.min() >= 0 and w.max() <= 1


def test_hillsides_drain_too() -> None:
    """'Down the valley and off the hillsides': an open valley side 300-400 m from the floor drains, less than the
    floor itself, and the same at a coarser grid."""
    drains = []
    for res in (10.0, 20.0):
        n = int(1600 / res)
        r, c = np.mgrid[0:n, 0:n] * res  # the same valley as valley(160) at 10 m, in metres
        ctx = toys.context((0.3 * np.abs(c - 800) + 0.08 * (1600 - r)).astype("float32") + 1000, res=res)
        run(ctx, "terrain", "wind")
        A = ctx.layers
        side, floor = A["drain_mid"][int(0.75 * n), int(0.25 * n)], A["drain_mid"][int(0.75 * n), n // 2]
        assert 0.1 < side < floor
        drains.append(side)
    assert drains[0] == pytest.approx(drains[1], abs=0.08)


def test_drain_strength_scale() -> None:
    assert F.drain_strength(np.array([1.0, 50.0])).max() == 0
    assert F.drain_strength(np.array([10**5.5, 1e7])).min() == 1
    assert 0.2 < F.drain_strength(np.array([400.0]))[0] < 0.35  # a 400 m open slope


# ---- pinch points -------------------------------------------------------------------------------------------


def test_cliff_base_bank_and_fence() -> None:
    n = 160
    z = np.zeros((n, n), "float32")
    z[:, 100:] = 30.0  # a 30 m step facing west
    z += np.arange(n, dtype="float32")[:, None] * 0.05
    g = toys.grid(n)
    x0, y0 = g.xy(0, 0)
    fence = toys.feature(g, LineString([(x0, y0 - 20 * RES), (x0 + 60 * RES, y0 - 20 * RES)]), barrier="fence")
    lake = toys.feature(g, box(x0 + 10 * RES, y0 - 140 * RES, x0 + 40 * RES, y0 - 110 * RES), ftype=390)
    water = Water(points=[], flowlines=[], waterbodies=[lake])
    ctx = toys.context(z, osm=[fence], water=water)
    ctx.layers["meadow"] = np.zeros(z.shape, bool)  # no openings (compute_edges)
    run(ctx, "terrain", "pinch")
    A = ctx.layers
    assert A["cliff"][:, 99:101].any()
    assert A["pinch_cliffbase"][80, 96] > 0.5 and A["pinch_clifftop"][80, 104] <= 0.4
    assert A["pinch_fence"][20, 30] == pytest.approx(0.35, abs=0.02) and A["pinch_fence"][60, 30] == 0
    assert A["pinch_bank"][125, 45] > 0 and A["pinch_bank"][125, 25] == 0  # beside the lake, not in it
    assert A["pinch"].min() >= 0 and A["pinch"].max() <= 1 and isinstance(A["saddle_points"], list)


def test_saddle_is_painted_and_listed() -> None:
    n = 201
    y, x = np.mgrid[0:n, 0:n].astype(float)
    x, y = (x - 100) * 10, -(y - 100) * 10
    z = (200 - 0.4 * np.abs(x) + 40 * (1 - np.exp(-(y**2) / (2 * 300**2)))).astype("float32")
    ctx = toys.context(z, res=10.0)
    ctx.layers["meadow"] = np.zeros(z.shape, bool)
    run(ctx, "terrain", "pinch")
    sp = ctx.layers["saddle_points"]
    assert sp and {"x", "y", "rise_m"} <= set(sp[0])
    best = max(sp, key=lambda s: s["rise_m"])
    assert ctx.layers["pinch_saddle"][best["row"], best["col"]] >= 0.25


def pond_pinch_ctx() -> Context:
    """A 2.5 ha pond (cols 80-104, rows 80-119) between an opening to the west (cols 40-69) and a 39 degree slope
    rising from col 112, with a seasonal creek coming in from the north; the ground tilts gently south."""
    n = 200
    rows = np.arange(n, dtype="float32")[:, None]
    cols = np.arange(n, dtype="float32")[None, :]
    z = (0.02 * rows + np.clip(cols - 112, 0, None) * 4.0).astype("float32")
    g = toys.grid(n)
    x0, y0 = g.xy(0, 0)
    h = RES / 2
    pond = toys.feature(g, box(x0 + 80 * RES - h, y0 - 120 * RES + h, x0 + 105 * RES - h, y0 - 80 * RES + h), ftype=390)
    creek = toys.feature(g, LineString([(x0 + 92 * RES, y0 - 20 * RES), (x0 + 92 * RES, y0 - 100 * RES)]), fcode=46003)
    ctx = toys.context(z, water=Water(points=[], flowlines=[creek], waterbodies=[pond]))
    meadow = np.zeros((n, n), bool)
    meadow[60:140, 40:70] = True
    ctx.layers["meadow"] = meadow
    return run(ctx, "terrain", "pinch")


def test_a_pond_between_a_slope_and_an_opening_is_a_pinch() -> None:
    A = pond_pinch_ctx().layers
    pw, kind = A["pinch_water"], A["pinch_water_kind"]
    # the 35-45 m strip between the shore and the foot of the slope: nearly full strength
    assert pw[100, 108] == pytest.approx(0.8 * (100 - 50) / 60, abs=0.15) and kind[100, 108] == F.WATER_PINCH_STEEP
    # the 50 m gap to the opening: 0.8 x (100 - 55) / 60 at its middle
    assert pw[100, 75] == pytest.approx(0.6, abs=0.01) and kind[100, 75] == F.WATER_PINCH_OPENING
    assert pw[100, 90] == 0 and pw[100, 60] == 0  # not on the water, not in the opening itself
    assert pw[74, 84] == 0  # beside the pond with the opening off to the side, not across: no squeeze
    assert pw[78, 92] > 0.5 and kind[78, 92] == F.WATER_PINCH_END  # where the creek comes in
    assert pw[30, 92] == 0  # the creek far from the pond is no pinch
    assert A["pinch"][100, 108] >= pw[100, 108] + 0.1  # beside the bank too: the extra-component bonus


def test_water_squeeze_needs_the_barrier_across_and_a_narrow_gap() -> None:
    n = 100
    water = np.zeros((n, n), bool)
    water[:, 40:50] = True  # a long pond, shore at col 50 (east) and col 39 (west)
    barrier = np.zeros((n, n), np.int8)
    barrier[:50, 57:] = F.WATER_PINCH_CLIFF  # north half: a cliff 35 m east of the shore
    barrier[50:75, 61:] = F.WATER_PINCH_CLIFF  # then 55 m east
    barrier[75:, 70:] = F.WATER_PINCH_CLIFF  # then 100 m east
    v, kind = F.water_squeeze(water, barrier, 5.0)
    assert v[25, 52] == pytest.approx(0.8) and kind[25, 52] == F.WATER_PINCH_CLIFF  # gap 40 m: full
    assert v[25, 55] == pytest.approx(0.8)  # the gap is the same across the strip
    assert v[60, 52] == pytest.approx(0.8 * (100 - 60) / 60, abs=0.01)  # gap 60 m
    assert v[90, 52] == 0 and kind[90, 52] == 0  # gap 105 m
    assert not v[:, :40].any()  # west shore: the cliff is on the same side as the water, not across
    assert v[25, 58] == 0  # the cliff itself
    two = np.zeros((n, n), bool)
    two[:, 20:30], two[:, 42:52] = True, True  # two ponds 60 m apart
    v2, k2 = F.water_squeeze(two, np.zeros((n, n), np.int8), 5.0)
    assert v2[50, 35] == pytest.approx(0.8 * (100 - 65) / 60, abs=0.01) and k2[50, 35] == F.WATER_PINCH_POND
    assert not F.water_squeeze(np.zeros((n, n), bool), barrier, 5.0)[0].any()


# ---- water --------------------------------------------------------------------------------------------------


def spring_ctx(perennial: bool = False, n: int = 200) -> Context:
    g = toys.grid(n)
    x0, y0 = g.xy(0, 0)
    spring = Feature(Point(*g.to_lonlat(x0 + 100 * RES, y0 - 100 * RES)), {"ftype": 458})
    lines = []
    if perennial:
        lines.append(toys.feature(g, LineString([(x0, y0 - 20 * RES), (x0 + 199 * RES, y0 - 20 * RES)]), fcode=46006))
    lake = toys.feature(g, box(x0 + 140 * RES, y0 - 190 * RES, x0 + 190 * RES, y0 - 140 * RES), ftype=390)  # 6 ha
    return toys.context(flat(n), water=Water(points=[spring], flowlines=lines, waterbodies=[lake]))


def test_spring_strength_by_distance_and_lakes_score_zero() -> None:
    ctx = run(spring_ctx(), "water")
    A = ctx.layers
    w = A["water"]
    assert w[100, 100] == pytest.approx(w[100, 107], rel=1e-6)  # full strength within 40 m
    assert w[100, 100] > w[100, 130] > w[100, 155] and w[100, 165] == 0  # fading, gone past 300 m
    assert A["water_labels"][A["water_kind"][100, 100]] == "spring/seep (NHD)"
    assert A["lake"][170, 170] and w[170, 170] == 0
    assert A["water_kind"][5, 5] == -1


def test_scarcity_is_a_bonus_where_permanent_water_is_rare() -> None:
    dry = run(spring_ctx(), "water").layers["water_scarcity"][100, 100]
    wet = run(spring_ctx(perennial=True), "water").layers["water_scarcity"][100, 100]
    assert 0.75 <= wet < dry <= 1.0


def pond_ctx(side_m: float) -> Context:
    """A flat 1 km square with one square pond (side_m on a side) in the middle and nothing else."""
    g = toys.grid(200)
    x0, y0 = g.xy(0, 0)
    cx, cy, h = x0 + 100 * RES, y0 - 100 * RES, side_m / 2
    pond = toys.feature(g, box(cx - h, cy - h, cx + h, cy + h), ftype=390)
    return toys.context(flat(), water=Water(points=[], flowlines=[], waterbodies=[pond]))


def test_small_ponds_are_drinking_water_and_only_big_lakes_are_permanent() -> None:
    none = run(toys.context(flat()), "water").layers["water_scarcity"]
    pond = run(pond_ctx(math.sqrt(3e4)), "water").layers  # 3 ha
    edge = 100 + round(math.sqrt(3e4) / 2 / RES) + 4  # ~20 m off the shore
    assert pond["lake"][100, 100] and pond["water"][100, 100] == 0  # no camera in the pond itself
    assert pond["water"][100, edge] == pytest.approx(0.85 * pond["water_scarcity"][100, edge], rel=1e-3)
    assert pond["water_labels"][pond["water_kind"][100, edge]] == "small pond/marsh (under 5 ha)"
    assert np.allclose(pond["water_scarcity"], none)  # a pond is no permanent water
    lake = run(pond_ctx(math.sqrt(6e4)), "water").layers  # 6 ha
    assert lake["water"].max() == 0 and lake["water_scarcity"][100, 100] < none[100, 100]


def test_user_water_pins_count() -> None:
    ctx = toys.context(flat())
    lon, lat = ctx.fine.to_lonlat(*ctx.fine.xy(50, 50))
    ctx.aoi.user_points = [dict(name="Guzzler", lat=lat, lon=lon, kind="water", note="")]
    run(ctx, "water")
    assert ctx.layers["water"][50, 50] > 0.7
    assert ctx.layers["water_labels"][ctx.layers["water_kind"][50, 50]] == "known water (your pin)"


# ---- winter -------------------------------------------------------------------------------------------------


def ridge_and_floor(n: int = 160, res: float = 10.0) -> Floats:
    """An east-west ridge (20 deg flanks facing north and south) above a flat valley floor at 300 m."""
    i = np.arange(n, dtype="float32")
    prof = np.maximum(500 - math.tan(math.radians(20)) * res * np.abs(i - n // 2), 300)
    return prof[:, None].repeat(n, 1).astype("float32")


def test_winter_favours_low_sun_facing_ground() -> None:
    ctx = toys.context(ridge_and_floor(), res=10.0)
    ctx.month = 1
    run(ctx, "season")
    A = ctx.layers
    w, season = A["winter_mid"], A["season"]
    assert season[110, 80] > season[50, 80]  # the south flank over the north flank, at the same height
    assert season[150, 80] > season[80, 80]  # the valley floor over the summit
    assert season.min() >= WINTER.floor and season.max() <= 1 and w.max() <= 1
    assert not A["winter_range_mid"].any()  # no mapped range here


def test_snow_ramp_and_winter_range_feather() -> None:
    f = F.snow_factor(np.array([0, 25, 42.5, 60, 100, np.nan], "float32"))
    assert np.allclose(f, [1, 1, 0.6, 0.2, 0.2, 1])
    inside = np.zeros((1, 100), bool)
    inside[0, :20] = True
    r = F.winter_range_factor(inside, 10.0)
    assert r[0, 10] == 1 and r[0, 39] == pytest.approx(0.9) and r[0, 80] == pytest.approx(0.8)
    assert np.all(F.winter_range_factor(np.zeros((3, 3), bool), 10.0) == pytest.approx(0.8))


def test_snow_and_winter_range_enter_the_season() -> None:
    def season(snow_cm: float | None, phs: bool, month: int = 1) -> Context:
        ctx = toys.context(flat(), res=10.0)
        ctx.month = month
        if snow_cm is not None:
            w, s_, e, n = ctx.fine.lonlat_bounds(pad_m=500)
            ctx.snow_cm = (np.full((40, 40), snow_cm, "float32"), Affine((e - w) / 40, 0, w, 0, -(n - s_) / 40, n))
        if phs:
            x0, y0 = ctx.fine.xy(0, 0)
            ctx.phs = [toys.feature(ctx.fine, box(x0, y0 - 500, x0 + 500, y0), occurrence_name="Elk")]
            vec.project_features(ctx.phs, ctx.fine)
        return run(ctx, "season")

    shallow, deep = season(10.0, False).layers["season"], season(80.0, False).layers["season"]
    assert deep[100, 100] < shallow[100, 100] == pytest.approx(season(None, False).layers["season"][100, 100])
    ranged = season(10.0, True).layers
    assert ranged["winter_range_mid"][20, 20] and not ranged["winter_range_mid"][150, 150]
    assert ranged["season"][20, 20] == pytest.approx(shallow[20, 20]) and ranged["season"][150, 150] < shallow[150, 150]
    assert not season(10.0, True, month=4).layers["winter_range_mid"].any()  # April: deer are off it


# ---- people: buildings, traffic -----------------------------------------------------------------------------


def test_houses_are_counted_within_the_radius_and_sheds_are_not() -> None:
    ctx = toys.context(flat())
    ctx.opts.houses_radius_m = 150.0  # the toy grid is small; the rule is the same at 500 m

    def at(r: int, c: int, area: float = 150.0) -> tuple[float, float, float]:
        lon, lat = ctx.fine.to_lonlat(*ctx.fine.xy(r, c))
        return float(lon), float(lat), area

    ctx.buildings = [at(50, 50), at(52, 50), at(50, 53), at(48, 49), at(150, 150), (0.0, 0.0, 150.0)]
    ctx.buildings.append(at(100, 100, area=28.0))  # a shed, blind or trailer: not a house
    run(ctx, "houses")
    n = ctx.layers["houses"]
    assert n[50, 50] == 4 and n[150, 150] == 1 and n[100, 100] == 0
    assert n[50, 90] == 0  # 185-200 m from the cluster: outside the radius


def test_recreation_distance_on_the_grid() -> None:
    ctx = toys.context(flat())
    assert run(ctx, "recreation").layers["rec_dist"].min() == F.PAVED_DIST_CAP_M  # no site at all: the cap
    lon, lat = ctx.fine.to_lonlat(*ctx.fine.xy(50, 50))
    ctx.rec_points = [Feature(Point(lon, lat), {"highway": "trailhead"})]
    vec.project_features(ctx.rec_points, ctx.fine)
    d = run(ctx, "recreation").layers["rec_dist"]
    assert d[50, 50] == pytest.approx(0, abs=1) and d[50, 90] == pytest.approx(40 * RES, abs=2)
    assert d.dtype == np.float32


def test_quiet_roads_and_trails_for_the_placement_suggestion() -> None:
    g = toys.grid(200)
    x0, y0 = g.xy(0, 0)

    def row_line(r: int, **props: object) -> Feature:
        return toys.feature(g, LineString([(x0, y0 - r * RES), (x0 + 199 * RES, y0 - r * RES)]), **props)

    osm = [
        row_line(20, highway="track"),
        row_line(60, highway="path"),
        row_line(100, highway="track", surface="asphalt"),  # paved: never quiet
        row_line(140, highway="track", access="private"),
        row_line(180, highway="footway"),  # sidewalks and town paths
    ]
    ctx = toys.context(flat(), osm=osm)
    seasonal = dict(passengervehicle="open", passengervehicle_datesopen="05/01-11/30")
    ctx.mvum = [row_line(40, **seasonal), row_line(80, passengervehicle="open")]
    vec.project_features(ctx.mvum, ctx.fine)
    ctx.layers["usfs_mid"] = np.ones(g.shape, bool)
    run(ctx, "trails")
    kind = ctx.layers["trail_kind"]
    names = {r: F.TRAIL_KINDS[kind[r, 100] - 1] if kind[r, 100] else None for r in (20, 40, 60, 80, 100, 140, 180)}
    # the track is in national forest and off the MVUM: a closed forest road. The seasonal forest road is open in
    # October (May-November), so it is on the drivable network then, not quiet
    assert names == {
        20: "closed forest road",
        40: None,
        60: "trail",
        80: None,
        100: None,
        140: None,
        180: None,
    }
    ctx.month = 1  # closed for the winter: quiet
    run(ctx, "trails")
    assert F.TRAIL_KINDS[ctx.layers["trail_kind"][40, 100] - 1] == "seasonal forest road"
    assert kind[23, 100] and not kind[24, 100]  # within 15 m of the line


def test_paved_distance_and_closed_forest_roads() -> None:
    g = toys.grid(200)
    x0, y0 = g.xy(0, 0)
    paved = toys.feature(g, LineString([(x0, y0), (x0 + 199 * RES, y0)]), highway="secondary")
    gated = toys.feature(g, LineString([(x0, y0 - 100 * RES), (x0 + 199 * RES, y0 - 100 * RES)]), highway="track")
    open_track = toys.feature(g, LineString([(x0, y0 - 150 * RES), (x0 + 199 * RES, y0 - 150 * RES)]), highway="track")
    ctx = toys.context(flat(), osm=[paved, gated, open_track])
    mv = toys.feature(g, LineString([(x0, y0 - 150 * RES), (x0 + 199 * RES, y0 - 150 * RES)]))
    mv.props.update(passengervehicle="open", passengervehicle_datesopen="05/01-11/30")
    vec.project_features([mv], g)
    ctx.mvum = [mv]
    ctx.layers["usfs_mid"] = np.ones(g.shape, bool)
    run(ctx, "traffic")
    A = ctx.layers
    assert A["paved_dist"][0, 50] < RES and A["paved_dist"][100, 50] == pytest.approx(500, abs=RES)
    assert A["paved_dist"].max() <= F.PAVED_DIST_CAP_M
    assert A["closed_track_mid"][100, 50] and not A["closed_track_mid"][150, 50]


# ---- access and land ------------------------------------------------------------------------------------------


def access_ctx(public_strip: bool = True) -> Context:
    n, width = 120, 520
    g = toys.grid(n, width=width)
    x0, y0 = g.xy(0, 0)
    road = toys.feature(g, LineString([(x0, y0), (x0, y0 - (n - 1) * RES)]), highway="tertiary")
    lake = toys.feature(g, box(x0 + 200 * RES, y0 - 60 * RES, x0 + 230 * RES, y0 - 30 * RES), ftype=390)
    ctx = toys.context(flat(n, width), osm=[road], water=Water(points=[], flowlines=[], waterbodies=[lake]))
    A = ctx.layers
    A["slope_mid"] = np.zeros(g.shape, "float32")
    A["usfs_mid"] = np.zeros(g.shape, bool)
    A["public_mid"] = np.ones(g.shape, bool)
    if public_strip:
        A["public_mid"][:, 100:110] = False  # a private strip the public walk can't cross
    return ctx


def test_walk_grows_from_the_road_and_respects_private_land_and_lakes() -> None:
    ctx = run(access_ctx(), "access")
    A = ctx.layers
    row = A["walk_any_m"][100]
    assert np.all(np.diff(row[:150]) > 0)  # farther east, longer walk
    assert A["walk_any_m"][100, 50] == pytest.approx(50 * RES, rel=0.01)
    assert A["walk_m"][100, 150] >= 1e7 and A["walk_any_m"][100, 150] < MILE_M  # beyond the private strip
    assert A["walk_any_m"][45, 215] >= 1e7  # in the lake
    assert A["walk_any_m"][100, 519] >= 1e7  # 2.6 km: past the walk limit (x 1.5)
    assert A["road_dist"][100, 20] == pytest.approx(20 * RES, abs=RES) and A["road_mid"][:, 0].all()
    assert A["walk_pred_mid"].shape == (120 * 520,)


def test_no_roads_is_noted() -> None:
    ctx = access_ctx()
    ctx.osm = []
    run(ctx, "access")
    assert ctx.notes == ["No open roads found near this area."] and (ctx.layers["walk_any_m"] >= 1e7).all()


def test_open_access_wins_where_land_overlaps() -> None:
    g = toys.grid(100)
    x0, y0 = g.xy(0, 0)
    ctx = toys.context(flat(100))
    oa = toys.feature(g, box(x0, y0 - 99 * RES, x0 + 60 * RES, y0), unit_nm="Test NF", mngnm_desc="Forest Service")
    oa.props["pub_access"] = "OA"
    ra = toys.feature(g, box(x0 + 40 * RES, y0 - 99 * RES, x0 + 99 * RES, y0), unit_nm="Closed", pub_access="RA")
    ctx.land = [ra, oa]
    vec.project_features(ctx.land, g)
    run(ctx, "land")
    A = ctx.layers
    assert A["public"][50, 50] and not A["public"][50, 80] and A["usfs_mid"][50, 10]
    assert A["land_names"][A["land_id"][50, 50]] == "Test NF (Forest Service)"
    assert A["land_names"][0] == "private / unknown" and A["land_access"][0] is None
