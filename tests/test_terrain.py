import math

import numpy as np
import pytest

from cougarmap import terrain as T
from cougarmap.arrays import Floats

RES = 5.0


def xy(n: int = 201) -> tuple[Floats, Floats]:
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return (x - n // 2) * RES, -(y - n // 2) * RES  # east, north (m)


def test_downslope_points_downhill() -> None:
    x, _y = xy()
    z = 0.1 * x  # rises to the east
    ux, un, _m = T.downslope_unit(z, RES)
    assert np.allclose(ux[50:150, 50:150], -1, atol=1e-6)
    assert np.allclose(un[50:150, 50:150], 0, atol=1e-6)


def test_flow_accumulation_valley_collects() -> None:
    x, y = xy()
    z = np.abs(x) * 0.2 + y * 0.05  # V valley along x=0 draining south
    acc = T.flow_accumulation(z, RES)
    c = z.shape[1] // 2
    assert acc[190, c] > 50 * acc[190, c + 30]


@pytest.mark.parametrize("res", [5.0, 10.0, 20.0])
def test_specific_catchment_grows_down_an_open_slope(res: float) -> None:
    """On a plane every cell passes all its area one row down, so the area per metre of contour is the slope length
    above the cell (plus the cell itself), whatever the cell size; D8 sees only the single-file line above it."""
    n = 121
    rows = np.arange(n, dtype=float)[:, None]
    z = np.broadcast_to(-rows * res * 0.3, (n, n)).copy()  # 30% slope falling south
    sca = T.specific_catchment(z, res)
    c = n // 2
    for r in (5, 20, 50):
        assert sca[r, c] == pytest.approx((r + 1) * res, rel=1e-6)
    # what matters is the distance down the slope, not the cell size
    assert sca[int(400 / res), c] == pytest.approx(400 + res, rel=1e-6)


def test_specific_catchment_collects_in_a_valley_and_conserves_area() -> None:
    x, y = xy()
    z = np.abs(x) * 0.2 + y * 0.05  # V valley along x=0 draining south
    sca = T.specific_catchment(z, RES)
    c = z.shape[1] // 2
    assert sca[190, c] > 20 * sca[190, c + 30] > 0
    # every cell's area ends somewhere: what drains off the grid (edge cells and pits) is the whole grid
    acc = sca / RES
    lower = np.zeros(z.shape, bool)  # cells with a strictly lower neighbour pass their area on
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                nb = np.full(z.shape, np.inf)
                nb[max(dr, 0) : z.shape[0] + min(dr, 0), max(dc, 0) : z.shape[1] + min(dc, 0)] = z[
                    max(-dr, 0) : z.shape[0] + min(-dr, 0), max(-dc, 0) : z.shape[1] + min(-dc, 0)
                ]
                lower |= nb < z
    assert acc[~lower].sum() == pytest.approx(z.size, rel=1e-9)


def test_geomorphons_peak_and_pit() -> None:
    x, y = xy(121)
    r2 = x**2 + y**2
    peak = 100 * np.exp(-r2 / (2 * 80**2))
    g = T.geomorphons(peak, RES, search_m=200)
    assert g[60, 60] == T.PEAK
    g2 = T.geomorphons(-peak, RES, search_m=200)
    assert g2[60, 60] == T.PIT


def test_geomorphons_ridge() -> None:
    x, _y = xy(121)
    ridge = 80 - 0.3 * np.abs(x)  # N-S ridge along x=0
    g = T.geomorphons(ridge, RES, search_m=200)
    assert g[60, 60] == T.RIDGE


def test_saddle_detected_at_center() -> None:
    x, y = xy()
    # ridge running N-S with a dip at the center: z = along-ridge up, across-ridge down
    z = 200 - 0.4 * np.abs(x) + 40 * (1 - np.exp(-(y**2) / (2 * 150**2)))
    s = T.saddles(z, RES)
    assert s, "no saddle found"
    best = max(s, key=lambda d: d["rise_m"])
    assert abs(best["row"] - 100) <= 3 and abs(best["col"] - 100) <= 3
    assert best["rise_m"] > 15
    # ridge axis runs north-south
    assert abs(best["ridge_axis_deg"] - 90) < 15


def test_cliff_base_below_cliff() -> None:
    x, y = xy(101)
    z = np.where(x > 0, 30.0, 0.0) + 0.01 * y
    z = z + np.clip(x, 0, RES) * 0  # sharp step
    cm, base, top = T.cliffs(z, RES, min_area_m2=10)
    assert cm.any()
    col_cliff = np.nonzero(cm[50])[0].min()
    assert base[50, col_cliff - 2] > 0.3  # just west (downhill side)
    assert base[50, col_cliff + 4] == 0
    assert top[50, col_cliff + 4] > 0


def test_walking_distance_flat_is_euclidean() -> None:
    cost = np.full((101, 101), 1.0)
    src = np.zeros((101, 101), bool)
    src[50, 0] = True
    t, ln, pred = T.walking(cost, src, 10.0, 5000)
    assert abs(ln[50, 100] - 1000) < 1e-6
    assert abs(t[50, 100] - 1000) < 1e-6
    path = T.trace_route(pred, cost.shape, 50, 100)
    assert path[0] == (50, 0) and path[-1] == (50, 100)


def test_walking_goes_around_obstacle() -> None:
    cost = np.full((101, 101), 1.0)
    cost[20:101, 50] = np.inf  # wall with a gap at the top
    src = np.zeros((101, 101), bool)
    src[90, 0] = True
    _t, ln, _pred = T.walking(cost, src, 1.0, 10000)
    assert ln[90, 100] > 150  # must detour through the gap


def test_upwind_shelter_leeward_positive() -> None:
    x, _y = xy(121)
    z = np.where(x < 0, 50.0, 0.0)  # high ground to the west
    sx = T.upwind_shelter(z, RES, from_deg=270, reach_m=200)
    assert sx[60, 70] > 5  # east of the step, wind from west -> sheltered
    sx2 = T.upwind_shelter(z, RES, from_deg=90, reach_m=200)
    assert sx2[60, 70] < 1


# ---- percentile rank, fast blur, travel lines ----------------------------------------------------------


def test_percentile_rank_monotone_and_bounded() -> None:
    a = np.random.default_rng(0).normal(size=(300, 300)).astype("float32")
    p = T.percentile_rank(a)
    assert p.dtype == np.float32 and p.min() >= 0 and p.max() <= 1
    order = np.argsort(a.ravel())
    assert np.all(np.diff(p.ravel()[order]) >= -1e-6)  # monotone in a
    assert abs(float(np.median(p)) - 0.5) < 0.01


def test_percentile_rank_nan_constant_empty() -> None:
    a = np.array([[1.0, np.nan], [3.0, 2.0]], "float32")
    p = T.percentile_rank(a)
    assert np.isnan(p[0, 1]) and p[0, 0] < p[1, 1] < p[1, 0]
    assert np.all(T.percentile_rank(np.full((4, 4), 7.0)) == 0.5)  # all tied -> the middle
    assert np.all(T.percentile_rank(np.full((3, 3), np.nan)) != 0)  # all invalid: NaN, never a fake extreme
    assert T.percentile_rank(np.zeros((0, 5))).shape == (0, 5)


def test_blur_matches_exact_gaussian() -> None:
    from scipy import ndimage

    rng = np.random.default_rng(1)
    a = ndimage.gaussian_filter(rng.random((600, 600)), 3).astype("float32")  # a smooth random field
    fast = T.blur(a, 250.0, RES)
    exact = ndimage.gaussian_filter(a, 250.0 / RES)
    c = slice(150, 450)  # away from the edges, where the two pad differently
    assert np.abs(fast - exact)[c, c].max() < 0.01 * float(exact[c, c].mean())


def test_blur_small_sigma_is_exact() -> None:
    from scipy import ndimage

    a = np.random.default_rng(2).random((80, 80)).astype("float32")
    assert np.allclose(T.blur(a, 20.0, RES), ndimage.gaussian_filter(a, 4.0))


def _sine_terrain(res: float = 10.0, shape: tuple[int, int] = (300, 400)) -> Floats:
    x = np.arange(shape[1]) * res
    return (100 * np.sin(2 * np.pi * x / 2000.0))[None, :].repeat(shape[0], 0)  # spines at 500 m, bottoms at 1500 m


def test_travel_lines_on_spines_and_bottoms() -> None:
    line, pos = T.travel_lines(_sine_terrain(), 10.0)
    row, prow = line[150], pos[150]
    spine, bottom, mid = 50, 150, 100
    assert row[spine] > 0.8 and row[bottom] > 0.8 and row[mid] < 0.1
    assert prow[spine] > 0 > prow[bottom]


def test_travel_parts_split_the_line() -> None:
    z = _sine_terrain()
    bottom, spine, pos = T.travel_parts(z, 10.0)
    line, pos2 = T.travel_lines(z, 10.0)
    assert not ((bottom > 0) & (spine > 0)).any()
    assert np.allclose(bottom + spine, line) and np.array_equal(pos, pos2)
    assert bottom[150, 150] > 0.8 and spine[150, 150] == 0 and spine[150, 50] > 0.8 and bottom[150, 50] == 0


def test_spine_gate_at_saddles_and_junctions() -> None:
    pos = np.full((200, 200), -1.0, "float32")
    pos[98:103, 0:100] = 1  # a Y of ridge spines meeting at (100, 100), 5 cells wide
    for i in range(100):
        pos[max(0, 100 - i - 2) : 100 - i + 3, 100 + i] = 1  # to the north-east
        pos[100 + i - 2 : min(200, 100 + i + 3), 100 + i] = 1  # to the south-east
    g = T.spine_gate(pos, [(20, 160)], 10.0, saddle_m=150.0, junction_m=100.0)
    assert g[20, 160] == 1 and g[20, 150] == pytest.approx(1 - 100 / 150) and g[20, 140] == 0
    assert g[100, 100] > 0.8  # the junction
    assert g[100, 30] == 0 and g[60, 20] == 0  # 700 m along an arm; far from both
    assert T.spine_gate(pos, [], 10.0)[20, 160] == 0  # no saddle there, and no junction near


def _tent(axis: str, n: int = 160, res: float = 10.0, deg: float = 20.0) -> Floats:
    """A ridge through the middle of an n x n grid, both flanks deg steep: 'ew' runs east-west (flanks face north
    and south), 'ns' north-south (flanks face east and west)."""
    i = np.arange(n, dtype="float32")
    prof = 500 - math.tan(math.radians(deg)) * res * np.abs(i - n // 2)
    return (prof[:, None] if axis == "ew" else prof[None, :]).repeat(n, 1 if axis == "ew" else 0).astype("float32")


def test_thermal_slopes_above_south_slopes_only() -> None:
    south = T.thermal_slopes(_tent("ew"), 10.0)  # crest with a south flank
    east_west = T.thermal_slopes(_tent("ns"), 10.0)  # crest with east and west flanks
    assert south[80, 80] == 1 and east_west[80, 80] == 0
    assert south[20, 80] == 0  # deep on the north flank: no sunny slope within 300 m
    assert T.thermal_slopes(_tent("ew", deg=5.0), 10.0).max() == 0  # too gentle to warm


def test_gentle_grade() -> None:
    assert np.all(T.gentle_grade(np.zeros((50, 50)), 10.0) == 1)
    steep = T.gentle_grade(_tent("ew", deg=30.0), 10.0)
    assert steep[40, 80] == pytest.approx(0.5)
    mid = T.gentle_grade(_tent("ew", deg=16.5), 10.0)
    assert mid[40, 80] == pytest.approx(0.75, abs=0.01)


def test_travel_lines_flat_ground_has_none() -> None:
    noise = np.random.default_rng(3).uniform(-0.5, 0.5, (200, 200))
    line, _ = T.travel_lines(noise, 10.0)
    assert float(line.max()) < 0.2  # DEM noise is under the relief floor
    flat, _ = T.travel_lines(np.zeros((50, 50)), 10.0)
    assert float(flat.max()) == 0


def test_travel_lines_tiny_area_finite() -> None:
    line, pos = T.travel_lines(_sine_terrain(shape=(5, 7)), 10.0)
    assert np.isfinite(line).all() and np.isfinite(pos).all()
