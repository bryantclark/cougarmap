"""The fast kernels must match the scipy/numpy code they replace exactly (the correctness gate for every speedup)."""

import math
import pickle
from typing import Any, Literal

import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import transform as shp_transform

from cougarmap import terrain as T
from cougarmap.arrays import Floats
from cougarmap.grid import Grid
from cougarmap.sources import vector as vec

SHAPES = [((300, 401), 0.01), ((57, 1000), 0.001), ((1000, 33), 0.2), ((5, 5), 0.5), ((1, 9), 0.3), ((200, 200), 5e-4)]


@pytest.mark.parametrize(("shape", "p"), SHAPES)
@pytest.mark.parametrize("res", [1.0, 4.5])
def test_edt_matches_scipy(shape: tuple[int, int], p: float, res: float) -> None:
    m = np.random.default_rng(1).random(shape) < p
    m.flat[len(m.flat) // 2] = True
    d0, (i0, j0) = ndimage.distance_transform_edt(~m, sampling=res, return_indices=True)
    d1, (i1, j1) = T.edt_nearest(m, res)
    assert np.array_equal(d0, d1)
    assert np.array_equal(T.edt(m, res), d0)
    # nearest-cell indices point at a feature at exactly that distance (and match scipy's tie-breaking here)
    assert m[i1, j1].all()
    # sqrt(a*a + b*b) like scipy and the kernel: np.hypot can differ in the last bit (glibc, not macOS)
    di, dj = (i1 - np.arange(shape[0])[:, None]) * res, (j1 - np.arange(shape[1])) * res
    assert np.array_equal(np.sqrt(di * di + dj * dj), d1)
    assert np.array_equal(i0, i1) and np.array_equal(j0, j1)


def test_edt_without_features_is_inf() -> None:
    d, (ir, ic) = T.edt_nearest(np.zeros((4, 6), bool), 2.0)
    assert np.isinf(d).all() and (ir == -1).all() and (ic == -1).all()


def test_labeled_minmax_matches_scipy() -> None:
    rng = np.random.default_rng(2)
    v = rng.normal(size=(120, 90)).astype("float32")
    lab, n = ndimage.label(rng.random((120, 90)) < 0.4)
    lab[lab == 3] = 0  # a label with no cells gets 0, as in scipy
    idx = np.arange(1, n + 1)
    lo, hi = T.labeled_minmax(v, lab, n)
    assert lo.dtype == v.dtype
    assert np.array_equal(lo, ndimage.minimum(v, lab, idx))
    assert np.array_equal(hi, ndimage.maximum(v, lab, idx))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_median3_matches_scipy(dtype: str) -> None:
    a = np.random.default_rng(3).gamma(2.0, 5.0, size=(77, 131)).astype(dtype)
    a[10:20, 10:20] = 7.0  # ties
    assert np.array_equal(T.median_filter(a, 3), ndimage.median_filter(a, size=3))
    assert np.array_equal(T.median_filter(a, 5), ndimage.median_filter(a, size=5))  # falls back to scipy


def _saddles_reference(
    z: Floats, res: float, sigma_m: float = 25.0, reach_m: float = 250.0, min_rise_m: float = 8.0
) -> list[dict[str, Any]]:
    """The per-point loop terrain.saddles replaced (kept here as the oracle)."""

    zs = T.smooth(z, sigma_m, res)
    gy, gx = np.gradient(zs, res)
    gyy, gyx = np.gradient(gy, res)
    gxy, gxx = np.gradient(gx, res)
    hxy = 0.5 * (gxy + gyx)
    det = gxx * gyy - hxy**2
    s = np.where((det < 0) & (np.hypot(gx, gy) < np.tan(np.radians(8))), np.sqrt(-np.minimum(det, 0)), 0)
    win = max(3, round(60 / res) | 1)
    rows, cols = np.nonzero((s == ndimage.maximum_filter(s, size=win)) & (s > 0))
    steps = np.linspace(res, reach_m, max(4, int(reach_m / (2 * res))))
    H, W = zs.shape

    def ray(r: int, c: int, dx: float, dy: float) -> Floats:
        rr = np.clip(np.round(r + dy * steps / res).astype(int), 0, H - 1)
        cc = np.clip(np.round(c + dx * steps / res).astype(int), 0, W - 1)
        out: Floats = zs[rr, cc]
        return out

    out = []
    for r, c in zip(rows, cols, strict=True):
        w, v = np.linalg.eigh(np.array([[gxx[r, c], hxy[r, c]], [hxy[r, c], gyy[r, c]]]))
        if not (w[0] < 0 < w[1]):
            continue
        ex, ey, nx, ny = v[0, 1], v[1, 1], v[0, 0], v[1, 0]
        z0 = zs[r, c]
        rise = min(ray(r, c, ex, ey).max() - z0, ray(r, c, -ex, -ey).max() - z0)
        drop = min(z0 - ray(r, c, nx, ny).min(), z0 - ray(r, c, -nx, -ny).min())
        if rise >= min_rise_m and drop >= min_rise_m * 0.75:
            ang = math.degrees(math.atan2(-ey, ex)) % 180
            out.append(dict(row=int(r), col=int(c), rise_m=float(rise), drop_m=float(drop), ridge_axis_deg=ang))
    return out


def test_vectorized_saddles_match_loop() -> None:
    rng = np.random.default_rng(4)
    z = ndimage.gaussian_filter(rng.normal(size=(260, 300)), 12) * 4000
    ref = _saddles_reference(z, 10.0, 30, 300, 8)
    assert len(ref) > 3
    assert T.saddles(z, 10.0, 30, 300, 8) == ref
    assert T.saddles(np.zeros((50, 50)), 10.0) == []


def test_geomorphons_unchanged_by_parallel_rows() -> None:
    z = ndimage.gaussian_filter(np.random.default_rng(5).normal(size=(90, 110)), 6) * 800
    serial = T._geomorphons.py_func(z, 10.0, 25, np.radians(1.5)).astype(np.int8)  # type: ignore[attr-defined]  # numba's Dispatcher.py_func
    assert np.array_equal(T.geomorphons(z, 10.0, search_m=250), serial)


# ---- grid -------------------------------------------------------------------------------------------------

G = Grid(32611, 400_000.0, 5_300_000.0, 5.0, 300, 200)


def test_project_all_matches_shapely_ops() -> None:
    geoms = [
        Point(-117.7, 47.8),
        LineString([(-117.7, 47.8), (-117.69, 47.81), (-117.68, 47.805)]),
        Polygon(
            [(-117.7, 47.8), (-117.69, 47.8), (-117.69, 47.81)],
            [[(-117.698, 47.801), (-117.697, 47.801), (-117.697, 47.802)]],
        ),
        MultiPolygon([Polygon([(-117.6, 47.7), (-117.59, 47.7), (-117.59, 47.71)])]),
    ]
    ref = [shp_transform(G._from_ll.transform, g) for g in geoms]
    for a, b in zip(G.project_all(geoms), ref, strict=True):
        assert a.geom_type == b.geom_type and a.equals_exact(b, 0)
    assert G.project(geoms[1]).equals_exact(ref[1], 0)
    assert G.project_all([]) == []


def test_project_features_sets_xy() -> None:
    f = vec.Feature(LineString([(-117.7, 47.8), (-117.69, 47.81)]), {"highway": "track"})
    with pytest.raises(RuntimeError):
        _ = f.xy
    vec.project_features([f], G)
    assert f.xy.equals_exact(G.project(f.geom), 0)


def test_grid_pickles_without_cached_helpers() -> None:
    G.to_lonlat(400_100.0, 5_299_900.0)  # fills the cached transformer
    assert "_to_ll" in G.__dict__
    g2 = pickle.loads(pickle.dumps(G))
    assert g2 == G and "_to_ll" not in g2.__dict__
    assert g2.to_lonlat(400_100.0, 5_299_900.0) == G.to_lonlat(400_100.0, 5_299_900.0)
    # states saved before kept a full coordinate mesh in the pickled grid: it is dropped on load
    old = pickle.dumps(G)
    legacy = Grid.__new__(Grid)
    legacy.__setstate__({**G.__getstate__(), "XY": np.zeros((2, 2))})
    assert legacy == G and "XY" not in legacy.__dict__
    assert pickle.loads(old) == G


def test_cell_centers_match_xy() -> None:
    xs, ys = G.cell_centers()
    x, _ = G.xy(0, np.arange(G.width))
    _, y = G.xy(np.arange(G.height), 0)
    assert xs.shape == (G.width,) and ys.shape == (G.height,)
    assert np.array_equal(xs, x) and np.array_equal(ys, y)


def test_geojson_matches_shapely_mapping() -> None:
    from shapely.geometry import MultiLineString, mapping

    from cougarmap.grid import geojson

    poly = Polygon([(0, 0), (4, 0), (4, 4)], [[(1, 0.5), (2, 0.5), (2, 1.5)]])
    for g in (
        LineString([(0, 0), (1, 2.5), (3, 1)]),
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 1), (4, 4)]]),
        poly,
        MultiPolygon([poly, Polygon([(10, 10), (11, 10), (11, 12)])]),
        Point(1.5, 2.5),
    ):
        ref = mapping(g)
        got = geojson(g)
        assert got["type"] == ref["type"]
        assert _lists(got["coordinates"]) == _lists(ref["coordinates"])
    lines = [LineString([(400_010.0, 5_299_990.0), (400_900.0, 5_299_200.0)]), poly]
    assert G.rasterize(lines, 1, dtype="uint8").sum() > 0


def _lists(x: Any) -> Any:
    return [_lists(v) for v in x] if isinstance(x, (list, tuple)) else x


def _accumulate_reference(z: Floats, res: float) -> Floats:
    """The argsort-based D8 accumulation flow_accumulation used to run (the oracle)."""
    f, _ = T._fill_eps(z.astype(np.float64))
    H, W = f.shape
    rec = np.full(H * W, -1)
    for r in range(H):
        for c in range(W):
            best, bi = 0.0, -1
            for d in range(8):
                rr, cc = r + T.DR[d], c + T.DC[d]
                if 0 <= rr < H and 0 <= cc < W:
                    s = (f[r, c] - f[rr, cc]) / (T.DL[d] * res)
                    if s > best:
                        best, bi = s, rr * W + cc
            rec[r * W + c] = bi
    acc = np.ones(H * W)
    for i in np.argsort(-f.ravel()):
        if rec[i] >= 0:
            acc[rec[i]] += acc[i]
    return acc.reshape(H, W) * res * res


def test_flow_accumulation_matches_sorted_order() -> None:
    z = ndimage.gaussian_filter(np.random.default_rng(6).normal(size=(60, 70)), 3) * 50
    z[20:25, 30:35] = z.min() - 5  # a pit the fill has to flood
    assert np.array_equal(T.flow_accumulation(z, 10.0), _accumulate_reference(z, 10.0))


def _mfd_reference(z: Floats, res: float, p: float = 1.1) -> Floats:
    """Multiple-flow-direction accumulation in sorted order, with numpy weights per cell (the oracle)."""
    f, _ = T._fill_eps(z.astype(np.float64))
    H, W = f.shape
    acc = np.ones((H, W))
    for i in np.argsort(-f.ravel(), kind="stable"):
        r, c = divmod(int(i), W)
        nb = [(r + T.DR[d], c + T.DC[d], d) for d in range(8)]
        nb = [(rr, cc, d) for rr, cc, d in nb if 0 <= rr < H and 0 <= cc < W and f[rr, cc] < f[r, c]]
        if not nb:
            continue
        w = np.array(
            [((f[r, c] - f[rr, cc]) / (T.DL[d] * res)) ** p * (0.5 if d % 2 == 0 else 0.354) for rr, cc, d in nb]
        )
        for (rr, cc, _), share in zip(nb, w / w.sum(), strict=True):
            acc[rr, cc] += acc[r, c] * share
    return acc * res


def test_specific_catchment_matches_sorted_order() -> None:
    z = ndimage.gaussian_filter(np.random.default_rng(7).normal(size=(40, 50)), 3) * 50
    z[10:14, 20:24] = z.min() - 5  # a pit the fill has to flood
    assert np.allclose(T.specific_catchment(z, 10.0), _mfd_reference(z, 10.0), rtol=1e-9)


@pytest.mark.parametrize(("sigma", "mode", "dtype"), [(4.4, "reflect", "float32"), (20.0, "nearest", "float64")])
def test_banded_gaussian_matches_scipy(sigma: float, mode: Literal["reflect", "nearest"], dtype: str) -> None:
    a = np.random.default_rng(7).random((1600, 1300)).astype(dtype)  # big enough to be split into bands
    assert a.size >= T.BAND_MIN_CELLS
    got = T.gaussian(a, sigma, mode=mode)
    assert got.dtype == a.dtype
    assert np.array_equal(got, ndimage.gaussian_filter(a, sigma, mode=mode))


def test_upwind_shelter_matches_serial_loop() -> None:
    z = (ndimage.gaussian_filter(np.random.default_rng(8).normal(size=(120, 140)), 6) * 900).astype("float32")
    rr, cc = np.mgrid[0:120, 0:140].astype("float32")
    best = np.full(z.shape, -90.0, dtype="float32")
    ux, uy = np.sin(np.radians(250.0)), np.cos(np.radians(250.0))
    for d in np.linspace(10, 300, 12):
        zr = ndimage.map_coordinates(z, [rr - uy * d / 10.0, cc + ux * d / 10.0], order=1, mode="nearest")
        best = np.maximum(best, np.degrees(np.arctan((zr - z) / d)))
    assert np.array_equal(T.upwind_shelter(z, 10.0, 250.0, 300.0), best)


def test_label_counts_and_sums_match_scipy() -> None:
    rng = np.random.default_rng(9)
    lab, n = ndimage.label(rng.random((150, 170)) < 0.45)
    v = rng.normal(size=lab.shape).astype("float32")
    idx = np.arange(1, n + 1)
    assert np.array_equal(T.label_counts(lab, n) * 9.0, ndimage.sum(lab > 0, lab, idx) * 9.0)
    assert np.array_equal(T.label_sums(v, lab, n), ndimage.sum(v, lab, idx))
    assert np.array_equal(T.label_sums(v, lab, n) / T.label_counts(lab, n).astype(float), ndimage.mean(v, lab, idx))
    assert np.array_equal(T.label_sums(v > 0, lab, n), ndimage.sum(v > 0, lab, idx))


def test_banded_binary_opening_matches_scipy() -> None:
    m = np.random.default_rng(10).random((1700, 1250)) < 0.55
    for k in (1, 3):
        ref = ndimage.binary_opening(m, structure=np.ones((2 * k + 1, 2 * k + 1), bool))
        assert np.array_equal(T.binary_opening(m, k), ref)


def _heapq_tree(cost: np.ndarray, starts: np.ndarray, res: float, cutoff: float) -> tuple[Any, Any, Any]:
    """The least-cost tree with Python's heapq on (cost, cell) tuples: the reference _least_cost_tree replaces."""
    import heapq

    H, W = cost.shape
    cf = cost.ravel()
    D = np.full(H * W, np.inf)
    pred = np.full(H * W, -1, np.int64)
    done = np.zeros(H * W, bool)
    order = []
    heap = [(0.0, int(s)) for s in starts]
    for s in starts:
        D[s] = 0.0
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        order.append(i)
        r, c = divmod(i, W)
        for k in range(8):
            rr, cc = r + int(T.DR[k]), c + int(T.DC[k])
            j = rr * W + cc
            if 0 <= rr < H and 0 <= cc < W and not done[j]:
                nd = d + T.DL[k] * res * 0.5 * (cf[i] + cf[j])
                if nd < D[j] and nd <= cutoff:
                    D[j], pred[j] = nd, i
                    heapq.heappush(heap, (nd, j))
    return D.reshape(H, W), pred, np.array(order, np.int64)


@pytest.mark.parametrize("flat", [False, True])  # flat: equal costs everywhere, so ties settle by cell index
def test_least_cost_tree_matches_heapq(flat: bool) -> None:
    rng = np.random.default_rng(3)
    cost = np.ones((60, 45)) if flat else rng.random((60, 45)) * 5 + 0.5
    starts = np.flatnonzero(rng.random(cost.size) < 0.01).astype(np.int64)
    ref = _heapq_tree(cost, starts, 3.0, 25.0)
    got = T._least_cost_tree(cost, starts, 3.0, 25.0)
    assert np.isinf(ref[0]).any() and np.isfinite(ref[0]).any()  # the cutoff stops some routes
    for a, b in zip(ref, got, strict=True):
        assert np.array_equal(a, b)
    # the distances are the graph's shortest paths
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    H, W = cost.shape
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    cf: Floats = cost.ravel()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    w: list[np.ndarray] = []
    for dr, dc, dl in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        a = idx[: H - dr, max(0, -dc) : W - max(0, dc)].ravel()
        b = idx[dr:, max(0, dc) : W - max(0, -dc) or None].ravel()
        e = dl * 3.0 * 0.5 * (cf[a] + cf[b])
        rows += [a, b]
        cols += [b, a]
        w += [e, e]
    g = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W))
    d = np.asarray(dijkstra(g, indices=starts)).min(axis=0)
    ok = np.isfinite(got[0].ravel())
    assert np.allclose(got[0].ravel()[ok], d[ok]) and (d[~ok] > 25).all()


def test_tree_flow_matches_walking_each_route() -> None:
    rng = np.random.default_rng(4)
    cost = rng.random((30, 40)) + 0.5
    dest = np.zeros(cost.shape, bool)
    dest[5, 5] = dest[20, 30] = True
    weight = (rng.random(cost.shape) < 0.2).astype(float)
    covered = rng.random(cost.shape) < 0.6
    t = T.cost_tree(cost, dest, weight, covered, 2.0, 1e9, 9.0)
    _, pred, _ = T._least_cost_tree(cost, np.flatnonzero(dest).astype(np.int64), 2.0, 1e9)
    W = cost.shape[1]
    flow = weight.ravel().copy() * 0
    for i in range(cost.size):  # walk every cell's route to its destination
        path = [i]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        for j in path:
            flow[j] += weight.flat[i]
        assert t.root.flat[i] == path[-1]
        steps = [(a, b) for a, b in zip(path[::-1][:-1], path[::-1][1:], strict=True)]  # root outward
        length = uncovered = 0.0
        for a, b in steps:
            step = 2.0 * (math.sqrt(2) if (a // W != b // W and a % W != b % W) else 1.0)
            length += step
            if not covered.flat[b] and length <= 9.0:
                uncovered += step
        share = 1 - uncovered / min(length, 9.0) if length > 0 else 1.0
        assert t.covered_share.flat[i] == pytest.approx(share)
    assert np.allclose(t.flow.ravel(), flow)
