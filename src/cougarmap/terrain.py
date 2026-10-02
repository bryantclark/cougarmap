"""Terrain math on a DEM: slope, smoothing, flow routing, landforms (geomorphons), saddles, cliffs, sheltering,
and least-cost walking distance. Hot loops are numba-compiled.

All functions take `res` (cell size, m) where distances matter. Vector outputs are (east, north).

The parallel numba kernels (parallel=True) are meant to be called from one thread at a time: numba's fallback
threading layer (workqueue) does not support concurrent parallel launches. Threads are used here only around
scipy/numpy code, which releases the GIL.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, NamedTuple, TypedDict, TypeVar, cast

import numpy as np
import numpy.typing as npt
from numba import njit
from numba import prange as _prange
from scipy import ndimage, signal
from skimage.morphology import skeletonize

from .arrays import Floats, Ints, Mask
from .config import THREADS

# numba's parallel range: the same object (numba resolves it by identity), typed as a range for mypy
prange = cast("Callable[[int], range]", _prange)

_Fn = TypeVar("_Fn", bound=Callable[..., Any])


def jit(*, parallel: bool = False, nogil: bool = False) -> Callable[[_Fn], _Fn]:
    """numba.njit with an on-disk cache, typed as returning the function it compiles (so callers and the kernel
    bodies are type-checked as the plain Python they also are)."""
    return cast("Callable[[_Fn], _Fn]", njit(parallel=parallel, nogil=nogil, cache=True))


I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]

# 8 neighbours: (drow, dcol). Order matters for geomorphons (N, NE, E, SE, S, SW, W, NW).
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
DL = np.array([1, math.sqrt(2)] * 4)

FLAT, PEAK, RIDGE, SHOULDER, SPUR, SLOPE, HOLLOW, FOOTSLOPE, VALLEY, PIT = range(1, 11)
LANDFORM_NAMES = {
    1: "flat",
    2: "peak",
    3: "ridge",
    4: "shoulder",
    5: "spur",
    6: "slope",
    7: "hollow",
    8: "footslope",
    9: "valley",
    10: "pit",
}


# ---- basics ---------------------------------------------------------------------------------------


def gradient(z: Floats, res: float) -> tuple[Floats, Floats]:
    """dz/d(east), dz/d(north)."""
    gy, gx = (np.asarray(g) for g in np.gradient(z, res))
    return gx, -gy


def slope_deg(z: Floats, res: float) -> Floats:
    gx, gn = gradient(z, res)
    out: Floats = np.degrees(np.arctan(np.hypot(gx, gn)))
    return out


BAND_MIN_CELLS = 2_000_000  # below this, threading a filter costs more than it saves


def _banded(fn: Callable[[np.ndarray], np.ndarray], a: np.ndarray, halo: int) -> np.ndarray | None:
    """fn(a) computed on overlapping row bands on threads (ndimage releases the GIL), or None when a is too small
    to gain. Exact whenever each output row depends only on input rows within `halo` of it, as for
    scipy's separable direct-sum filters and binary morphology (but not running-sum filters like uniform_filter)."""
    H = a.shape[0]
    n = min(THREADS, H // max(1, 4 * halo))
    if a.ndim != 2 or a.size < BAND_MIN_CELLS or n < 2:
        return None
    edges = np.linspace(0, H, n + 1).astype(int)
    parts: list[np.ndarray] = [np.empty(0)] * n

    def band(i: int) -> None:
        lo, hi = edges[i], edges[i + 1]
        top, bot = max(0, lo - halo), min(H, hi + halo)
        parts[i] = fn(a[top:bot])[lo - top : hi - top]

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(band, range(n)))
    return np.concatenate(parts)


def _gauss(a: Floats, sigma: float, mode: Literal["reflect", "nearest"]) -> Floats:
    out: Floats = ndimage.gaussian_filter(a, sigma, mode=mode)
    return out


def gaussian(a: Floats, sigma: float, mode: Literal["reflect", "nearest"] = "reflect") -> Floats:
    """ndimage.gaussian_filter(a, sigma, mode=mode), bit-identical, on threads for big arrays: each output of
    scipy's separable filter is a direct weighted sum over the input within its radius."""
    a = np.asarray(a)
    radius = int(4.0 * float(sigma) + 0.5)  # scipy's default truncate=4.0
    out = _banded(lambda b: _gauss(b, sigma, mode), a, radius) if a.dtype.kind == "f" else None
    return _gauss(a, sigma, mode) if out is None else out


def binary_opening(mask: Mask, k: int) -> Mask:
    """ndimage.binary_opening(mask, a (2k+1) x (2k+1) square), bit-identical, on threads for big masks (an
    erosion then a dilation: each output depends on input within 2k rows)."""
    st = np.ones((2 * k + 1, 2 * k + 1), bool)

    def opening(b: Mask) -> Mask:
        out: Mask = ndimage.binary_opening(b, structure=st)
        return out

    banded = _banded(opening, mask, 2 * k + 1)
    return opening(mask) if banded is None else banded


def smooth(z: Floats, sigma_m: float, res: float) -> Floats:
    s = sigma_m / res
    return gaussian(z, s, mode="nearest") if s > 0.3 else z.copy()


def downslope_unit(z: Floats, res: float) -> tuple[Floats, Floats, Floats]:
    """Unit vector pointing downhill (east, north) and the slope magnitude (rise/run)."""
    gx, gn = gradient(z, res)
    m = np.hypot(gx, gn)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, un = np.where(m > 1e-6, -gx / m, 0.0), np.where(m > 1e-6, -gn / m, 0.0)
    return ux, un, m


def tpi(z: Floats, radius_m: float, res: float) -> Floats:
    """Topographic position index: elevation minus the mean elevation around it (m)."""
    r = max(1, round(radius_m / res))
    out: Floats = z - ndimage.uniform_filter(z, size=2 * r + 1, mode="nearest")
    return out


def percentile_rank(
    a: Floats, valid: Mask | None = None, n_q: int = 201, sample: int = 400_000, seed: int = 0
) -> Floats:
    """Percentile (0-1) of every cell among the finite `valid` cells, from sampled quantiles.

    Monotone in `a`. Non-finite cells come out NaN; if every cell ties (or none is valid) the result is a constant
    0.5. Sampling keeps it ~0.1 s on tens of millions of cells and makes it deterministic (fixed seed)."""
    a = np.asarray(a, dtype="float32")
    v = a[valid] if valid is not None else a.ravel()
    v = v[np.isfinite(v)]
    if v.size > sample:
        v = v[np.random.default_rng(seed).choice(v.size, sample, replace=False)]
    if v.size == 0 or float(v.max() - v.min()) <= 1e-12:
        return np.where(np.isfinite(a), 0.5, np.nan).astype("float32")
    qs = np.percentile(v, np.linspace(0, 100, n_q))
    qs = qs + np.arange(n_q) * 1e-9 * max(1.0, float(np.abs(qs).max()))  # strictly increasing for np.interp
    return np.interp(a, qs, np.linspace(0, 1, n_q)).astype("float32")


def blur(a: Floats, sigma_m: float, res: float) -> Floats:
    """Gaussian blur (sigma in metres), fast for big sigmas.

    When sigma spans 8+ cells the array is block-averaged by f = sigma/4 cells, blurred there, and bilinearly
    interpolated back: under 1% from an exact Gaussian, and seconds instead of minutes for a 250 m sigma on a
    3-5 m grid. Smaller sigmas run scipy's gaussian_filter directly."""
    s = sigma_m / res
    f = int(s // 4)
    a = np.asarray(a, dtype="float32")
    if f < 2:
        return gaussian(a, s)
    H, W = a.shape
    p = np.pad(a, ((0, (-H) % f), (0, (-W) % f)), mode="edge")
    small = p.reshape(p.shape[0] // f, f, p.shape[1] // f, f).mean(axis=(1, 3))
    small = _gauss(small, s / f, "reflect")
    return _upsample_linear(small, f, (H, W))


def _upsample_linear(small: Floats, f: int, shape: tuple[int, int]) -> Floats:
    """Bilinear interpolation of a block-averaged grid (block size f) back to `shape`, edges held constant.
    Separable, so it needs no full-size coordinate arrays."""

    def axis(n: int, m: int) -> tuple[Ints, Ints, Floats]:
        x = np.clip((np.arange(n) + 0.5) / f - 0.5, 0, m - 1)
        i0 = np.floor(x).astype(np.intp)
        return i0, np.minimum(i0 + 1, m - 1), (x - i0).astype("float32")

    r0, r1, wr = axis(shape[0], small.shape[0])
    rows = small[r0] * (1 - wr)[:, None] + small[r1] * wr[:, None]
    c0, c1, wc = axis(shape[1], small.shape[1])
    out = rows[:, c0] * (1 - wc)
    out += rows[:, c1] * wc
    return out.astype("float32", copy=False)


def travel_parts(
    z: Floats, res: float, tpi_m: float = 300.0, power: float = 2.0, relief_m: float = 3.0
) -> tuple[Floats, Floats, Floats]:
    """The landscape's skeleton, split: (drainage bottoms 0-1, ridge spines 0-1, signed position -1 bottom .. +1
    spine). Position is the TPI percentile within this grid, so the lines adapt to the local relief: 1 on this
    landscape's most extreme bottoms and spines, 0 on the average mid-slope. A relief floor keeps DEM noise on
    flat ground from making lines. Bottoms and spines are disjoint and sum to travel_lines' line."""
    t = tpi(np.asarray(z, dtype="float32"), tpi_m, res)
    pos = np.nan_to_num(2 * percentile_rank(t, np.isfinite(t)) - 1)
    line = np.abs(pos) ** power * np.clip(np.nan_to_num(np.abs(t)) / relief_m, 0, 1)
    bottom, spine = np.where(pos < 0, line, 0), np.where(pos > 0, line, 0)
    return bottom.astype("float32"), spine.astype("float32"), pos.astype("float32")


def travel_lines(
    z: Floats, res: float, tpi_m: float = 300.0, power: float = 2.0, relief_m: float = 3.0
) -> tuple[Floats, Floats]:
    """Drainage bottoms and ridge spines together, where animals travel through broken country (both run level
    along their axis, give cover or sight lines, and connect hunting ground; mid-slopes are crossed, not
    followed). Returns (line 0-1, signed position -1 bottom .. +1 spine); see travel_parts."""
    bottom, spine, pos = travel_parts(z, res, tpi_m, power, relief_m)
    return (bottom + spine).astype("float32"), pos


def _soft_near(points: Mask, reach_m: float, res: float) -> Floats:
    """1 on the points, fading linearly to 0 at reach_m (0 everywhere with no point)."""
    if not points.any():
        return np.zeros(points.shape, "float32")
    out: Floats = np.clip(1 - edt(points, res) / reach_m, 0, 1).astype("float32")
    return out


def spine_gate(
    pos: Floats,
    saddle_rc: list[tuple[int, int]],
    res: float,
    saddle_m: float = 150.0,
    junction_m: float = 100.0,
    junction_pos: float = 0.5,
) -> Floats:
    """0-1: where a ridge spine is a crossing, not just a ridge. 1 at a saddle (fading to 0 at saddle_m) or at a
    junction of the spine skeleton (cells of the skeleton of pos > junction_pos with 3+ skeleton neighbours,
    fading to 0 at junction_m): a lion crossing from one drainage to the next goes over there."""
    sad = np.zeros(pos.shape, bool)
    for r, c in saddle_rc:
        if 0 <= r < pos.shape[0] and 0 <= c < pos.shape[1]:
            sad[r, c] = True
    sk: Mask = skeletonize(pos > junction_pos)  # type: ignore[no-untyped-call]  # scikit-image is untyped
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    junction = sk & (nb >= 3)
    out: Floats = np.maximum(_soft_near(sad, saddle_m, res), _soft_near(junction, junction_m, res))
    return out


def disc_kernel(radius_m: float, res: float) -> Floats:
    """float32 0/1 disc of radius_m on a res grid (cells whose centre is within radius_m of the middle one)."""
    k = max(1, math.ceil(radius_m / res))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    disc: Floats = ((xx * res) ** 2 + (yy * res) ** 2 <= radius_m**2).astype("float32")
    return disc


def disc_frac(mask: Mask, radius_m: float, res: float) -> Floats:
    """Fraction of a disc of radius_m around each cell where mask is true."""
    disc = disc_kernel(radius_m, res)
    frac: Floats = np.clip(signal.fftconvolve(mask.astype("float32"), disc / disc.sum(), mode="same"), 0, 1)
    return frac


def thermal_slopes(
    z: Floats,
    res: float,
    aspect_deg: tuple[float, float] = (100.0, 215.0),
    min_slope_deg: float = 10.0,
    radius_m: float = 300.0,
    frac: tuple[float, float] = (0.2, 0.45),
) -> Floats:
    """0-1: how much of the ground within radius_m is a sun-facing slope (aspect within aspect_deg, compass
    degrees, and at least min_slope_deg steep, on the DEM smoothed 30 m), ramped from frac[0] to frac[1] of the
    disc. Winter sun warms such slopes and draws air up them; a ridge above them is a winter travel line."""
    dx, dn, sl = downslope_unit(smooth(np.asarray(z, dtype="float32"), 30, res), res)
    aspect = (np.degrees(np.arctan2(dx, dn)) + 360) % 360  # the compass direction the slope faces
    sunny = (aspect >= aspect_deg[0]) & (aspect <= aspect_deg[1]) & (np.degrees(np.arctan(sl)) >= min_slope_deg)
    out: Floats = np.clip((disc_frac(sunny, radius_m, res) - frac[0]) / (frac[1] - frac[0]), 0, 1).astype("float32")
    return out


def gentle_grade(z: Floats, res: float, deg: tuple[float, float] = (8.0, 25.0), min_: float = 0.5) -> Floats:
    """min_..1: 1 on grades up to deg[0] (slope of the DEM smoothed 20 m), falling linearly to min_ at deg[1] and
    steeper. Travelling lions follow the gentler line (Dickson et al. 2005, Dunford et al. 2020)."""
    sl = slope_deg(smooth(np.asarray(z, dtype="float32"), 20, res), res)
    out: Floats = np.clip(1 - (1 - min_) * (sl - deg[0]) / (deg[1] - deg[0]), min_, 1).astype("float32")
    return out


# ---- distance transform, labeled extrema, 3x3 median (numba, parallel) ------------------------------


@jit(parallel=True)
def _nearest_in_column(feat: Mask) -> I32:
    """Per cell: row of the nearest feature cell in the same column (-1 when the column has none)."""
    H, W = feat.shape
    near = np.full((H, W), -1, np.int32)
    for c in prange(W):
        last = -1
        for r in range(H):
            if feat[r, c]:
                last = r
            near[r, c] = last
        nxt = -1
        for r in range(H - 1, -1, -1):
            if feat[r, c]:
                nxt = r
            if nxt >= 0 and (near[r, c] < 0 or nxt - r < r - near[r, c]):
                near[r, c] = nxt
    return near


@jit(parallel=True)
def _edt_rows(near: I32, res: float, want_idx: bool) -> tuple[F64, I32, I32]:  # noqa: C901 - one numba loop nest
    """Felzenszwalb-Huttenlocher lower envelope along each row over the per-column nearest rows."""
    H, W = near.shape
    dist = np.empty((H, W), np.float64)
    ir = np.empty((H, W) if want_idx else (1, 1), np.int32)
    ic = np.empty((H, W) if want_idx else (1, 1), np.int32)
    for r in prange(H):
        v = np.empty(W, np.int64)  # columns whose parabolas form the envelope
        z = np.empty(W + 1, np.float64)  # envelope breakpoints
        f = np.empty(W, np.float64)
        k = -1
        for q in range(W):
            nr = near[r, q]
            if nr < 0:
                continue
            fq = float((r - nr) * (r - nr))
            f[q] = fq
            if k < 0:
                k = 0
                v[0] = q
                z[0] = -np.inf
                z[1] = np.inf
                continue
            sx = 0.0
            while k >= 0:
                p = v[k]
                sx = ((fq + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)
                if sx > z[k]:
                    break
                k -= 1
            k += 1
            v[k] = q
            z[k] = -np.inf if k == 0 else sx
            z[k + 1] = np.inf
        if k < 0:  # no feature anywhere (callers check first)
            for c in range(W):
                dist[r, c] = np.inf
                if want_idx:
                    ir[r, c] = -1
                    ic[r, c] = -1
            continue
        j = 0
        for c in range(W):
            while z[j + 1] < c:
                j += 1
            q = v[j]
            fr = near[r, q]
            a = (r - fr) * res
            b = (c - q) * res
            dist[r, c] = math.sqrt(a * a + b * b)
            if want_idx:
                ir[r, c] = fr
                ic[r, c] = q
    return dist, ir, ic


def edt(features: Mask, res: float = 1.0) -> F64:
    """Exact Euclidean distance (m) from every cell to the nearest True cell of `features`, i.e.
    scipy.ndimage.distance_transform_edt(~features, sampling=res), on all cores and ~10x faster. With no True
    cell at all, distances are inf."""
    feat = np.ascontiguousarray(features, dtype=np.bool_)
    return _edt_rows(_nearest_in_column(feat), float(res), False)[0]


def edt_nearest(features: Mask, res: float = 1.0) -> tuple[F64, tuple[I32, I32]]:
    """edt, plus the (rows, cols) of each cell's nearest True cell (int32; -1 with no True cell at all)."""
    feat = np.ascontiguousarray(features, dtype=np.bool_)
    d, ir, ic = _edt_rows(_nearest_in_column(feat), float(res), True)
    return d, (ir, ic)


@jit()
def _labeled_minmax(vals: Floats, lab: Ints, n: int) -> tuple[Floats, Floats]:
    mn = np.zeros(n + 1, vals.dtype)
    mx = np.zeros(n + 1, vals.dtype)
    seen = np.zeros(n + 1, np.bool_)
    for i in range(vals.size):
        lb = lab[i]
        if lb < 1 or lb > n:
            continue
        v = vals[i]
        if not seen[lb]:
            mn[lb] = v
            mx[lb] = v
            seen[lb] = True
        else:
            mn[lb] = min(mn[lb], v)
            mx[lb] = max(mx[lb], v)
    return mn[1:], mx[1:]


def labeled_minmax(values: Floats, labels: Ints, n: int) -> tuple[Floats, Floats]:
    """Per label 1..n: (min, max) of values, in one pass (ndimage.minimum/maximum sort). Labels with no cell
    get 0, as in scipy. Values must not be NaN."""
    return _labeled_minmax(np.ravel(values), np.ravel(labels), int(n))


def label_counts(labels: Ints, n: int) -> I64:
    """Cells per label 1..n (int64): ndimage.sum(labels > 0, labels, 1..n) without its extra passes."""
    return np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]


def label_sums(values: npt.NDArray[Any], labels: Ints, n: int) -> F64:
    """Sum of values per label 1..n (float64): the same bincount ndimage.sum(values, labels, 1..n) runs."""
    return np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)[1 : n + 1]


@jit(parallel=True)
def _median3(p: Floats) -> Floats:
    H, W = p.shape[0] - 2, p.shape[1] - 2
    out = np.empty((H, W), p.dtype)
    for r in prange(H):
        buf = np.empty(9, p.dtype)
        for c in range(W):
            k = 0
            for i in range(3):
                for j in range(3):
                    buf[k] = p[r + i, c + j]
                    k += 1
            for i in range(1, 9):  # insertion sort of 9 values
                v = buf[i]
                j = i - 1
                while j >= 0 and buf[j] > v:
                    buf[j + 1] = buf[j]
                    j -= 1
                buf[j + 1] = v
            out[r, c] = buf[4]
    return out


def median_filter(a: Floats, size: int) -> Floats:
    """ndimage.median_filter(a, size) (reflect edges); the common 3x3 NaN-free case runs in parallel numba."""
    if size == 3 and a.ndim == 2 and a.dtype.kind == "f" and not np.isnan(a).any():
        return _median3(np.pad(a, 1, mode="symmetric"))  # numpy 'symmetric' == scipy 'reflect'
    out: Floats = ndimage.median_filter(a, size=size)
    return out


# ---- heap helpers (numba) -------------------------------------------------------------------------


@jit()
def _hpush(hk: F64, hv: I64, n: int, k: float, v: int) -> int:
    i = n
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1


@jit()
def _hpop(hk: F64, hv: I64, n: int) -> tuple[float, int, int]:
    k, v = hk[0], hv[0]
    n -= 1
    hk[0], hv[0] = hk[n], hv[n]
    i = 0
    while True:
        left = 2 * i + 1
        if left >= n:
            break
        c = left
        if left + 1 < n and hk[left + 1] < hk[left]:
            c = left + 1
        if hk[i] <= hk[c]:
            break
        hk[c], hk[i] = hk[i], hk[c]
        hv[c], hv[i] = hv[i], hv[c]
        i = c
    return k, v, n


# ---- hydrology-style routing (used for cold air drainage and valley bottoms) ----------------------


@jit()
def _fill_eps(z: F64) -> tuple[F64, I64]:
    """Priority-flood depression filling with a tiny gradient so every cell drains to the edge. Also returns the
    cells in the order they left the queue: nondecreasing filled elevation, so reversed it is a valid
    upstream-first order for accumulation (no sort needed)."""
    H, W = z.shape
    out = z.copy()
    done = np.zeros((H, W), np.bool_)
    cap = H * W + 8 * (H + W)
    hk = np.empty(cap, np.float64)
    hv = np.empty(cap, np.int64)
    order = np.empty(H * W, np.int64)
    m = 0
    n = 0
    for r in range(H):
        for c in range(W):
            if r == 0 or c == 0 or r == H - 1 or c == W - 1:
                n = _hpush(hk, hv, n, out[r, c], r * W + c)
                done[r, c] = True
    while n > 0:
        k, v, n = _hpop(hk, hv, n)
        order[m] = v
        m += 1
        r, c = v // W, v % W
        for d in range(8):
            rr, cc = r + DR[d], c + DC[d]
            if rr < 0 or cc < 0 or rr >= H or cc >= W or done[rr, cc]:
                continue
            done[rr, cc] = True
            nz = out[rr, cc]
            lim = k + 1e-4
            if nz < lim:
                nz = lim
                out[rr, cc] = nz
            n = _hpush(hk, hv, n, nz, rr * W + cc)
    return out, order


@jit(parallel=True)
def _d8_receivers(z: F64, res: float) -> I64:
    """Each cell's D8 receiver (flat index of its steepest strictly lower neighbour, -1 = none)."""
    H, W = z.shape
    rec = np.full(H * W, -1, np.int64)
    for r in prange(H):
        for c in range(W):
            best, bi = 0.0, -1
            for d in range(8):
                rr, cc = r + DR[d], c + DC[d]
                if rr < 0 or cc < 0 or rr >= H or cc >= W:
                    continue
                s = (z[r, c] - z[rr, cc]) / (DL[d] * res)
                if s > best:
                    best, bi = s, rr * W + cc
            rec[r * W + c] = bi
    return rec


@jit()
def _d8_accumulate(rec: I64, low_first: I64) -> F64:
    """D8 contributing cells. low_first: cells by nondecreasing z (the fill's queue order). Every receiver is
    strictly lower than its donor, so walking that order backwards passes each cell's total on only after all its
    donors (ties in z cannot be donor and receiver, and sums of whole cell counts are exact in any order)."""
    acc = np.ones(rec.size, np.float64)
    for t in range(rec.size - 1, -1, -1):
        i = low_first[t]
        j = rec[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc


def d8_flow(z: Floats, res: float) -> tuple[I64, F64]:
    """D8 on a depression-filled DEM: each cell's receiver (flat index, -1 at the edge) and its upslope
    contributing area (m^2)."""
    f, low_first = _fill_eps(z.astype(np.float64))
    rec = _d8_receivers(f, float(res))
    return rec, _d8_accumulate(rec, low_first).reshape(z.shape) * res * res


def flow_accumulation(z: Floats, res: float) -> F64:
    """Upslope contributing area (m^2) using D8 on a depression-filled DEM."""
    return d8_flow(z, res)[1]


@jit()
def _mfd_accumulate(z: F64, res: float, low_first: I64, p: float) -> F64:
    """Multiple-flow-direction contributing cells (Quinn et al. 1991 / Freeman 1991): each cell passes its total
    to every lower neighbour in proportion to slope**p x contour length (0.5 cell for an edge neighbour, 0.354
    for a diagonal one). Walks low_first backwards like _d8_accumulate (every receiver is strictly lower)."""
    H, W = z.shape
    acc = np.ones(H * W, np.float64)
    wts = np.zeros(8, np.float64)
    for t in range(H * W - 1, -1, -1):
        i = low_first[t]
        r, c = i // W, i % W
        tot = 0.0
        for d in range(8):
            wts[d] = 0.0
            rr, cc = r + DR[d], c + DC[d]
            if rr < 0 or cc < 0 or rr >= H or cc >= W:
                continue
            s = (z[r, c] - z[rr, cc]) / (DL[d] * res)
            if s > 0:
                wts[d] = s**p * (0.5 if DL[d] == 1.0 else 0.354)
                tot += wts[d]
        if tot > 0:
            for d in range(8):
                if wts[d] > 0:
                    acc[(r + DR[d]) * W + c + DC[d]] += acc[i] * wts[d] / tot
    return acc.reshape(H, W)


def specific_catchment(z: Floats, res: float, p: float = 1.1) -> F64:
    """Upslope area per metre of contour width (m^2/m, the 'specific catchment area'), by multiple flow directions
    on a depression-filled DEM. Unlike a single-path (D8) area it grows steadily down an open hillside (about the
    slope length above the cell) instead of staying one cell until flow gathers into a line, and it does not
    depend on the cell size."""
    f, low_first = _fill_eps(z.astype(np.float64))
    return _mfd_accumulate(f, float(res), low_first, float(p)) * res


# ---- landforms (geomorphons, Jasiewicz & Stepinski 2013) ------------------------------------------

_FL, _PK, _RI, _SH, _SP, _SL, _HL, _FS, _VL, _PT = (
    FLAT,
    PEAK,
    RIDGE,
    SHOULDER,
    SPUR,
    SLOPE,
    HOLLOW,
    FOOTSLOPE,
    VALLEY,
    PIT,
)
# rows: number of "lower" directions (0-8); cols: number of "higher" directions (0-8)
_FORMS = np.array(
    [
        [_FL, _FL, _FL, _FS, _FS, _VL, _VL, _VL, _PT],
        [_FL, _FL, _FS, _FS, _FS, _VL, _VL, _VL, 0],
        [_FL, _SH, _SL, _SL, _HL, _HL, _VL, 0, 0],
        [_SH, _SH, _SL, _SL, _SL, _HL, 0, 0, 0],
        [_SH, _SH, _SP, _SL, _SL, 0, 0, 0, 0],
        [_RI, _RI, _SP, _SP, 0, 0, 0, 0, 0],
        [_RI, _RI, _RI, 0, 0, 0, 0, 0, 0],
        [_RI, _RI, 0, 0, 0, 0, 0, 0, 0],
        [_PK, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)


@jit(parallel=True)
def _geomorphons(z: F64, res: float, L: int, flat_rad: float) -> I64:
    H, W = z.shape
    out = np.zeros((H, W), np.int64)
    for r in prange(H):
        for c in range(W):
            z0 = z[r, c]
            npl, nmi = 0, 0
            for d in range(8):
                up, dn = -10.0, -10.0
                for k in range(1, L + 1):
                    rr, cc = r + DR[d] * k, c + DC[d] * k
                    if rr < 0 or cc < 0 or rr >= H or cc >= W:
                        break
                    dist = DL[d] * k * res
                    a = math.atan((z[rr, cc] - z0) / dist)
                    up = max(up, a)
                    dn = max(dn, -a)
                if up == -10.0:
                    continue
                if up > flat_rad or dn > flat_rad:
                    if up > dn:
                        npl += 1
                    elif dn > up:
                        nmi += 1
            f = _FORMS[nmi, npl]
            out[r, c] = f if f > 0 else _SL
    return out


def geomorphons(z: Floats, res: float, search_m: float = 250.0, flat_deg: float = 1.5) -> Ints:
    L = max(2, round(search_m / res))
    return _geomorphons(z.astype(np.float64), float(res), L, math.radians(flat_deg)).astype(np.int8)


# ---- saddles ---------------------------------------------------------------------------------------


def _rays(zs: Floats, r: Ints, c: Ints, dx: Floats, dy: Floats, steps: Floats, res: float) -> Floats:
    """Elevations sampled outward from each (r, c) along its array-space direction (dx=col, dy=row):
    one row of samples per point."""
    H, W = zs.shape
    rr = np.clip(np.round(r[:, None] + dy[:, None] * steps / res).astype(int), 0, H - 1)
    cc = np.clip(np.round(c[:, None] + dx[:, None] * steps / res).astype(int), 0, W - 1)
    out: Floats = zs[rr, cc]
    return out


class Saddle(TypedDict):
    row: int
    col: int
    rise_m: float  # climb saved by crossing here (the lower of the ridge rises on either side)
    drop_m: float
    ridge_axis_deg: float  # from east, 0-180


def saddles(
    z: Floats, res: float, sigma_m: float = 25.0, reach_m: float = 250.0, min_rise_m: float = 8.0
) -> list[Saddle]:
    """Saddles (low points on a ridge line). A saddle is a Hessian saddle point of the smoothed surface that
    is a local minimum along the ridge and a local maximum across it. Returns points with the rise of the
    ridge on either side (the climb an animal saves by crossing here)."""
    zs = smooth(z, sigma_m, res)
    gy, gx = np.gradient(zs, res)
    gyy, gyx = np.gradient(gy, res)
    gxy, gxx = np.gradient(gx, res)
    hxy = 0.5 * (gxy + gyx)
    det = gxx * gyy - hxy**2
    grad = np.hypot(gx, gy)
    # saddle-ness: strongly negative determinant with near-zero gradient
    s = np.where((det < 0) & (grad < np.tan(np.radians(8))), np.sqrt(-np.minimum(det, 0)), 0)
    win = max(3, round(60 / res) | 1)
    peaks = (s == ndimage.maximum_filter(s, size=win)) & (s > 0)
    rows, cols = np.nonzero(peaks)
    if not len(rows):
        return []
    steps = np.linspace(res, reach_m, max(4, int(reach_m / (2 * res))))
    # Hessian per candidate; the eigenvector of the positive eigenvalue = along the ridge (surface curves up)
    Hm = np.stack(
        [np.stack([gxx[rows, cols], hxy[rows, cols]], -1), np.stack([hxy[rows, cols], gyy[rows, cols]], -1)], -2
    )
    w, v = np.linalg.eigh(Hm)
    ex, ey = v[:, 0, 1], v[:, 1, 1]  # along ridge (x=col dir, y=row dir in array space)
    nx, ny = v[:, 0, 0], v[:, 1, 0]  # across ridge
    z0 = zs[rows, cols]
    rise = np.minimum(
        _rays(zs, rows, cols, ex, ey, steps, res).max(1) - z0, _rays(zs, rows, cols, -ex, -ey, steps, res).max(1) - z0
    )
    drop = np.minimum(
        z0 - _rays(zs, rows, cols, nx, ny, steps, res).min(1), z0 - _rays(zs, rows, cols, -nx, -ny, steps, res).min(1)
    )
    keep = (w[:, 0] < 0) & (w[:, 1] > 0) & (rise >= min_rise_m) & (drop >= min_rise_m * 0.75)
    return [
        Saddle(
            row=int(rows[i]),
            col=int(cols[i]),
            rise_m=float(rise[i]),
            drop_m=float(drop[i]),
            ridge_axis_deg=math.degrees(math.atan2(-ey[i], ex[i])) % 180,  # from east (array rows go south)
        )
        for i in np.flatnonzero(keep)
    ]


# ---- cliffs ----------------------------------------------------------------------------------------


def cliffs(
    z: Floats,
    res: float,
    cliff_deg: float = 50.0,
    steep_deg: float = 35.0,
    min_area_m2: float = 60.0,
    min_relief_m: float = 12.0,
    base_reach_m: float = 60.0,
    top_reach_m: float = 30.0,
) -> tuple[Mask, Floats, Floats]:
    """Cliffs and escarpments.

    A barrier is a connected band of steep ground (>= steep_deg) with at least min_relief_m of vertical relief;
    bands that contain bare rock faces (>= cliff_deg) count fully, plain steep slopes count 70%. Returns the rock
    cliff mask, a base score (gentle ground at the foot of the band, within base_reach_m) and a rim score.
    """
    sl = slope_deg(smooth(z, 3.0, res), res)
    rock = sl >= cliff_deg
    steep = sl >= steep_deg
    lab, n = ndimage.label(steep, structure=np.ones((3, 3)))
    zero = np.zeros_like(z, dtype="float32")
    if n == 0:
        return rock, zero, zero
    lo, hi = labeled_minmax(z, lab, n)
    zmax = np.concatenate([[0], hi])
    zmin = np.concatenate([[0], lo])
    rock_area = np.concatenate([[0], label_sums(rock, lab, n)]) * res * res
    relief = zmax - zmin
    keep = relief >= min_relief_m
    keep[0] = False
    barrier = keep[lab]
    # rock cliff mask: rock cells in kept bands, minus specks
    rock &= barrier
    rl, rn = ndimage.label(rock)
    if rn:
        sizes = label_counts(rl, rn) * res * res  # rock is exactly the labeled cells
        k2 = np.zeros(rn + 1, bool)
        k2[1:] = sizes >= min_area_m2
        rock = k2[rl]
    if not barrier.any():
        return rock, zero, zero
    dist, (ir, ic) = edt_nearest(barrier, res)
    L = lab[ir, ic]
    mid = 0.5 * (zmax[L] + zmin[L])
    tall = np.clip(relief[L] / 30.0, 0.35, 1.0) * np.where(rock_area[L] >= min_area_m2, 1.0, 0.7)
    gentle = sl < 25
    base = np.where(gentle & (z < mid) & (dist > 0), np.clip(1 - dist / base_reach_m, 0, 1) * tall, 0)
    top = np.where(gentle & (z >= mid) & (dist > 0), np.clip(1 - dist / top_reach_m, 0, 1) * tall, 0)
    return rock, base.astype("float32"), top.astype("float32")


# ---- wind sheltering (Winstral et al. 2002 "Sx") -----------------------------------------------------


def upwind_shelter(z: Floats, res: float, from_deg: float, reach_m: float = 300.0) -> Floats:
    """Max upwind horizon angle (degrees). >0 means terrain upwind is higher (sheltered/leeward)."""
    th = math.radians(from_deg)
    ux, uy = math.sin(th), math.cos(th)  # direction the wind comes FROM (east, north)
    H, W = z.shape
    rr, cc = np.mgrid[0:H, 0:W].astype("float32")

    def angle(d: float) -> Floats:
        zr = ndimage.map_coordinates(z, [rr - uy * d / res, cc + ux * d / res], order=1, mode="nearest")
        out: Floats = np.degrees(np.arctan((zr - z) / d))
        return out

    best = np.full(z.shape, -90.0, dtype="float32")
    with ThreadPoolExecutor(min(THREADS, 6)) as ex:  # ndimage releases the GIL; max is order-independent
        for a in ex.map(angle, np.linspace(max(res, 10), reach_m, 12)):
            best = np.maximum(best, a)
    return best


# ---- least-cost walking -----------------------------------------------------------------------------


@jit(nogil=True)  # releases the GIL, so independent walks can run on threads at once
def _dijkstra(cost: F64, src: Mask, res: float, max_len: float) -> tuple[F64, F64, I64]:
    """cost: seconds per metre (inf = impassable). src: bool sources. Returns (time_s, route_len_m, pred)."""
    H, W = cost.shape
    N = H * W
    t = np.full(N, np.inf)
    ln = np.full(N, np.inf)
    pred = np.full(N, -1, np.int64)
    done = np.zeros(N, np.bool_)
    cap = 8 * N + 1
    hk = np.empty(cap, np.float64)
    hv = np.empty(cap, np.int64)
    n = 0
    for i in range(N):
        if src.ravel()[i] and np.isfinite(cost.ravel()[i]):
            t[i] = 0.0
            ln[i] = 0.0
            n = _hpush(hk, hv, n, 0.0, i)
    cf = cost.ravel()
    while n > 0:
        k, v, n = _hpop(hk, hv, n)
        if done[v] or k > t[v]:
            continue
        done[v] = True
        if ln[v] > max_len:
            continue
        r, c = v // W, v % W
        for d in range(8):
            rr, cc = r + DR[d], c + DC[d]
            if rr < 0 or cc < 0 or rr >= H or cc >= W:
                continue
            u = rr * W + cc
            if done[u] or not np.isfinite(cf[u]):
                continue
            step = DL[d] * res
            nt = t[v] + 0.5 * (cf[v] + cf[u]) * step
            if nt < t[u]:
                t[u] = nt
                ln[u] = ln[v] + step
                pred[u] = v
                if n < cap:
                    n = _hpush(hk, hv, n, nt, u)
    return t.reshape(H, W), ln.reshape(H, W), pred


@jit()
def _before(ka: float, va: int, kb: float, vb: int) -> bool:
    """Settling order (cost, cell): equal costs settle the lower cell index first, so routes are reproducible."""
    return bool(ka < kb or (ka == kb and va < vb))


@jit()
def _grow(hk: F64, hv: I64) -> tuple[F64, I64]:
    n = hk.size
    hk2, hv2 = np.empty(2 * n, np.float64), np.empty(2 * n, np.int64)
    hk2[:n], hv2[:n] = hk, hv
    return hk2, hv2


@jit(nogil=True)
def _least_cost_tree(  # noqa: C901, PLR0912, PLR0915 - one numba loop: heap helper calls cost ~15%
    cost: F64, starts: I64, res: float, cutoff: float
) -> tuple[F64, I64, I64]:
    """Multi-source Dijkstra (8 neighbours, step x mean cost of its two cells) from the start cells (sorted), up to
    cutoff. Returns the cost distance (inf past the cutoff), each cell's predecessor (-1 at a start or unreached)
    and the reached cells in the order they settled (a parent always before its children). Cells settle in
    (cost, index) order; the start cells, all at cost 0, settle first in index order without entering the heap."""
    H, W = cost.shape
    N = H * W
    cf = cost.ravel()
    D = np.full(N, np.inf)
    pred = np.full(N, -1, np.int64)
    done = np.zeros(N, np.bool_)
    order = np.empty(N, np.int64)
    n_done = 0
    hk, hv = np.empty(1 << 16, np.float64), np.empty(1 << 16, np.int64)
    n = 0
    for s in starts:
        D[s] = 0.0
    s_next = 0
    while n > 0 or s_next < starts.size:
        if s_next < starts.size:
            d, i = 0.0, starts[s_next]
            s_next += 1
        else:  # pop the top of the 4-ary heap (shallower than a binary one), sifting the last entry down
            d, i = hk[0], hv[0]
            n -= 1
            last_k, last_v = hk[n], hv[n]
            h = 0
            while True:
                first = 4 * h + 1
                if first >= n:
                    break
                best = first
                for c in range(first + 1, min(first + 4, n)):
                    if _before(hk[c], hv[c], hk[best], hv[best]):
                        best = c
                if not _before(hk[best], hv[best], last_k, last_v):
                    break
                hk[h], hv[h] = hk[best], hv[best]
                h = best
            hk[h], hv[h] = last_k, last_v
        if done[i]:
            continue
        done[i] = True
        order[n_done] = i
        n_done += 1
        r = i // W
        c = i - r * W
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W:
                continue
            j = rr * W + cc
            if done[j]:
                continue
            nd = d + DL[k] * res * 0.5 * (cf[i] + cf[j])
            if nd < D[j] and nd <= cutoff:
                D[j] = nd
                pred[j] = i
                if n == hk.size:
                    hk, hv = _grow(hk, hv)
                h = n  # push: sift the hole up from the end
                while h > 0:
                    p = (h - 1) >> 2
                    if not _before(nd, j, hk[p], hv[p]):
                        break
                    hk[h], hv[h] = hk[p], hv[p]
                    h = p
                hk[h], hv[h] = nd, j
                n += 1
    return D.reshape(H, W), pred, order[:n_done]


@jit(nogil=True)
def _tree_flow(
    pred: I64, order: I64, weight: F64, covered: npt.NDArray[np.bool_], W: int, res: float, last_m: float
) -> tuple[I64, F64, F64]:
    """Over a least-cost tree (pred, order): each cell's root (start cell), the weight flowing through it (its own
    plus everything upstream: origins send their weight down the tree to the root), and the covered share of the
    route's last_m metres into the root (1 where the route is shorter than one step)."""
    N = pred.size
    root = np.full(N, -1, np.int64)
    length = np.zeros(N)  # route length from the root
    uncovered = np.zeros(N)  # uncovered metres of the route's first last_m from the root
    for t in range(order.size):
        i = order[t]
        p = pred[i]
        if p < 0:
            root[i] = i
            continue
        root[i] = root[p]
        diag = (i // W != p // W) and (i % W != p % W)
        step = res * (math.sqrt(2.0) if diag else 1.0)
        length[i] = length[p] + step
        uncovered[i] = uncovered[p]
        if not covered[i] and length[i] <= last_m:
            uncovered[i] += step
    acc = weight.copy()
    for t in range(order.size - 1, -1, -1):
        i = order[t]
        p = pred[i]
        if p >= 0:
            acc[p] += acc[i]
    share = np.ones(N)
    for i in range(N):
        if length[i] > 0:
            share[i] = 1.0 - uncovered[i] / min(length[i], last_m)
    return root, acc, share


class CostTree(NamedTuple):
    """A least-cost tree grown from destination cells (terrain.cost_tree), on the grid it was grown on."""

    dist: F64  # cost distance to the nearest destination (inf past the cutoff)
    root: Ints  # flat index of the destination each cell's route ends at (-1 = unreached)
    flow: F64  # origin weight passing through each cell on its way to a destination
    covered_share: F64  # covered share of each route's last metres into its destination


def cost_tree(
    cost: Floats, starts: Mask, origin_weight: Floats, covered: Mask, res: float, cutoff: float, last_m: float
) -> CostTree:
    """Least-cost routes from every cell to the nearest start cell (by cost distance, up to cutoff), the flow of
    origin_weight down them, and the covered share of each route's last_m metres."""
    H, W = cost.shape
    src = np.flatnonzero(starts.ravel()).astype(np.int64)
    D, pred, order = _least_cost_tree(cost.astype(np.float64), src, float(res), float(cutoff))
    root, acc, share = _tree_flow(
        pred, order, origin_weight.ravel().astype(np.float64), covered.ravel().astype(np.bool_), W, float(res), last_m
    )
    return CostTree(D, root.reshape(H, W), acc.reshape(H, W), share.reshape(H, W))


def tobler_sec_per_m(slope: Floats) -> Floats:
    """Walking pace from Tobler's hiking function, averaged uphill/downhill (seconds per metre)."""
    s = np.tan(np.radians(slope))
    v = 0.5 * (6 * np.exp(-3.5 * np.abs(s + 0.05)) + 6 * np.exp(-3.5 * np.abs(-s + 0.05)))  # km/h
    out: Floats = 3.6 / np.maximum(v, 0.05)
    return out


def walking(cost: Floats, sources: Mask, res: float, max_len_m: float) -> tuple[F64, F64, I64]:
    """Least-cost walk from the source cells: (seconds, route length m, predecessor flat index or -1) per cell.
    Routes stop growing past 1.5 x max_len_m."""
    return _dijkstra(cost.astype(np.float64), sources.astype(np.bool_), float(res), float(max_len_m) * 1.5)


def trace_route(
    pred: Ints, shape: tuple[int, int], row: int, col: int, max_steps: int = 100000
) -> list[tuple[int, int]]:
    _, W = shape
    i = row * W + col
    path: list[tuple[int, int]] = []
    while i >= 0 and len(path) < max_steps:
        path.append((i // W, i % W))
        i = pred[i]
    return path[::-1]
