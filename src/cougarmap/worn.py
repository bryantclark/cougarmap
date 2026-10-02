"""Worn trails from 1 m bare-earth lidar: game trails, cattle trails and old two-tracks, mapped or not.

A worn tread shows in a 1 m DEM as a narrow trough on gentle ground and as a bench on a sidehill (a cut slope
above, a flat tread, a fill slope below). The detector (parameters in config.WornTrails):

1. Oriented trough filters: the second derivative across an anisotropic Gaussian (1-2 m across, 6 m along) in
   12 directions, by FFT, chunk by chunk. The long support lifts faint treads out of the lidar noise; each scale's
   response is divided by its robust spread over the area (the noise), so thresholds are in noise units.
2. Fall-line test: on ground steeper than 8 deg only directions at least 40 deg off the fall line count
   (drainages run down the fall line; trails traverse).
3. Bench test on sidehills: the ground just below the line must be flatter than the cut above and the fill below
   (a concave slope break alone is a toe slope or a terrace edge).
4. Hysteresis, a 1-cell skeleton, lines of at least 40 m.
5. Creek banks: a line within ~10 m of a mapped flowline or waterbody, or of a channel the fine DEM drains into,
   and roughly parallel to it, is a channel edge, not a trail, and is dropped.

Known miss: faint two-tracks across flat open meadows (two shallow ruts on smooth ground) mostly stay under the
noise floor. The layer is a suggestion for where to hang a camera; it changes no score.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, TypedDict

import numpy as np
import shapely
from scipy import fft as sfft
from scipy import ndimage

from .arrays import Floats, Ints, Mask
from .config import THREADS, WORN, WornTrails
from .grid import Grid

HALO = 32  # cells of context each chunk reads past its edges (filter support, slope smoothing, bench offsets)
CHUNK = 1024


class WornLine(TypedDict):
    """One detected worn line, as kept in state.pkl and drawn in the KMZ."""

    lonlat: list[tuple[float, float]]
    mapped: bool  # on (within WornTrails.mapped_m of) a mapped road or trail
    length_m: float


@dataclass(frozen=True)
class Detection:
    """What the detector found on a 1 m grid."""

    lines: Mask  # skeleton cells of the kept worn lines
    theta: Ints  # direction index of each cell's strongest filter (0..n_dirs-1; angle = i * pi / n_dirs)
    noise: tuple[float, ...]  # robust spread of each scale's raw response
    creek_cells: int  # skeleton cells dropped as creek banks / channel edges


# ---- filters ---------------------------------------------------------------------------------------------------


@lru_cache(maxsize=64)
def kernel(across: float, along: float, theta: float) -> Floats:
    """Second derivative across a line at angle theta (x = column, y = row, radians) of an anisotropic Gaussian,
    zero-mean and scaled so the response is about 10x the across-curvature in m (positive = trough)."""
    h = int(3 * along)
    y, x = np.mgrid[-h : h + 1, -h : h + 1].astype(np.float64)
    v = x * math.cos(theta) + y * math.sin(theta)
    u = -x * math.sin(theta) + y * math.cos(theta)
    G = np.exp(-(u**2) / (2 * across**2) - v**2 / (2 * along**2))
    K = (u**2 / across**4 - 1 / across**2) * G
    K -= K.mean()
    K /= np.abs(K).sum()
    out: Floats = (K * across * across * 10).astype(np.float32)
    return out


KernelFFTs = dict[tuple[float, int], np.ndarray]  # (across_m, direction index) -> kernel spectrum


def _fft_shape() -> tuple[int, int]:
    n = sfft.next_fast_len(CHUNK + 2 * HALO, real=True)
    return n, n


def kernel_ffts(p: WornTrails) -> KernelFFTs:
    """Every filter's spectrum on the chunk FFT size (built once per detection: ~5 MB each)."""
    shape = _fft_shape()
    return {
        (su, i): _kernel_fft(kernel(su, p.along_m, th), shape) for su in p.across_m for i, th in enumerate(_thetas(p))
    }


def _kernel_fft(K: Floats, shape: tuple[int, int]) -> np.ndarray:
    h = K.shape[0] // 2
    pad = np.zeros(shape, np.float32)
    pad[: K.shape[0], : K.shape[1]] = K
    pad = np.roll(pad, (-h, -h), axis=(0, 1))  # centred at (0, 0): a circular correlation = convolution
    return sfft.rfft2(pad, workers=-1)


def _slope_fall(z: Floats, sigma: float) -> tuple[Floats, Floats]:
    """Slope (degrees) and uphill direction (radians, x = column, y = row) of z smoothed at sigma cells."""
    gy, gx = np.gradient(ndimage.gaussian_filter(z, sigma))
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32), np.arctan2(gy, gx).astype(np.float32)


def _chunks(shape: tuple[int, int], inside: Mask) -> Iterator[tuple[slice, slice]]:
    """Core windows of CHUNK x CHUNK cells that hold any inside cell."""
    H, W = shape
    for r0 in range(0, H, CHUNK):
        for c0 in range(0, W, CHUNK):
            win = (slice(r0, min(H, r0 + CHUNK)), slice(c0, min(W, c0 + CHUNK)))
            if inside[win].any():
                yield win


def _halo(zp: Floats, win: tuple[slice, slice]) -> Floats:
    """The chunk with HALO cells of context, from the HALO-padded array zp."""
    rs, cs = win
    return zp[rs.start : rs.stop + 2 * HALO, cs.start : cs.stop + 2 * HALO]


def _core(a: np.ndarray) -> np.ndarray:
    return a[HALO:-HALO, HALO:-HALO]


def raw_responses(zp: Floats, win: tuple[slice, slice], ks: KernelFFTs, p: WornTrails) -> list[tuple[Floats, Ints]]:
    """Per scale: the best response over the directions the fall-line test allows (-inf where none is) and its
    direction index, on the chunk's core cells."""
    zc = _halo(zp, win).astype(np.float32)
    slope, fall = (_core(a) for a in _slope_fall(zc, p.fall_smooth_m))
    shape = _fft_shape()  # every chunk the same size, so the kernel spectra are shared (edge chunks: zero pad)
    F = sfft.rfft2(zc, s=shape)
    flat = slope < p.flat_deg
    cf, sf = np.cos(fall), np.sin(fall)
    cos_max = math.cos(math.radians(p.off_fall_deg))
    # the directions each cell may use (the fall-line test), shared by both scales
    allowed = [flat | (np.abs(math.cos(t) * cf + math.sin(t) * sf) <= cos_max) for t in _thetas(p)]
    out: list[tuple[Floats, Ints]] = []
    for su in p.across_m:
        best = np.full(flat.shape, -np.inf, np.float32)
        idx = np.zeros(flat.shape, np.int8)
        for i in range(p.n_dirs):
            R = _core(sfft.irfft2(F * ks[su, i], s=shape)[: zc.shape[0], : zc.shape[1]])
            better = allowed[i] & (best < R)
            np.copyto(best, R, where=better)
            idx[better] = i
        out.append((best, idx))
    return out


def _thetas(p: WornTrails) -> list[float]:
    return [i * math.pi / p.n_dirs for i in range(p.n_dirs)]


def robust_spread(a: Floats) -> float:
    """1.4826 x the median absolute deviation of the finite values (a noise level that ignores the lines)."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 1.0
    v = v[:: max(1, v.size // 2_000_000)]
    s = float(np.median(np.abs(v - np.median(v))) * 1.4826)
    return s if s > 0 else 1.0


def bench_ok(zc: Floats, fall: Floats, rows: Ints, cols: Ints, p: WornTrails) -> Mask:
    """The bench test at cells (rows, cols) of a chunk zc whose uphill directions are fall: the ground 0-2 m
    below each is flatter (under bench_ratio) than both the fill slope 3-7 m below and the cut slope 2-7 m
    above."""
    fine_slope, _ = _slope_fall(zc, 1.0)
    f = fall[rows, cols]
    dc, dr = np.cos(f), np.sin(f)  # uphill unit vector

    def at(d: float) -> Floats:  # slope d m downhill (negative: uphill) of each cell
        coords = np.stack([rows - d * dr, cols - d * dc])
        out: Floats = ndimage.map_coordinates(fine_slope, coords, order=1, mode="nearest")
        return out

    tread = np.minimum.reduce([at(d) for d in (0.0, 1.0, 2.0)])
    fill = np.maximum.reduce([at(d) for d in (3.0, 5.0, 7.0)])
    cut = np.maximum.reduce([at(-d) for d in (2.0, 4.0, 7.0)])
    ok: Mask = (tread < p.bench_ratio * fill) & (tread < p.bench_ratio * cut)
    return ok


def _combine(
    zp: Floats,
    win: tuple[slice, slice],
    parts: list[tuple[Floats, Ints]],
    noise: tuple[float, ...],
    inside: Mask,
    p: WornTrails,
) -> tuple[Floats, Ints]:
    """A chunk's response in noise units (the stronger scale) and its direction, after the bench test."""
    scaled = np.stack([b.astype(np.float32) / n for (b, _), n in zip(parts, noise, strict=True)])
    k = np.argmax(scaled, axis=0)[None]
    rc = np.take_along_axis(scaled, k, 0)[0]
    th = np.take_along_axis(np.stack([i for _, i in parts]), k, 0)[0]
    rc = np.where(np.isfinite(rc) & inside, rc, 0).astype(np.float32)
    # the bench test, only where it can matter: candidate cells on sloping ground
    zc = _halo(zp, win).astype(np.float32)
    slope, fall = _slope_fall(zc, p.fall_smooth_m)
    rr, cc = np.nonzero((rc >= p.low) & (_core(slope) >= p.flat_deg))
    if rr.size:
        ok = bench_ok(zc, fall, rr + HALO, cc + HALO, p)
        rc[rr[~ok], cc[~ok]] = 0
    return rc, th


class Response(NamedTuple):
    r: Floats  # trough strength in noise units (0 where a test rejects it, and outside)
    theta: Ints  # direction index of the strongest filter
    noise: tuple[float, ...]


def response(z: Floats, inside: Mask, p: WornTrails = WORN) -> Response:
    """The trough response of a 1 m DEM (no NaN) over the inside cells, after the fall-line and bench tests.
    Chunks run on threads (FFTs, filters and array math release the GIL)."""
    zp = np.pad(np.asarray(z, np.float32), HALO, mode="reflect", reflect_type="odd")  # planes stay planes
    wins = list(_chunks(z.shape, inside))
    ks = kernel_ffts(p)

    def first(w: tuple[slice, slice]) -> list[tuple[Floats, Ints]]:  # float16 keeps the memory down
        return [(b.astype(np.float16), i) for b, i in raw_responses(zp, w, ks, p)]

    with ThreadPoolExecutor(THREADS) as ex:
        raws = dict(zip(wins, ex.map(first, wins), strict=True))
        noise = tuple(
            robust_spread(np.concatenate([raws[w][k][0][inside[w]][::7].astype(np.float32) for w in wins]))
            if wins
            else 1.0
            for k in range(len(p.across_m))
        )
        r = np.zeros(z.shape, np.float32)
        theta = np.zeros(z.shape, np.int8)
        for w, (rc, th) in zip(
            wins, ex.map(lambda w: _combine(zp, w, raws[w], noise, inside[w], p), wins), strict=True
        ):
            r[w], theta[w] = rc, th
            del raws[w]
    return Response(r, theta, noise)


def skeleton_lines(r: Floats, p: WornTrails = WORN) -> Mask:
    """Hysteresis (low/high noise units), a 1-cell skeleton, and only lines of at least min_len_m cells."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import remove_small_objects, skeletonize

    m = apply_hysteresis_threshold(r, p.low, p.high)  # type: ignore[no-untyped-call]  # scikit-image is untyped
    m = remove_small_objects(m, max_size=15)
    sk: Mask = skeletonize(m)  # type: ignore[no-untyped-call]
    return keep_long(sk, p.min_len_m)


def keep_long(sk: Mask, min_cells: float) -> Mask:
    """The 8-connected pieces of a skeleton with at least min_cells cells (about metres on a 1 m grid)."""
    lab, n = ndimage.label(sk, structure=np.ones((3, 3)))
    if n == 0:
        return sk.copy()
    size = np.bincount(lab.ravel())
    keep = size >= min_cells
    keep[0] = False
    out: Mask = keep[lab]
    return out


def along_channel(theta: Ints, to_channel: tuple[Floats, Floats, Floats], p: WornTrails = WORN) -> Mask:
    """Which line cells (their filter directions theta) run along a channel: within creek_m of it and within
    creek_deg of parallel to it (on it, within 1.5 m, always). to_channel: each cell's distance (m) to the
    nearest channel cell and the vector to it (x = column, y = row)."""
    d, vx, vy = to_channel
    th = theta * (math.pi / p.n_dirs)
    lx, ly = np.cos(th), np.sin(th)
    n = np.maximum(np.hypot(vx, vy), 1e-9)
    across = np.abs(lx * vx + ly * vy) / n  # |cos| between the line and the way to the channel
    parallel = across <= math.sin(math.radians(p.creek_deg))
    out: Mask = (d <= 1.5) | ((d <= p.creek_m) & parallel)
    return out


def detect(
    z: Floats,
    inside: Mask,
    to_channel: Callable[[Ints, Ints], tuple[Floats, Floats, Floats]] | None = None,
    p: WornTrails = WORN,
) -> Detection:
    """Worn lines on a 1 m DEM (NaN = no data, left out with a margin). to_channel(rows, cols) gives each cell's
    distance (m) and vector (x = column, y = row) to the nearest channel or mapped water; lines along one are
    dropped as creek banks."""
    bad = ~np.isfinite(z)
    if bad.all():
        return Detection(np.zeros(z.shape, bool), np.zeros(z.shape, np.int8), (1.0,) * len(p.across_m), 0)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
        inside = inside & ~ndimage.binary_dilation(bad, iterations=HALO)
    res = response(z, inside, p)
    sk = skeleton_lines(res.r, p)
    creek = 0
    if to_channel is not None:
        rr, cc = np.nonzero(sk)
        drop = along_channel(res.theta[rr, cc], to_channel(rr, cc), p)
        creek = int(drop.sum())
        if creek:
            sk[rr[drop], cc[drop]] = False
            sk = keep_long(sk, p.min_len_m)
    return Detection(sk, res.theta, res.noise, creek)


# ---- lines -----------------------------------------------------------------------------------------------------


def vectorize(sk: Mask, grid: Grid, min_m: float = 5.0, tolerance_m: float = 1.0) -> list[shapely.LineString]:
    """A skeleton as lines in grid's CRS: neighbouring cells joined (a diagonal only where no square step does),
    merged between junctions, simplified, and pieces shorter than min_m dropped."""
    rr, cc = np.nonzero(sk)
    if rr.size == 0:
        return []
    H, W = sk.shape
    segs = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 < H) & (c2 >= 0) & (c2 < W)
        a_r, a_c, b_r, b_c = rr[ok], cc[ok], r2[ok], c2[ok]
        on = sk[b_r, b_c]
        if dr and dc:  # a diagonal step only where neither square path exists
            on &= ~sk[a_r, b_c] & ~sk[b_r, a_c]
        segs.append(np.stack([a_r[on], a_c[on], b_r[on], b_c[on]], axis=1))
    s = np.concatenate(segs)
    if s.size == 0:
        return []
    x0, y0 = grid.xy(s[:, 0], s[:, 1])
    x1, y1 = grid.xy(s[:, 2], s[:, 3])
    coords = np.stack([np.stack([x0, y0], 1), np.stack([x1, y1], 1)], 1)
    segments: np.ndarray = np.asarray(shapely.linestrings(coords), dtype=object)
    merged = shapely.line_merge(shapely.multilinestrings(segments))
    parts = list(getattr(merged, "geoms", [merged]))
    out = [shapely.simplify(g, tolerance_m) for g in parts if g.length >= min_m]
    return [g for g in out if isinstance(g, shapely.LineString)]


def fine_cells(rows: Ints, cols: Ints, src: Grid, dst: Grid) -> tuple[Ints, Ints, Mask]:
    """The dst cells holding src cells (rows, cols) (same CRS), and which of them fall inside dst."""
    x, y = src.xy(rows, cols)
    r2, c2 = dst.rowcol(x, y)
    return r2, c2, dst.contains_rc(r2, c2)
