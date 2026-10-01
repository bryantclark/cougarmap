"""Lion truth: how the model scores against what lions actually did in the field (the field log, fieldlog.py).

The human-pick check (evaluate.py) measures whether the tool agrees with spots people picked by hand. These tests
measure whether its scores line up with lions, each against its own null so a result can't come from where people
happen to go:

- **Cameras** (`camera_test`): cougar detections per 100 camera-nights for each arm, against the base rate of
  random on-trail cameras in NE Washington (Bassing et al. 2023), and, where a model (or human) camera and a
  control camera share a zone, the within-zone rate ratio with a paired permutation test (zones are compared
  with themselves, so a good zone can't flatter one arm). An on-feature camera (the pick's "alternate on the
  trail") is compared with the pick's own camera in its zone (arm model or human), or with an off-feature one.
- **Snow tracks** (`path_test`): the mean score along a followed track, against the same track shape rotated and
  shifted 100-1,500 m inside the area. A track scoring above its shifted copies walked where the model says
  lions walk. Across tracks, a sign test, also split by whether the track was found from a road (the copies
  don't keep their distance to roads, so road-found tracks are only partly corrected for where people go).
- **Crossing transects** (`transect_test`): the score where lions crossed a fixed route, against every point of
  the route (an AUC: 0.5 = no better than anywhere along the route). Surveys with no crossings still count
  toward the crossing rates.
- **Sample size** (`camera_power`, `zones_needed`, `tracks_needed`): how many paired zones or tracks it takes to
  see an effect of a given size.

Everything here works on a score raster and points in fractional (row, col) cells, so it is tested on synthetic
rasters; `area_truth` ties it to an analyzed area and the field log.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from . import fieldlog
from .arrays import Floats, Mask
from .config import TRUTH, Truth
from .evaluate import AreaPins, production_score, vs_random
from .fieldlog import Camera, Record
from .grid import Grid
from .state import ModelState

JSON = dict[str, Any]
type Points = Floats  # (n, 2) fractional (row, col) cells
# (treatment, comparison) arms compared within zones. The trail alternate's camera (on-feature) pairs with the
# pick's own camera, logged as model or human, so the pick's zone stays in the model-vs-control test too.
TREATMENTS = (
    ("model", "control"),
    ("human", "control"),
    ("on-feature", "model"),
    ("on-feature", "human"),
    ("on-feature", "off-feature"),
)


# ---- shared helpers -------------------------------------------------------------------------------------------


def densify(line: Points, step: float) -> Points:
    """Points every `step` cells along a polyline (its first and last vertex included)."""
    line = np.asarray(line, float)
    if len(line) < 2:
        return line
    seg = np.hypot(*np.diff(line, axis=0).T)
    at = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.append(np.arange(0.0, at[-1], max(step, 1e-9)), at[-1])
    return np.stack([np.interp(s, at, line[:, 0]), np.interp(s, at, line[:, 1])], axis=1)


def inside(mask: Mask, pts: Points) -> Mask:
    """Which points fall in a cell of the area."""
    r = np.floor(pts[:, 0]).astype(int)
    c = np.floor(pts[:, 1]).astype(int)
    ok: Mask = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    ok[ok] = mask[r[ok], c[ok]]
    return ok


def values_at(score: Floats, mask: Mask, pts: Points) -> tuple[Floats, Mask]:
    """(score at each point, whether the point is inside the area). Outside points get NaN."""
    ok = inside(mask, pts)
    v = np.full(len(pts), np.nan)
    v[ok] = score[np.floor(pts[ok, 0]).astype(int), np.floor(pts[ok, 1]).astype(int)]
    return v, ok


def percentile_of(x: float, ref: Floats) -> float:
    """Share of ref below x, ties counting half (0.5 = typical, 1 = above all of ref)."""
    ref = np.asarray(ref)
    return float(np.mean(ref < x) + 0.5 * np.mean(ref == x))


def to_cells(g: Grid, latlon: list[list[float]]) -> Points:
    """[[lat, lon], ...] -> fractional (row, col) cells of g."""
    if not latlon:
        return np.zeros((0, 2))
    a = np.asarray(latlon, float)
    x, y = g.from_lonlat(a[:, 1], a[:, 0])
    return np.stack([(g.y0 - np.asarray(y)) / g.res, (np.asarray(x) - g.x0) / g.res], axis=1)


def sign_test(percentiles: list[float]) -> JSON:
    """Do more tracks beat their shifted copies than not? One-sided binomial sign test (ties at 0.5 dropped)."""
    above = sum(p > 0.5 for p in percentiles)
    below = sum(p < 0.5 for p in percentiles)
    n = above + below
    p = float(stats.binomtest(above, n, 0.5, alternative="greater").pvalue) if n else None
    return dict(n=len(percentiles), above_half=above, below_half=below, p_value=p)


# ---- snow tracks ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathResult:
    mean_score: float  # along the track (inside the area)
    percentile: float  # share of shifted copies scoring lower (0.5 = chance)
    null_mean: float
    n_null: int
    n_points: int
    inside_share: float


def path_test(
    score: Floats, mask: Mask, lines: list[Points], res: float, rng: np.random.Generator, cfg: Truth = TRUTH
) -> PathResult | None:
    """A track (one or more lines, in cells) against cfg.n_shifted copies of itself rotated by a random angle
    about its centre and shifted cfg.shift_min_m-cfg.shift_max_m in a random direction, each keeping at least
    cfg.min_inside of its points inside the area. None when the track is not in the area."""
    parts = [densify(np.asarray(ln, float), cfg.sample_m / res) for ln in lines if len(ln) >= 2]
    if not parts:
        return None
    pts = np.concatenate(parts)
    v, ok = values_at(score, mask, pts)
    if not ok.any():
        return None
    obs = float(np.nanmean(v))
    centre = pts.mean(axis=0)
    rel = pts - centre
    null: list[float] = []
    for _ in range(cfg.n_shifted * 20):
        if len(null) >= cfg.n_shifted:
            break
        th, phi = rng.uniform(0, 2 * math.pi, 2)
        d = rng.uniform(cfg.shift_min_m, cfg.shift_max_m) / res
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        moved = rel @ rot.T + centre + d * np.array([math.cos(phi), math.sin(phi)])
        nv, nok = values_at(score, mask, moved)
        if nok.mean() >= cfg.min_inside:
            null.append(float(np.nanmean(nv)))
    if not null:
        return None
    ref = np.array(null)
    return PathResult(obs, percentile_of(obs, ref), float(ref.mean()), len(null), len(pts), float(ok.mean()))


def tracks_needed(p_above: float, cfg: Truth = TRUTH, max_n: int = 200) -> int | None:
    """Tracks needed for the sign test to reach cfg.power_target, when each track beats its shifted copies with
    probability p_above (exact binomial)."""
    for n in range(1, max_n + 1):
        sig = stats.binom.sf(np.arange(n + 1) - 1, n, 0.5) <= cfg.alpha  # P(at least k above) under chance
        if sig.any():
            k = int(np.argmax(sig))  # the fewest tracks above half that is significant
            if stats.binom.sf(k - 1, n, p_above) >= cfg.power_target:
                return n
    return None


# ---- crossing transects ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Survey:
    """One transect survey in cells: the route's lines and the crossing points (none = an absence)."""

    lines: list[Points]
    crossings: Points


def transect_test(
    score: Floats, mask: Mask, surveys: list[Survey], res: float, rng: np.random.Generator, cfg: Truth = TRUTH
) -> JSON:
    """Score at the crossings against every point of their own route. auc: the mean share of route points a
    crossing out-scores (0.5 = chance); p_value: one-sided, from crossings placed at random along the same
    routes. Also crossings per 10 km of route in the top and bottom third of route scores (absences count)."""
    routes: list[Floats] = []
    pcts: list[list[float]] = []
    cross_vals: list[float] = []
    km = 0.0
    for s in surveys:
        parts = [densify(np.asarray(ln, float), cfg.sample_m / res) for ln in s.lines if len(ln) >= 2]
        rv, rok = values_at(score, mask, np.concatenate(parts)) if parts else (np.zeros(0), np.zeros(0, bool))
        route = rv[rok]
        cv, cok = values_at(score, mask, np.asarray(s.crossings, float).reshape(-1, 2))
        routes.append(route)
        km += len(route) * cfg.sample_m / 1000
        pcts.append([percentile_of(x, route) for x in cv[cok]] if len(route) else [])
        cross_vals += list(cv[cok])
    flat = [p for ps in pcts for p in ps]
    out: JSON = dict(
        surveys=len(surveys),
        with_crossings=sum(bool(p) for p in pcts),
        crossings=len(flat),
        route_km=round(km, 2),
        auc=round(float(np.mean(flat)), 3) if flat else None,
        p_value=None,
    )
    if flat:
        obs = float(np.mean(flat))
        null = np.zeros(cfg.n_permutations)
        for route, ps in zip(routes, pcts, strict=True):
            if ps:  # random positions along the route: the route's own percentiles (ties count half)
                own = (stats.rankdata(route) - 0.5) / len(route)
                null += rng.choice(own, (cfg.n_permutations, len(ps))).sum(axis=1)
        null /= len(flat)
        out["p_value"] = round(float((1 + np.sum(null >= obs)) / (1 + cfg.n_permutations)), 4)
    pooled = np.concatenate(routes) if routes else np.zeros(0)
    if len(pooled) >= 3:
        lo, hi = np.quantile(pooled, [1 / 3, 2 / 3])
        cvals = np.asarray(cross_vals)
        top_km = float(np.sum(pooled >= hi)) * cfg.sample_m / 1000
        bot_km = float(np.sum(pooled <= lo)) * cfg.sample_m / 1000
        out["crossings_per_10km"] = dict(
            top_third=round(10 * float(np.sum(cvals >= hi)) / top_km, 2) if top_km else None,
            bottom_third=round(10 * float(np.sum(cvals <= lo)) / bot_km, 2) if bot_km else None,
        )
    return out


# ---- cameras --------------------------------------------------------------------------------------------------


def base_rate(month: int, cfg: Truth = TRUTH) -> float:
    """Cougar detections per 100 camera-nights at random on-trail cameras in that month (Bassing et al. 2023)."""
    if month in cfg.summer_months:
        return cfg.base_summer_per_100
    if month in cfg.winter_months:
        return cfg.base_winter_per_100
    return (cfg.base_summer_per_100 + cfg.base_winter_per_100) / 2


@dataclass(frozen=True)
class CamStat:
    """One camera's effort and detections, as the camera test needs them."""

    id: str
    arm: str
    zone: str | None
    nights: float
    cougar: int
    deer: int = 0
    elk: int = 0
    base_expected: float = 0.0  # cougar detections expected at the random on-trail base rate

    @classmethod
    def of(cls, cam: Camera, cfg: Truth = TRUTH) -> CamStat:
        d = cam.dep
        base = sum(n * base_rate(m, cfg) / 100 for m, n in cam.nights_by_month.items())
        det = cam.detections
        return cls(d["id"], d["arm"], d["zone"], cam.nights, det["cougar"], det["deer"], det["elk"], base)


def _per_100(det: float, nights: float) -> float | None:
    return round(100 * det / nights, 3) if nights > 0 else None


def arm_rates(cams: list[CamStat], cfg: Truth = TRUTH) -> JSON:
    """Detections per 100 camera-nights by arm, against the base rate: vs_base > 1 is better than a random
    on-trail camera; p_above_base is the one-sided Poisson p-value of doing that well by chance. Both are None
    (and too_few_nights True) below cfg.min_nights camera-nights, as in the summary."""
    out: JSON = {}
    for arm in fieldlog.ARMS:
        cs = [c for c in cams if c.arm == arm]
        if not cs:
            continue
        nights = sum(c.nights for c in cs)
        det = sum(c.cougar for c in cs)
        base = sum(c.base_expected for c in cs)
        few = nights < cfg.min_nights
        out[arm] = dict(
            cameras=len(cs),
            camera_nights=round(nights, 1),
            cougar=det,
            cougar_per_100=_per_100(det, nights),
            base_per_100=_per_100(base, nights),
            vs_base=round(det / base, 2) if base > 0 and not few else None,
            p_above_base=round(float(stats.poisson.sf(det - 1, base)), 4) if base > 0 and not few else None,
            too_few_nights=few,
            deer_per_100=_per_100(sum(c.deer for c in cs), nights),
            elk_per_100=_per_100(sum(c.elk for c in cs), nights),
        )
    return out


def sign_flip_p(diffs: Floats, rng: np.random.Generator, n_perm: int) -> float:
    """One-sided p-value that the paired differences sum this high by chance (each zone's sign flipped at
    random; exact for up to 16 zones)."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n == 0:
        return 1.0
    if n <= 16:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        null = signs @ d
        return float(np.mean(null >= d.sum() - 1e-12))
    signs = rng.integers(0, 2, (n_perm, n)) * 2 - 1
    null = signs @ d
    return float((1 + np.sum(null >= d.sum() - 1e-12)) / (1 + n_perm))


def paired_test(
    cams: list[CamStat], treat: str, control: str, rng: np.random.Generator, cfg: Truth = TRUTH
) -> JSON | None:
    """Within-zone comparison of two arms: zones holding both, each zone's rate difference (per 100 nights),
    the pooled rate ratio (0.5 added to each arm's count when either has none) and the permutation p-value."""
    zones: dict[str, tuple[list[CamStat], list[CamStat]]] = {}
    for c in cams:
        if c.zone and c.arm in (treat, control):
            t, k = zones.setdefault(c.zone, ([], []))
            (t if c.arm == treat else k).append(c)
    rows = []
    for z, (t, k) in sorted(zones.items()):
        nt, nk = sum(c.nights for c in t), sum(c.nights for c in k)
        if nt > 0 and nk > 0:
            dt_, dk = sum(c.cougar for c in t), sum(c.cougar for c in k)
            rows.append((z, dt_, nt, dk, nk))
    if not rows:
        return None
    det_t, n_t = sum(r[1] for r in rows), sum(r[2] for r in rows)
    det_c, n_c = sum(r[3] for r in rows), sum(r[4] for r in rows)
    pad = 0.5 if min(det_t, det_c) == 0 else 0.0
    diffs = np.array([100 * (r[1] / r[2] - r[3] / r[4]) for r in rows])
    return dict(
        zones=len(rows),
        treatment=treat,
        control=control,
        treatment_per_100=_per_100(det_t, n_t),
        control_per_100=_per_100(det_c, n_c),
        rate_ratio=round(((det_t + pad) / n_t) / ((det_c + pad) / n_c), 2),
        zones_better=int(np.sum(diffs > 0)),
        zones_worse=int(np.sum(diffs < 0)),
        p_value=round(sign_flip_p(diffs, rng, cfg.n_permutations), 4),
    )


def camera_test(cams: list[CamStat], rng: np.random.Generator, cfg: Truth = TRUTH) -> JSON:
    """Per-arm rates against the base rate, and every within-zone comparison the data allows."""
    pairs = [paired_test(cams, t, c, rng, cfg) for t, c in TREATMENTS]
    return dict(by_arm=arm_rates(cams, cfg), paired=[p for p in pairs if p])


def camera_power(
    zones: int,
    rate_ratio: float,
    nights: float | None = None,
    base_per_100: float | None = None,
    cfg: Truth = TRUTH,
    sims: int = 1000,
    perms: int = 400,
    seed: int = 0,
) -> float:
    """Chance the paired permutation test finds a model-vs-control difference (p < cfg.alpha) with this many
    zones, when model cameras truly detect rate_ratio x as often. Both cameras in a zone run `nights`
    (default cfg.nights_per_camera); true rates vary between sites (gamma, CV cfg.site_cv)."""
    rng = np.random.default_rng(seed)
    nights = nights or cfg.nights_per_camera
    lam = (base_per_100 if base_per_100 is not None else cfg.base_summer_per_100) / 100 * nights
    k = 1 / cfg.site_cv**2
    c = rng.poisson(lam * rng.gamma(k, 1 / k, (sims, zones)))
    m = rng.poisson(rate_ratio * lam * rng.gamma(k, 1 / k, (sims, zones)))
    d = (m - c).astype(float)
    signs = rng.integers(0, 2, (perms, zones)) * 2 - 1
    null = d @ signs.T  # (sims, perms)
    obs = d.sum(axis=1)
    p = (1 + np.sum(null >= obs[:, None] - 1e-12, axis=1)) / (1 + perms)
    return float(np.mean((p < cfg.alpha) & (obs > 0)))


ZONE_STEPS = (4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60)


def zones_needed(rate_ratio: float, cfg: Truth = TRUTH, **kw: Any) -> int | None:
    """Fewest paired zones (of ZONE_STEPS) with camera_power >= cfg.power_target; None if 60 aren't enough."""
    return next((z for z in ZONE_STEPS if camera_power(z, rate_ratio, cfg=cfg, **kw) >= cfg.power_target), None)


# ---- an analyzed area ------------------------------------------------------------------------------------------


def _confident(level: str) -> bool:
    return fieldlog.CONFIDENCE.index(level) <= fieldlog.CONFIDENCE.index(TRUTH.min_confidence)


def _share_inside(g: Grid, mask: Mask, latlon: list[list[float]]) -> float:
    """Share of the points inside the area."""
    if not latlon:
        return 0.0
    return float(inside(mask, to_cells(g, latlon)).mean())


def _camera_section(st: ModelState, spots: AreaPins, records: list[Record], rng: np.random.Generator) -> JSON:
    g, m = st.fine, st.aoi_mask
    cams = [
        c
        for c in fieldlog.cameras(records, TRUTH.independent_min)
        if _share_inside(g, m, [[c.dep["lat"], c.dep["lon"]]])
    ]
    stats_ = [CamStat.of(c) for c in cams]
    rows = []
    for c, s in zip(cams, stats_, strict=True):
        r, col = (int(v) for v in np.floor(to_cells(g, [[c.dep["lat"], c.dep["lon"]]])[0]))
        rows.append(
            dict(
                id=s.id,
                arm=s.arm,
                zone=s.zone,
                camera_nights=round(s.nights, 1),
                cougar=s.cougar,
                deer=s.deer,
                elk=s.elk,
                last_check=c.last_check,
                model_rank=round(spots.spots.rank(r, col)[0], 3),
            )
        )
    out = camera_test(stats_, rng)
    paired_zones = max((p["zones"] for p in out["paired"]), default=0)
    month = st.month
    out.update(
        cameras=rows,
        sample_size=dict(
            paired_zones=paired_zones,
            zones_for_3x=zones_needed(3.0, base_per_100=base_rate(month)),
            zones_for_2x=zones_needed(2.0, base_per_100=base_rate(month)),
            note=f"zones (model + control camera, {TRUTH.nights_per_camera} nights each) for an 80% chance to "
            f"see a 3x / 2x gain over the control at this season's base rate",
        ),
    )
    return out


def _road_m_at(st: ModelState, latlon: list[float]) -> float | None:
    """Metres from a point to the nearest road open in the area's month (the access model's road_dist); None
    outside the grid."""
    r, c = (int(v) for v in np.floor(to_cells(st.fine, [latlon])[0]))
    d = st.layers["road_dist"]
    return float(d[r, c]) if 0 <= r < d.shape[0] and 0 <= c < d.shape[1] else None


def _track_section(st: ModelState, score: Floats, records: list[Record], rng: np.random.Generator) -> JSON:
    g, m = st.fine, st.aoi_mask
    rows: list[JSON] = []
    for r in records:
        if r["type"] != "track" or r["species"] != "cougar":
            continue
        pts = [p for ln in r["lines"] for p in ln]
        if _share_inside(g, m, pts) < 0.5:
            continue
        sure = _confident(r["confidence"])
        res = path_test(score, m, [to_cells(g, ln) for ln in r["lines"]], g.res, rng) if sure else None
        road_m = _road_m_at(st, r["lines"][0][0])  # tracks are followed backwards: the first point is the find
        rows.append(
            dict(
                id=r["id"],
                date=r["date"],
                confidence=r["confidence"],
                km=round(sum(fieldlog.line_length_m(ln) for ln in r["lines"]) / 1000, 2),
                start_road_m=round(road_m) if road_m is not None else None,
                found_near_road=road_m is not None and road_m <= TRUTH.near_road_m,
                percentile=round(res.percentile, 3) if res else None,
                mean_score=round(res.mean_score, 1) if res else None,
                shifted_mean_score=round(res.null_mean, 1) if res else None,
            )
        )
    pcts = [x["percentile"] for x in rows if x["percentile"] is not None]
    near = [x["percentile"] for x in rows if x["percentile"] is not None and x["found_near_road"]]
    away = [x["percentile"] for x in rows if x["percentile"] is not None and not x["found_near_road"]]
    return dict(
        tracks=rows,
        mean_percentile=round(float(np.mean(pcts)), 3) if pcts else None,
        sign_test=sign_test(pcts) if pcts else None,
        by_start=dict(
            near_road=sign_test(near) if near else None,
            away_from_road=sign_test(away) if away else None,
            note=f"tracks found within {TRUTH.near_road_m:g} m of an open road vs farther: the shifted copies "
            "don't keep their distance to roads, so a road-found track can beat them because of where people "
            "drive; the away-from-road tracks are the cleaner test",
        ),
        tracks_needed=dict(
            if_70pct_beat_shifted=tracks_needed(0.7),
            if_80pct_beat_shifted=tracks_needed(0.8),
            note="tracks for an 80% chance the sign test shows it, if that share of lion tracks walks higher-scoring "
            "ground than its shifted copies",
        ),
    )


def _transect_section(st: ModelState, score: Floats, records: list[Record], rng: np.random.Generator) -> JSON:
    g, m = st.fine, st.aoi_mask
    surveys = []
    routes: set[str] = set()
    for r in records:
        if r["type"] != "transect" or r["species"] != "cougar":
            continue
        if _share_inside(g, m, [p for ln in r["lines"] for p in ln]) < 0.5:
            continue
        surveys.append(Survey([to_cells(g, ln) for ln in r["lines"]], to_cells(g, r["crossings"])))
        routes.add(r["route"])
    out = transect_test(score, m, surveys, g.res, rng)
    out["routes"] = sorted(routes)
    return out


def _human_section(a: AreaPins) -> JSON | None:
    if not a.ranks:
        return None
    rnd = np.array([x[0] for x in a.random])
    return dict(
        picks=len(a.ranks),
        median_rank=round(float(np.median(a.ranks)), 3),
        random_median_rank=round(float(np.median(rnd)), 3),
        vs_random=round(vs_random(np.array(a.ranks), rnd), 3),
        by_pick={n: round(r, 3) for n, r in zip(a.names, a.ranks, strict=True)},
        note="rank = share of the area's candidate spots scoring at least as well as the pick (lower is better); "
        "vs_random = chance a pick outranks a random point (0.5 = chance). This checks the tool agrees with the "
        "person who chose the picks, not with lions.",
    )


def _summary(cam: JSON, tracks: JSON, transects: JSON, human: JSON | None) -> list[str]:
    lines = [
        f"{p['treatment']} vs {p['control']} cameras in {p['zones']} zone(s): rate ratio {p['rate_ratio']}, "
        f"better in {p['zones_better']}, worse in {p['zones_worse']} (p = {p['p_value']})"
        for p in cam["paired"]
    ]
    for arm, v in cam["by_arm"].items():
        if v["camera_nights"] < TRUTH.min_nights:
            lines.append(
                f"{arm} cameras: {v['cougar']} cougar detections in {v['camera_nights']:g} camera-nights "
                f"(too few nights to compare with the base rate; {TRUTH.min_nights:g}+ needed)"
            )
        elif v["vs_base"] is not None:
            lines.append(
                f"{arm} cameras: {v['cougar']} cougar detections in {v['camera_nights']:g} camera-nights, "
                f"{v['vs_base']}x a random on-trail camera (p = {v['p_above_base']})"
            )
    st = tracks["sign_test"]
    if st:
        lines.append(
            f"snow tracks: {st['above_half']} of {st['n']} walked higher-scoring ground than their shifted copies "
            f"(mean percentile {tracks['mean_percentile']}, p = {st['p_value']})"
        )
        near, away = tracks["by_start"]["near_road"], tracks["by_start"]["away_from_road"]
        if near:
            lines.append(
                f"  of those, found from a road: {near['above_half']} of {near['n']}; found away from roads: "
                + (f"{away['above_half']} of {away['n']} (p = {away['p_value']})" if away else "none yet")
                + " (road-found tracks are only partly corrected for where people go)"
            )
    if transects["surveys"]:
        lines.append(
            f"crossing transects: {transects['surveys']} survey(s), {transects['route_km']} km, "
            f"{transects['crossings']} crossing(s)"
            + (f", AUC {transects['auc']} (p = {transects['p_value']})" if transects["auc"] is not None else "")
        )
    if not lines:
        lines.append("no lion truth logged in this area yet: see docs/FIELD_PROTOCOL.md for what to record")
    if human:
        lines.append(
            f"human camera picks ({human['picks']}): median rank {human['median_rank']:.0%} of the area's "
            f"spots (random points {human['random_median_rank']:.0%}), vs random {human['vs_random']}"
        )
    return lines


def area_truth(
    st: ModelState,
    records: list[Record],
    pins: list[JSON],
    score_fn: Callable[[ModelState], Floats] = production_score,
    seed: int = 0,
) -> JSON:
    """Every lion-truth test the field log allows for one analyzed area, plus the human-pick check for the camera
    pins (pins: [{name, lat, lon}]) inside it."""
    rng = np.random.default_rng(seed)
    score = np.where(st.aoi_mask, score_fn(st), 0).astype("float32")
    a = AreaPins.rank(st, score, pins)
    cam = _camera_section(st, a, records, rng)
    tracks = _track_section(st, score, records, rng)
    transects = _transect_section(st, score, records, rng)
    human = _human_section(a)
    return dict(
        area=st.aoi.name,
        month=st.month,
        as_of=dt.date.today().isoformat(),
        summary=_summary(cam, tracks, transects, human),
        cameras=cam,
        snow_tracks=tracks,
        crossing_transects=transects,
        human_picks=human,
        how_to_read="cameras: detections per 100 camera-nights vs random on-trail cameras in NE Washington "
        "(Bassing 2023: summer 0.87, winter 0.37) and model vs control in the same zone; snow tracks: "
        "percentile vs the same track rotated and shifted 100-1,500 m (0.5 = chance); transects: AUC of the "
        "crossings vs the whole route (0.5 = chance). p-values are one-sided; small samples say little.",
    )
