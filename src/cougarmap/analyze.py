"""Run the whole model on an area: factors -> camera-spot score -> ranked camera spots with reasons."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
from skimage.feature import peak_local_max

from . import context, factors
from . import terrain as T
from .aoi import AOI
from .arrays import Floats, Ints, Mask
from .config import MILE_M, OUT_DIR, PLACEMENT, WINTER, WORN, Options, Weights
from .context import Log
from .sources import weather
from .sources.weather import compass
from .state import Layers, ModelState, SaddlePoint, to_saved_precision

FACTORS = ("wind", "edges", "pinch", "water")


class TrailAlternate(TypedDict):
    """The best cell beside a quiet road or trail near a spot: an extra suggestion, never the default pick."""

    lat: float
    lon: float
    score: float
    distance_m: int
    direction: str  # compass direction from the spot
    kind: str  # factors.TRAIL_KINDS
    open_to_vehicles: bool  # on a road open to vehicles that month (more traffic, theft)
    walk_miles: float | None
    reason: str


class WornTrail(TypedDict):
    """The nearest unmapped worn line (1 m lidar) near a spot: where to face the camera, offered beside the spot."""

    lat: float  # the nearest point of the line
    lon: float
    distance_m: int
    direction: str  # compass direction from the spot
    reason: str


class Spot(TypedDict):
    """One camera spot (or any explained cell): where it is, its score and factors, and why."""

    lat: float
    lon: float
    elevation_m: float
    score: float  # 0-100 after the land/access rules and site penalties
    raw_score: float  # 0-100 before them
    factors: dict[str, float]
    factors_on: int
    reasons: list[str]
    walk_miles: float | None  # None: not reachable within the walk limit
    walk_minutes: int | None
    road_distance_m: int
    paved_road_distance_m: int | None  # None: the state's model version did not map pavement
    land: str
    public: bool
    landform: str
    canopy_m: float
    slope_deg: float
    trail_alternate: TrailAlternate | None  # None: none within reach, or the spot already watches one
    worn_trail: WornTrail | None  # None: no unmapped worn line within WORN.hint_m (or the layer is off)
    row: int
    col: int
    rank: NotRequired[int]
    name: NotRequired[str]
    zone: NotRequired[int]


class Result(TypedDict):
    summary: dict[str, Any]
    candidates: list[Spot]
    private_candidates: list[Spot]
    state: ModelState


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "area"


def ramp(a: Floats, lo: float, hi: float) -> Floats:
    """0 below lo, 1 above hi, linear between."""
    out: Floats = np.clip((a - lo) / (hi - lo), 0, 1)
    return out


def combine(A: Layers, w: Weights, res: float) -> tuple[Floats, Ints]:
    """The camera-spot score (0-100) and the number of factors "on" at each cell.

    score = scale x zone( spot x habitat ):
    - spot: the four factors at the spot, weighted (wind first) plus the terrain travel line, times a smooth
      stacking bonus per factor ("more factors at one spot is better", with no jump at any threshold). The
      travel line is added in but not stacked: it is where a lion walks, not a reason to stop.
    - habitat: how much hunting edge and water the surrounding ~500 m has. A lion spends its time in a
      timber/opening mosaic with water around; the same spot in solid timber or dry flats sees less of it.
      In winter, times the season layer (low, sun-facing ground with shallow snow, where deer winter);
      `context` keeps the habitat without it.
    - zone: averaged over the camera's ~20 m view (which is also the pin and GPS error), so a patch of good
      ground beats one lucky cell.
    Also stores edge_density, water_density and context (the habitat multiplier) for reasons and KMZ layers."""
    h = w.habitat
    F = [A["wind"], A["edges"], A["pinch"], A["water"]]
    spot = w.wind * F[0] + w.edges * F[1] + w.pinch * F[2] + w.water * F[3] + w.travel * A["travel"]
    for f in F:
        spot = spot * (1 + (w.stack_multiplier - 1) * ramp(f, w.stack_from, w.stack_to))
    A["edge_density"] = T.blur(A["edge_meadow"], h.context_m, res)
    A["water_density"] = T.blur(A["water"], h.context_m, res)
    habitat = (h.edge_floor + np.clip(A["edge_density"] / h.edge_sat, 0, 1)) * (
        h.water_floor + np.clip(A["water_density"] / h.water_sat, 0, 1)
    )
    A["context"] = habitat.astype("float32")
    s = (spot * habitat * A["season"]).astype("float32")  # the winter module: 1 outside Nov-Apr
    if h.zone_m > 0:
        s = T.gaussian(s, h.zone_m / res)
    top = (w.wind + w.edges + w.pinch + w.water + w.travel) * w.stack_multiplier**4 * (1 + h.edge_floor)
    top *= 1 + h.water_floor
    score = np.clip(100 * h.score_scale * s / top, 0, 100).astype("float32")
    n_on = sum((f >= w.stack_threshold).astype("uint8") for f in F)
    return score, np.asarray(n_on, dtype="uint8")


def traffic_penalty(paved_dist: Floats, opts: Options) -> Floats:
    """Score multiplier for nearness to a paved road: 1 - paved_penalty right beside it, back to 1 at paved_reach_m."""
    near = ramp(paved_dist, opts.paved_reach_m, opts.paved_full_m)  # 1 at the road, 0 beyond reach
    return (1 - opts.paved_penalty * near).astype("float32")


def site_penalty(A: Layers, opts: Options) -> Floats:
    """Score multiplier for people around a spot: paved-road traffic, houses (a little for each, a lot in
    populated areas: people, dogs, theft) and recreation sites (trailheads, campgrounds, parking)."""
    pen = np.ones(A["score"].shape, "float32")
    pen *= traffic_penalty(A["paved_dist"], opts)
    pen *= 1 - opts.houses_penalty * ramp(A["houses"], opts.houses_from, opts.houses_full)
    if opts.houses_exponent:
        pen *= np.power(1 + np.clip(A["houses"], 0, None), -opts.houses_exponent).astype("float32")
    pen *= 1 - opts.rec_penalty * ramp(A["rec_dist"], opts.rec_reach_m, opts.rec_full_m)
    return pen


def run(aoi: AOI, opts: Options | None = None, log: Log = print, out_dir: Path | None = None) -> Result:
    """Analyze an area: download, compute every factor, score, pick spots; write the outputs to out_dir
    (default OUT_DIR/<area slug>)."""
    opts = opts or Options()
    t0 = time.perf_counter()
    ctx = context.build(aoi, opts, log=log)
    for compute in (
        factors.compute_terrain,
        factors.compute_land,
        factors.compute_wind,
        factors.compute_edges,
        factors.compute_pinch,
        factors.compute_travel,  # after compute_pinch: spine crossings need the saddles
        factors.compute_water,
        factors.compute_season,
        factors.compute_access,
        factors.compute_traffic,
        factors.compute_trails,
        factors.compute_worn_trails,  # after compute_terrain (the fine DEM's channels)
        factors.compute_houses,
        factors.compute_recreation,
    ):
        compute(ctx, log)
    A = ctx.layers
    A["score"], A["n_on"] = combine(A, opts.weights, ctx.fine.res)
    to_saved_precision(ctx)  # pick and describe from the values a re-pick will load, so both report the same
    apply_masks(ctx)

    cands, priv = pick_all(ctx)
    summary = summarize(ctx, cands, 0.0, priv)
    result = Result(summary=summary, candidates=cands, private_candidates=priv, state=ctx)
    from .export import write_outputs, write_summary

    out = out_dir or (OUT_DIR / slug(aoi.name))
    paths = write_outputs(result, out, log=log)
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)  # the whole run, writing the outputs included
    write_summary(result, out)
    summary["outputs"] = {k: str(v) for k, v in paths.items()}
    log(f"done in {summary['runtime_s']:.0f}s: {len(cands)} camera spots")
    return result


def apply_masks(st: ModelState) -> None:
    """Usable ground from the land/access rules. usable/final: the main spot list, open-access public land within
    the walk limit along public-access routes. usable_private/final_private: private ground, reached by any route
    (a separate list and hidden KMZ layers)."""
    A, o = st.layers, st.opts
    limit = o.max_walk_miles * MILE_M
    base = st.aoi_mask & ~A["cliff"] & (A["road_dist"] > 8) & ~A["lake"]
    pen = site_penalty(A, o)
    usable = base & A["public"] & (A["walk_m"] <= limit)
    priv = base & ~A["public"] & (A["walk_any_m"] <= limit)
    A["usable"], A["usable_private"] = usable, priv
    A["final"] = np.where(usable, A["score"] * pen, 0).astype("float32")
    A["final_private"] = np.where(priv, A["score"] * pen, 0).astype("float32")


# ---- candidates -----------------------------------------------------------------------------------


def pick_all(st: ModelState) -> tuple[list[Spot], list[Spot]]:
    """The main (public-land) spot list, and the private-land spots (a separate, hidden-by-default layer)."""
    A = st.layers
    cands = pick_candidates(st, A["final"], st.opts, A["usable"])
    priv = pick_candidates(st, A["final_private"], st.opts, A["usable_private"], prefix="P")
    return cands, priv


def spread(xy: list[tuple[float, float]], per_zone: int, zone_radius_m: float, n: int) -> list[int]:
    """Indices of the points to keep, best first (xy is sorted best first): at most per_zone within
    zone_radius_m of each other, n in all."""
    picked: list[int] = []
    for i, (x, y) in enumerate(xy):
        near = sum(math.hypot(x - xy[j][0], y - xy[j][1]) < zone_radius_m for j in picked)
        if near >= per_zone:
            continue
        picked.append(i)
        if len(picked) >= n:
            break
    return picked


def zones(xy: list[tuple[float, float]], zone_radius_m: float) -> list[int]:
    """Hotspot zone number (1, 2, ... in order of first appearance) of each point: a point joins the zone of the
    first earlier point within zone_radius_m."""
    zone = list(range(len(xy)))
    for i in range(len(xy)):
        for j in range(i):
            if math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) < zone_radius_m:
                zone[i] = zone[j]
                break
    zmap: dict[int, int] = {}
    return [zmap.setdefault(z, len(zmap) + 1) for z in zone]


# Peaks below PEAK_MIN are not offered as spots, unless the whole area scores low (dry ground, solid timber):
# then the cutoff drops to PEAK_REL of the area's best peak (never below PEAK_FLOOR), so the best of weak ground
# is still returned, with a note (WEAK_AREA_SCORE) saying so.
PEAK_MIN = 5.0
PEAK_REL = 0.2
PEAK_FLOOR = 0.5
WEAK_AREA_SCORE = 20.0


def peak_threshold(best: float) -> float:
    """The smallest smoothed score offered as a spot, given the best smoothed score of the usable ground."""
    return min(PEAK_MIN, max(PEAK_FLOOR, PEAK_REL * best))


def pick_candidates(st: ModelState, final: Floats, opts: Options, usable: Mask, prefix: str = "#") -> list[Spot]:
    """Up to opts.n_candidates spots: smoothed score peaks at least candidate_spacing_m apart (and above
    peak_threshold), snapped to the best usable cell within ~10 m, spread over hotspot zones (per_zone within
    zone_radius_m), ranked and named."""
    r = st.fine.res
    if not usable.any():
        return []
    sm = T.gaussian(final, max(0.5, 6 / r))
    pk = peak_local_max(  # type: ignore[no-untyped-call]  # scikit-image is untyped
        sm,
        min_distance=max(1, int(opts.candidate_spacing_m / r)),
        threshold_abs=peak_threshold(float(sm[usable].max())),
        num_peaks=opts.n_candidates * 8,
        exclude_border=False,
        labels=usable.astype(int),
    )
    cands = []
    k = max(1, int(10 / r))
    for pr, pc in pk:
        # snap to the best usable cell within ~10 m
        r0, c0 = max(pr - k, 0), max(pc - k, 0)
        win = final[r0 : pr + k + 1, c0 : pc + k + 1]
        dr, dc = np.unravel_index(np.argmax(win), win.shape)
        cands.append(describe(st, int(r0 + dr), int(c0 + dc), final))
    cands.sort(key=lambda c: -c["score"])
    xy = [(float(x), float(y)) for x, y in (st.fine.xy(c["row"], c["col"]) for c in cands)]
    keep = spread(xy, opts.per_zone, opts.zone_radius_m, opts.n_candidates)
    picked = [cands[i] for i in keep]
    for i, (c, z) in enumerate(zip(picked, zones([xy[i] for i in keep], opts.zone_radius_m), strict=True), 1):
        c["rank"], c["name"], c["zone"] = i, f"{prefix}{i}", z
    return picked


# ---- explaining a cell ------------------------------------------------------------------------------


class Cell(NamedTuple):
    """Where one fine cell is, on both grids."""

    row: int
    col: int
    mr: int  # mid-grid row/col (clipped to the grid)
    mc: int
    x: float
    y: float


def _cell(st: ModelState, row: int, col: int) -> Cell:
    x, y = st.fine.xy(row, col)
    mr, mc = st.mid.rowcol(x, y)
    mr, mc = int(np.clip(mr, 0, st.mid.height - 1)), int(np.clip(mc, 0, st.mid.width - 1))
    return Cell(row, col, mr, mc, float(x), float(y))


def describe(st: ModelState, row: int, col: int, final: Floats | None = None) -> Spot:
    """Everything known about one cell, with plain-English reasons (final: the score layer it is ranked by,
    default the main one)."""
    A = st.layers
    c = _cell(st, row, col)
    lon, lat = st.fine.to_lonlat(c.x, c.y)
    v = dict(
        wind=float(A["wind"][row, col]),
        edges=float(A["edges"][row, col]),
        pinch=float(A["pinch"][row, col]),
        water=float(A["water"][row, col]),
    )
    if "travel" not in st.unmodeled:
        v["travel"] = float(A["travel"][row, col])
    if "context" not in st.unmodeled:  # the habitat multiplier, shown 0-1
        h = st.opts.weights.habitat
        v["habitat"] = float(A["context"][row, col]) / ((1 + h.edge_floor) * (1 + h.water_floor))
    if float(A["season"][row, col]) != 1:  # the winter multiplier, in winter months
        v["season"] = float(A["season"][row, col])
    walk_m, walk_s = (
        (A["walk_any_m"][row, col], A["walk_any_s"][row, col])
        if st.any_route(row, col)
        else (A["walk_m"][row, col], A["walk_s"][row, col])
    )
    final = A["final"] if final is None else final
    paved = None if "paved_dist" in st.unmodeled else round(float(A["paved_dist"][row, col]))
    return Spot(
        lat=round(lat, 6),
        lon=round(lon, 6),
        elevation_m=round(float(st.z[row, col]), 1),
        score=round(float(final[row, col]), 1),
        raw_score=round(float(A["score"][row, col]), 1),
        factors={k: round(val, 2) for k, val in v.items()},
        factors_on=int(A["n_on"][row, col]),
        reasons=reasons(st, c),
        walk_miles=round(float(walk_m) / MILE_M, 2) if walk_m < 1e6 else None,
        walk_minutes=round(float(walk_s) / 60) if walk_s < 1e6 else None,
        road_distance_m=round(float(A["road_dist"][row, col])),
        paved_road_distance_m=paved,
        land=A["land_names"][int(A["land_id"][row, col])],
        public=bool(A["public"][row, col]),
        landform=T.LANDFORM_NAMES.get(int(A["landform_mid"][c.mr, c.mc]), "?"),
        canopy_m=round(float(st.chm[row, col]), 1),
        slope_deg=round(float(A["slope"][row, col]), 1),
        trail_alternate=trail_alternate(st, c, final),
        worn_trail=worn_hint(st, c),
        row=row,
        col=col,
    )


def _walk_miles(st: ModelState, row: int, col: int) -> float | None:
    A = st.layers
    walk_m = A["walk_any_m"][row, col] if st.any_route(row, col) else A["walk_m"][row, col]
    return round(float(walk_m) / MILE_M, 2) if walk_m < 1e6 else None


def trail_alternate(st: ModelState, c: Cell, final: Floats) -> TrailAlternate | None:
    """The best-scoring usable cell beside a quiet road or trail (factors.compute_trails) within
    PLACEMENT.search_m of the cell and away from trailheads, scoring at least PLACEMENT.min_score and
    PLACEMENT.min_score_frac of the cell. Cells off the roads open to vehicles that month come first; one on an
    open road only when no quiet one qualifies (and the reason says so). None when the cell already watches one,
    or there is none."""
    A, p, r = st.layers, PLACEMENT, st.fine.res
    if "trail_kind" in st.unmodeled or A["trail_kind"][c.row, c.col]:
        return None
    k = int(p.search_m / r)
    r0, c0 = max(c.row - k, 0), max(c.col - k, 0)
    win = (slice(r0, c.row + k + 1), slice(c0, c.col + k + 1))
    rr, cc = np.ogrid[win[0], win[1]]
    near = np.hypot((rr - c.row) * r, (cc - c.col) * r) <= p.search_m
    floor = max(p.min_score, p.min_score_frac * float(final[c.row, c.col]))
    ok = near & (A["trail_kind"][win] > 0) & (final[win] >= floor) & (final[win] > 0)
    ok &= A["rec_dist"][win] >= p.min_rec_m
    on_open = A["road_dist"][win] <= p.open_road_m
    pick = ok & ~on_open if (ok & ~on_open).any() else ok
    if not pick.any():
        return None
    sc = np.where(pick, final[win], -1.0)
    dr, dc = np.unravel_index(int(np.argmax(sc)), sc.shape)
    br, bc = r0 + int(dr), c0 + int(dc)
    best = float(final[br, bc])
    is_open = bool(on_open[dr, dc])
    x, y = st.fine.xy(br, bc)
    lon, lat = st.fine.to_lonlat(x, y)
    d = math.hypot(x - c.x, y - c.y)
    direction = compass(math.degrees(math.atan2(x - c.x, y - c.y)))
    kind = factors.TRAIL_KINDS[int(A["trail_kind"][br, bc]) - 1]
    line = f"{kind}, on or right beside a road open to vehicles this month," if is_open else kind
    why = (
        "cameras on dirt roads and trails catch more of the lions passing, but this one is on the drivable "
        "network: more traffic and theft risk"
        if is_open
        else "cameras on quiet dirt roads and trails catch more of the lions passing"
    )
    return TrailAlternate(
        lat=round(float(lat), 6),
        lon=round(float(lon), 6),
        score=round(best, 1),
        distance_m=round(d),
        direction=direction,
        kind=kind,
        open_to_vehicles=is_open,
        walk_miles=_walk_miles(st, br, bc),
        reason=(
            f"alternate spot on the {line} through this zone, {d:.0f} m {direction} (score {best:.0f}): {why}; "
            "the spot itself stays the pick"
            if d > 2 * p.on_m
            else f"the {line} passes {d:.0f} m {direction} of this spot: a camera can face it from here ({why})"
        ),
    )


def worn_hint(st: ModelState, c: Cell) -> WornTrail | None:
    """The nearest unmapped worn line (factors.compute_worn_trails) within WORN.hint_m of the cell, as a hint on
    where to face the camera. It changes no score and never moves the spot."""
    mask = st.layers["worn_unmapped"]
    reach = int(WORN.hint_m / st.fine.res) + 1
    rc = _nearest(mask, c.row, c.col, reach)
    if rc is None:
        return None
    x, y = st.fine.xy(*rc)
    d = math.hypot(float(x) - c.x, float(y) - c.y)
    if d > WORN.hint_m:
        return None
    lon, lat = st.fine.to_lonlat(x, y)
    direction = compass(math.degrees(math.atan2(float(x) - c.x, float(y) - c.y)))
    what = "a worn line on no map (1 m lidar: a game trail or old track; check it on the ground)"
    if d <= st.fine.res:
        how = f"{what} runs through this spot: hang the camera beside it, facing along it"
    elif d <= WORN.facing_m:
        how = f"{what} {d:.0f} m {direction}: hang the camera facing it"
    else:
        how = f"{what} {d:.0f} m {direction}: hang the camera a few metres toward it, facing it"
    return WornTrail(
        lat=round(float(lat), 6), lon=round(float(lon), 6), distance_m=round(d), direction=direction, reason=how
    )


def reasons(st: ModelState, c: Cell) -> list[str]:
    """The plain-English reasons a cell is (or is not) a good camera spot, factor by factor."""
    return [
        *_wind_reasons(st, c),
        *_edge_reasons(st, c),
        *_habitat_reasons(st.layers, c),
        *_season_reasons(st, c),
        *_pinch_reasons(st, c),
        *_water_reasons(st.layers, c),
        *_people_reasons(st, c),
    ]


# Reason thresholds: each fires on roughly the top quarter of an area (checked on the validation areas: hunting
# edge around 19-54% of the area (the most mosaic area the most), water 20-33%, travel line 21-28%).
EDGE_DENSITY_REASON = 0.10
WATER_DENSITY_REASON = 0.2
TRAVEL_REASON = 0.6
# The per-factor reason thresholds: a part of a factor "counts" at these strengths.
CONVERGENCE_REASON = 0.25
DRAINAGE_REASON = 0.5
WINDWARD_REASON = 0.3
ALIGNED_DEG = 45.0  # drainage and wind directions within this agree
MEADOW_REASON = 0.3
DOWNWIND_REASON = 0.3
DOWNWIND_CURRENT_REASON = 0.7  # an opening's end counts as downwind for one air current from this position
SPINE_GATE_REASON = 0.5  # a ridge spine counts as a crossing (saddle, junction, winter thermals) from this gate
VALLEY_REASON = 0.2
RIDGE_REASON = 0.15
WATER_EDGE_REASON = 0.25
ROUTE_REASON = 0.25
WATER_REASON = 0.15
SCARCE_REASON = 0.85
WINTER_REASON = 0.6  # the winter module's W (0-1) from which a spot is "where deer winter"


def wind_name(st: ModelState, current: Literal["dawn", "day"]) -> str:
    """Which wind a reason means, named so two reasons never seem to disagree: convergence and the windward side
    use the dawn/dusk high-pressure wind, the downwind ends of openings the daytime one. The user's own wind
    (wind_from_deg) stands in for both."""
    if st.opts.wind_from_deg is not None:
        return f"the {compass(st.opts.wind_from_deg % 360)} wind"
    if current == "dawn":
        return f"the dawn/dusk {st.wind['from_compass']} high-pressure wind"
    return f"the daytime {compass(factors.daytime_wind_deg(st.wind, None))} high-pressure wind"


def _unsteady(st: ModelState) -> str:
    """' (an unsteady wind here: consistency R 0.26)' when the modeled dawn/dusk wind is unsteady, else ''."""
    R = st.wind["consistency"]
    if st.opts.wind_from_deg is not None or R >= weather.UNSTEADY_R:
        return ""
    return f" (an unsteady wind here: consistency R {R:.2f})"


def _wind_reasons(st: ModelState, c: Cell) -> list[str]:
    A, wind = st.layers, st.wind
    out = []
    name = wind_name(st, "dawn")
    shaky = _unsteady(st)
    dx, dn = A["drainx_mid"][c.mr, c.mc], A["drainn_mid"][c.mr, c.mc]
    drains_to = compass(math.degrees(math.atan2(dx, dn)))
    wind_to = compass(wind["from_deg"] + 180)
    if A["conv_mid"][c.mr, c.mc] > CONVERGENCE_REASON:
        th = math.radians(wind["from_deg"])
        raw_ang = math.degrees(math.acos(np.clip(dx * -math.sin(th) + dn * -math.cos(th), -1, 1)))
        if raw_ang <= ALIGNED_DEG:
            out.append(
                f"cold air drains {drains_to} here and {name} pushes {wind_to} "
                f"- they line up within {raw_ang:.0f} deg, so air flow is consistent most of the day{shaky}"
            )
        else:
            local = compass(math.degrees(math.atan2(A["windx_mid"][c.mr, c.mc], A["windn_mid"][c.mr, c.mc])))
            out.append(
                f"{name} gets channeled along this valley (toward {local}), the same "
                f"way cold air drains ({drains_to}) - consistent air flow most of the day{shaky}"
            )
    elif A["drain_mid"][c.mr, c.mc] > DRAINAGE_REASON:
        out.append(f"strong cold-air drainage heading {drains_to} at dawn/dusk")
    if A["windward_mid"][c.mr, c.mc] > WINDWARD_REASON:
        out.append(f"windward side of a ridge in {name}{shaky}")
    return out


def _meadow_size(A: Layers, row: int, col: int) -> str:
    """'3.2 ha ' for the opening nearest the cell (within 40 cells), '' when there is none."""
    rc = _nearest(A["meadow"], row, col)
    ha = float(A["meadow_ha"][A["meadow_label"][rc]]) if rc is not None else None
    return f"{ha:.1f} ha " if ha else ""


def _downwind_currents(st: ModelState, c: Cell) -> str:
    """'the evening cold-air drainage (toward S) and the daytime WSW wind': the air currents this cell is the
    downwind end of an opening for (each above DOWNWIND_CURRENT_REASON, or else the stronger one)."""
    A = st.layers
    qd, qw = float(A["edge_q_drain"][c.row, c.col]), float(A["edge_q_wind"][c.row, c.col])
    drains_to = compass(math.degrees(math.atan2(A["drainx_mid"][c.mr, c.mc], A["drainn_mid"][c.mr, c.mc])))
    names = []
    if qd > DOWNWIND_CURRENT_REASON or qd >= qw:
        names.append(f"the evening cold-air drainage (toward {drains_to})")
    if qw > DOWNWIND_CURRENT_REASON or qw > qd:
        names.append(wind_name(st, "day"))
    return " and ".join(names)


def _edge_reasons(st: ModelState, c: Cell) -> list[str]:
    A = st.layers
    row, col = c.row, c.col
    out = []
    if A["edge_meadow"][row, col] > MEADOW_REASON:
        size = _meadow_size(A, row, col)
        downwind = A["edge_downwind"][row, col] > DOWNWIND_REASON
        if "edge_q_drain" in st.unmodeled:  # older states: one blended dawn/dusk flow, timber side only
            if downwind:
                air_to = compass(math.degrees(math.atan2(A["flowx"][row, col], A["flown"][row, col])))
                out.append(
                    f"in timber on the most downwind edge of a {size}opening - can watch prey in it without "
                    f"being seen or smelled (air leaves the opening toward {air_to})"
                )
            else:
                out.append(f"in timber cover overlooking a {size}opening - can watch prey without being seen")
        else:
            end = f", at its most downwind end for {_downwind_currents(st, c)}" if downwind else ""
            if A["meadow"][row, col]:
                out.append(f"at the timber edge, in the open of a {size}opening{end} - where kills cluster")
            elif downwind:
                out.append(
                    f"in timber overlooking a {size}opening{end} - can watch prey in it without being seen or smelled"
                )
            else:
                out.append(f"in timber cover overlooking a {size}opening - can watch prey without being seen")
    # the terrain travel line supersedes the older valley-bottom / ridgeline tie-breakers in the reasons
    if A["travel"][row, col] >= TRAVEL_REASON:
        out.append(_travel_reason(st, c))
    else:
        if A["edge_valley"][row, col] > VALLEY_REASON:
            out.append("valley bottom")
        if A["edge_ridge"][row, col] > RIDGE_REASON:
            out.append("ridgeline")
    if A["edge_water"][row, col] > WATER_EDGE_REASON:
        out.append("water edge")
    if A["closed_track_mid"][c.mr, c.mc]:
        out.append("on a closed/gated forest road: quiet travel route")
    elif A["edge_route"][row, col] > ROUTE_REASON:
        out.append("on a trail/two-track through timber (travel route)")
    return out


def _travel_reason(st: ModelState, c: Cell) -> str:
    A = st.layers
    line = "a natural travel line lions follow through this country"
    if A["travel_pos_mid"][c.mr, c.mc] < 0:
        return f"drainage bottom - {line}"
    gate, thermal = float(A["travel_gate_mid"][c.mr, c.mc]), float(A["travel_thermal_mid"][c.mr, c.mc])
    if "travel_gate_mid" not in st.unmodeled:
        if thermal > SPINE_GATE_REASON and thermal >= gate:
            return "ridgeline above large south/southeast slopes - winter thermals"
        if gate > SPINE_GATE_REASON:
            return "ridge spine at a saddle/junction - lions cross here"
    return f"ridge spine - {line}"


def _habitat_reasons(A: Layers, c: Cell) -> list[str]:
    out = []
    if A["edge_density"][c.row, c.col] >= EDGE_DENSITY_REASON:
        out.append(
            "in a patchwork of timber and openings - lots of hunting edge within ~500 m (deer and stalking cover)"
        )
    if A["water_density"][c.row, c.col] >= WATER_DENSITY_REASON:
        out.append("water around (streams/springs within a few hundred m)")
    return out


def _season_reasons(st: ModelState, c: Cell) -> list[str]:
    A = st.layers
    out = []
    if st.month in WINTER.months and A["winter_mid"][c.mr, c.mc] >= WINTER_REASON:
        out.append("winter: low, sun-facing ground with shallow snow, where deer winter")
    if A["winter_range_mid"][c.mr, c.mc]:
        out.append("inside WDFW-mapped deer/elk winter range")
    return out


def _pinch_reasons(st: ModelState, c: Cell) -> list[str]:
    A = st.layers
    row, col = c.row, c.col
    out = []
    if A["pinch_saddle"][row, col] > 0.2:
        s = _nearest_saddle(A["saddle_points"], c.x, c.y)
        if s:
            out.append(f"saddle - crossing here saves about {s[0]['rise_m']:.0f} m of climbing ({s[1]:.0f} m away)")
    if A["pinch_cliffbase"][row, col] > 0.2:
        out.append("base of a cliff - animals moving along cover are funneled here")
    if A["pinch_clifftop"][row, col] > 0.15:
        out.append("cliff rim")
    if A["pinch_bank"][row, col] > 0.2:
        out.append("bank of a lake/stream that animals travel along")
    if A["pinch_funnel"][row, col] > 0.35:
        out.append("natural travel funnel (movement concentrates here)")
    if A["pinch_fence"][row, col] > 0.15:
        out.append("along a fence/rail line")
    return out


def _water_reasons(A: Layers, c: Cell) -> list[str]:
    if A["water"][c.row, c.col] <= WATER_REASON:
        return []
    kind = int(A["water_kind"][c.row, c.col])
    label = A["water_labels"][kind] if kind >= 0 else "water"
    scarce = float(A["water_scarcity"][c.row, c.col]) > SCARCE_REASON
    return [f"near {label}" + (" - few other water sources within a mile" if scarce else "")]


def _people_reasons(st: ModelState, c: Cell) -> list[str]:
    A, o = st.layers, st.opts
    out = []
    paved = float(A["paved_dist"][c.row, c.col])
    if paved < o.paved_reach_m and "paved_dist" not in st.unmodeled:
        out.append(f"{paved:.0f} m from a paved road: traffic and people (score reduced)")
    rec = float(A["rec_dist"][c.row, c.col])
    if rec < o.rec_reach_m and "rec_dist" not in st.unmodeled:
        out.append(f"{rec:.0f} m from a trailhead/campground/parking area: people (score reduced)")
    n = int(A["houses"][c.row, c.col])
    e = o.houses_exponent
    if n > o.houses_from:
        out.append(
            f"populated area: {n} houses within {o.houses_radius_m:.0f} m - people, dogs, camera theft (score reduced)"
        )
    elif n >= 1 and e:
        out.append(
            f"{n} house{'s' if n > 1 else ''} within {o.houses_radius_m:.0f} m: people, dogs, camera theft "
            f"(score x{(1 + n) ** -e:.2f})"
        )
    return out


def _nearest(mask: Mask, row: int, col: int, reach: int = 40) -> tuple[int, int] | None:
    """The nearest True cell of mask within `reach` cells (a window around the cell), or None."""
    r0, c0 = max(row - reach, 0), max(col - reach, 0)
    sub = mask[r0 : row + reach + 1, c0 : col + reach + 1]
    if not sub.any():
        return None
    _, (ir, ic) = T.edt_nearest(sub)
    lr, lc = row - r0, col - c0
    return int(r0 + ir[lr, lc]), int(c0 + ic[lr, lc])


def _nearest_saddle(saddles: list[SaddlePoint], x: float, y: float) -> tuple[SaddlePoint, float] | None:
    """The nearest saddle point and its distance (m), if within 120 m."""
    best: tuple[SaddlePoint, float] | None = None
    for s in saddles:
        d = math.hypot(s["x"] - x, s["y"] - y)
        if best is None or d < best[1]:
            best = (s, d)
    return best if best and best[1] < 120 else None


def _zones(cands: list[Spot]) -> list[dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for c in cands:
        z = out.setdefault(
            c.get("zone", 0),
            dict(
                zone=c.get("zone", 0),
                best_rank=c["rank"],
                best_score=c["score"],
                spots=0,
                lat=c["lat"],
                lon=c["lon"],
                land=c["land"],
            ),
        )
        z["spots"] += 1
    return list(out.values())


# ---- summary --------------------------------------------------------------------------------------


def pick_notes(cands: list[Spot], usable: Mask) -> list[str]:
    """Notes on a spot list an agent should pass on: no spots on usable ground, or only weak ones."""
    if not usable.any():
        return []
    if not cands:
        return ["No camera spots: the usable ground scores near zero everywhere."]
    best = cands[0]["score"]
    if best >= WEAK_AREA_SCORE:
        return []
    return [
        f"Every spot here scores low (best {best:.0f} of 100): little mapped water and few timber/opening edges "
        "around, so these are the best of weak ground. Small water is often unmapped: pins for springs, ponds "
        "or guzzlers (import_kml) count as water."
    ]


def worn_summary(st: ModelState, cands: list[Spot]) -> dict[str, Any]:
    """The worn-trail layer in the summary: whether it was computed, and how much it found."""
    lines = st.layers["worn_lines"]
    if not st.opts.worn_trails:
        return dict(
            on=False, how="off in a fast run (fast=True, CLI --fast); a normal run adds worn trails from 1 m lidar"
        )
    km = sum(ln["length_m"] for ln in lines) / 1000
    return dict(
        on=True,
        km=round(km, 1),
        unmapped_km=round(sum(ln["length_m"] for ln in lines if not ln["mapped"]) / 1000, 1),
        spots_with_hint=sum(c["worn_trail"] is not None for c in cands),
        layer="hidden in the KMZ: 'Worn trails (lidar)' folder",
    )


def summarize(st: ModelState, cands: list[Spot], runtime: float, private_cands: list[Spot]) -> dict[str, Any]:
    A = st.layers
    w = st.wind
    inside = st.aoi_mask
    usable = A["usable"]
    any_inside = bool(inside.any())
    limit = st.opts.max_walk_miles * MILE_M
    return dict(
        area=st.aoi.name,
        area_km2=round(st.aoi.area_km2(), 1),
        month=st.month,
        resolution_m=st.fine.res,
        lidar_fraction=st.dem_info["lidar_fraction"],
        wind=dict(
            **weather.describe(w),
            all_days_from=w.get("all_days_from_compass"),
            daytime_from=w.get("day_from_compass"),
            how_used="two winds, both from high-pressure days: prevailing_from (dawn/dusk) for where it lines up "
            "with cold-air drainage and for the windward side of ridges; daytime_from for the downwind ends of "
            "openings. Each reason names which one it means. A wind override stands in for both. consistency is "
            "how steady prevailing_from is (R 0-1); ground_level (10 m) is shown for reference, not scored.",
        ),
        options=dict(
            max_walk_miles=st.opts.max_walk_miles,
            wind_override=st.opts.wind_from_deg,
        ),
        coverage=dict(
            public_fraction=round(float(A["public"][inside].mean()), 2) if any_inside else 0,
            reachable_fraction=round(float((A["walk_m"][inside] <= limit).mean()), 2) if any_inside else 0,
            usable_fraction=round(float(usable[inside].mean()), 2) if any_inside else 0,
        ),
        best_score=cands[0]["score"] if cands else 0,
        mean_top5=round(float(np.mean([c["score"] for c in cands[:5]])), 1) if cands else 0,
        n_candidates=len(cands),
        zones=_zones(cands),
        private_land=dict(
            n_spots=len(private_cands),
            best_score=private_cands[0]["score"] if private_cands else 0,
            layer="hidden in the KMZ: 'Private land spots' folder and 'Lion score on private land' layer",
        ),
        worn_trails=worn_summary(st, cands),
        notes=[*st.notes, *pick_notes(cands, A["usable"])],
        runtime_s=round(runtime, 1),
    )
