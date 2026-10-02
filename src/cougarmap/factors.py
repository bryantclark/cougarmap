"""The four factors (wind, pinch points, edges, limited water), terrain travel lines, walking access, traffic and
land status.

Each compute_* function reads a Context and stores its layers (state.Layers) in ctx.layers, keeping the
sub-components so candidate spots can be explained ("saddle, 32 m climb saved", "downwind end of a 3 ha
meadow", ...). The pieces they are built from are pure functions of arrays and the cell size.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, cast

import numpy as np
from rasterio.enums import Resampling
from scipy import ndimage, signal
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from . import terrain as T
from .arrays import Floats, Ints, Mask
from .config import (
    APPROACH,
    MILE_M,
    PLACEMENT,
    WATER,
    WATER_PINCH,
    WINTER,
    WORN,
    Approach,
    EdgeBand,
    WaterPinch,
    Winter,
)
from .context import Context, Log
from .corridor import funnel_score
from .grid import Grid
from .sources import vector as vec
from .sources.vector import Feature, Props
from .sources.weather import Wind
from .sources.weather import label as wind_label
from .state import PAVED_DIST_CAP_M, SaddlePoint, stored_chm

__all__ = [
    "APPROACH_TO",
    "CLOSED_TRACK_M",
    "PAVED_DIST_CAP_M",
    "TRAIL_KINDS",
    "compute_access",
    "compute_approach",
    "compute_edges",
    "compute_houses",
    "compute_land",
    "compute_pinch",
    "compute_recreation",
    "compute_season",
    "compute_terrain",
    "compute_traffic",
    "compute_trails",
    "compute_travel",
    "compute_water",
    "compute_wind",
    "compute_worn_trails",
    "daytime_wind_deg",
    "destination_approach",
    "drain_strength",
    "water_pinch",
    "water_squeeze",
    "wind_terms",
]


def _unit(ax: Floats, an: Floats) -> tuple[Floats, Floats]:
    m = np.hypot(ax, an)
    m = np.where(m < 1e-9, 1, m)
    return ax / m, an / m


def wind_toward(from_deg: float) -> tuple[float, float]:
    """Unit vector the wind blows TOWARD (east, north) for a meteorological 'from' direction."""
    th = math.radians(from_deg)
    return -math.sin(th), -math.cos(th)


def _any_props(_props: Props) -> bool:
    return True


def _lines(feats: list[Feature], pred: Callable[[Props], bool] = _any_props) -> list[BaseGeometry]:
    """Projected line geometries (Feature.xy) of the features matching pred."""
    return [f.xy for f in feats if pred(f.props) and isinstance(f.geom, (LineString, MultiLineString))]


def _polys(feats: list[Feature], pred: Callable[[Props], bool] = _any_props) -> list[BaseGeometry]:
    """Projected polygon geometries (Feature.xy) of the features matching pred."""
    return [f.xy for f in feats if pred(f.props) and isinstance(f.geom, (Polygon, MultiPolygon))]


def _ramp(a: Floats, lo: float, hi: float) -> Floats:
    """0 below lo, 1 above hi, linear between."""
    out: Floats = np.clip((a - lo) / (hi - lo), 0, 1)
    return out


_disc_kernel = T.disc_kernel
_disc_frac = T.disc_frac


def _soft(mask: Mask, reach_m: float, res: float, exclude: Mask | None = None) -> Floats:
    """1 on the mask, fading linearly to 0 at reach_m away (0 on `exclude`)."""
    if not mask.any():
        return np.zeros(mask.shape, "float32")
    out = np.clip(1 - T.edt(mask, res) / reach_m, 0, 1)
    if exclude is not None:
        out[exclude] = 0
    return out.astype("float32")


# ======================================================================================================
# Terrain basics shared by several factors
# ======================================================================================================


def compute_terrain(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    r, rm = ctx.fine.res, ctx.mid.res
    log("terrain: slope, landforms, drainage...")
    A["slope"] = T.slope_deg(ctx.z, r)
    A["slope_mid"] = T.slope_deg(ctx.z_mid, rm)
    A["landform_mid"] = T.geomorphons(ctx.z_mid, rm, search_m=250)
    z20 = T.smooth(ctx.z_mid, 20, rm)
    A["d8_rec_mid"], A["acc_mid"] = T.d8_flow(z20, rm)
    A["sca_mid"] = T.specific_catchment(z20, rm)
    A["tpi_mid"] = T.tpi(ctx.z_mid, 400, rm)


def compute_travel(ctx: Context, log: Log = print) -> None:
    """Terrain travel lines (a TPI percentile over the whole padded 10 m grid, so small areas and property scans
    rank their terrain against the same surroundings): drainage bottoms all year, on gentle grades most; ridge
    spines at full strength only where they are crossings (saddles, spine junctions, and in winter ridges above
    big sun-facing slopes), at travel_spine_floor elsewhere. Needs compute_pinch's saddle points."""
    A, h = ctx.layers, ctx.opts.weights.habitat
    rm = ctx.mid.res
    log("travel lines: drainage bottoms, ridge-spine crossings...")
    bottom, spine, pos = T.travel_parts(ctx.z_mid, rm, h.travel_tpi_m, h.travel_power, h.travel_relief_m)
    saddles = [(int(s["row"]), int(s["col"])) for s in A["saddle_points"]]
    gate = T.spine_gate(pos, saddles, rm, h.travel_saddle_m, h.travel_junction_m, h.travel_junction_pos)
    thermal = np.zeros(pos.shape, "float32")
    if ctx.month in h.thermal_months:
        thermal = T.thermal_slopes(
            ctx.z_mid, rm, h.thermal_aspect_deg, h.thermal_min_slope_deg, h.thermal_radius_m, h.thermal_frac
        )
    gentle = T.gentle_grade(ctx.z_mid, rm, h.travel_gentle_deg, h.travel_gentle_min)
    k = h.travel_spine_floor + (1 - h.travel_spine_floor) * np.maximum(gate, thermal)
    A["travel_pos_mid"], A["travel_gate_mid"], A["travel_thermal_mid"] = pos, gate, thermal
    A["travel"] = np.nan_to_num(ctx.up((bottom * gentle + spine * k).astype("float32"))).astype("float32")


# ======================================================================================================
# 1. Wind: cold air drainage + prevailing high-pressure wind, and where they agree
# ======================================================================================================


# Cold-air drainage strength from the specific catchment area (upslope area per metre of contour width, MFD):
# 0 below ~50 m2/m (the top ~50 m of an open slope), 1 at ~300,000 m2/m (a main valley floor). Hillsides drain
# too ("down the valley and off the hillsides"): a long open slope reaches ~0.3-0.4.
DRAIN_SCA_LOG10 = (1.7, 5.5)


class WindTerms(NamedTuple):
    """compute_wind's layers on the 10 m grid (see compute_wind)."""

    score: Floats
    drain: Floats
    dx: Floats  # cold-air drainage direction (unit east, north)
    dn: Floats
    ax: Floats  # along-valley / exposed wind direction
    an: Floats
    fx: Floats  # the air actually moving at dawn/dusk
    fn: Floats
    cos: Floats
    conv: Floats
    windward: Floats


def drain_strength(sca: Floats) -> Floats:
    """0-1 cold-air drainage strength from the specific catchment area (m2/m, terrain.specific_catchment)."""
    lo, hi = DRAIN_SCA_LOG10
    out: Floats = np.clip((np.log10(np.maximum(sca, 1)) - lo) / (hi - lo), 0, 1)
    return out


def wind_terms(
    z_mid: Floats, rm: float, from_deg: float, consistency: float, landform: Ints, tpi: Floats, sca: Floats
) -> WindTerms:
    """The wind factor and its parts from the 10 m DEM, its landforms, TPI and specific catchment area."""
    wx, wn = wind_toward(from_deg)

    # Cold air drainage: downhill direction of a smoothed surface, strength from the upslope area per metre of
    # slope width, so open hillsides drain as well as channels.
    dx, dn, sl = T.downslope_unit(T.smooth(z_mid, 60, rm), rm)
    strength = drain_strength(sca)
    moving = np.clip(sl / 0.05, 0.3, 1.0)  # on flats cold air pools instead of flowing
    # cold air moves as a layer tens of metres deep and wide, not a 10 m channel: spread it across the valley floor
    spread: Floats = np.maximum(strength, ndimage.maximum_filter(strength, size=max(3, int(40 / rm) | 1)) * 0.85)
    drain = T.gaussian(spread, 20 / rm) * moving

    # Exposure: ridges see the prevailing wind; valley bottoms get it channelled along the valley.
    E = 1 / (1 + np.exp(-tpi / 12.0))
    vx, vn, _ = T.downslope_unit(T.smooth(z_mid, 200, rm), rm)  # large-scale valley axis
    # In a valley the wind is channeled along the axis, and only as strongly as the valley lines up with it:
    # a valley running across the wind gets little consistent along-valley flow.
    along = vx * wx + vn * wn  # cos(valley axis, wind)
    ax, an = _unit(E * wx + (1 - E) * along * vx + 1e-6 * dx, E * wn + (1 - E) * along * vn + 1e-6 * dn)
    shelter = np.clip(T.upwind_shelter(z_mid, rm, from_deg, 300) / 10.0, 0, 1)

    # Air actually moving at dawn/dusk: drainage dominates low and sheltered, wind dominates high and exposed.
    beta = np.clip(0.3 + 0.5 * (1 - E) + 0.2 * shelter, 0, 0.95)
    fx, fn = _unit(beta * dx + (1 - beta) * ax, beta * dn + (1 - beta) * an)

    # Convergence (the pro-tip): drainage heading the same way the wind blows = consistent flow most of the day.
    # On exposed ground compare drainage with the wind itself; in valleys, with the wind's along-valley part.
    cos = np.clip(E * (dx * wx + dn * wn) + (1 - E) * (dx * vx + dn * vn) * along, -1, 1)
    conv = ((1 + cos) / 2) ** 2
    valleyish = np.clip(1.2 - E, 0, 1)
    conv_score = conv * drain * valleyish

    # Windward side of a ridge: slope faces into the wind, within ~100 m below the crest.
    facing = np.clip(-(dx * wx + dn * wn), 0, 1)
    crest = np.isin(landform, (T.RIDGE, T.PEAK, T.SHOULDER))
    d_crest = T.edt(crest, rm) if crest.any() else np.full(crest.shape, 1e6)
    near_crest = np.clip(1 - d_crest / 100.0, 0, 1)
    steep = np.clip(sl / 0.15, 0, 1)
    windward = facing * near_crest * steep * (1 - shelter)

    R = float(np.clip(consistency, 0, 1))
    # consistent moving air at dawn/dusk, boosted where drainage and the prevailing wind agree
    flow_consistency = drain * (0.5 + 0.5 * (1 + cos) / 2)
    score = np.clip(np.maximum(flow_consistency, 0.6 * windward) * (0.7 + 0.3 * R), 0, 1)

    def f32(a: Floats) -> Floats:
        return a.astype("float32")

    return WindTerms(*(f32(a) for a in (score, drain, dx, dn, ax, an, fx, fn, cos, conv_score, windward)))


def compute_wind(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    w = ctx.wind
    log(f"wind: prevailing {wind_label(w)}, cold air drainage...")
    t = wind_terms(
        ctx.z_mid, ctx.mid.res, w["from_deg"], w["consistency"], A["landform_mid"], A["tpi_mid"], A["sca_mid"]
    )
    A["windx_mid"], A["windn_mid"], A["wind_mid"] = t.ax, t.an, t.score
    A["conv_mid"], A["windward_mid"], A["drain_mid"] = t.conv, t.windward, t.drain
    A["drainx_mid"], A["drainn_mid"] = t.dx, t.dn
    A["flowx_mid"], A["flown_mid"], A["cos_mid"] = t.fx, t.fn, t.cos
    A["wind"] = ctx.up(t.score)
    A["flowx"] = ctx.up(t.fx)
    A["flown"] = ctx.up(t.fn)


# ======================================================================================================
# 2. Edges: the hunting edge - standing in timber, looking out over an opening where prey feeds. The most downwind
#    part of each edge is best. Ridgelines, valley bottoms, water and trails are secondary travel lines.
# ======================================================================================================

MIN_OPENING_M2 = 1500.0  # smaller openings don't hold prey to watch
TRAIL_HIGHWAYS = ("track", "path", "footway", "bridleway")  # two-tracks, old logging roads, game/foot trails


class Openings(NamedTuple):
    meadow: Mask  # openings big enough to hunt
    label: Ints  # 1..n per opening, 0 elsewhere
    n: int
    area_m2: Floats  # per opening 1..n


def openings(chm: Floats, lakes: Mask, res: float) -> Openings:
    """Meadows, clearcuts, burns, rock/grass glades: open ground (canopy < 3 m, not lake) without narrow strips
    (roads, trails, powerline cuts), in patches of at least MIN_OPENING_M2."""
    open_ = T.binary_opening((chm < 3.0) & ~lakes, max(1, round(6 / res)))
    lab, n = ndimage.label(open_)
    if not n:
        return Openings(open_, lab, 0, np.array([]))
    keep = np.zeros(n + 1, bool)
    keep[1:] = T.label_counts(lab, n) * res * res >= MIN_OPENING_M2
    meadow = keep[lab]
    lab, n = ndimage.label(meadow)
    return Openings(meadow, lab, n, T.label_counts(lab, n) * res * res)


def hunting_edge(
    chm: Floats, meadow: Mask, res: float, band: EdgeBand | None = None
) -> tuple[Floats, tuple[Ints, Ints]]:
    """The hunting edge (0-1): a lion hidden in cover, close enough to an opening to watch prey in it, and the
    opening's own rim beside real cover, where kills cluster (EdgeBand); plus each cell's nearest opening cell
    (rows, cols; an opening cell is its own)."""
    band = band or EdgeBand()
    cover = chm >= 4.0
    # Patchy cover counts: open pine parkland and small-glade mosaics are prime ground, so no solid stand needed.
    hide = np.clip((_disc_frac(cover, 8, res) - 0.35) / 0.45, 0, 1)  # in cover right here
    view = np.clip(_disc_frac(meadow, 60, res) / 0.30, 0, 1)  # open ground in sight
    d_open, nearest = T.edt_nearest(meadow, res)
    prox = np.clip(1 - (d_open - 35) / 40, 0, 1) * (d_open > 0)  # 0-35 m back in the timber, gone by 75 m
    in_cover = hide * view * prox
    full, reach = band.open_full_m, band.open_reach_m
    near_cover = np.clip(1 - (T.edt(cover, res) - full) / (reach - full), 0, 1)
    real_cover = np.clip(_disc_frac(cover, band.open_cover_m, res) / band.open_cover_frac, 0, 1)
    in_open = meadow * near_cover * real_cover * band.open_weight
    return np.maximum(in_cover, in_open).astype("float32"), nearest


def downwind_position(
    hunt: Floats, op: Openings, nearest: tuple[Ints, Ints], flowx: Floats, flown: Floats, grid: Grid
) -> Floats:
    """0-1 position of each hunting-edge cell along the mean dawn/dusk air flow over the opening it looks onto
    (1 = the most downwind end: air leaves the opening there, so prey can't smell the lion)."""
    q = np.zeros(grid.shape, "float32")
    n, lab = op.n, op.label
    if not n:
        return q
    ir, ic = nearest
    xs, ys = grid.cell_centers()
    cells = T.label_counts(lab, n).astype(np.float64)
    mfx = T.label_sums(flowx, lab, n) / cells  # mean air flow over each opening
    mfn = T.label_sums(flown, lab, n) / cells
    nrm = np.hypot(mfx, mfn) + 1e-9
    lut_x = np.concatenate([[0], mfx / nrm])
    lut_n = np.concatenate([[0], mfn / nrm])
    band = hunt > 0.05
    blab = np.where(band, lab[ir, ic], 0)
    p = xs[None, :] * lut_x[blab] + ys[:, None] * lut_n[blab]
    lo, hi = T.labeled_minmax(p, blab, n)  # blab is 0 off the band, so only band cells count
    pmin = np.concatenate([[0], lo])[blab]
    pmax = np.concatenate([[1], hi])[blab]
    return np.where(band, np.clip((p - pmin) / np.maximum(pmax - pmin, 1e-6), 0, 1), 0).astype("float32")


class TravelEdges(NamedTuple):
    """Secondary travel lines, kept below the "factor on" level so on their own they only break ties."""

    ridge: Floats
    valley: Floats
    water: Floats
    route: Floats


def travel_edges(ctx: Context, chm: Floats, lakes: Mask) -> TravelEdges:
    A, r = ctx.layers, ctx.fine.res
    lf = ctx.fine.resample_from(A["landform_mid"].astype("float32"), ctx.mid, Resampling.nearest)
    acc = ctx.up(np.log10(np.maximum(A["acc_mid"], 1)))
    ridge = np.isin(lf, (T.RIDGE, T.PEAK))
    valley = np.isin(lf, (T.VALLEY, T.PIT, T.HOLLOW)) & (acc > 4.5)
    water = _perennial_mask(ctx, ctx.fine) | lakes
    routes = ctx.fine.mask([f.xy for f in ctx.osm if f.props.get("highway") in TRAIL_HIGHWAYS])
    return TravelEdges(
        ridge=0.25 * _soft(ridge, 15, r),
        valley=0.3 * _soft(valley, 40, r),
        water=0.4 * _soft(water, 15, r, exclude=lakes),
        route=0.35 * _soft(routes, 12, r) * np.clip(chm / 5.0, 0, 1),  # trails through timber
    )


def daytime_wind_deg(wind: Wind, override_from_deg: float | None) -> float:
    """The daytime high-pressure wind (direction FROM, degrees): the user's wind when set, else the climatology's
    daytime wind, else (no daytime data) the dawn/dusk one."""
    if override_from_deg is not None:
        return float(override_from_deg) % 360
    day = float(wind["day_from_deg"])
    return day if math.isfinite(day) else float(wind["from_deg"])


def compute_edges(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    r = ctx.fine.res
    band = ctx.opts.weights.edge_band
    log("edges: timber edges overlooking openings, downwind ends...")
    chm = T.median_filter(ctx.chm, size=max(3, round(5 / r) | 1))
    lakes = _lake_mask(ctx, ctx.fine)
    op = openings(chm, lakes, r)
    hunt, nearest = hunting_edge(chm, op.meadow, r, band)
    # Downwind ends per air current (our method names two): the evening cold-air drainage, which runs
    # downhill, and the daytime high-pressure wind; an end downwind in both is best.
    drain = (np.nan_to_num(ctx.up(A["drainx_mid"])), np.nan_to_num(ctx.up(A["drainn_mid"])))
    qd = downwind_position(hunt, op, nearest, *drain, ctx.fine)
    wx, wn = wind_toward(daytime_wind_deg(ctx.wind, ctx.opts.wind_from_deg))
    day = (np.full(ctx.fine.shape, wx, "float32"), np.full(ctx.fine.shape, wn, "float32"))
    qw = downwind_position(hunt, op, nearest, *day, ctx.fine)
    q = (qd**2 + qw**2) / 2
    downwind = (hunt * q).astype("float32")
    hunt_edge = (hunt * (1 - band.downwind_bonus + band.downwind_bonus * q)).astype("float32")
    tr = travel_edges(ctx, chm, lakes)

    # a hunting edge that is also on a travel line (trail along the timber, creek past a meadow) is better still
    lines = np.stack(tr)
    n_lines = (lines > 0.15).sum(0)
    edge = np.maximum(hunt_edge, lines.max(0)) + 0.15 * np.minimum(n_lines, 2) * (hunt_edge > 0.3)
    A["meadow_label"] = op.label.astype("int32")
    A["meadow_ha"] = np.concatenate([[0], op.area_m2 / 1e4]).astype("float32")
    A["edge_q"] = q
    A["edge_q_drain"], A["edge_q_wind"] = qd, qw
    A["edge_route"] = tr.route.astype("float32")
    A["edges"] = np.clip(edge, 0, 1).astype("float32")
    A["meadow"] = op.meadow
    A["edge_downwind"], A["edge_meadow"] = downwind, hunt_edge
    A["edge_ridge"], A["edge_valley"], A["edge_water"] = tr.ridge, tr.valley, tr.water


# ======================================================================================================
# water layers (shared)
# ======================================================================================================

PERENNIAL_FCODES = (46006, 55800)  # perennial streams, artificial paths through lakes


def _is_lake(p: Props) -> bool:
    ft = p.get("ftype")
    return ft in (390, 436) or p.get("fcode") in (39000, 39004, 39009, 39010, 39011, 39012, 43600, 43601)


def _lake_polys(ctx: Context, min_ha: float) -> list[BaseGeometry]:
    """Lakes, ponds and reservoirs (NHD) of at least min_ha, projected."""
    return [g for g in _polys(ctx.water["waterbodies"], _is_lake) if g.area >= min_ha * 1e4]


def _lake_mask(ctx: Context, grid: Grid, min_ha: float = 0.5) -> Mask:
    """Lakes and reservoirs of at least min_ha on the grid (computed once per run; read-only)."""

    def build() -> Mask:
        return grid.mask(_lake_polys(ctx, min_ha), all_touched=False)

    return ctx.memo(("lake", grid, min_ha), build)


def _perennial_mask(ctx: Context, grid: Grid) -> Mask:
    """Perennial streams (and paths through lakes) on the grid (computed once per run; read-only)."""

    def build() -> Mask:
        return grid.mask(_lines(ctx.water["flowlines"], lambda p: p.get("fcode") in PERENNIAL_FCODES))

    return ctx.memo(("perennial", grid), build)


# ======================================================================================================
# 3. Topographic pinch points: saddles, cliff bases, water banks, fences, movement funnels, and the squeeze
#    between ponds and the barriers across from them
# ======================================================================================================


def _saddles(ctx: Context) -> tuple[Floats, list[SaddlePoint]]:
    """Saddles (10 m grid) painted as soft discs scaled by how much climb they save, and the saddle points."""
    rm = ctx.mid.res
    sad_mid = np.zeros(ctx.mid.shape, "float32")
    H, W = ctx.mid.shape
    yy, xx = np.mgrid[-6:7, -6:7]
    disc = np.exp(-(xx**2 + yy**2) * rm * rm / (2 * 25.0**2))
    points: list[SaddlePoint] = []
    for s in T.saddles(ctx.z_mid, rm, sigma_m=30, reach_m=300, min_rise_m=8):
        val = float(np.clip((s["rise_m"] - 8) / 40, 0.25, 1.0))
        r0, c0 = s["row"], s["col"]
        rs, cs = slice(max(r0 - 6, 0), min(r0 + 7, H)), slice(max(c0 - 6, 0), min(c0 + 7, W))
        d = disc[rs.start - r0 + 6 : rs.stop - r0 + 6, cs.start - c0 + 6 : cs.stop - c0 + 6]
        sad_mid[rs, cs] = np.maximum(sad_mid[rs, cs], val * d)
        x, y = ctx.mid.xy(r0, c0)
        points.append(SaddlePoint(**s, x=float(x), y=float(y)))
    return ctx.up(sad_mid), points


def _funnels(ctx: Context, cliff: Mask, lakes: Mask) -> Floats:
    """Circuit-theory movement funnels, solved on a coarse (>= 30 m, <= 160k cell) grid."""
    A = ctx.layers
    cres = max(30.0, 3 * ctx.fine.res)
    cg = ctx.fine.coarsen(cres)
    while cg.width * cg.height > 160_000:
        cres *= 1.25
        cg = ctx.fine.coarsen(cres)
    sl_c = cg.resample_from(A["slope"], ctx.fine, Resampling.average)
    open_c = cg.resample_from((ctx.chm < 2.5).astype("float32"), ctx.fine, Resampling.average)
    cliff_c = cg.resample_from(cliff.astype("float32"), ctx.fine, Resampling.max)
    lake_c = cg.resample_from(lakes.astype("float32"), ctx.fine, Resampling.average)
    roads_c = cg.rasterize([f.xy for f in ctx.osm if vec.osm_is_drivable(f.props)], 1)
    R = (
        1
        + 25 * np.clip(np.nan_to_num(sl_c) / 45, 0, 1.5) ** 2
        + 150 * np.nan_to_num(cliff_c)
        + 400 * np.nan_to_num(lake_c)
        + 1.5 * np.nan_to_num(open_c)
        + 4 * roads_c
    )
    funnel: Floats = np.nan_to_num(0.8 * ctx.fine.resample_from(funnel_score(R, cres), cg))
    return funnel


# pinch_water_kind: what a pond or lake squeezes travel against there (0 = no water pinch)
WATER_PINCH_CLIFF, WATER_PINCH_STEEP, WATER_PINCH_OPENING, WATER_PINCH_POND, WATER_PINCH_END = range(1, 6)


def water_squeeze(water: Mask, barrier: Ints, res: float, cfg: WaterPinch = WATER_PINCH) -> tuple[Floats, Ints]:
    """The squeeze between water and another barrier: land within cfg.shore_m of the water whose nearest other
    barrier lies across from the water (cosine of the angle between the two directions <= cfg.max_cos), by the
    width of the gap (shore distance + barrier distance): cfg.strength up to cfg.gap_full_m, 0 from cfg.gap_zero_m.
    barrier: a kind code per cell (WATER_PINCH_CLIFF/STEEP/OPENING, 0 = none); other water bodies (8-connected
    components of `water`) are barriers too (WATER_PINCH_POND). Barrier cells within cfg.skip_m of the water are
    its own bank and do not count; barrier cells themselves get 0. Returns the value and the kind of the barrier
    across (int8, 0 where the value is 0)."""
    val = np.zeros(water.shape, "float32")
    kind = np.zeros(water.shape, np.int8)
    if not water.any():
        return val, kind
    d_water, (wr, wc) = T.edt_nearest(water, res)
    B = (barrier > 0) & (d_water > cfg.skip_m)
    rr, cc = np.nonzero(~water & (d_water <= cfg.shore_m))
    if not len(rr):
        return val, kind
    # the nearest barrier: on land first ...
    best_d = np.full(len(rr), np.inf)
    best_r, best_c = np.zeros(len(rr)), np.zeros(len(rr))
    best_k = np.zeros(len(rr), np.int8)
    if B.any():
        d_b, (br, bc) = T.edt_nearest(B, res)
        best_d = d_b[rr, cc]
        best_r, best_c = br[rr, cc].astype(np.float64), bc[rr, cc].astype(np.float64)
        best_k = barrier[br[rr, cc], bc[rr, cc]].astype(np.int8)
    # ... then another water body, per body in a window around it (exact: water farther than the window only
    # ever gives a gap of at least gap_zero_m)
    lab, _ = ndimage.label(water, structure=np.ones((3, 3), bool))
    near = lab[wr[rr, cc], wc[rr, cc]]
    boxes = ndimage.find_objects(lab)
    pad = math.ceil((cfg.gap_zero_m + cfg.shore_m) / res) + 1
    H, W = water.shape
    order = np.argsort(near, kind="stable")
    labels, starts = np.unique(near[order], return_index=True)
    for li, sel in zip(labels, np.split(order, starts[1:]), strict=True):
        sl = boxes[li - 1]
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        win = lab[r0:r1, c0:c1]
        other = (win > 0) & (win != li)
        if not other.any():
            continue
        d_o, (orr, occ) = T.edt_nearest(other, res)
        lr, lc = rr[sel] - r0, cc[sel] - c0
        d = d_o[lr, lc]
        upd = d < best_d[sel]
        s2 = sel[upd]
        best_d[s2] = d[upd]
        best_r[s2] = orr[lr[upd], lc[upd]] + r0
        best_c[s2] = occ[lr[upd], lc[upd]] + c0
        best_k[s2] = WATER_PINCH_POND
    # across: the directions to the water and to the barrier point (nearly) opposite ways
    vwr, vwc = (wr[rr, cc] - rr).astype(np.float64), (wc[rr, cc] - cc).astype(np.float64)
    vbr, vbc = best_r - rr, best_c - cc
    nb = np.hypot(vbr, vbc)
    cos = (vwr * vbr + vwc * vbc) / np.maximum(np.hypot(vwr, vwc) * nb, 1e-9)
    gap = d_water[rr, cc] + best_d
    ok = (cos <= cfg.max_cos) & np.isfinite(best_d) & (nb > 0)
    v = cfg.strength * np.clip((cfg.gap_zero_m - gap) / (cfg.gap_zero_m - cfg.gap_full_m), 0, 1) * ok
    val[rr, cc] = v
    kind[rr, cc] = np.where(v > 0, best_k, 0)
    val[B] = 0
    kind[B] = 0
    return val, kind


def _points_of(geom: BaseGeometry) -> list[Point]:
    """The points where a line meets a shore (a piece of line running along it: its two ends)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, LineString):
        return [Point(geom.coords[0]), Point(geom.coords[-1])]
    return [p for part in getattr(geom, "geoms", ()) for p in _points_of(part)]


def _mid_cells_on_fine(ctx: Context, m_mid: Mask) -> Mask:
    """The fine cells holding the centres of the marked mid-grid cells."""
    out = np.zeros(ctx.fine.shape, bool)
    r, c = np.nonzero(m_mid)
    rr, cc = ctx.fine.rowcol(*ctx.mid.xy(r, c))
    ok = ctx.fine.contains_rc(rr, cc)
    out[rr[ok], cc[ok]] = True
    return out


def _pond_ends(ctx: Context, cfg: WaterPinch = WATER_PINCH) -> Mask:
    """Where streams come into or go out of the ponds under cfg.end_max_ha (fine grid): NHD flowlines crossing
    the shore, and D8 channels draining at least cfg.channel_min_m2 that cross it (10 m grid)."""
    A, g = ctx.layers, ctx.fine
    small = [p for p in _lake_polys(ctx, cfg.min_ha) if p.area < cfg.end_max_ha * 1e4]
    ends = np.zeros(g.shape, bool)
    if not small:
        return ends
    lines = _lines(ctx.water["flowlines"])
    if lines:
        tree = STRtree(lines)
        pts = [q for p in small for k in tree.query(p) for q in _points_of(lines[k].intersection(p.boundary))]
        if pts:
            rr, cc = g.rowcol([q.x for q in pts], [q.y for q in pts])
            ok = g.contains_rc(rr, cc)
            ends[rr[ok], cc[ok]] = True
    pond_mid = np.nan_to_num(ctx.mid.resample_from(g.mask(small, all_touched=False), g, Resampling.average)) >= 0.5
    inside = pond_mid.ravel()
    rec = A["d8_rec_mid"]
    i = np.nonzero((A["acc_mid"].ravel() >= cfg.channel_min_m2) & (rec >= 0))[0]
    j = rec[i]
    m = np.zeros(inside.size, bool)
    m[i[~inside[i] & inside[j]]] = True  # inlets: the last channel cell above the pond
    m[j[inside[i] & ~inside[j]]] = True  # outlets: the first channel cell below it
    return ends | _mid_cells_on_fine(ctx, m.reshape(ctx.mid.shape))


def water_pinch(ctx: Context, cliff: Mask, cfg: WaterPinch = WATER_PINCH) -> tuple[Floats, Ints]:
    """Ponds and lakes as barriers (config.WaterPinch): the squeeze against cliffs, steep ground, openings and
    other water, and the ends of small ponds, max-combined. Returns the component and its kind
    (WATER_PINCH_*, 0 = none). Needs compute_edges' openings and compute_terrain's D8 channels."""
    A, r = ctx.layers, ctx.fine.res
    water = _lake_mask(ctx, ctx.fine, cfg.min_ha)
    z = T.smooth(np.asarray(ctx.z, np.float64), cfg.steep_smooth_m, r)
    steep = T.binary_opening(T.slope_deg(z, r) >= cfg.steep_deg, cfg.steep_open_cells)
    barrier = np.zeros(water.shape, np.int8)
    barrier[A["meadow"]] = WATER_PINCH_OPENING
    barrier[steep] = WATER_PINCH_STEEP
    barrier[cliff] = WATER_PINCH_CLIFF
    squeeze, kind = water_squeeze(water, barrier, r, cfg)
    ends = cfg.strength * _soft(_pond_ends(ctx, cfg), cfg.end_reach_m, r, exclude=water)
    kind[(ends > 0) & (ends >= squeeze)] = WATER_PINCH_END
    return np.maximum(squeeze, ends).astype("float32"), kind


def _is_barrier(p: Props) -> bool:
    return p.get("barrier") in ("fence", "wall") or bool(p.get("railway")) or p.get("man_made") == "pipeline"


def compute_pinch(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    r = ctx.fine.res
    log("pinch points: saddles, cliffs, funnels...")
    saddle, A["saddle_points"] = _saddles(ctx)

    # cliffs on the fine grid
    cliff_deg, steep_deg = (50.0, 35.0) if r <= 4 else (42.0, 30.0)
    cm, base, top = T.cliffs(ctx.z, r, cliff_deg=cliff_deg, steep_deg=steep_deg)
    A["cliff"] = cm

    # banks of lakes and perennial streams, fences/rail/pipelines
    lakes = _lake_mask(ctx, ctx.fine)
    peren = _perennial_mask(ctx, ctx.fine)
    bank = np.maximum(0.7 * _soft(lakes, 30, r, exclude=lakes), 0.45 * _soft(peren, 25, r, exclude=peren))
    fence = 0.35 * _soft(ctx.fine.mask([f.xy for f in ctx.osm if _is_barrier(f.props)]), 12, r)
    funnel = _funnels(ctx, cm, lakes)
    water, A["pinch_water_kind"] = water_pinch(ctx, cm)

    stack = np.stack([saddle, base, 0.4 * top, bank, fence, funnel, water])
    pinch = np.clip(stack.max(0) + 0.1 * np.maximum((stack > 0.3).sum(0) - 1, 0), 0, 1)
    A["pinch"] = pinch.astype("float32")
    A["pinch_saddle"], A["pinch_cliffbase"] = saddle.astype("float32"), base.astype("float32")
    A["pinch_clifftop"], A["pinch_bank"] = (0.4 * top).astype("float32"), bank.astype("float32")
    A["pinch_fence"], A["pinch_funnel"] = fence.astype("float32"), funnel.astype("float32")
    A["pinch_water"] = water


# ======================================================================================================
# 4. Limited water: springs, seeps, small ponds, seasonal streams -- the scarcer, the better
# ======================================================================================================


class WaterSource(NamedTuple):
    mask: Mask  # fine grid
    weight: float  # how much this kind of water counts
    label: str
    limited: bool  # counts toward the "distinct limited sources within a mile" uniqueness
    name: str  # what it is, in a reason ("the pond")


def water_sources(ctx: Context) -> list[WaterSource]:
    """Every kind of water on the fine grid, most valuable first (springs, the user's pins, small ponds,
    seasonal and perennial streams)."""
    g = ctx.fine
    pts = [f for f in ctx.water["points"] if f.props.get("ftype") == 458 or f.props.get("fcode") == 45800]
    user = [p for p in ctx.opts.user_points + ctx.aoi.user_points if p.get("kind") in ("water", "seasonal_water")]

    def ptmask(geoms: list[Point]) -> Mask:
        m = np.zeros(g.shape, bool)
        for geom in geoms:
            rr, cc = g.rowcol(*g.from_lonlat(geom.x, geom.y))
            if g.contains_rc(rr, cc):
                m[rr, cc] = True
        return m

    def pins(kind: str) -> Mask:
        return ptmask([Point(p["lon"], p["lat"]) for p in user if p["kind"] == kind])

    small = [p for p in _polys(ctx.water["waterbodies"]) if p.area < WATER.pond_max_ha * 1e4]
    seasonal = _lines(ctx.water["flowlines"], lambda p: p.get("fcode") in (46003, 46007))
    perennial = _lines(ctx.water["flowlines"], lambda p: p.get("fcode") == 46006)
    return [
        WaterSource(ptmask([cast("Point", f.geom) for f in pts]), 1.0, "spring/seep (NHD)", True, "spring"),
        WaterSource(pins("water"), 1.0, "known water (your pin)", True, "water"),
        WaterSource(pins("seasonal_water"), 0.8, "seasonal water (your pin)", True, "water"),
        WaterSource(g.mask(small), 0.85, f"small pond/marsh (under {WATER.pond_max_ha:g} ha)", True, "pond"),
        WaterSource(g.mask(seasonal), 0.45, "seasonal stream", True, "seasonal creek"),
        WaterSource(g.mask(perennial), 0.3, "perennial stream", False, "creek"),
    ]


def water_scarcity(ctx: Context, sources: list[WaterSource]) -> Floats:
    """0.75-1 multiplier: more where there is little permanent water (perennial streams, lakes of at least
    WATER.permanent_min_ha) within a mile and few other limited sources (computed on the 10 m grid). Water always
    matters; scarcity is a bonus, not a cut, since nearly everywhere has some stream within a mile."""
    rm = ctx.mid.res
    lakes = _lake_mask(ctx, ctx.mid, min_ha=WATER.permanent_min_ha)
    wet = ndimage.binary_dilation(lakes | _perennial_mask(ctx, ctx.mid), iterations=max(1, int(20 / rm)))
    rad = int(MILE_M / rm)
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    disc = ((xx**2 + yy**2) <= rad * rad).astype("float32")
    disc /= disc.sum()
    frac = signal.fftconvolve(wet.astype("float32"), disc, mode="same")
    scarcity_mid = np.clip(1 - frac / 0.04, 0.0, 1.0)
    # uniqueness: number of distinct limited sources within a mile
    anysrc_mid = np.zeros(ctx.mid.shape, "float32")
    for s in sources:
        if s.limited:
            mm = ctx.mid.resample_from(s.mask.astype("float32"), ctx.fine, Resampling.max)
            anysrc_mid = np.maximum(anysrc_mid, np.nan_to_num(mm))
    lab, _ = ndimage.label(anysrc_mid > 0, structure=np.ones((3, 3)))
    seeds = np.zeros_like(anysrc_mid)
    if lab.max():
        com = ndimage.center_of_mass(anysrc_mid > 0, lab, np.arange(1, lab.max() + 1))
        for a, b in cast("list[tuple[float, float]]", com):
            seeds[int(a), int(b)] = 1
    count = signal.fftconvolve(seeds, (disc > 0).astype("float32"), mode="same")
    unique_mid = 1 / (1 + 0.15 * np.maximum(np.round(count) - 1, 0))
    scarcity: Floats = np.nan_to_num(ctx.up(0.75 + 0.25 * scarcity_mid * unique_mid), nan=0.75)
    return scarcity


def water_proximity(sources: list[WaterSource], res: float, shape: tuple[int, int]) -> tuple[Floats, Ints]:
    """Best weighted nearness to any source (full strength within 40 m: cameras sit on the approach trails, not
    in the water; fading, gone past 300 m) and which source gives it (-1 = none)."""
    best = np.zeros(shape, "float32")
    which = np.full(shape, -1, "int8")
    for i, s in enumerate(sources):
        if not s.mask.any():
            continue
        d = T.edt(s.mask, res)
        near = d <= 300
        dn = d[near]
        v = np.zeros(shape)
        v[near] = s.weight * np.where(dn <= 40, 1.0, np.exp(-(dn - 40) / 100.0))
        upd = v > best
        best = np.where(upd, v, best)
        which = np.where(upd, i, which)
    return best, which


def compute_water(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    log("limited water: springs, seeps, seasonal water, scarcity...")
    sources = water_sources(ctx)
    scarcity = water_scarcity(ctx, sources)
    best, which = water_proximity(sources, ctx.fine.res, ctx.fine.shape)
    score = best * scarcity
    lakes = _lake_mask(ctx, ctx.fine)
    score[lakes] = 0
    A["water"] = np.clip(score, 0, 1).astype("float32")
    A["water_kind"] = which
    A["water_scarcity"] = scarcity.astype("float32")
    A["water_labels"] = [s.label for s in sources]
    A["lake"] = lakes
    to = np.zeros(ctx.fine.shape, np.int8)  # destinations for the approaches: limited water, the first source wins
    for s in reversed(sources):
        if s.limited:
            to[s.mask] = APPROACH_TO.index(s.name) + 1
    A["approach_water"] = to


# ======================================================================================================
# Destination approaches: covered routes from bedding timber to water and meadows (part of the travel line)
# ======================================================================================================

# What an approach leads to (state.Layers.travel_approach_to codes 1.., 0 = none): limited water by kind, openings.
APPROACH_TO = ("spring", "water", "pond", "seasonal creek", "meadow")


class Approaches(NamedTuple):
    value: Floats  # 0-1: a converging, covered approach near its destination, downwind of it
    root: Ints  # flat fine-grid index of the destination each cell's route ends at (-1 = none within reach)


def destination_approach(
    chm: Floats,
    slope: Floats,
    cliff: Mask,
    lake: Mask,
    line: Floats,
    trails: Mask,
    flowx: Floats,
    flown: Floats,
    dest: Mask,
    res: float,
    cfg: Approach = APPROACH,
) -> Approaches:
    """Covered approaches to the destination cells (config.Approach): least-cost routes from bedding cover to the
    nearest destination over a cost that favours cover, gentle ground and travel lines (line: the terrain travel
    line; trails: mapped two-tracks and paths), scored where the routes converge, run under cover, near the
    destination (taper) and downwind of it at dawn/dusk (flowx/flown: unit vector the air moves toward)."""
    shape = dest.shape
    if not dest.any():
        return Approaches(np.zeros(shape, "float32"), np.full(shape, -1, np.int64))
    r = res
    canopy = T.median_filter(chm.astype("float32"), size=max(3, round(cfg.cover_median_m / r) | 1))
    cover = canopy >= cfg.cover_m
    near_cover = T.disc_frac(cover, cfg.cover_near_m, r)
    t = np.maximum(np.clip(line, 0, 1), trails.astype("float32"))
    step_cost = (
        (1 + cfg.open_cost * (1 - near_cover))
        * (1 + cfg.slope_cost * _ramp(slope, *cfg.slope_deg))
        * (1 - cfg.line_discount * t)
    )
    cost = np.where(cliff, step_cost * cfg.cliff_cost, step_cost).astype(np.float64)
    cost[lake & ~dest] = cfg.lake_cost
    d_cover = T.edt(cover, r) if cover.any() else np.full(shape, 1e6)
    covered = d_cover <= cfg.covered_m
    by_cover = np.clip(1 - (d_cover - cfg.near_cover_m) / (cfg.far_cover_m - cfg.near_cover_m), 0, 1)
    d_dest = T.edt(dest, r)
    bed = T.disc_frac(cover, cfg.bed_m, r) >= cfg.bed_frac
    origins = bed & (d_dest >= cfg.start_m[0]) & (d_dest <= cfg.start_m[1]) & ~lake
    tree = T.cost_tree(cost, dest, origins * (r * r), covered, r, cfg.max_cost_m, cfg.last_m)
    conv = np.clip(np.log10(np.maximum(tree.flow, 1e-9) / (cfg.front_m * r)), 0, 1)
    conv[~np.isfinite(tree.dist)] = 0
    conv = ndimage.maximum_filter(conv, size=3)
    under = np.sqrt(np.clip(tree.covered_share, 0, 1) * by_cover)
    ok = tree.root >= 0
    W = shape[1]
    rr, cc = np.where(ok, tree.root // W, 0), np.where(ok, tree.root % W, 0)
    fx, fn = flowx[rr, cc].astype("float32"), flown[rr, cc].astype("float32")
    ve = (np.arange(W)[None, :] - cc).astype("float32")  # destination -> cell, east and north
    vn = (rr - np.arange(shape[0])[:, None]).astype("float32")
    nv = np.hypot(ve, vn)
    q = np.where(nv > 0, np.clip(0.5 + 0.5 * (ve * fx + vn * fn) / np.maximum(nv, 1e-6), 0, 1), 0.5)
    downwind = 0.5 + 0.5 * q
    taper = np.clip(1 - (d_dest - cfg.taper_m[0]) / (cfg.taper_m[1] - cfg.taper_m[0]), 0, 1)
    value = conv * under * downwind * taper * ok
    value[lake | dest] = 0
    return Approaches(value.astype("float32"), tree.root)


def compute_approach(ctx: Context, log: Log = print) -> None:
    """The destination approaches (destination_approach) to limited water and openings, folded into the travel
    line: travel = max(travel line, approach). Needs compute_travel, compute_water, compute_edges, compute_wind,
    compute_pinch (cliffs) and compute_trails (mapped two-tracks and paths lower the route cost)."""
    A = ctx.layers
    log("destination approaches: covered routes from timber to water and meadows...")
    to = np.where(A["meadow"] & (A["approach_water"] == 0), APPROACH_TO.index("meadow") + 1, A["approach_water"])
    trails = np.isin(A["trail_kind"], (TRAIL_KINDS.index("two-track") + 1, TRAIL_KINDS.index("trail") + 1))
    ap = destination_approach(
        stored_chm(ctx.chm),
        A["slope"],
        A["cliff"],
        A["lake"],
        A["travel"],
        trails,
        A["flowx"],
        A["flown"],
        to > 0,
        ctx.fine.res,
    )
    A["travel_approach"] = ap.value
    A["travel_approach_to"] = np.where((ap.value > 0) & (ap.root >= 0), to.ravel()[ap.root], 0).astype(np.int8)
    A["travel"] = np.maximum(A["travel"], ap.value)


# ======================================================================================================
# Season: where deer winter (a multiplier on the habitat around a spot, Nov-Apr)
# ======================================================================================================


def snow_factor(snow_cm: Floats, cfg: Winter = WINTER) -> Floats:
    """1 up to snow_full_cm of snow, falling linearly to snow_min at snow_zero_cm and deeper (NaN: no data, 1)."""
    f = 1 - (1 - cfg.snow_min) * (np.nan_to_num(snow_cm) - cfg.snow_full_cm) / (cfg.snow_zero_cm - cfg.snow_full_cm)
    out: Floats = np.clip(f, cfg.snow_min, 1).astype("float32")
    return out


def winter_range_factor(inside: Mask, res: float, cfg: Winter = WINTER) -> Floats:
    """1 inside mapped winter range, feathered over phs_feather_m to phs_outside elsewhere."""
    near = np.clip(1 - T.edt(inside, res) / cfg.phs_feather_m, 0, 1) if inside.any() else np.zeros(inside.shape)
    out: Floats = (cfg.phs_outside + (1 - cfg.phs_outside) * near).astype("float32")
    return out


def low_ground(z: Floats, res: float, cfg: Winter = WINTER) -> Floats:
    """0-1 position in the local elevation range (1 at the bottom of what lies within relief_radius_m), from the
    min/max on a 50 m decimation, smoothed relief_smooth_m."""
    f = max(1, round(50 / res))
    k = max(3, int(cfg.relief_radius_m / (f * res)))
    zc = z[::f, ::f]

    def back(a: Floats) -> Floats:
        up = np.repeat(np.repeat(a, f, axis=0), f, axis=1)[: z.shape[0], : z.shape[1]]
        return T.gaussian(up, cfg.relief_smooth_m / res)

    lo = back(ndimage.minimum_filter(zc, size=2 * k + 1))
    hi = back(ndimage.maximum_filter(zc, size=2 * k + 1))
    out: Floats = np.clip(1 - (z - lo) / np.maximum(hi - lo, cfg.min_relief_m), 0, 1).astype("float32")
    return out


def sun_facing(z: Floats, res: float, cfg: Winter = WINTER) -> Floats:
    """0-1: 0.5 on flats, toward 1 on steep south slopes, toward 0 on steep north ones (DEM smoothed 60 m)."""
    _, dn, sl = T.downslope_unit(T.smooth(z, 60, res), res)
    out: Floats = (0.5 + 0.5 * np.clip(-dn, -1, 1) * np.clip(sl / cfg.south_steep_tan, 0, 1)).astype("float32")
    return out


def compute_season(ctx: Context, log: Log = print) -> None:
    """The winter module (config.Winter): W = shallow snow x low ground x sun-facing slopes (0-1) on the 10 m
    grid, and the habitat multiplier floor + (1 - floor) x W, times the deer/elk winter range prior in its months.
    Exactly 1 outside the winter months."""
    A, g, cfg = ctx.layers, ctx.mid, WINTER
    if ctx.month not in cfg.months:
        A["season"] = np.ones(ctx.fine.shape, "float32")
        A["winter_mid"] = np.zeros(g.shape, "float32")
        A["winter_range_mid"] = np.zeros(g.shape, bool)
        return
    log("winter: low ground, sun-facing slopes, snow depth, deer winter range...")
    z = ctx.z_mid.astype("float32")
    f_snow = np.ones(g.shape, "float32")
    if ctx.snow_cm is not None:
        f_snow = snow_factor(g.resample_lonlat(*ctx.snow_cm))
    w = f_snow * (0.5 + 0.5 * low_ground(z, g.res, cfg)) * (0.5 + 0.5 * sun_facing(z, g.res, cfg))
    mult = cfg.floor + (1 - cfg.floor) * w
    inside = np.zeros(g.shape, bool)
    if ctx.month in cfg.phs_months and ctx.phs:
        inside = g.mask([f.xy for f in ctx.phs], all_touched=False)
        if inside.any():
            mult = mult * winter_range_factor(inside, g.res, cfg)
    A["winter_mid"], A["winter_range_mid"] = w.astype("float32"), inside
    A["season"] = np.nan_to_num(ctx.up(mult.astype("float32")), nan=1.0).astype("float32")


# ======================================================================================================
# Access: walking distance from an open road, and land status
# ======================================================================================================


def _land_name(p: Props) -> str:
    return f"{p.get('unit_nm') or ''} ({p.get('mngnm_desc') or p.get('mngtp_desc') or ''})".strip()


def compute_land(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    log("land status...")
    feats = [f for f in ctx.land if isinstance(f.geom, (Polygon, MultiPolygon))]
    names = [_land_name(f.props) for f in feats]
    access: list[str | None] = [f.props.get("pub_access") for f in feats]
    A["land_names"] = ["private / unknown", *names]
    A["land_access"] = [None, *access]
    # draw restricted first so open-access wins where they overlap
    order = sorted(range(len(feats)), key=lambda i: 0 if access[i] == "OA" else 1, reverse=True)
    ids = {
        key: grid.rasterize([feats[i].xy for i in order], [i + 1 for i in order], dtype="int32", all_touched=False)
        for grid, key in ((ctx.fine, "land_id"), (ctx.mid, "land_id_mid"))
    }
    A["land_id"], A["land_id_mid"] = ids["land_id"], ids["land_id_mid"]
    acc = np.array(A["land_access"], dtype=object)
    A["public"] = acc[A["land_id"]] == "OA"
    A["public_mid"] = acc[A["land_id_mid"]] == "OA"
    usfs = np.array([False] + ["forest service" in (n.lower()) for n in names])
    A["usfs_mid"] = usfs[A["land_id_mid"]]


def compute_houses(ctx: Context, log: Log = print) -> None:
    """Houses within houses_radius_m of each cell: how many people are around (each house costs a little, towns
    and subdivisions a lot: analyze.site_penalty). Footprints under house_min_m2 (sheds, blinds, trailers) don't
    count."""
    g, o = ctx.fine, ctx.opts
    log("populated areas (houses)...")
    m = np.zeros(g.shape, "float32")
    for lon, lat, area in ctx.buildings:
        if area < o.house_min_m2:
            continue
        rr, cc = g.rowcol(*g.from_lonlat(lon, lat))
        if g.contains_rc(rr, cc):
            m[int(rr), int(cc)] += 1
    disc = _disc_kernel(o.houses_radius_m, g.res)
    n = signal.fftconvolve(m, disc, mode="same") if m.any() else m
    ctx.layers["houses"] = np.round(np.clip(n, 0, None)).astype("float32")


def compute_recreation(ctx: Context, log: Log = print) -> None:
    """Distance to the nearest recreation site (trailhead, campground, picnic site, parking, toilets, shelter), on
    the 10 m grid (which includes the walking pad, so sites just outside the area count), capped like paved_dist."""
    g = ctx.mid
    log("recreation sites (trailheads, campgrounds, parking)...")
    sites = g.mask([f.xy for f in ctx.rec_points])
    d = np.minimum(T.edt(sites, g.res), PAVED_DIST_CAP_M)  # inf (so the cap) with no site at all
    ctx.layers["rec_dist"] = np.nan_to_num(ctx.up(d.astype("float32")), nan=PAVED_DIST_CAP_M).astype("float32")


CLOSED_TRACK_M = 15.0  # a spot this close to a closed forest road is "on" it


def compute_traffic(ctx: Context, log: Log = print) -> None:
    """Human traffic. Distance to the nearest paved road (on the 10 m grid, which includes the walking pad so
    roads just outside the area count; 10 m is ample for an 800 m ramp), and closed forest roads (quiet travel
    routes, for explanations only)."""
    A, g = ctx.layers, ctx.mid
    log("traffic: paved roads, closed forest roads...")
    paved = g.mask([f.xy for f in ctx.osm if vec.osm_is_paved(f.props) and isinstance(f.geom, LineString)])
    d = np.minimum(T.edt(paved, g.res), PAVED_DIST_CAP_M)  # inf (so the cap) with no paved road at all
    A["paved_dist"] = np.nan_to_num(ctx.up(d.astype("float32")), nan=PAVED_DIST_CAP_M).astype("float32")
    closed = _closed_tracks(ctx)
    A["closed_track_mid"] = T.edt(closed, g.res) <= CLOSED_TRACK_M if closed.any() else closed


def _closed_tracks(ctx: Context) -> Mask:
    """OSM tracks inside national forest that are not on the Motor Vehicle Use Map's roads open this month: gated
    or decommissioned forest roads, quiet travel routes through timber. Mid-grid mask; empty without MVUM data."""
    A, g = ctx.layers, ctx.mid
    tracks = [f.xy for f in ctx.osm if f.props.get("highway") == "track" and _quiet(f.props)]
    if not ctx.mvum or not tracks:
        return np.zeros(g.shape, bool)
    ids = g.rasterize(tracks, list(range(1, len(tracks) + 1)), dtype="int32")
    mv = g.rasterize([f.xy for f in ctx.mvum if vec.mvum_open_in_month(f.props, ctx.month)], 1)
    near_mv = T.edt(mv != 0, g.res) <= 20 if mv.any() else np.zeros(g.shape, bool)
    idx = np.arange(1, len(tracks) + 1)
    with np.errstate(invalid="ignore"):  # tracks too short to rasterize have no cells
        on_mvum = np.nan_to_num(np.asarray(ndimage.mean(near_mv, ids, idx)), nan=1.0)
        in_forest = np.nan_to_num(np.asarray(ndimage.mean(A["usfs_mid"], ids, idx)))
    lut = np.concatenate([[False], (on_mvum < 0.5) & (in_forest >= 0.5)])
    out: Mask = lut[ids]
    return out


# Quiet linear features a camera can watch, quietest first (state.Layers.trail_kind codes 1.., 0 = none).
TRAIL_KINDS = ("closed forest road", "seasonal forest road", "two-track", "trail")


def _quiet(p: Props) -> bool:
    return p.get("access") not in ("private", "no") and not vec.osm_is_paved(p)


def compute_trails(ctx: Context, log: Log = print) -> None:
    """Cells within PLACEMENT.on_m of a quiet linear feature, by kind (TRAIL_KINDS, quietest wins): closed or gated
    forest roads (OSM tracks in national forest off the Motor Vehicle Use Map), forest roads closed part of the year
    (MVUM, only in a month they are closed), OSM two-tracks, and paths and bridleways. Paved roads never count.
    Two-tracks may be open to vehicles: analyze.trail_alternate checks that against the open-road map. For the
    "alternate on the trail" suggestion only: it changes no score."""
    g, A = ctx.fine, ctx.layers
    log("trails and quiet roads (placement suggestions)...")
    closed_mid = _closed_tracks(ctx)
    lines = [
        g.resample_from(closed_mid.astype("float32"), ctx.mid, Resampling.nearest) > 0.5,
        g.mask(
            [f.xy for f in ctx.mvum if vec.mvum_seasonal(f.props) and not vec.mvum_open_in_month(f.props, ctx.month)]
        ),
        g.mask(_lines(ctx.osm, lambda p: p.get("highway") == "track" and _quiet(p))),
        g.mask(_lines(ctx.osm, lambda p: p.get("highway") in ("path", "bridleway") and _quiet(p))),
    ]
    kind = np.zeros(g.shape, np.int8)
    for code in range(len(lines), 0, -1):  # the quietest last, so it wins
        m = lines[code - 1]
        if m.any():
            kind[T.edt(m, g.res) <= PLACEMENT.on_m] = code
    A["trail_kind"] = kind


def compute_worn_trails(ctx: Context, log: Log = print) -> None:
    """Worn trails from the area's 1 m lidar (worn.py; Options.worn_trails): every line for the KMZ, and the fine
    cells of the lines on no map (not within WORN.mapped_m of an OpenStreetMap highway or an MVUM road) for the
    per-spot hint. Lines running along a channel are dropped as creek banks: mapped flowlines and waterbody shores
    (NHD), and channels the fine DEM drains at least WORN.channel_min_m2 into. For the KMZ layer and the hint
    only: it changes no score."""
    from . import worn

    A, g, L = ctx.layers, ctx.fine, ctx.lidar1m
    A["worn_lines"], A["worn_unmapped"] = [], np.zeros(g.shape, bool)
    if L is None:  # the option is off, or there is no 1 m lidar (context notes why)
        return
    log("worn trails (1 m lidar)...")
    water = [f.xy for f in ctx.water["flowlines"]] + [f.xy.boundary for f in ctx.water["waterbodies"]]
    channels = g.mask(water) | (T.flow_accumulation(ctx.z, g.res) >= WORN.channel_min_m2)
    near_d, (near_r, near_c) = T.edt_nearest(channels, g.res)

    def to_channel(rows: Ints, cols: Ints) -> tuple[Floats, Floats, Floats]:
        fr, fc, ok = worn.fine_cells(rows, cols, L.grid, g)
        fr, fc = np.clip(fr, 0, g.height - 1), np.clip(fc, 0, g.width - 1)
        x, y = L.grid.xy(rows, cols)
        cx, cy = g.xy(near_r[fr, fc], near_c[fr, fc])
        vx, vy = cx - x, y - cy  # towards the channel, x = column (east), y = row (south)
        far = ~ok | (near_r[fr, fc] < 0) | ~np.isfinite(near_d[fr, fc])
        return np.where(far, np.inf, np.hypot(vx, vy)), vx, vy

    inside = L.grid.mask([L.grid.project(ctx.aoi.geom)])
    det = worn.detect(L.z, inside, to_channel)
    ctx.lidar1m = None  # nothing else reads the 1 m DEM: free it before the rest of the run
    mapped = _lines(ctx.osm, lambda p: bool(p.get("highway"))) + _lines(ctx.mvum)
    mapped_d = T.edt(g.mask(mapped), g.res)
    rr, cc = np.nonzero(det.lines)
    fr, fc, ok = worn.fine_cells(rr, cc, L.grid, g)
    fr, fc = fr[ok], fc[ok]
    on_map = mapped_d[fr, fc] <= WORN.mapped_m
    by_kind = np.zeros(det.lines.shape, np.int8)
    by_kind[rr[ok], cc[ok]] = np.where(on_map, 1, 2)
    A["worn_unmapped"][fr[~on_map], fc[~on_map]] = True
    lines = []
    for code, is_mapped in ((1, True), (2, False)):
        for ln in worn.vectorize(by_kind == code, L.grid):
            lon, lat = g.to_lonlat(*np.asarray(ln.coords).T)
            pts = [(round(float(a), 6), round(float(b), 6)) for a, b in zip(lon, lat, strict=True)]
            lines.append(worn.WornLine(lonlat=pts, mapped=is_mapped, length_m=round(ln.length, 1)))
    A["worn_lines"] = lines
    km = sum(ln["length_m"] for ln in lines) / 1000
    off = sum(ln["length_m"] for ln in lines if not ln["mapped"]) / 1000
    log(f"  {km:.0f} km of worn lines, {off:.0f} km on no map ({det.creek_cells} m along channels dropped)")


class Roads(NamedTuple):
    open_roads: Mask  # roads a car can reach this month (mid grid)
    trails: Mask


def _roads(ctx: Context) -> Roads:
    """Inside national forest only roads on the Motor Vehicle Use Map (open this month) count, since OSM tracks
    there are often gated. Elsewhere, public tracks and service roads count."""
    g = ctx.mid
    yes, maybe, foot = [], [], []
    for f in ctx.osm:
        d = vec.osm_is_drivable(f.props)
        if d == "yes":
            yes.append(f.xy)
        elif d == "maybe":
            maybe.append(f.xy)
        elif f.props.get("highway") in vec.FOOT:
            foot.append(f.xy)
    mv_open = [f.xy for f in ctx.mvum if vec.mvum_open_in_month(f.props, ctx.month)]
    road = g.mask(yes + mv_open)
    maybe_r = g.mask(maybe)
    if ctx.mvum:
        maybe_r &= ~ctx.layers["usfs_mid"]
    return Roads(road | maybe_r, g.mask(foot))


def compute_access(ctx: Context, log: Log = print) -> None:
    A = ctx.layers
    g, rm = ctx.mid, ctx.mid.res
    log(f"access: walking distance from roads open in month {ctx.month}...")
    road, trail = _roads(ctx)
    sl = A["slope_mid"]
    cost = T.tobler_sec_per_m(sl)
    cost = np.where(trail, cost * 0.85, cost * 1.1)  # off-trail brush
    cost[sl > 45] = np.inf
    cost[_lake_mask(ctx, g)] = np.inf
    cost[_perennial_mask(ctx, g)] *= 4
    if not road.any():
        ctx.notes.append("No open roads found near this area.")
    # two sets of routes, so private land is just a layer: "walk_*" never cuts across private land to reach
    # public ground (roads themselves are fine); "walk_any_*" goes anywhere (e.g. your own property)
    cost_pub = np.where(~A["public_mid"] & ~road, np.inf, cost)
    A["road_mid"] = road
    limit = ctx.opts.max_walk_miles * MILE_M
    with ThreadPoolExecutor(2) as ex:  # the two walks are independent
        pub, anyw = ex.submit(T.walking, cost_pub, road, rm, limit), ex.submit(T.walking, cost, road, rm, limit)
        (t_pub, len_pub, pred_pub), (t_any, len_any, pred_any) = pub.result(), anyw.result()

    def fine(v: Floats) -> Floats:
        v_mid = np.where(np.isfinite(v), v, 1e7).astype("float32")
        return ctx.fine.resample_from(v_mid, g, Resampling.nearest)

    A["walk_pred_mid"], A["walk_m"], A["walk_s"] = pred_pub, fine(len_pub), fine(t_pub)
    A["walk_any_pred_mid"], A["walk_any_m"], A["walk_any_s"] = pred_any, fine(len_any), fine(t_any)
    d_road = T.edt(road, rm) if road.any() else np.full(g.shape, 1e7)
    A["road_dist"] = ctx.fine.resample_from(d_road.astype("float32"), g)
