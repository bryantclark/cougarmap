"""Scout: a fast, coarse screen of a whole region ("near Missoula", 40 km radius) that ranks ~3 km blocks worth a
detailed look. It uses the same ideas as the full model but at 30 m and with regional proxies:

- terrain: ruggedness, saddles, cliffs/escarpments, valleys (lions use broken country)
- edges: mix of timber and openings (edge density), not solid forest or open farmland
- wind: cold-air drainage aligned with the prevailing wind
- water: wetness from land shape
- access: share of the block within walking distance of a public road (straight line at this scale)
- land: share that is open-access public land
- people: penalty for dense road networks (towns, subdivisions)
- prior: iNaturalist cougar observations nearby (locations are blurred for privacy; used only as a weak prior)
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from . import aoi as aoi_mod
from . import terrain as T
from .arrays import Floats, Ints
from .config import MILE_M
from .context import Log
from .factors import wind_toward
from .grid import Grid
from .net import cached, get_json
from .sources import canopy as canopy_src
from .sources import dem as dem_src
from .sources import vector as vec
from .sources import weather
from .sources.dem import LonLatBounds
from .sources.vector import Feature
from .sources.weather import Wind

ROAD_TYPES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|track|service|road"


def _osm_roads(lb: LonLatBounds) -> list[Feature]:
    """Roads only, tiled so Overpass copes with big regions (two tiles at a time, Overpass's per-client limit)."""
    minx, miny, maxx, maxy = lb
    step = 0.25
    tiles = [
        (float(x0), float(y0), min(x0 + step, maxx), min(y0 + step, maxy))
        for x0 in np.arange(minx, maxx, step)
        for y0 in np.arange(miny, maxy, step)
    ]

    def tile(t: tuple[float, float, float, float]) -> list[Feature]:
        q = f'[out:json][timeout:180];way["highway"~"^({ROAD_TYPES})$"]({t[1]},{t[0]},{t[3]},{t[2]});out tags geom qt;'

        def build() -> list[Feature]:
            try:
                j = vec.overpass_json(q)
            except RuntimeError:  # every mirror failed: this tile's roads are missing (scouting is approximate)
                return []
            return [
                Feature(LineString([(p["lon"], p["lat"]) for p in el["geometry"]]), el.get("tags", {}))
                for el in j.get("elements", [])
                if len(el.get("geometry", [])) >= 2
            ]

        roads: list[Feature] = cached("osmroads", [round(v, 4) for v in t], build)
        return roads

    with ThreadPoolExecutor(2) as ex:
        return [f for part in ex.map(tile, tiles) for f in part]


def _inat_cougars(lb: LonLatBounds) -> list[tuple[float, float]]:
    """(lat, lon) of iNaturalist cougar observations in the bounds (at most 1000), [] if the service is down."""

    def build() -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        page = 1
        while page <= 5:
            j = get_json(
                "https://api.inaturalist.org/v1/observations",
                params=dict(
                    taxon_id=41944,
                    swlat=lb[1],
                    swlng=lb[0],
                    nelat=lb[3],
                    nelng=lb[2],
                    per_page=200,
                    page=page,
                    quality_grade="research,needs_id",
                ),
                timeout=60,
            )
            res = j.get("results", [])
            for r in res:
                if r.get("geojson"):
                    lon, lat = r["geojson"]["coordinates"]
                    pts.append((lat, lon))
            if len(res) < 200:
                break
            page += 1
        return pts

    try:
        obs: list[tuple[float, float]] = cached("inat", [round(v, 3) for v in lb], build)
    except (RuntimeError, ValueError, KeyError):  # the service is down or answered garbage: the prior is optional
        return []
    return obs


class Inputs(NamedTuple):
    z: Floats
    chm: Floats
    roads: list[Feature]
    land: list[Feature]
    wind: Wind
    cougars: list[tuple[float, float]]  # (lat, lon) sightings


def _fetch(g: Grid, lat: float, lon: float, month: int) -> Inputs:
    lb = g.lonlat_bounds()
    with ThreadPoolExecutor(6) as ex:
        f_z = ex.submit(dem_src.fetch_dem, g, allow_lidar=False)
        f_chm = ex.submit(canopy_src.fetch_canopy, g)
        f_roads = ex.submit(_osm_roads, lb)
        f_land = ex.submit(vec.fetch_public_land, lb)
        f_wind = ex.submit(weather.prevailing, lat, lon, month)
        f_obs = ex.submit(_inat_cougars, lb)
        z, _ = f_z.result()
        return Inputs(z, f_chm.result(), f_roads.result(), f_land.result(), f_wind.result(), f_obs.result())


class Cells(NamedTuple):
    """Per-cell (30-45 m) regional proxies."""

    inside: Floats
    rugged: Floats
    saddle: Floats  # saddle strength at each saddle cell
    steep: Floats
    forest: Floats
    edge: Floats  # timber/opening boundary
    conv: Floats  # cold-air drainage lined up with the wind
    wet: Floats  # wet draws from land shape (top 8% of the wetness index)
    reach: Floats  # within a (discounted) straight-line walk of a drivable road
    public: Floats
    land_id: Ints  # 1-based index into the public land names
    developed: Floats  # residential and major roads
    cougars: Floats  # sightings per cell


def _cells(g: Grid, inp: Inputs, region: aoi_mod.AOI, max_walk_miles: float) -> tuple[Cells, list[str]]:
    res = g.res
    z = inp.z
    inside = g.mask([g.project(region.geom)])
    sl = T.slope_deg(z, res)
    # terrain
    relief = ndimage.maximum_filter(z, 17) - ndimage.minimum_filter(z, 17)  # ~500 m window
    sad_r = np.zeros(g.shape, "float32")
    for s in T.saddles(z, res, sigma_m=60, reach_m=450, min_rise_m=15):
        sad_r[s["row"], s["col"]] = np.clip((s["rise_m"] - 15) / 60, 0.3, 1)
    # edges
    forest = inp.chm >= 3
    edge = forest ^ ndimage.binary_erosion(forest, iterations=1)
    # wind: drainage aligned with prevailing wind
    wx, wn = wind_toward(inp.wind["from_deg"])
    dx, dn, m = T.downslope_unit(T.smooth(z, 90, res), res)
    acc = T.flow_accumulation(T.smooth(z, 30, res), res)
    strength = np.clip((np.log10(np.maximum(acc, 1)) - 4.5) / 2.5, 0, 1)
    strength = ndimage.gaussian_filter(np.maximum(strength, 0.85 * ndimage.maximum_filter(strength, 3)), 1.0)
    drain = strength * np.clip(m / 0.04, 0.2, 1)
    conv = drain * ((1 + dx * wx + dn * wn) / 2) ** 2
    # water from land shape
    twi = np.log(np.maximum(acc / res, 1) / np.tan(np.radians(np.maximum(sl, 0.3))))
    wet = twi > np.percentile(twi[inside], 92)
    # access, land, people
    vec.project_features(inp.roads, g)
    road_r = g.mask([f.xy for f in inp.roads if vec.osm_is_drivable(f.props)])
    d_road = T.edt(road_r, res) if road_r.any() else np.full(g.shape, 1e7)
    feats = [f for f in inp.land if isinstance(f.geom, (Polygon, MultiPolygon)) and f.props.get("pub_access") == "OA"]
    feats_xy = g.project_all([f.geom for f in feats])
    dev = [f.xy for f in inp.roads if f.props.get("highway") in DEVELOPED]
    cnt = np.zeros(g.shape, "float32")
    for la, lo in inp.cougars:
        rr, cc = g.rowcol(*g.from_lonlat(lo, la))
        if g.contains_rc(rr, cc):
            cnt[rr, cc] += 1

    def f32(a: np.ndarray) -> Floats:
        return a.astype("float32")

    cells = Cells(
        inside=f32(inside),
        rugged=f32(np.clip((relief - 20) / 150, 0, 1)),
        saddle=sad_r,
        steep=f32(sl >= 30),
        forest=f32(forest),
        edge=f32(edge),
        conv=f32(conv),
        wet=f32(wet),
        reach=f32(d_road <= max_walk_miles * MILE_M * 0.8),  # straight line, discounted for terrain
        public=f32(g.mask(feats_xy, all_touched=False)),
        land_id=g.rasterize(feats_xy, list(range(1, len(feats) + 1)), dtype="int32", all_touched=False),
        developed=g.rasterize(dev, 1, dtype="uint8").astype("float32"),
        cougars=cnt,
    )
    return cells, [f.props.get("unit_nm") or "" for f in feats]


DEVELOPED = ("residential", "service", "primary", "secondary", "trunk", "motorway")


class Blocks(NamedTuple):
    """Per-block (~3 km) features and the block score."""

    B: int  # block size in cells
    inside: Floats
    rugged: Floats
    saddles: Floats
    saddle_count: Floats
    steep: Floats
    forest: Floats
    edge: Floats
    conv: Floats
    wet: Floats
    reach: Floats
    public: Floats
    developed: Floats
    prior: Floats
    terrain: Floats
    edges: Floats
    score: Floats


def _p97(a: Floats, axis: tuple[int, int]) -> Floats:
    out: Floats = np.percentile(a, 97, axis=axis)
    return out


def _blocks(c: Cells, res: float, block_km: float) -> Blocks:
    B = max(2, round(block_km * 1000 / res))
    H, W = c.inside.shape
    nby, nbx = H // B, W // B

    def blk(a: np.ndarray, fn: Callable[..., Floats] = np.mean) -> Floats:
        a = a[: nby * B, : nbx * B].astype("float32").reshape(nby, B, nbx, B)
        return fn(a, axis=(1, 3))

    f_in = blk(c.inside)
    f_forest = blk(c.forest)
    f_mix = 1 - np.abs(f_forest - 0.6) / 0.6
    f_rug, f_sad, f_steep = blk(c.rugged), np.clip(blk(c.saddle, np.sum) / 3, 0, 1), np.clip(blk(c.steep) / 0.08, 0, 1)
    f_edge, f_conv = np.clip(blk(c.edge) / 0.06, 0, 1), np.clip(blk(c.conv, _p97) / 0.35, 0, 1)
    f_wet, f_reach, f_pub = np.clip(blk(c.wet) / 0.12, 0, 1), blk(c.reach), blk(c.public)
    f_dev = np.clip(blk(c.developed) * res / 1000 * (1e6 / (res * res)) / 4.0, 0, 1)  # km road per km2 / 4
    f_prior = np.zeros_like(f_in)
    if c.cougars.any():
        f_prior = ndimage.gaussian_filter(blk(c.cougars, np.sum), 10_000 / (B * res))
        f_prior = f_prior / max(float(f_prior.max()), 1e-9)

    terrain = 0.45 * f_rug + 0.3 * f_sad + 0.25 * f_steep
    edges = 0.6 * f_edge + 0.4 * np.clip(f_mix, 0, 1)
    score = 0.30 * terrain + 0.25 * edges + 0.20 * f_conv + 0.10 * f_wet + 0.15 * f_prior
    score *= (1 - 0.8 * f_dev) * np.clip(f_reach / 0.4, 0, 1)
    score *= np.clip(f_pub / 0.5, 0, 1)
    score *= f_in > 0.6
    return Blocks(
        B=B,
        inside=f_in,
        rugged=f_rug,
        saddles=f_sad,
        saddle_count=blk(c.saddle > 0, np.sum),
        steep=f_steep,
        forest=f_forest,
        edge=f_edge,
        conv=f_conv,
        wet=f_wet,
        reach=f_reach,
        public=f_pub,
        developed=f_dev,
        prior=f_prior,
        terrain=terrain,
        edges=edges,
        score=100 * score,
    )


def _pick(score: Floats, top: int) -> list[tuple[int, int]]:
    """The best blocks (block row, col), none adjacent to another."""
    picked: list[tuple[int, int]] = []
    taken = np.zeros_like(score, bool)
    for i in np.argsort(-score, axis=None):
        by, bx = (int(v) for v in np.unravel_index(i, score.shape))
        if score[by, bx] <= 1 or taken[by, bx]:
            continue
        picked.append((by, bx))
        taken[max(by - 1, 0) : by + 2, max(bx - 1, 0) : bx + 2] = True
        if len(picked) >= top:
            break
    return picked


def _why(b: Blocks, by: int, bx: int, wind: Wind) -> list[str]:
    why = []
    if b.rugged[by, bx] > 0.5:
        why.append("broken, rugged terrain")
    if b.saddles[by, bx] > 0.3:
        why.append(f"{round(b.saddle_count[by, bx])} saddles")
    if b.steep[by, bx] > 0.5:
        why.append("cliffs/escarpments")
    if b.edge[by, bx] > 0.5:
        why.append("lots of timber/opening edge")
    if b.conv[by, bx] > 0.5:
        why.append(f"drainages lined up with the {wind['from_compass']} wind")
    if b.wet[by, bx] > 0.5:
        why.append("wet draws / likely water")
    if b.prior[by, bx] > 0.3:
        why.append("cougar sightings reported in the area")
    return why


def _block(
    g: Grid, b: Blocks, cells: Cells, names: list[str], rank: int, at: tuple[int, int], block_km: float, wind: Wind
) -> dict[str, Any]:
    """One picked block as JSON (at: block row, col)."""
    B, res = b.B, g.res
    by, bx = at
    x0 = g.x0 + bx * B * res
    y1 = g.y0 - by * B * res
    poly = g.unproject(box(x0, y1 - B * res, x0 + B * res, y1))
    c = poly.centroid
    sub = cells.land_id[by * B : (by + 1) * B, bx * B : (bx + 1) * B]
    ids, counts = np.unique(sub[sub > 0], return_counts=True)
    land_name = names[ids[np.argmax(counts)] - 1] if len(ids) else "mostly private"
    return dict(
        rank=rank,
        name=f"Scout block {rank}",
        score=round(float(b.score[by, bx]), 1),
        center=dict(lat=round(c.y, 5), lon=round(c.x, 5)),
        bbox=[round(v, 5) for v in poly.bounds],
        block_km=block_km,
        public_fraction=round(float(b.public[by, bx]), 2),
        reachable_fraction=round(float(b.reach[by, bx]), 2),
        forest_fraction=round(float(b.forest[by, bx]), 2),
        land=land_name,
        why=_why(b, by, bx, wind),
        parts=dict(
            terrain=round(float(b.terrain[by, bx]), 2),
            edges=round(float(b.edges[by, bx]), 2),
            wind=round(float(b.conv[by, bx]), 2),
            water=round(float(b.wet[by, bx]), 2),
            prior=round(float(b.prior[by, bx]), 2),
            development=round(float(b.developed[by, bx]), 2),
        ),
    )


def scout(
    lat: float,
    lon: float,
    radius_km: float = 40.0,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    block_km: float = 3.0,
    top: int = 8,
    log: Log = print,
) -> dict[str, Any]:
    """Rank ~block_km blocks within radius_km of lat/lon by regional lion-habitat proxies; the top ones are worth
    a detailed analysis."""
    month = month or dt.date.today().month
    region = aoi_mod.circle(lat, lon, radius_km)
    res = 30.0 if radius_km <= 45 else 45.0
    g = Grid.around_geometry(region.geom, res, pad_m=500)
    log(f"scout: {radius_km:g} km around {lat:.4f},{lon:.4f} at {res:g} m ({g.width}x{g.height})")
    log("  elevation, canopy, roads, public land, wind, cougar sightings (in parallel)...")
    inp = _fetch(g, lat, lon, month)
    cells, names = _cells(g, inp, region, max_walk_miles)
    b = _blocks(cells, res, block_km)
    picked = _pick(b.score, top)
    blocks = [_block(g, b, cells, names, k, at, block_km, inp.wind) for k, at in enumerate(picked, 1)]
    return dict(
        center=dict(lat=lat, lon=lon),
        radius_km=radius_km,
        month=month,
        wind=weather.describe(inp.wind),
        cougar_observations=len(inp.cougars),
        blocks=blocks,
        next_step="run analyze on the top blocks (bbox) for camera spots",
    )
