"""Vector data: water (NHD), public land (PAD-US), Forest Service roads (MVUM), OpenStreetMap roads/fences/trails.

Every fetcher returns a list of Feature(geom=<shapely, lon/lat>, props=<dict>) and is cached on disk.
"""

from __future__ import annotations

import contextlib
import math
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from shapely.errors import ShapelyError
from shapely.geometry import LineString, Point, shape
from shapely.geometry.base import BaseGeometry

from ..net import cached, get_json
from .dem import LonLatBounds

if TYPE_CHECKING:
    from ..grid import Grid

Props = dict[str, Any]  # a feature's attributes (OSM tags, NHD/PAD-US/MVUM fields, lower-cased)

NHD = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
PADUS = "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/PADUS_Public_Access/FeatureServer/0"
MVUM = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MVUM_01/MapServer/1"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
# Overpass allows about two concurrent requests per client; parallel downloads share these slots.
_OVERPASS_SLOTS = threading.BoundedSemaphore(2)


@dataclass
class Feature:
    geom: BaseGeometry  # lon/lat
    props: Props = field(default_factory=dict)
    _xy: BaseGeometry | None = field(default=None, repr=False, compare=False)

    @property
    def xy(self) -> BaseGeometry:
        """The geometry in the analysis grid's CRS (set once per run by project_features)."""
        if self._xy is None:
            raise RuntimeError("feature not projected yet: call project_features (Context.build does)")
        return self._xy


def project_features(feats: Iterable[Feature], grid: Grid) -> None:
    """Project every feature onto the grid's CRS once, in one vectorized call (sets Feature.xy). All grids of
    one analysis share a CRS, so each factor reuses these instead of re-projecting per grid."""
    feats = list(feats)
    for f, g in zip(feats, grid.project_all([f.geom for f in feats]), strict=True):
        f._xy = g


def overpass_json(query: str, timeout: float = 240) -> Any:
    """Run an Overpass query, trying each mirror in turn (at most two requests in flight per process)."""
    last: Exception | None = None
    with _OVERPASS_SLOTS:
        for url in OVERPASS:
            try:
                return get_json(url, data={"data": query}, timeout=timeout)
            except Exception as ex:
                last = ex
    raise RuntimeError(f"overpass failed: {last}")


def _bbox(lb: LonLatBounds) -> str:
    return ",".join(f"{v:.5f}" for v in lb)


def arcgis_query(
    url: str,
    lb: LonLatBounds,
    out_fields: str = "*",
    where: str = "1=1",
    page: int = 1000,
    simplify_deg: float | None = None,
) -> list[Feature]:
    """Every feature of an ArcGIS REST layer intersecting the bounds (paged), attribute names lower-cased."""
    feats: list[Feature] = []
    offset = 0
    while True:
        params: dict[str, Any] = dict(
            geometry=_bbox(lb),
            geometryType="esriGeometryEnvelope",
            inSR=4326,
            spatialRel="esriSpatialRelIntersects",
            where=where,
            outFields=out_fields,
            returnGeometry="true",
            outSR=4326,
            f="geojson",
            resultOffset=offset,
            resultRecordCount=page,
        )
        if simplify_deg:
            params["maxAllowableOffset"] = simplify_deg
        j = get_json(f"{url}/query", params=params, timeout=120)
        if "error" in j:
            raise RuntimeError(f"{url}: {j['error']}")
        got = j.get("features", [])
        for f in got:
            if f.get("geometry"):
                with contextlib.suppress(ShapelyError, ValueError, TypeError, AttributeError):  # malformed: skip
                    feats.append(
                        Feature(shape(f["geometry"]), {k.lower(): v for k, v in (f.get("properties") or {}).items()})
                    )
        offset += len(got)
        if (
            len(got) < page
            and not j.get("exceededTransferLimit")
            and not j.get("properties", {}).get("exceededTransferLimit")
        ):
            break
        if not got:
            break
    return feats


def arcgis_query_tiled(
    url: str, lb: LonLatBounds, out_fields: str, tile_deg: float = 0.05, id_field: str | None = None, workers: int = 6
) -> list[Feature]:
    """Some services time out on big boxes but answer small ones in a second or two: split and merge."""
    minx, miny, maxx, maxy = lb
    nx, ny = max(1, math.ceil((maxx - minx) / tile_deg)), max(1, math.ceil((maxy - miny) / tile_deg))
    tiles = [
        (
            minx + i * (maxx - minx) / nx,
            miny + j * (maxy - miny) / ny,
            minx + (i + 1) * (maxx - minx) / nx,
            miny + (j + 1) * (maxy - miny) / ny,
        )
        for i in range(nx)
        for j in range(ny)
    ]
    fields = out_fields + ("," + id_field if id_field else "")
    failed: list[LonLatBounds] = []

    def fetch(t: LonLatBounds, depth: int = 0) -> list[Feature]:
        try:
            return arcgis_query(url, t, fields, page=1000)
        except Exception:  # a timeout or service error on this tile: split it (twice), then give up on it
            if depth >= 2:
                failed.append(t)
                return []
            x0, y0, x1, y1 = t
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            quads = [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]
            return [f for q in quads for f in fetch(q, depth + 1)]

    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(fetch, tiles))
    if failed:
        print(
            f"  warning: {len(failed)} small tiles of {url.rsplit('/', 2)[-2]}/{url.rsplit('/', 1)[-1]} "
            f"could not be fetched; water there may be missing",
            file=sys.stderr,
        )
    seen: set[Any] = set()
    out: list[Feature] = []
    for ch in chunks:
        for f in ch:
            key = f.props.get(id_field.lower()) if id_field else None
            key = key or (f.geom.wkb if hasattr(f.geom, "wkb") else id(f))
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
    return out


# ---- water ------------------------------------------------------------------------------------
# NHD FCodes: 46006 perennial stream, 46003 intermittent, 46007 ephemeral, 55800 artificial path (through lakes),
# 33600 canal/ditch. FTypes: 458 spring/seep (point), 390 lake/pond, 466 swamp/marsh, 436 reservoir.


class Water(TypedDict):
    points: list[Feature]  # springs and seeps
    flowlines: list[Feature]  # streams, canals, paths through lakes
    waterbodies: list[Feature]  # lakes, ponds, reservoirs, marshes


def fetch_water(lb: LonLatBounds) -> Water:
    def build() -> Water:
        return Water(
            points=arcgis_query_tiled(f"{NHD}/0", lb, "fcode,ftype,gnis_name", 0.1, "permanent_identifier"),
            flowlines=arcgis_query_tiled(
                f"{NHD}/6", lb, "fcode,ftype,gnis_name,lengthkm", 0.05, "permanent_identifier"
            ),
            waterbodies=arcgis_query_tiled(
                f"{NHD}/12", lb, "fcode,ftype,gnis_name,areasqkm", 0.1, "permanent_identifier"
            ),
        )

    water: Water = cached("nhd", _bbox(lb), build)
    return water


# ---- public land ------------------------------------------------------------------------------


def fetch_public_land(lb: LonLatBounds) -> list[Feature]:
    fields = "Category,FeatClass,Unit_Nm,Pub_Access,MngNm_Desc,MngTp_Desc,DesTp_Desc"
    # simplify to ~5 m for small areas, ~25 m for regional scouting (boundaries are huge multipolygons)
    span = max(lb[2] - lb[0], lb[3] - lb[1])
    tol = 0.00005 if span < 0.5 else 0.00025
    land: list[Feature] = cached(
        "padus", (_bbox(lb), tol), lambda: arcgis_query(PADUS, lb, fields, page=500, simplify_deg=tol)
    )
    return land


# ---- deer and elk winter range (Washington) ---------------------------------------------------

PHS = "https://geodataservices.wdfw.wa.gov/arcgis/rest/services/PHSOnTheWeb/PHSOnTheWebPublic/MapServer/3"


def is_winter_range(p: Props) -> bool:
    """A WDFW Priority Habitats and Species polygon of deer or elk winter range (or a regular concentration)."""
    name = (p.get("occurrence_name") or "").lower()
    winter = "WINTER" in (p.get("notes") or "").upper() or p.get("priorityarea_desc") == "Regular Concentration"
    return winter and ("deer" in name or "elk" in name)


def fetch_phs_winter_range(lb: LonLatBounds) -> list[Feature]:
    """Mapped deer and elk winter range from WDFW's public PHS on the Web layer (empty outside Washington)."""
    feats: list[Feature] = cached(
        "phs", _bbox(lb), lambda: arcgis_query(PHS, lb, "Occurrence_Name,PriorityArea_Desc,Notes")
    )
    return [f for f in feats if is_winter_range(f.props)]


# ---- forest service roads ---------------------------------------------------------------------


def fetch_mvum(lb: LonLatBounds) -> list[Feature]:
    fields = (
        "id,name,symbol,mvum_symbol_name,seasonal,passengervehicle,passengervehicle_datesopen,"
        "highclearancevehicle,highclearancevehicle_datesopen,forestname"
    )
    roads: list[Feature] = cached("mvum", _bbox(lb), lambda: arcgis_query(MVUM, lb, fields))
    return roads


def mvum_open_in_month(props: Props, month: int) -> bool:
    for kind in ("passengervehicle", "highclearancevehicle"):
        if (props.get(kind) or "").lower() != "open":
            continue
        dates = props.get(f"{kind}_datesopen") or ""
        if not dates.strip():
            return True
        for span in dates.split(","):
            try:
                a, b = span.strip().split("-")
                m0, m1 = int(a.split("/")[0]), int(b.split("/")[0])
            except ValueError:
                return True
            if (m0 <= month <= m1) if m0 <= m1 else (month >= m0 or month <= m1):
                return True
    return False


def mvum_seasonal(props: Props) -> bool:
    """A Forest Service road closed to vehicles for part of the year (quiet then)."""
    return not all(mvum_open_in_month(props, m) for m in range(1, 13))


# ---- OpenStreetMap ----------------------------------------------------------------------------


def fetch_osm(lb: LonLatBounds) -> list[Feature]:
    s, w, n, e = lb[1], lb[0], lb[3], lb[2]
    q = (
        f"[out:json][timeout:180];("
        f'way["highway"]({s},{w},{n},{e});'
        f'way["barrier"~"fence|wall"]({s},{w},{n},{e});'
        f'way["railway"~"rail|abandoned|disused"]({s},{w},{n},{e});'
        f'way["man_made"="pipeline"]({s},{w},{n},{e});'
        f");out tags geom;"
    )

    def build() -> list[Feature]:
        j = overpass_json(q)
        feats = []
        for el in j.get("elements", []):
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 2:
                feats.append(Feature(LineString(pts), el.get("tags", {})))
        return feats

    osm: list[Feature] = cached("osm", _bbox(lb), build)
    return osm


# Where people gather: trailheads, campgrounds, picnic and caravan sites, parking, toilets, shelters (the source
# method's "low pressure from humans"; heavy hiker use is what cougars avoid, Arthurs et al. 2025).
RECREATION_TAGS = {
    "highway": ("trailhead",),
    "tourism": ("camp_site", "picnic_site", "caravan_site"),
    "amenity": ("parking", "toilets", "shelter"),
}


def fetch_osm_points(lb: LonLatBounds) -> list[Feature]:
    """Recreation sites (RECREATION_TAGS) as points: nodes where they are, ways and relations at their centre."""
    s, w, n, e = lb[1], lb[0], lb[3], lb[2]
    sel = "".join(f'nwr["{k}"~"^({"|".join(v)})$"]({s},{w},{n},{e});' for k, v in RECREATION_TAGS.items())
    q = f"[out:json][timeout:180];({sel});out tags center;"

    def build() -> list[Feature]:
        feats = []
        for el in overpass_json(q).get("elements", []):
            c = el.get("center") or (el if "lon" in el and "lat" in el else None)
            if c is not None:
                feats.append(Feature(Point(c["lon"], c["lat"]), el.get("tags", {})))
        return feats

    pts: list[Feature] = cached("osmpoints", _bbox(lb), build)
    return pts


DRIVABLE = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "road",
}
MAYBE_DRIVABLE = {"track", "service"}  # counted unless inside national forest (MVUM decides there) or tagged private
FOOT = {"path", "footway", "bridleway", "steps", "cycleway"}


def osm_is_drivable(tags: Props) -> str | None:
    """'yes', 'maybe' (track/service), or None."""
    hw = tags.get("highway")
    if tags.get("access") in ("private", "no") or tags.get("motor_vehicle") in ("private", "no"):
        return None
    if tags.get("service") in ("driveway", "parking_aisle", "drive-through"):
        return None
    if hw in DRIVABLE:
        return "yes"
    if hw in MAYBE_DRIVABLE:
        return "maybe"
    return None


# Paved roads carry traffic: lions avoid the noise and people out to hundreds of metres, and cameras there get
# stolen. Gravel county roads, forest roads, tracks and trails are quiet (lions walk them), so they don't count.
PAVED_HIGHWAY = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
PAVED_SURFACE = {"paved", "asphalt", "concrete", "chipseal"}


def osm_is_paved(tags: Props) -> bool:
    """A paved, trafficked road: a major highway class, or any road tagged with a paved surface."""
    hw = tags.get("highway")
    return hw in PAVED_HIGHWAY or (hw is not None and tags.get("surface") in PAVED_SURFACE)


__all__ = [
    "DRIVABLE",
    "FOOT",
    "MAYBE_DRIVABLE",
    "PAVED_HIGHWAY",
    "PAVED_SURFACE",
    "RECREATION_TAGS",
    "Feature",
    "Props",
    "Water",
    "arcgis_query",
    "arcgis_query_tiled",
    "fetch_mvum",
    "fetch_osm",
    "fetch_osm_points",
    "fetch_phs_winter_range",
    "fetch_public_land",
    "fetch_water",
    "is_winter_range",
    "mvum_open_in_month",
    "mvum_seasonal",
    "osm_is_drivable",
    "osm_is_paved",
    "overpass_json",
    "project_features",
]
