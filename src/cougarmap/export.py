"""Outputs: a Google Earth KMZ (heatmap, factor layers, camera pins with reasons, walking routes, air-flow
arrows), plus summary.json / candidates.geojson, and a compact saved state so spots can be explained later."""

from __future__ import annotations

import html
import json
import math
import zipfile
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from . import terrain as T
from .arrays import Floats, Mask
from .config import MILE_M, THREADS
from .state import ModelState, save_state

if TYPE_CHECKING:
    from .analyze import Result, Spot, TrailAlternate
    from .context import Log
    from .grid import Grid

LonLat = tuple[float, float]
Bounds = tuple[float, float, float, float]  # west, south, east, north

# perceptual-ish ramp (dark purple -> red -> yellow), alpha grows with value
_RAMP = np.array(
    [[0.0, 20, 11, 52], [0.25, 101, 21, 110], [0.5, 188, 55, 84], [0.75, 249, 142, 9], [1.0, 252, 255, 164]]
)
_BLUE = np.array([[0.0, 8, 48, 107], [0.5, 33, 113, 181], [1.0, 198, 219, 239]])


def _colorize(v: Floats, ramp: Floats = _RAMP, alpha_max: float = 210, floor: float = 0.08) -> npt.NDArray[np.uint8]:
    v = np.clip(np.nan_to_num(v), 0, 1)
    rgba = np.zeros((*v.shape, 4), np.uint8)
    for ch in range(3):
        rgba[..., ch] = np.interp(v, ramp[:, 0], ramp[:, ch + 1]).astype(np.uint8)
    a = np.clip((v - floor) / (1 - floor), 0, 1) ** 0.7 * alpha_max
    rgba[..., 3] = np.where(v > floor, a, 0).astype(np.uint8)
    return rgba


def _to_wgs84(
    arr: np.ndarray, grid: Grid, max_px: int = 2400, resampling: Resampling = Resampling.bilinear
) -> tuple[Floats, Bounds]:
    """arr warped to lon/lat (at most max_px on a side) and its bounds."""
    src_t, src_crs = grid.transform, grid.crs
    b = grid.bounds
    dst_t, w, h = calculate_default_transform(src_crs, "EPSG:4326", grid.width, grid.height, *b)
    scale = max(1.0, max(w, h) / max_px)
    w2, h2 = int(w / scale), int(h / scale)
    dst_t = dst_t @ Affine.scale(w / w2, h / h2)
    out = np.full((h2, w2), np.nan, np.float32)
    reproject(
        arr.astype(np.float32),
        out,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs="EPSG:4326",
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        num_threads=THREADS,
    )
    west, north = dst_t.c, dst_t.f
    east, south = west + dst_t.a * w2, north + dst_t.e * h2
    return out, (west, south, east, north)


def _png(rgba: npt.NDArray[np.uint8]) -> bytes:
    # default zlib level, no exhaustive optimize pass: ~10x faster for files only ~6% bigger
    bio = BytesIO()
    Image.fromarray(rgba, "RGBA").save(bio, "PNG")
    return bio.getvalue()


def _render(ll: Floats, ramp: Floats, floor: float) -> bytes:
    return _png(_colorize(ll, ramp, floor=floor))


def _desc(c: Spot) -> str:
    bars = "".join(
        f"<tr><td>{k}</td><td><div style='background:#e8590c;height:9px;width:{int(v * 120)}px'></div></td>"
        f"<td>{v:.2f}</td></tr>"
        for k, v in c["factors"].items()
    )
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in c["reasons"]) or "<li>general stacking of factors</li>"
    walk = (
        f"{c['walk_miles']:.2f} mi walk (~{c['walk_minutes']} min) from the nearest open road"
        if c.get("walk_miles") is not None
        else "walk distance unknown"
    )
    alt = c.get("trail_alternate")
    alt_p = f"<p><i>Optional: {html.escape(alt['reason'])}; {alt['lat']:.6f}, {alt['lon']:.6f}</i></p>" if alt else ""
    worn = c.get("worn_trail")
    if worn:
        alt_p += f"<p><i>Placement hint: {html.escape(worn['reason'])}; {worn['lat']:.6f}, {worn['lon']:.6f}</i></p>"
    return (
        f"<b>Score {c['score']:.0f}</b> ({c['factors_on']} of 4 factors strong)<br/>"
        f"<ul>{reasons}</ul><table>{bars}</table>{alt_p}"
        f"<p>{walk}<br/>{html.escape(c['land'])}{'' if c['public'] else ' - PRIVATE'}<br/>"
        f"{c['landform']}, slope {c['slope_deg']:.0f} deg, canopy {c['canopy_m']:.0f} m, "
        f"elev {c['elevation_m']:.0f} m<br/>"
        f"{c['lat']:.6f}, {c['lon']:.6f}</p>"
    )


def _alt_desc(c: Spot, a: TrailAlternate) -> str:
    walk = f"{a['walk_miles']:.2f} mi walk" if a["walk_miles"] is not None else "walk distance unknown"
    return (
        f"<b>Score {a['score']:.0f}</b>, for {html.escape(c['name'])} (score {c['score']:.0f})<br/>"
        f"{html.escape(a['reason'])}<br/>{walk}<br/>{a['lat']:.6f}, {a['lon']:.6f}"
    )


def _arrows(st: ModelState, spacing_m: float = 250.0, length_m: float = 120.0) -> list[list[LonLat]]:
    """Dawn/dusk air-flow arrows (lon/lat polylines: shaft and head) on a spacing_m lattice inside the area."""
    A = st.layers
    g = st.mid
    step = max(1, int(spacing_m / g.res))
    inside = st.fine.rasterize([st.fine.project(st.aoi.geom)], 1, dtype="uint8")  # fine
    out = []
    for r in range(step // 2, g.height, step):
        for c in range(step // 2, g.width, step):
            x, y = g.xy(r, c)
            fr, fc = st.fine.rowcol(x, y)
            if not st.fine.contains_rc(fr, fc) or not inside[fr, fc]:
                continue
            fx, fn = float(A["flowx_mid"][r, c]), float(A["flown_mid"][r, c])
            if abs(fx) + abs(fn) < 1e-6:
                continue
            x0, y0 = x - fx * length_m / 2, y - fn * length_m / 2
            x1, y1 = x + fx * length_m / 2, y + fn * length_m / 2
            ang = math.atan2(fn, fx)
            h1 = (x1 - 35 * math.cos(ang - 0.45), y1 - 35 * math.sin(ang - 0.45))
            h2 = (x1 - 35 * math.cos(ang + 0.45), y1 - 35 * math.sin(ang + 0.45))
            pts = [(x0, y0), (x1, y1), h1, (x1, y1), h2]
            out.append([st.fine.to_lonlat(px, py) for px, py in pts])
    return out


def _route(st: ModelState, cand: Spot) -> list[LonLat]:
    """The walking route from the road to a spot (lon/lat, at most ~200 points), [] if there is none."""
    A = st.layers
    g = st.mid
    x, y = st.fine.xy(cand["row"], cand["col"])
    r, c = g.rowcol(x, y)
    if not g.contains_rc(r, c):
        return []
    pred = A["walk_any_pred_mid"] if st.any_route(cand["row"], cand["col"]) else A["walk_pred_mid"]
    path = T.trace_route(pred, g.shape, int(r), int(c))
    if len(path) < 2:
        return []
    return [g.to_lonlat(*g.xy(pr, pc)) for pr, pc in path[:: max(1, len(path) // 200)]] + [(cand["lon"], cand["lat"])]


class Overlay(NamedTuple):
    file: str  # PNG name inside the KMZ
    bounds: Bounds
    visible: bool


def kml_doc(result: Result, overlays: dict[str, Overlay]) -> str:
    st = result["state"]
    s = result["summary"]
    cands = result["candidates"]
    priv = result["private_candidates"]
    esc = html.escape
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f"<name>CougarMap - {esc(s['area'])}</name>",
        f"<description>{esc(_doc_description(s))}</description>",
        '<Style id="top"><IconStyle><color>ff0000ff</color><scale>1.3</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>',
        '<Style id="cand"><IconStyle><color>ff0080ff</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon></IconStyle></Style>',
        '<Style id="private"><IconStyle><color>ffff80ff</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/purple-circle.png</href></Icon></IconStyle></Style>',
        '<Style id="trail"><IconStyle><color>ff00c000</color><scale>0.9</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-diamond.png</href></Icon></IconStyle></Style>',
        '<Style id="worn_unmapped"><LineStyle><color>ff0060ff</color><width>2.5</width></LineStyle></Style>',
        '<Style id="worn_mapped"><LineStyle><color>a0c8c8c8</color><width>1.5</width></LineStyle></Style>',
        '<Style id="route"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle></Style>',
        '<Style id="arrow"><LineStyle><color>b0ffcc66</color><width>1.5</width></LineStyle></Style>',
        '<Style id="aoi"><LineStyle><color>ff2dc0fb</color><width>2</width></LineStyle>'
        "<PolyStyle><fill>0</fill></PolyStyle></Style>",
        '<Style id="saddle"><IconStyle><scale>0.6</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>',
    ]
    # AOI outline
    geoms = getattr(st.aoi.geom, "geoms", [st.aoi.geom])
    for g in geoms:
        if hasattr(g, "exterior"):
            coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in g.exterior.coords)
            parts.append(
                f"<Placemark><name>Area</name><styleUrl>#aoi</styleUrl><Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
            )
    # camera spots
    parts.append("<Folder><name>Camera spots (ranked)</name><open>1</open>")
    parts.extend(
        f"<Placemark><name>{esc(c['name'])} ({c['score']:.0f}) zone {c.get('zone', '')}</name>"
        f"<styleUrl>#{'top' if c['rank'] <= 5 else 'cand'}</styleUrl>"
        f"<description><![CDATA[{_desc(c)}]]></description>"
        f"<Point><coordinates>{c['lon']:.6f},{c['lat']:.6f},0</coordinates></Point></Placemark>"
        for c in cands
    )
    parts.append("</Folder>")
    # private-land spots: off by default, tick the folder to see them
    if priv:
        parts.append("<Folder><name>Private land spots (need landowner permission)</name><visibility>0</visibility>")
        parts.extend(
            f"<Placemark><name>{esc(c['name'])} ({c['score']:.0f})</name><visibility>0</visibility>"
            f"<styleUrl>#private</styleUrl><description><![CDATA[{_desc(c)}]]></description>"
            f"<Point><coordinates>{c['lon']:.6f},{c['lat']:.6f},0</coordinates></Point></Placemark>"
            for c in priv
        )
        parts.append("</Folder>")
    # the optional alternates on a quiet road or trail: off by default, never the pick
    alts = [(c, c["trail_alternate"]) for c in cands + priv if c.get("trail_alternate")]
    if alts:
        parts.append(
            "<Folder><name>Alternate spots on a trail/two-track (optional, not the pick)</name>"
            "<visibility>0</visibility>"
        )
        for c, a in alts:
            if a is None:
                continue
            parts.append(
                f"<Placemark><name>{esc(c['name'])} trail alt ({a['score']:.0f})</name><visibility>0</visibility>"
                f"<styleUrl>#trail</styleUrl><description><![CDATA[{_alt_desc(c, a)}]]></description>"
                f"<Point><coordinates>{a['lon']:.6f},{a['lat']:.6f},0</coordinates></Point></Placemark>"
            )
        parts.append("</Folder>")
    parts.extend(_worn_folder(st))
    # routes
    parts.append("<Folder><name>Walking routes from road</name><visibility>0</visibility>")
    for c in cands + priv:
        rt = _route(st, c)
        if rt:
            coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in rt)
            parts.append(
                f"<Placemark><name>Route to {esc(c['name'])}</name><visibility>0</visibility>"
                "<styleUrl>#route</styleUrl>"
                f"<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>"
            )
    parts.append("</Folder>")
    # airflow arrows
    parts.append("<Folder><name>Dawn/dusk air flow</name><visibility>0</visibility>")
    for arr in _arrows(st):
        coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in arr)
        parts.append(
            f"<Placemark><visibility>0</visibility><styleUrl>#arrow</styleUrl><LineString><tessellate>1</tessellate>"
            f"<coordinates>{coords}</coordinates></LineString></Placemark>"
        )
    parts.append("</Folder>")
    # saddles
    parts.append("<Folder><name>Saddles</name><visibility>0</visibility>")
    for sd in st.layers["saddle_points"]:
        lon, lat = st.mid.to_lonlat(sd["x"], sd["y"])
        parts.append(
            f"<Placemark><visibility>0</visibility><name>saddle {sd['rise_m']:.0f} m</name><styleUrl>#saddle</styleUrl>"
            f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>"
        )
    parts.append("</Folder>")
    # overlays
    parts.append("<Folder><name>Layers</name><open>1</open>")
    for i, (name, (fname, bnds, visible)) in enumerate(overlays.items()):
        w, s_, e, n = bnds
        parts.append(
            f"<GroundOverlay><name>{esc(name)}</name><visibility>{1 if visible else 0}</visibility>"
            f"<drawOrder>{i + 1}</drawOrder><Icon><href>{fname}</href></Icon>"
            f"<LatLonBox><north>{n:.7f}</north><south>{s_:.7f}</south><east>{e:.7f}</east><west>{w:.7f}</west></LatLonBox>"
            f"</GroundOverlay>"
        )
    parts.append("</Folder></Document></kml>")
    return "\n".join(parts)


def _worn_folder(st: ModelState) -> list[str]:
    """Worn trails from 1 m lidar, off by default: the ones on no map bright, those on a mapped road or trail grey."""
    lines = st.layers["worn_lines"]
    if not lines:
        return []
    parts = ["<Folder><name>Worn trails (lidar)</name><visibility>0</visibility>"]
    for mapped, name in ((False, "On no map (game trails, old tracks)"), (True, "On a mapped road or trail")):
        parts.append(f"<Folder><name>{name}</name><visibility>0</visibility>")
        for ln in lines:
            if ln["mapped"] is not mapped:
                continue
            coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in ln["lonlat"])
            parts.append(
                f"<Placemark><name>{ln['length_m']:.0f} m</name><visibility>0</visibility>"
                f"<styleUrl>#worn_{'mapped' if mapped else 'unmapped'}</styleUrl>"
                f"<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>"
            )
        parts.append("</Folder>")
    parts.append("</Folder>")
    return parts


def _common(w: dict[str, Any]) -> str:
    c = w.get("most_common_from")
    return f", most common {c}" if c and c != w["prevailing_from"] else ""


def _ground(w: dict[str, Any]) -> str:
    g = w.get("ground_level")
    if not g or not g.get("from_compass"):
        return ""
    return (
        f"Ground-level (HRRR 10 m) dawn/dusk wind for reference: from {g['from_compass']} "
        f"(R {g['consistency']:.2f}), not scored. "
    )


def _doc_description(s: dict[str, Any]) -> str:
    w = s["wind"]
    return (
        f"Month {s['month']}. Prevailing high-pressure dawn/dusk wind from {w['prevailing_from']} "
        f"(consistency R {w['consistency']:.2f}{_common(w)}; source: {w['source']}). "
        + _ground(w)
        + f"Resolution {s['resolution_m']:g} m, lidar {s['lidar_fraction']:.0%}. "
        "Spots on public land; private-land spots and score are in hidden layers. "
        f"Max walk {s['options']['max_walk_miles']} mi."
    )


def _percentile_display(a: Floats, mask: Mask, lo: float = 70) -> Floats:
    """Below the lo-th percentile (of the positive cells in mask) is transparent, the top few percent brightest."""
    v = a[mask & (a > 0)]
    if v.size == 0:
        return np.zeros_like(a)
    q = np.percentile(v, np.linspace(0, 100, 101))
    r = np.interp(a, q, np.linspace(0, 1, 101))
    return np.where(mask & (a > 0), np.clip((r - lo / 100) / (1 - lo / 100), 0, 1), 0).astype("float32")


def write_outputs(result: Result, out_dir: Path, log: Log = print, state: bool = True) -> dict[str, Path]:
    """KMZ, summary.json, candidates.geojson and (unless state=False, e.g. a re-pick of an unchanged analysis)
    state.pkl. Layers are warped here (GDAL threads each warp) while colorizing, PNG encoding and the state
    file run on a thread pool."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = result["state"]
    A = st.layers
    g = st.fine
    inside = st.aoi_mask
    pct = _percentile_display

    with ThreadPoolExecutor(4) as ex:
        state_job = ex.submit(save_state, st, out_dir / "state.pkl") if state else None
        layers: dict[str, tuple[Overlay, Future[bytes]]] = {}

        def add(
            name: str,
            arr: Floats,
            ramp: Floats = _RAMP,
            visible: bool = False,
            floor: float = 0.08,
            resampling: Resampling = Resampling.bilinear,
        ) -> None:
            ll, bnds = _to_wgs84(np.where(inside, arr, np.nan), g, resampling=resampling)
            layers[name] = (Overlay(f"layer_{len(layers)}.png", bnds, visible), ex.submit(_render, ll, ramp, floor))

        add("Lion score (usable ground, top 30%)", pct(A["final"], A["usable"]), visible=True, floor=0.02)
        add("Lion score on private land (top 30%)", pct(A["final_private"], A["usable_private"]), floor=0.02)
        add("Lion score (all ground, ignoring access/land rules)", pct(A["score"], inside), floor=0.02)
        add("Factor: wind", A["wind"])
        add("Factor: edges", A["edges"])
        add("Factor: pinch points", A["pinch"])
        add("Factor: limited water", A["water"], ramp=_BLUE)
        add("Hunting edge (timber edge and the open beside it; downwind ends brightest)", A["edge_meadow"])
        if "travel" not in st.unmodeled:
            add("Travel lines (drainage bottoms; ridge spines at crossings)", A["travel"])
        if "context" not in st.unmodeled:
            add("Lion habitat around (edge + water)", A["context"] / max(float(A["context"].max()), 1e-6))
        if float(A["season"].min()) < 1:  # winter months only
            add("Winter ground (low, sun-facing, shallow snow: where deer winter)", st.up(A["winter_mid"]))
        add(
            "Reachable within walk limit",
            (A["walk_m"] <= st.opts.max_walk_miles * MILE_M).astype("float32") * 0.35,
            ramp=_BLUE,
            floor=0.01,
            resampling=Resampling.nearest,
        )
        add("Public land", A["public"].astype("float32") * 0.3, ramp=_BLUE, floor=0.01, resampling=Resampling.nearest)
        add("Private land", (~A["public"]).astype("float32") * 0.3, floor=0.01, resampling=Resampling.nearest)

        kml = kml_doc(result, {k: ov for k, (ov, _) in layers.items()})
        kmz = out_dir / "cougarmap.kmz"
        with zipfile.ZipFile(kmz, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("doc.kml", kml)
            for ov, png in layers.values():
                z.writestr(ov.file, png.result())

        write_summary(result, out_dir)
        cands, priv = result["candidates"], result["private_candidates"]
        gj = dict(
            type="FeatureCollection",
            features=[
                dict(
                    type="Feature",
                    geometry=dict(type="Point", coordinates=[c["lon"], c["lat"]]),
                    properties={k: v for k, v in c.items() if k not in ("row", "col")},
                )
                for c in cands + priv
            ],
        )
        (out_dir / "candidates.geojson").write_text(json.dumps(gj, indent=2))
        if state_job is not None:
            state_job.result()
    log(f"wrote {kmz}")
    return dict(
        kmz=kmz, summary=out_dir / "summary.json", geojson=out_dir / "candidates.geojson", state=out_dir / "state.pkl"
    )


def write_summary(result: Result, out_dir: Path) -> Path:
    """summary.json: the summary and both spot lists."""
    path = Path(out_dir) / "summary.json"
    cands, priv = result["candidates"], result["private_candidates"]
    path.write_text(json.dumps(dict(summary=result["summary"], candidates=cands, private_candidates=priv), indent=2))
    return path


def merge_kmz(out: Path, title: str, parts: Sequence[tuple[str, Path]]) -> Path:
    """One KMZ holding several (each in its own folder b1/, b2/, ...), linked from a top document: one
    Google Earth file for a multi-block find_hotspots run. parts: (folder name, KMZ path)."""
    links = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (name, path) in enumerate(parts, 1):
            with zipfile.ZipFile(path) as src:
                for n in src.namelist():
                    z.writestr(f"b{i}/{n}", src.read(n))
            links.append(
                f"<NetworkLink><name>{html.escape(name)}</name>"
                f"<open>1</open><Link><href>b{i}/doc.kml</href></Link></NetworkLink>"
            )
        z.writestr(
            "doc.kml",
            '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
            f"<Document><name>{html.escape(title)}</name><open>1</open>" + "".join(links) + "</Document></kml>",
        )
    return out
