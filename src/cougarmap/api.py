"""The agent-facing operations. The CLI and the MCP server are thin wrappers over these functions; every
function returns plain JSON-able dicts."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import aoi as aoi_mod
from . import context, fieldlog
from .analyze import Result, Spot, apply_masks, describe, pick_all, run, slug, summarize
from .config import OBSERVATIONS_FILE, OUT_DIR, PRIVATE_DIR, TRUTH, Options
from .context import Log
from .evaluate import cameras
from .export import merge_kmz, write_outputs
from .state import ModelState, load_state, save_state, save_state_opts
from .truth import area_truth

JSON = dict[str, Any]  # what every operation returns


def _opts(
    month: int | None = None,
    max_walk_miles: float = 1.0,
    wind_from_deg: float | None = None,
    n_candidates: int = 15,
    fast: bool = False,
) -> Options:
    return Options(
        month=month,
        max_walk_miles=max_walk_miles,
        wind_from_deg=wind_from_deg,
        n_candidates=n_candidates,
        worn_trails=not fast,
    )


def _known_points(log: Log = print) -> list[aoi_mod.UserPoint]:
    """User pins (water, sign, cameras) from every KML/KMZ in the private data folder."""
    pts: list[aoi_mod.UserPoint] = []
    if PRIVATE_DIR.exists():
        for p in sorted([*PRIVATE_DIR.glob("*.kml"), *PRIVATE_DIR.glob("*.kmz")]):
            try:
                pts += aoi_mod.user_points_from_kml(p)
            except (OSError, ET.ParseError, zipfile.BadZipFile, StopIteration, ValueError) as e:
                log(f"  skipped pins in {p.name}: {e}")
    return pts


def resolve_area(
    location: str | None = None,
    radius_km: float | None = None,
    kml: str | None = None,
    area_name: str | None = None,
    bbox: list[float] | None = None,
) -> aoi_mod.AOI:
    if kml:
        return aoi_mod.from_kml(kml, area_name)
    if bbox:
        west, south, east, north = bbox
        return aoi_mod.bbox(west, south, east, north)
    if location:
        lat, lon, label = aoi_mod.parse_location(location)
        return aoi_mod.circle(
            lat, lon, radius_km or 3.0, name=area_name or f"{label.split(',')[0]} r{radius_km or 3:g}km"
        )
    raise ValueError("give a location (place or 'lat,lon'), a kml path, or a bbox")


def list_areas(kml: str) -> JSON:
    areas = aoi_mod.areas_in_kml(kml)
    pts = aoi_mod.user_points_from_kml(kml)
    return dict(
        areas=[
            dict(name=a.name, folder=a.folder, area_km2=round(aoi_mod.AOI(a.name, a.geom).area_km2(), 1)) for a in areas
        ],
        points=[
            dict(name=p["name"], kind=p["kind"], lat=round(p["lat"], 6), lon=round(p["lon"], 6), note=p.get("note", ""))
            for p in pts
        ],
    )


def analyze_area(
    location: str | None = None,
    radius_km: float | None = None,
    kml: str | None = None,
    area_name: str | None = None,
    bbox: list[float] | None = None,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    wind_from_deg: float | None = None,
    n_candidates: int = 15,
    log: Log = print,
    user_pins: bool = True,
    fast: bool = False,
    interactive: bool = False,
) -> JSON:
    """user_pins=False ignores the water and sign pins in the private KML files (validation reruns, when the pins
    were placed by the same person who chose the human-picked cameras). fast=True skips the slow extras: the worn
    trails from 1 m lidar (a hidden KMZ layer and each spot's worn_trail hint; they change no score).
    interactive=True also writes explore.html (summary outputs "explore"): a local page with a weight slider per
    factor that moves the top spots live."""
    a = resolve_area(location, radius_km, kml, area_name, bbox)
    known = _known_points(log) if user_pins else []
    a.user_points = [p for p in known if p["kind"] in ("water", "seasonal_water", "sign")]
    res = run(a, _opts(month, max_walk_miles, wind_from_deg, n_candidates, fast), log=log)
    if interactive:
        _explore_page(res["state"], res["candidates"], res["private_candidates"], res["summary"], log)
    return dict(
        summary=res["summary"],
        candidates=_strip(res["candidates"]),
        private_candidates=_strip(res["private_candidates"]),
    )


def _explore_page(st: ModelState, cands: list[Spot], private: list[Spot], summary: JSON, log: Log) -> None:
    """Write the interactive weights page next to the KMZ and list it in the summary's outputs."""
    from .explore import write_page

    path = write_page(st, cands, private, Path(summary["outputs"]["kmz"]).parent, log)
    summary["outputs"]["explore"] = str(path)
    log(f"wrote {path}")


def _strip(cands: list[Spot]) -> list[JSON]:
    """Spots as JSON, without their grid cell (internal)."""
    return [{k: v for k, v in c.items() if k not in ("row", "col")} for c in cands]


def _state_for(area: str) -> Path:
    """The saved state of an analyzed area: its name, its folder, or its state.pkl. Only files inside the results
    folder (OUT_DIR) are ever read: a state file is a pickle, and unpickling runs code."""
    root = OUT_DIR.resolve()
    p = Path(area).expanduser()
    for cand in (p, p / "state.pkl"):
        if cand.is_file():
            if not cand.resolve().is_relative_to(root):
                raise ValueError(f"{area!r} is outside the results folder {OUT_DIR}: give the area's name instead")
            return cand
    cand = OUT_DIR / slug(area) / "state.pkl"
    if cand.exists():
        return cand
    runs = sorted(OUT_DIR.glob("*/state.pkl"), key=lambda q: q.stat().st_mtime, reverse=True)
    for q in runs:
        if slug(area) in q.parent.name:
            return q
    raise FileNotFoundError(
        f"no saved analysis for {area!r}; run analyze first. Available: {[q.parent.name for q in runs]}"
    )


class _StateCache:
    """The most recently loaded saved state, kept in memory so follow-up calls on the same area (explain_point,
    repick, validate) in a long-running process such as the MCP server skip re-reading it (seconds on a big area).
    It is reloaded when the file changes on disk and dropped after IDLE_S without use. One operation at a time
    uses it (they adjust its options and masks in place)."""

    IDLE_S = 900.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key: tuple[str, int, int] | None = None
        self._state: ModelState | None = None
        self._timer: threading.Timer | None = None

    @staticmethod
    def _stat(path: Path) -> tuple[str, int, int]:
        st = path.stat()
        return (str(path.resolve()), st.st_mtime_ns, st.st_size)

    @contextlib.contextmanager
    def use(self, path: Path) -> Iterator[ModelState]:
        with self._lock:
            key = self._stat(path)
            if key != self._key or self._state is None:
                self._state = self._key = None  # free the old state before loading the next
                self._state, self._key = load_state(path), key
            try:
                yield self._state
            except BaseException:
                # the operation may have changed the state's options or masks in place without saving them
                self._state = self._key = None
                raise
            finally:
                self._arm()

    def saved(self, path: Path) -> None:
        """The in-memory state was just written to path (it still matches the file)."""
        with self._lock:
            self._key = self._stat(path)

    def clear(self) -> None:
        with self._lock:
            self._state = self._key = None

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.IDLE_S, self.clear)
        self._timer.daemon = True
        self._timer.start()


_STATES = _StateCache()


def explain_point(area: str, lat: float, lon: float, search_m: float = 25.0) -> JSON:
    """Why a spot scores the way it does (uses the best cell within search_m)."""
    with _STATES.use(_state_for(area)) as st:
        return _explain(st, lat, lon, search_m)


def _explain(st: ModelState, lat: float, lon: float, search_m: float) -> JSON:
    apply_masks(st)
    A = st.layers
    g = st.fine
    cell = g.cell(lon, lat)
    if cell is None:
        return dict(error="point is outside the analyzed area")
    r, c = cell
    k = max(0, int(search_m / g.res))
    r0, c0 = max(r - k, 0), max(c - k, 0)
    win = A["score"][r0 : r + k + 1, c0 : c + k + 1]
    dr, dc = np.unravel_index(np.argmax(win), win.shape)
    br, bc = r0 + int(dr), c0 + int(dc)

    def private_only(rr: int, cc: int) -> bool:  # private ground is on the private-land layer
        return bool(A["usable_private"][rr, cc] and not A["usable"][rr, cc])

    best = describe(st, br, bc, A["final_private"] if private_only(br, bc) else None)
    at = describe(st, r, c, A["final_private"] if private_only(r, c) else None)
    sc = A["score"][st.aoi_mask]
    d = _strip([best])[0]
    d["percentile_in_area"] = round(float((sc < best["raw_score"]).mean() * 100), 1)
    d["usable"] = bool(A["usable"][br, bc])
    d["on_private_layer"] = private_only(br, bc)
    return dict(best_nearby=d, exact_point=_strip([at])[0], search_m=search_m)


def validate(area: str, kml: str | None = None) -> JSON:
    """Every lion-truth test the field log allows for an analyzed area (cameras by arm and model-vs-control,
    snow tracks vs shifted copies, crossing transects), plus how the human camera picks rank there. kml: the
    file holding the picks (CamNN pins); default: the KML/KMZ files in the private data folder."""
    if kml:
        pins = cameras(Path(kml))
    else:
        pins = [dict(name=p["name"], lat=p["lat"], lon=p["lon"]) for p in _known_points() if p["kind"] == "camera"]
    records = fieldlog.load()
    with _STATES.use(_state_for(area)) as st:
        return area_truth(st, records, pins)


def read_observations() -> list[JSON]:
    """Every field-log record (schema version 2; older lines migrated)."""
    return [dict(r) for r in fieldlog.load()]


def log_camera(
    lat: float | None = None,
    lon: float | None = None,
    name: str | None = None,
    arm: str | None = None,
    zone: str | None = None,
    start: str | None = None,
    end: str | None = None,
    trail_type: str | None = None,
    height_m: float | None = None,
    facing_deg: float | None = None,
    lure: bool | None = None,
    downtime_nights: float | None = None,
    notes: str | None = None,
    deployment: str | None = None,
) -> JSON:
    """Record a camera put out (or, with deployment=<id>, update one: e.g. its end date; end="" reopens one that
    is still out). arm: model | human |
    control | on-feature | off-feature | unpaired; cameras sharing a zone are compared with each other.
    trail_type: paved | open-dirt | closed-road | hiking-trail | game-trail | none; the result carries a warning
    while it is missing (placement decides which cameras validate can compare)."""
    recs = fieldlog.load()
    d = fieldlog.new_deployment(
        recs,
        lat,
        lon,
        name,
        arm,
        zone,
        start,
        end,
        downtime_nights,
        trail_type,
        height_m,
        facing_deg,
        lure,
        notes,
        deployment,
    )
    total = fieldlog.append([d])
    merged = fieldlog.deployments([*recs, d])[d["id"]]
    out: JSON = dict(saved=True, updated=deployment is not None, deployment=merged, total=total)
    if not merged["trail_type"]:
        out["warning"] = (
            "no trail_type: ask what the camera watches (game-trail, hiking-trail, closed-road, open-dirt, paved or "
            f'none) and set it with log_camera(deployment="{merged["id"]}", trail_type=...). On-trail cameras catch '
            "about 3x more lions, so validate compares a camera with the on-trail base rate, and with the other "
            "camera in its zone, only when its placement is known."
        )
    return out


def log_check(
    deployment: str,
    date: str | None = None,
    events: list[JSON] | None = None,
    downtime_nights: float = 0.0,
    removed: bool = False,
    notes: str | None = None,
) -> JSON:
    """Record a camera check: what it caught since the last check (events: [{datetime, species, count}],
    species cougar | deer | elk | other, one entry per visit), nights it was not recording, and whether it was
    taken down (removed=True ends the deployment on that date). No events = it caught nothing."""
    recs = fieldlog.load()
    new = fieldlog.new_check(recs, deployment, date, downtime_nights, removed, events, notes)
    total = fieldlog.append(new)
    cam = next(c for c in fieldlog.cameras([*recs, *new], TRUTH.independent_min) if c.dep["id"] == deployment)
    return dict(
        saved=True,
        deployment=deployment,
        events_logged=len(new) - 1,
        camera_nights=round(cam.nights, 1),
        detections=cam.detections,
        total=total,
    )


def log_track(
    file: str | None = None,
    points: list[list[float]] | None = None,
    date: str | None = None,
    snow_age_h: float | None = None,
    snow_depth_cm: float | None = None,
    confidence: str = "probable",
    species: str = "cougar",
    name: str | None = None,
    notes: str | None = None,
) -> JSON:
    """Record a lion track followed in snow or mud: a GPX/KML/KMZ file from a phone GPS app (its track, or its
    waypoints in order) and/or points [[lat, lon], ...]. confidence: certain | probable | possible (possible
    tracks are kept but left out of the test)."""
    recs = fieldlog.load()
    t = fieldlog.new_track(recs, file, points, date, snow_age_h, snow_depth_cm, confidence, species, name, notes)
    total = fieldlog.append([t])
    km = sum(fieldlog.line_length_m(ln) for ln in t["lines"]) / 1000
    return dict(saved=True, id=t["id"], date=t["date"], lines=len(t["lines"]), km=round(km, 2), total=total)


def log_transect(
    route: str,
    file: str | None = None,
    line: list[list[float]] | None = None,
    crossings: list[list[float]] | None = None,
    date: str | None = None,
    surface: str = "snow",
    snow_age_h: float | None = None,
    species: str = "cougar",
    notes: str | None = None,
) -> JSON:
    """Record one survey of a fixed route (walked or driven after fresh snow, or on mud/dust): the route from a
    GPX/KML/KMZ file or line [[lat, lon], ...] (a route surveyed before can leave both out), and every place a
    lion trail crossed it (the file's waypoints named "lion" and/or crossings [[lat, lon], ...]; other waypoints
    such as parking or start are left out and listed back as ignored_waypoints). No crossings is a result."""
    recs = fieldlog.load()
    t, ignored = fieldlog.new_transect(recs, route, file, line, crossings, date, surface, snow_age_h, species, notes)
    total = fieldlog.append([t])
    km = sum(fieldlog.line_length_m(ln) for ln in t["lines"]) / 1000
    out: JSON = dict(saved=True, id=t["id"], route=route, km=round(km, 2), crossings=len(t["crossings"]), total=total)
    if ignored:
        out["ignored_waypoints"] = ignored
        out["note"] = (
            f"{len(ignored)} waypoint(s) in the file were not counted as crossings because their name doesn't say "
            f"lion ({', '.join(ignored[:5])}{', ...' if len(ignored) > 5 else ''}). If any was a lion crossing, "
            "rename it 'lion' in the GPS app and log the survey again."
        )
    return out


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cougarmap")
    except PackageNotFoundError:  # a source tree that was never installed
        return "unknown"


def share_results(name: str, out: str | None = None) -> JSON:
    """Write the user's own field log (cameras, checks, detections, tracks, transect surveys) to one file for
    someone they trust, who adds it to theirs with import_results. name: whose results these are. The file holds
    camera locations: send it privately (email, a message), never publish it."""
    doc = fieldlog.share(fieldlog.load(), name, _version())
    recs = doc["records"]
    path = (
        Path(out).expanduser()
        if out
        else PRIVATE_DIR / f"cougarmap-field-log-{doc['shared_by']}-{dt.date.today().isoformat()}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    kinds = {k: sum(r["type"] == k for r in recs) for k in ("deployment", "check", "event", "track", "transect")}
    return dict(
        file=str(path),
        shared_by=doc["shared_by"],
        cameras=len(fieldlog.deployments(recs)),
        records=kinds,
        note="This file has your camera locations. Send it privately (email or a message) to someone you trust; "
        "they add it with `cougarmap import-results <file>`. Don't post it anywhere public.",
    )


def import_results(path: str, name: str | None = None) -> JSON:
    """Add someone's shared field log (from share_results) to this one. Their ids, zones and routes get their name
    as a prefix ("sam/M1"); a newer file from the same person replaces what was imported from them before. name
    overrides the name in the file."""
    doc = json.loads(Path(path).expanduser().read_text())
    tag = fieldlog.share_tag(name or str(doc.get("shared_by") or ""))
    recs = fieldlog.from_share(doc, tag)
    replaced, total = fieldlog.replace_shared(tag, recs)
    return dict(
        shared_by=tag,
        imported=len(recs),
        replaced=replaced,
        cameras=len(fieldlog.deployments(recs)),
        total_records=total,
        note=f"Their cameras show up as {tag}/<id> in field_log and count in validate.",
    )


def field_log() -> JSON:
    """What's in the field log: cameras (effort and detections so far), tracks and transect surveys."""
    recs = fieldlog.load()
    cams = [
        dict(
            id=c.dep["id"],
            name=c.dep["name"],
            arm=c.dep["arm"],
            zone=c.dep["zone"],
            lat=c.dep["lat"],
            lon=c.dep["lon"],
            start=c.dep["start"],
            end=c.dep["end"],
            last_check=c.last_check,
            camera_nights=round(c.nights, 1),
            detections=c.detections,
        )
        for c in fieldlog.cameras(recs, TRUTH.independent_min)
    ]
    tracks = [
        dict(
            id=r["id"],
            date=r["date"],
            confidence=r["confidence"],
            species=r["species"],
            km=round(sum(fieldlog.line_length_m(ln) for ln in r["lines"]) / 1000, 2),
        )
        for r in recs
        if r["type"] == "track"
    ]
    transects = [
        dict(
            id=r["id"],
            route=r["route"],
            date=r["date"],
            surface=r["surface"],
            crossings=len(r["crossings"]),
            km=round(sum(fieldlog.line_length_m(ln) for ln in r["lines"]) / 1000, 2),
        )
        for r in recs
        if r["type"] == "transect"
    ]
    return dict(file=str(OBSERVATIONS_FILE), cameras=cams, tracks=tracks, transects=transects)


def scout_region(
    location: str,
    radius_km: float = 40.0,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    top: int = 8,
    log: Log = print,
) -> JSON:
    from .scout import scout

    lat, lon, label = aoi_mod.parse_location(location)
    r = scout(lat, lon, radius_km, month, max_walk_miles, top=top, log=log)
    r["location"] = label
    return r


def repick(
    area: str,
    n_candidates: int = 15,
    per_zone: int = 3,
    spacing_m: float = 150.0,
    max_walk_miles: float | None = None,
    interactive: bool = False,
) -> JSON:
    """Re-select camera spots from a saved analysis (e.g. more spread out, more spots) and rewrite the KMZ,
    without recomputing. max_walk_miles re-applies the walk rule. interactive=True also writes the weights page
    (explore.html, see analyze_area) from the saved state."""
    path = _state_for(area)
    with _STATES.use(path) as st:
        o = st.opts
        o.n_candidates, o.per_zone, o.candidate_spacing_m = n_candidates, per_zone, spacing_m
        if max_walk_miles is not None:
            o.max_walk_miles = max_walk_miles
        apply_masks(st)
        cands, priv = pick_all(st)
        summary = summarize(st, cands, 0.0, priv)
        result = Result(summary=summary, candidates=cands, private_candidates=priv, state=st)
        # The layers are unchanged: only the options are stored (states in the old format are rewritten once).
        paths = write_outputs(result, path.parent, log=lambda *_: None, state=False)
        if not save_state_opts(path, o):
            save_state(st, path)
        _STATES.saved(path)
        summary["outputs"] = {k: str(v) for k, v in paths.items()}
        if interactive:
            _explore_page(st, cands, priv, summary, lambda *_: None)
    return dict(summary=summary, candidates=_strip(cands), private_candidates=_strip(priv))


def find_hotspots(
    location: str | None = None,
    radius_km: float | None = None,
    kml: str | None = None,
    area_name: str | None = None,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    blocks: int = 3,
    bbox: list[float] | None = None,
    wind_from_deg: float | None = None,
    log: Log = print,
    fast: bool = False,
    interactive: bool = False,
) -> JSON:
    """The one-call answer to "find cougar hotspots near X": scout the region, analyze the best blocks in detail,
    merge into one ranked list and one Google Earth file. A small radius (<= 4 km), a KML area or a bbox
    [west, south, east, north] is analyzed directly, without scouting. wind_from_deg overrides the prevailing wind
    (degrees it blows FROM). fast=True skips the slow extras and interactive=True writes the weights page for each
    analyzed block (see analyze_area); the result's "explore" lists the pages."""
    if kml or bbox or (location and radius_km is not None and radius_km <= 4):
        r = analyze_area(
            location,
            radius_km,
            kml,
            area_name,
            bbox,
            month,
            max_walk_miles,
            wind_from_deg,
            log=log,
            fast=fast,
            interactive=interactive,
        )
        r["how"] = "analyzed the whole area in detail"
        if interactive:
            r["explore"] = [r["summary"]["outputs"]["explore"]]
        return r
    if not location:
        raise ValueError("give a location (place name or 'lat,lon') or a kml + area_name")
    radius_km = radius_km or 25.0
    log(f"scouting {radius_km:g} km around {location}...")
    sc = scout_region(location, radius_km, month, max_walk_miles, top=max(blocks, 3), log=log)
    picks = sc["blocks"][:blocks]
    if not picks:
        return dict(
            location=sc.get("location"),
            error="no promising public, road-accessible blocks found; try a larger radius",
            scout=sc,
        )
    runs = []
    # Download every block's data at once (network-bound on a first run); analyze each as its data arrives.
    with ThreadPoolExecutor(len(picks)) as ex:
        opts = _opts(month, max_walk_miles, n_candidates=8, fast=fast)
        fetched: list[Future[None]] = [
            ex.submit(context.prefetch, resolve_area(bbox=b["bbox"]), opts, lambda *_: None) for b in picks
        ]
        for i, (b, fut) in enumerate(zip(picks, fetched, strict=True), 1):
            with contextlib.suppress(Exception):  # the analysis retries anything that failed, and reports it
                fut.result()
            log(f"detailed analysis of block {i}/{len(picks)} ({b['land']})...")
            r = analyze_area(
                bbox=b["bbox"],
                month=month,
                max_walk_miles=max_walk_miles,
                wind_from_deg=wind_from_deg,
                n_candidates=8,
                area_name=f"{sc['location'].split(',')[0]} block {i}",
                log=log,
                fast=fast,
                interactive=interactive,
            )
            runs.append((b, r))

    place = sc["location"].split(",")[0]
    out = OUT_DIR / f"hotspots-{slug(place)}-r{radius_km:g}"
    out.mkdir(parents=True, exist_ok=True)
    kmz = merge_kmz(
        out / "hotspots.kmz",
        f"Cougar hotspots near {place}",
        [
            (f"Block {i}: {b['land']} (scout score {b['score']:.0f})", Path(r["summary"]["outputs"]["kmz"]))
            for i, (b, r) in enumerate(runs, 1)
        ],
    )

    spots = [
        dict(c, block=i, name=f"Block {i} {c['name']}") for i, (_, r) in enumerate(runs, 1) for c in r["candidates"]
    ]
    spots.sort(key=lambda c: -c["score"])
    priv = sorted(
        (
            dict(c, block=i, name=f"Block {i} {c['name']}")
            for i, (_, r) in enumerate(runs, 1)
            for c in r.get("private_candidates", [])
        ),
        key=lambda c: -c["score"],
    )
    result = dict(
        location=sc["location"],
        radius_km=radius_km,
        month=sc["month"],
        max_walk_miles=max_walk_miles,
        wind=sc["wind"],
        blocks=[
            dict(
                block=i,
                land=b["land"],
                scout_score=b["score"],
                why=b["why"],
                bbox=b["bbox"],
                center=b["center"],
                best_spot_score=r["candidates"][0]["score"] if r["candidates"] else None,
                kmz=r["summary"]["outputs"]["kmz"],
                **({"explore": r["summary"]["outputs"]["explore"]} if interactive else {}),
            )
            for i, (b, r) in enumerate(runs, 1)
        ],
        top_spots=spots[:12],
        private_spots=priv[:5],
        kmz=str(kmz),
        how=f"scouted {radius_km:g} km around {place}, analyzed the best {len(runs)} ~3 km blocks in detail",
    )
    if interactive:
        result["explore"] = [r["summary"]["outputs"]["explore"] for _, r in runs]
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def open_file(path: str) -> JSON:
    """Open a file (e.g. a KMZ in Google Earth) with the computer's default app."""
    p = Path(path)
    if not p.exists():
        return dict(opened=False, error=f"not found: {path}")
    if sys.platform == "win32":
        os.startfile(str(p))
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
    return dict(opened=True, path=str(p))


def import_kml(path: str) -> JSON:
    """Copy a Google Earth KML/KMZ into the private data folder so its areas and pins (water, sign, cameras)
    are used automatically, and list them."""
    src = Path(path).expanduser()
    if not src.exists():
        return dict(imported=False, error=f"not found: {path}")
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    dst = PRIVATE_DIR / src.name
    if not (dst.exists() and dst.samefile(src)):  # importing a file already there just lists it again
        shutil.copy2(src, dst)
    return dict(imported=True, path=str(dst), **list_areas(str(dst)))
