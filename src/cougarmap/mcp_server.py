"""MCP server exposing CougarMap to any MCP-capable agent (Claude Code, Claude Desktop/Cowork, Codex, Gemini CLI,
Antigravity, Cursor, ...). Run with `cougarmap-mcp` (stdio).

Long operations run as background jobs so no tool call outlives a harness's timeout: they wait up to
`wait_seconds`, then hand back a job_id to poll with job_status.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from importlib.resources import files
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from . import api, jobs

JSON = dict[str, Any]

PLAYBOOK = files("cougarmap").joinpath("PLAYBOOK.md").read_text()
# Short (< 2 KB) because some harnesses truncate server instructions; the full playbook is a tool.
INSTRUCTIONS = files("cougarmap").joinpath("INSTRUCTIONS.md").read_text()
# Some harnesses (Cursor) kill tool calls at ~60 s, others (Antigravity) at ~180 s: never block longer than this.
MAX_WAIT = 40

server = MCPServer("cougarmap", instructions=INSTRUCTIONS)
# Tools returning a JSON object answer with one JSON text block (structured_output=False): no output schema and
# no duplicate structured copy of results that run to tens of KB.


async def _bg[**P, R](fn: Callable[P, R], *a: P.args, **k: P.kwargs) -> R:
    """fn(*a, **k) on a worker thread, so a long call never blocks the server's event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *a, **k))


async def _job(kind: str, params: JSON, wait_seconds: float) -> JSON:
    job_id = jobs.start(kind, {k: v for k, v in params.items() if v is not None})
    return await _bg(jobs.status, job_id, min(max(wait_seconds, 0), MAX_WAIT))


# ---- the main entry point -------------------------------------------------------------------------------


@server.tool(structured_output=False)
async def find_hotspots(
    location: str | None = None,
    radius_km: float | None = None,
    kml: str | None = None,
    area_name: str | None = None,
    month: int | None = None,
    public_only: bool = True,
    max_walk_miles: float = 1.0,
    blocks: int = 3,
    wait_seconds: float = 30,
) -> JSON:
    """Find mountain lion hotspots and trail-camera spots. Use this for "find cougar hotspots near X".
    location: place name ("Missoula, MT") or "lat,lon". radius_km: search radius (default 25; <= 4 analyzes that
    circle directly, e.g. a property). Or kml + area_name for an area drawn in Google Earth.
    public_only=False includes private land (e.g. the user's own property) in the main list; with the default,
    private-land spots still come back separately (private_spots) and as hidden KMZ layers. month 1-12 (default: now).
    Runs in the background: if the result says state=running, call job_status(job_id) until done."""
    return await _job(
        "hotspots",
        dict(
            location=location,
            radius_km=radius_km,
            kml=kml,
            area_name=area_name,
            month=month,
            public_only=public_only,
            max_walk_miles=max_walk_miles,
            blocks=blocks,
        ),
        wait_seconds,
    )


@server.tool(structured_output=False)
async def job_status(job_id: str, wait_seconds: float = 30) -> JSON:
    """Check a background job (from find_hotspots / analyze_area / scout_region). Waits up to wait_seconds
    (max 40) for it to finish. state: queued | running | done (result included) | failed (error included)."""
    return await _bg(jobs.status, job_id, min(max(wait_seconds, 0), MAX_WAIT))


@server.tool()
async def list_jobs(limit: int = 10) -> list[JSON]:
    """Recent background jobs, newest first (use if you lost a job_id)."""
    return jobs.list_jobs(limit)


# ---- finer-grained operations ----------------------------------------------------------------------------


@server.tool(structured_output=False)
async def analyze_area(
    location: str | None = None,
    radius_km: float | None = None,
    kml: str | None = None,
    area_name: str | None = None,
    bbox: list[float] | None = None,
    month: int | None = None,
    public_only: bool = True,
    max_walk_miles: float = 1.0,
    wind_from_deg: float | None = None,
    n_candidates: int = 15,
    wait_seconds: float = 30,
) -> JSON:
    """Detailed analysis of ONE area (location+radius_km, kml+area_name, or bbox [west,south,east,north]) ->
    ranked camera spots with reasons + Google Earth KMZ. wind_from_deg overrides the prevailing wind (degrees
    it blows FROM). Background job: poll job_status if state=running."""
    return await _job(
        "analyze",
        dict(
            location=location,
            radius_km=radius_km,
            kml=kml,
            area_name=area_name,
            bbox=bbox,
            month=month,
            public_only=public_only,
            max_walk_miles=max_walk_miles,
            wind_from_deg=wind_from_deg,
            n_candidates=n_candidates,
        ),
        wait_seconds,
    )


@server.tool(structured_output=False)
async def scout_region(
    location: str,
    radius_km: float = 40.0,
    month: int | None = None,
    public_only: bool = True,
    max_walk_miles: float = 1.0,
    top: int = 8,
    wait_seconds: float = 30,
) -> JSON:
    """Coarse screen of a region -> ranked ~3 km blocks worth a detailed look (find_hotspots does this for you).
    Background job: poll job_status if state=running."""
    return await _job(
        "scout",
        dict(
            location=location,
            radius_km=radius_km,
            month=month,
            public_only=public_only,
            max_walk_miles=max_walk_miles,
            top=top,
        ),
        wait_seconds,
    )


@server.tool(structured_output=False)
async def repick(
    area: str,
    n_candidates: int = 15,
    per_zone: int = 3,
    spacing_m: float = 150.0,
    public_only: bool | None = None,
    max_walk_miles: float | None = None,
) -> JSON:
    """Re-select camera spots from an area already analyzed (more/fewer, more spread out with per_zone=1, a
    shorter walk limit, public_only on/off) and rewrite its KMZ. Takes seconds, no rerun, including switching
    private land in or out. area = the area's name or folder."""
    return await _bg(api.repick, area, n_candidates, per_zone, spacing_m, public_only, max_walk_miles)


@server.tool(structured_output=False)
async def explain_point(area: str, lat: float, lon: float, search_m: float = 25.0) -> JSON:
    """Factor-by-factor breakdown of why a spot scores the way it does (area must already be analyzed)."""
    return await _bg(api.explain_point, area, lat, lon, search_m)


@server.tool(structured_output=False)
async def wind_summary(location: str, month: int | None = None) -> JSON:
    """Prevailing high-pressure wind (dawn/dusk/day) for a place and month."""
    return await _bg(api.wind_summary, location, month)


@server.tool(structured_output=False)
async def validate(area: str, kml: str | None = None) -> JSON:
    """Test an analyzed area against the lion truth in the field log: camera detections per 100 camera-nights by
    arm (vs random on-trail cameras, and model vs control in the same zone), snow tracks vs shifted copies,
    crossing transects; plus how the human camera picks (CamNN pins in kml or the private KMLs) rank there."""
    return await _bg(api.validate, area, kml)


# ---- the field log: what lions actually did (see docs/FIELD_PROTOCOL.md) -------------------------------------


@server.tool(structured_output=False)
async def log_camera(
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
    """Record a camera put out: lat, lon, name, start date, arm (model = the tool's pick, human = a spot picked by hand,
    control = a nearby spot picked without the model, on-feature / off-feature, unpaired), zone (cameras compared with
    each other share one), trail_type (paved | open-dirt | closed-road | hiking-trail | game-trail | none), height_m,
    facing_deg, lure. deployment=<id> updates an existing one instead (e.g. end date when it comes down; end="" reopens
    one that is still out). For an "alternate on the trail" camera: arm on-feature, same zone as the pick's camera (arm
    model or human)."""
    return api.log_camera(
        lat,
        lon,
        name,
        arm,
        zone,
        start,
        end,
        trail_type,
        height_m,
        facing_deg,
        lure,
        downtime_nights,
        notes,
        deployment,
    )


@server.tool(structured_output=False)
async def log_check(
    deployment: str,
    date: str | None = None,
    events: list[JSON] | None = None,
    downtime_nights: float = 0.0,
    removed: bool = False,
    notes: str | None = None,
) -> JSON:
    """Record a camera check (deployment = its id from log_camera / field_log): events since the last check
    [{"datetime": "2026-11-03T05:40", "species": "cougar" | "deer" | "elk" | "other", "count": 1}], one per
    visit (a time with a zone is converted to local time); downtime_nights it wasn't recording; removed=True if it
    was taken down. No events = nothing came by. A check after the camera's end is refused (reopen it first)."""
    return api.log_check(deployment, date, events, downtime_nights, removed, notes)


@server.tool(structured_output=False)
async def log_track(
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
    """Record a lion track followed in snow or mud: file = a GPX/KML/KMZ from a phone GPS app, or points
    [[lat, lon], ...]; snow_age_h = hours since the snow fell; confidence certain | probable | possible."""
    return api.log_track(file, points, date, snow_age_h, snow_depth_cm, confidence, species, name, notes)


@server.tool(structured_output=False)
async def log_transect(
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
    """Record one survey of a fixed route (e.g. after fresh snow): route = its name; file = GPX/KML/KMZ of the
    route with a waypoint named "lion" at each lion crossing (other waypoints are ignored and listed back: ask the
    user about them), or line/crossings as [[lat, lon], ...]. A repeat survey of a known route can omit the line.
    Zero crossings is a valid (and useful) result. surface: snow | mud | dust."""
    return api.log_transect(route, file, line, crossings, date, surface, snow_age_h, species, notes)


@server.tool(structured_output=False)
async def field_log() -> JSON:
    """Everything logged in the field: cameras (ids, effort, detections so far), tracks and transect surveys."""
    return api.field_log()


@server.tool(structured_output=False)
async def log_result(
    lat: float,
    lon: float,
    lion_seen: bool,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    detections: int | None = None,
    times: list[str] | None = None,
    notes: str | None = None,
) -> JSON:
    """Quick record of what a camera caught (including nothing), when it isn't part of a designed test: saved as
    an unpaired camera, start through end (the last night covered). Prefer log_camera + log_check for cameras the
    user is still running."""
    return api.log_result(lat, lon, lion_seen, name, start, end, detections, times, notes)


@server.tool(structured_output=False)
async def list_areas(kml: str) -> JSON:
    """Named areas and pins (cameras, water, sign) in a Google Earth KML/KMZ."""
    return api.list_areas(kml)


@server.tool(structured_output=False)
async def import_kml(path: str) -> JSON:
    """Copy the user's Google Earth KML/KMZ into CougarMap's private folder so its water/sign pins are used."""
    return api.import_kml(path)


@server.tool(structured_output=False)
async def open_file(path: str) -> JSON:
    """Open a result file (e.g. the KMZ in Google Earth) with the default app."""
    return api.open_file(path)


@server.tool()
async def playbook() -> str:
    """How to use these tools and present results (read this first if your harness didn't show server
    instructions)."""
    return PLAYBOOK


@server.prompt()
def find_cougar_hotspots(place: str) -> str:
    """Find mountain lion hotspots and camera spots near a place."""
    return (
        f"Find mountain lion hotspots and good trail-camera spots near {place}. "
        f"Use the cougarmap tools (find_hotspots, then job_status until done) and report the best spots."
    )


def main() -> None:
    print("cougarmap MCP server ready", file=sys.stderr, flush=True)
    server.run("stdio")


if __name__ == "__main__":
    main()
