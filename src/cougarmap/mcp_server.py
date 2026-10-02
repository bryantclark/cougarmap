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
    bbox: list[float] | None = None,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    wind_from_deg: float | None = None,
    blocks: int = 3,
    wait_seconds: float = 30,
) -> JSON:
    """Find mountain lion hotspots and trail-camera spots: ranked spots with reasons and a Google Earth KMZ.
    Where: location (place name like "Missoula, MT", or "lat,lon") with radius_km (default 25: scouts the region
    and analyzes the best `blocks`; <= 4 analyzes that circle directly, e.g. a property), or kml + area_name (an
    area drawn in Google Earth), or bbox [west, south, east, north]. Spots are on public land within
    max_walk_miles of a road open that month; private-land spots come back separately (private_candidates).
    month 1-12 (default: now). wind_from_deg overrides the modeled prevailing wind (degrees it blows FROM).
    Runs in the background: if the result says state=running, call job_status(job_id) until done."""
    return await _job(
        "hotspots",
        dict(
            location=location,
            radius_km=radius_km,
            kml=kml,
            area_name=area_name,
            bbox=bbox,
            month=month,
            max_walk_miles=max_walk_miles,
            wind_from_deg=wind_from_deg,
            blocks=blocks,
        ),
        wait_seconds,
    )


@server.tool(structured_output=False)
async def job_status(job_id: str | None = None, wait_seconds: float = 30) -> JSON:
    """Check a background job from find_hotspots. Waits up to wait_seconds (max 40) for it to finish. state:
    queued | running | done (result included) | failed (error included). Without job_id: the recent jobs, newest
    first (if you lost the id)."""
    if job_id is None:
        return dict(jobs=jobs.list_jobs(10))
    return await _bg(jobs.status, job_id, min(max(wait_seconds, 0), MAX_WAIT))


# ---- follow-ups on an analyzed area ----------------------------------------------------------------------------


@server.tool(structured_output=False)
async def repick(
    area: str,
    n_candidates: int = 15,
    per_zone: int = 3,
    spacing_m: float = 150.0,
    max_walk_miles: float | None = None,
) -> JSON:
    """Re-select camera spots from an area already analyzed (more/fewer, more spread out with per_zone=1, a
    shorter walk limit) and rewrite its KMZ. Takes seconds, no rerun. area = the area's name or folder."""
    return await _bg(api.repick, area, n_candidates, per_zone, spacing_m, max_walk_miles)


@server.tool(structured_output=False)
async def explain_point(area: str, lat: float, lon: float, search_m: float = 25.0) -> JSON:
    """Factor-by-factor breakdown of why a spot scores the way it does (area must already be analyzed)."""
    return await _bg(api.explain_point, area, lat, lon, search_m)


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
async def share_results(name: str, out: str | None = None) -> JSON:
    """Write the user's field log to one file to send to someone they trust (e.g. "send my results to Sam"), who
    adds it with `cougarmap import-results`. name: whose results these are. The file holds camera locations: tell
    the user to send it privately (email or a message), never to post it."""
    return await _bg(api.share_results, name, out)


@server.tool(structured_output=False)
async def import_kml(path: str) -> JSON:
    """Copy the user's Google Earth KML/KMZ into CougarMap's private folder, so its water and sign pins are used,
    and list its named areas (for find_hotspots(kml=..., area_name=...)) and pins. Importing the same file again
    just lists them."""
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
