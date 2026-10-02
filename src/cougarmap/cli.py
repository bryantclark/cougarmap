"""Command line: `cougarmap hotspots|analyze|scout|explain|validate|wind|areas` and the field log
(`log-camera|log-check|log-track|log-transect|field-log`, `log` for a quick result). Every command prints JSON;
the ones that make a map also print a link to it, and at a terminal open it in Google Earth."""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from typing import Any

import typer

from . import api, jobs

EXAMPLES = """Examples:

\b
  cougarmap hotspots "Missoula, MT"
  cougarmap hotspots 47.3712, -116.1029 --radius-km 30
  cougarmap wind "Missoula, MT" --month 11
  cougarmap open <the .kmz path it prints>

Coordinates are latitude, longitude (west longitudes are negative, as Google Maps copies them)."""

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"Find mountain lion camera spots (wind, pinch points, edges, water).\n\n{EXAMPLES}",
    rich_markup_mode=None,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _map(result: dict[str, Any], open_it: bool) -> None:
    """After the JSON: a clickable link to the Google Earth file, and (for a person at a terminal) open it."""
    kmz = result.get("kmz") or result.get("summary", {}).get("outputs", {}).get("kmz")
    if not kmz:
        return
    from pathlib import Path

    _log(f"\nmap: {Path(kmz).resolve().as_uri()}")
    if open_it and sys.stdout.isatty() and sys.stderr.isatty():
        api.open_file(kmz)


def _note_fixes(location: str | None) -> None:
    """Says so when a coordinate gets corrected (see aoi.parse_coords)."""
    from .aoi import parse_coords

    c = parse_coords(location) if location else None
    if c and c[2]:
        _log(f"note: {c[2]}")


@app.command()
def hotspots(
    location: str | None = typer.Argument(None, help="place name, or coordinates: 47.37, -116.10"),
    near: str | None = typer.Option(None, "--near", hidden=True),
    radius_km: float | None = typer.Option(None, help="search radius (default 25; <=4 = that circle)"),
    kml: str | None = typer.Option(None, help="Google Earth KML/KMZ with the area outline (instead of a place)"),
    area: str | None = typer.Option(None, help="the area's name inside --kml"),
    month: int | None = typer.Option(None, help="1-12 (default this month): wind, snow and open roads change"),
    max_walk_miles: float = typer.Option(1.0, help="farthest walk from a road open that month"),
    blocks: int = typer.Option(3, help="how many ~3 km blocks to analyze in detail"),
    background: bool = typer.Option(False, help="start as a background job and print its id"),
    open_map: bool = typer.Option(True, "--open/--no-open", help="open the map in Google Earth when done"),
) -> None:
    """Find cougar hotspots near a place (scout + detailed analysis + one KMZ). The main command."""
    location = location or near
    _note_fixes(location)
    if background:
        params = dict(
            location=location,
            radius_km=radius_km,
            kml=kml,
            area_name=area,
            month=month,
            max_walk_miles=max_walk_miles,
            blocks=blocks,
        )
        jid = jobs.start("hotspots", {k: v for k, v in params.items() if v is not None})
        _out(dict(job_id=jid, state="running", next=f"cougarmap job {jid} --wait 45"))
    else:
        r = api.find_hotspots(location, radius_km, kml, area, month, max_walk_miles, blocks, log=_log)
        _out(r)
        _map(r, open_map)


@app.command()
def job(job_id: str, wait: float = typer.Option(0, help="seconds to wait for it to finish")) -> None:
    """Check a background job."""
    _out(jobs.status(job_id, wait))


@app.command("jobs")
def jobs_cmd(limit: int = 10) -> None:
    """List recent background jobs."""
    _out(jobs.list_jobs(limit))


@app.command("import-kml")
def import_kml_cmd(path: str) -> None:
    """Copy a Google Earth KML/KMZ into CougarMap's private folder (its pins are then used automatically)."""
    _out(api.import_kml(path))


@app.command("open")
def open_cmd(path: str) -> None:
    """Open a result in its app (a KMZ opens in Google Earth)."""
    _out(api.open_file(path))


@app.command()
def setup(
    only: list[str] | None = typer.Option(None, help="just these (e.g. --only codex --only cursor)"),
    all_: bool = typer.Option(False, "--all", help="set up every supported app, even ones not detected"),
    dry_run: bool = typer.Option(False, help="show what would change"),
    uninstall: bool = typer.Option(False, help="remove CougarMap from the AI apps"),
) -> None:
    """Connect CougarMap to the AI agent apps on this computer (Claude Code/Desktop/Cowork, Codex, Gemini CLI,
    Antigravity, Cursor)."""
    from .setup_harnesses import run

    r = run(only, all_, dry_run, uninstall)
    for name, msgs in r["harnesses"].items():
        print(f"- {name}:")
        for m in [msgs] if isinstance(msgs, str) else msgs:
            print(f"    {m}")
    print(f"- shared skill: {r['shared_skill']}")
    print(f"\nMCP server command: {' '.join(r['server_command'])}")
    print(f"Your results and private files go in: {r['data_folder']}")
    if not uninstall and not dry_run:
        print('\nRestart your AI apps, then ask: "Find me the cougar hotspots near <place>".')


@app.command()
def playbook() -> None:
    """Print the agent playbook (how to use CougarMap and report results)."""
    from importlib.resources import files

    print(files("cougarmap").joinpath("PLAYBOOK.md").read_text())


@app.command()
def analyze(
    location: str | None = typer.Option(None, "--near", help="place name or 'lat,lon'"),
    radius_km: float = typer.Option(3.0, help="radius around --near"),
    kml: str | None = typer.Option(None, help="KML/KMZ with the area outline"),
    area: str | None = typer.Option(None, help="area name inside the KML"),
    bbox: str | None = typer.Option(None, help="west,south,east,north"),
    month: int | None = None,
    max_walk_miles: float = 1.0,
    wind_from: float | None = typer.Option(None, help="override prevailing wind, degrees it blows FROM"),
    n: int = typer.Option(15, help="number of camera spots"),
    pins: bool = typer.Option(True, help="use the water/sign pins in your KML files (--no-pins for validation)"),
    open_map: bool = typer.Option(True, "--open/--no-open", help="open the map in Google Earth when done"),
) -> None:
    """Detailed analysis of one area -> KMZ + ranked camera spots."""
    bb = [float(v) for v in bbox.split(",")] if bbox else None
    _note_fixes(location)
    r = api.analyze_area(
        location,
        radius_km,
        kml,
        area,
        bb,
        month,
        max_walk_miles,
        wind_from,
        n,
        log=_log,
        user_pins=pins,
    )
    _out(r)
    _map(r, open_map)


@app.command()
def scout(
    location: str = typer.Argument(..., help="place name or 'lat,lon'"),
    radius_km: float = 40.0,
    month: int | None = None,
    max_walk_miles: float = 1.0,
    top: int = 8,
) -> None:
    """Coarse screen of a region -> the best ~3 km blocks to analyze."""
    _note_fixes(location)
    _out(api.scout_region(location, radius_km, month, max_walk_miles, top, log=_log))


@app.command()
def repick(
    area: str,
    n: int = 15,
    per_zone: int = 3,
    spacing_m: float = 150.0,
    max_walk_miles: float | None = None,
) -> None:
    """Re-select spots from a saved analysis (more/fewer, more spread out, other walk limit) without recomputing."""
    _out(api.repick(area, n, per_zone, spacing_m, max_walk_miles))


@app.command()
def explain(area: str, lat: float, lon: float, search_m: float = 25.0) -> None:
    """Why a spot scores the way it does (needs a prior analyze of that area)."""
    _out(api.explain_point(area, lat, lon, search_m))


@app.command()
def validate(
    area: str, kml: str | None = typer.Option(None, help="KML with human-picked camera pins, CamNN (default: yours)")
) -> None:
    """Test an analyzed area against the field log (cameras, snow tracks, transects) and human camera picks."""
    _out(api.validate(area, kml=kml))


@app.command()
def wind(location: str, month: int | None = None) -> None:
    """Prevailing high-pressure wind for a place and month."""
    _note_fixes(location)
    _out(api.wind_summary(location, month))


@app.command("log")
def log_cmd(
    lat: float,
    lon: float,
    lion: bool = typer.Option(..., "--lion/--no-lion"),
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    detections: int | None = None,
    notes: str | None = None,
) -> None:
    """Quick record of what a camera caught (an unpaired camera; use log-camera + log-check for a test)."""
    _out(api.log_result(lat, lon, lion, name, start, end, detections, None, notes))


def _event(text: str) -> dict[str, Any]:
    """'2026-11-03T05:40 cougar 2' (time, species, count; species and count optional) -> an event."""
    parts = text.split()
    if not parts:
        raise typer.BadParameter("give at least a time, e.g. '2026-11-03T05:40 cougar'")
    return dict(
        datetime=parts[0],
        species=parts[1] if len(parts) > 1 else "cougar",
        count=int(parts[2]) if len(parts) > 2 else 1,
    )


@app.command("log-camera")
def log_camera_cmd(
    lat: float | None = typer.Argument(None),
    lon: float | None = typer.Argument(None),
    name: str | None = None,
    arm: str | None = typer.Option(None, help="model | human | control | on-feature | off-feature | unpaired"),
    zone: str | None = typer.Option(None, help="cameras compared with each other share a zone"),
    start: str | None = typer.Option(None, help="date set (default today)"),
    end: str | None = typer.Option(None, help="date taken down; with --deployment, '' reopens it (still out)"),
    trail_type: str | None = typer.Option(
        None, help="paved | open-dirt | closed-road | hiking-trail | game-trail | none"
    ),
    height_m: float | None = None,
    facing_deg: float | None = None,
    lure: bool | None = typer.Option(None, "--lure/--no-lure"),
    downtime_nights: float | None = None,
    notes: str | None = None,
    deployment: str | None = typer.Option(None, help="update this existing deployment instead"),
) -> None:
    """Record a camera put out (or update one with --deployment ID, e.g. --end DATE)."""
    _out(
        api.log_camera(
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
    )


@app.command("log-check")
def log_check_cmd(
    deployment: str,
    date: str | None = typer.Option(None, help="date of the check (default today)"),
    event: list[str] | None = typer.Option(None, help="'TIME [SPECIES] [COUNT]', repeat for each visit"),
    downtime_nights: float = typer.Option(0.0, help="nights it wasn't recording since the last check"),
    removed: bool = typer.Option(False, help="the camera was taken down"),
    notes: str | None = None,
) -> None:
    """Record a camera check and what it caught (no --event = nothing came by)."""
    _out(api.log_check(deployment, date, [_event(e) for e in event or []], downtime_nights, removed, notes))


@app.command("log-track")
def log_track_cmd(
    file: str = typer.Argument(..., help="GPX, KML or KMZ from a phone GPS app"),
    date: str | None = None,
    snow_age_h: float | None = typer.Option(None, help="hours since the snow fell"),
    snow_depth_cm: float | None = None,
    confidence: str = typer.Option("probable", help="certain | probable | possible"),
    species: str = "cougar",
    name: str | None = None,
    notes: str | None = None,
) -> None:
    """Record a lion track followed in snow or mud."""
    _out(api.log_track(file, None, date, snow_age_h, snow_depth_cm, confidence, species, name, notes))


@app.command("log-transect")
def log_transect_cmd(
    route: str = typer.Argument(..., help="the fixed route's name"),
    file: str | None = typer.Argument(None, help="GPX/KML/KMZ of the route, a waypoint named 'lion' at each crossing"),
    date: str | None = None,
    surface: str = typer.Option("snow", help="snow | mud | dust"),
    snow_age_h: float | None = None,
    species: str = "cougar",
    notes: str | None = None,
) -> None:
    """Record one survey of a fixed route and its lion crossings (none is a result)."""
    _out(api.log_transect(route, file, None, None, date, surface, snow_age_h, species, notes))


@app.command("field-log")
def field_log_cmd() -> None:
    """Everything logged in the field: cameras, tracks, transect surveys."""
    _out(api.field_log())


@app.command()
def areas(kml: str) -> None:
    """List named areas and pins in a KML."""
    _out(api.list_areas(kml))


_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?\s*°?[NSEWnsew]?,?$")
_PLACE_COMMANDS = {"hotspots", "scout", "wind", "analyze"}


def _join_coords(args: list[str]) -> list[str]:
    """`hotspots 47.37, -116.10` arrives as two words, and click reads '-116.10' as an option: rejoin such a pair
    into one '47.37,-116.10' (only for the commands that take a place, skipping option values)."""
    cmd_at = next((i for i, a in enumerate(args) if not a.startswith("-")), None)
    if cmd_at is None or args[cmd_at] not in _PLACE_COMMANDS:
        return args
    cmd = typer.main.get_command(app).commands[args[cmd_at]]  # type: ignore[attr-defined]
    # options that take a value (arguments have no is_flag); --near's value is itself a place
    takes_value = {name for p in cmd.params if getattr(p, "is_flag", None) is False for name in p.opts} - {"--near"}
    out, rest = args[: cmd_at + 1], [a for a in args[cmd_at + 1 :] if a != "--"]
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in takes_value and i + 1 < len(rest):
            out += rest[i : i + 2]
            i += 2
        elif _NUMBER.match(a) and i + 1 < len(rest) and _NUMBER.match(rest[i + 1]):
            out.append(f"{a.rstrip(',')},{rest[i + 1].rstrip(',')}")
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def main() -> None:
    """The `cougarmap` command: a short message instead of a traceback (COUGARMAP_DEBUG=1 for the traceback)."""
    warnings.filterwarnings("ignore", message="Dataset has no geotransform")  # rasterio, harmless
    try:
        app(args=_join_coords(sys.argv[1:]), prog_name="cougarmap")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        if os.environ.get("COUGARMAP_DEBUG"):
            raise
        _log(f"error: {e}")
        _log("(COUGARMAP_DEBUG=1 shows the details)")
        sys.exit(1)


if __name__ == "__main__":
    main()
