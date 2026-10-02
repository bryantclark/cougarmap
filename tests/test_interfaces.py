"""The agent-facing surfaces (MCP tools, CLI) stay wired to the api functions they wrap."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
import typer.main
from mcp.types import CallToolResult, TextContent
from typer.core import TyperGroup
from typer.testing import CliRunner

import toys
from cougarmap import api, jobs, mcp_server
from cougarmap.cli import app
from cougarmap.config import OBSERVATIONS_FILE
from test_units import KML

# MCP-only parameters that the api functions do not take
_MCP_ONLY = {"wait_seconds"}
# MCP tools that are not a same-named api function
_NOT_API = {"job_status", "playbook"}


def _tools() -> dict[str, dict[str, Any]]:
    tools = anyio.run(mcp_server.server.list_tools)
    return {t.name: t.model_dump(mode="json")["input_schema"] for t in tools}


def test_mcp_tool_names() -> None:
    assert set(_tools()) == {
        "find_hotspots",
        "job_status",
        "repick",
        "explain_point",
        "log_camera",
        "log_check",
        "log_track",
        "log_transect",
        "field_log",
        "validate",
        "import_kml",
        "open_file",
        "playbook",
    }


@pytest.mark.parametrize("name", sorted(set(_tools()) - _NOT_API))
def test_mcp_params_are_api_params(name: str) -> None:
    """Every MCP argument must exist on the api function with the same name (catches drift between the
    three copies of each signature: api, mcp_server and cli)."""
    schema = _tools()[name]
    api_params = set(inspect.signature(getattr(api, name)).parameters)
    assert set(schema.get("properties", {})) - _MCP_ONLY <= api_params


def test_mcp_optional_params_not_required() -> None:
    schema = _tools()["find_hotspots"]
    assert not schema.get("required")  # every argument has a default
    loc = schema["properties"]["location"]
    assert {"type": "null"} in loc.get("anyOf", [])


def test_cli_help_for_every_command() -> None:
    runner = CliRunner()
    group = typer.main.get_command(app)
    assert isinstance(group, TyperGroup)
    commands = sorted(group.commands)
    assert {"hotspots", "analyze", "repick", "explain", "job", "jobs", "setup"} <= set(commands)
    for cmd in [None, *commands]:
        res = runner.invoke(app, [*([cmd] if cmd else []), "--help"])
        assert res.exit_code == 0, (cmd, res.output)


# ---- calling the tools and commands -------------------------------------------------------------------------


def _call(tool: str, **args: Any) -> Any:
    res = anyio.run(mcp_server.server.call_tool, tool, args)
    assert isinstance(res, CallToolResult) and not res.is_error, res
    text = res.content[0]
    assert isinstance(text, TextContent)
    return json.loads(text.text)


def test_mcp_tools_answer_in_json(analyzed: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    area = analyzed["dir"].name
    c = analyzed["candidates"][0]
    e = _call("explain_point", area=area, lat=c["lat"], lon=c["lon"])
    assert e["best_nearby"]["reasons"]
    imported = _call("import_kml", path=str(_kml_file(analyzed)))
    assert imported["imported"] and imported["areas"]
    assert _call("import_kml", path=imported["path"])["areas"] == imported["areas"]  # again: just lists it
    cam = _call("log_camera", lat=1.0, lon=2.0, name="M1", arm="model", zone="z1", start="2026-10-01")
    assert cam["deployment"]["id"] == "M1"
    ev = [dict(datetime="2026-10-03T05:00", species="cougar", count=1)]
    assert _call("log_check", deployment="M1", date="2026-10-11", events=ev)["camera_nights"] == 10
    assert _call("log_track", points=[[1.0, 2.0], [1.001, 2.0]], date="2026-12-01")["km"] > 0.1
    assert _call("log_transect", route="R1", line=[[1.0, 2.0], [1.01, 2.0]], date="2026-12-01")["crossings"] == 0
    assert [c["id"] for c in _call("field_log")["cameras"]] == ["M1"]
    assert _call("validate", area=area)["cameras"]["cameras"] == []  # all logged far outside the area
    OBSERVATIONS_FILE.unlink()
    assert _call("open_file", path="/no/such/file")["opened"] is False
    assert "find_hotspots" in anyio.run(mcp_server.playbook)
    started: list[tuple[str, dict[str, Any]]] = []

    def start(kind: str, params: dict[str, Any]) -> str:
        started.append((kind, params))
        return "job-1"

    monkeypatch.setattr(jobs, "start", start)
    monkeypatch.setattr(jobs, "status", lambda jid, wait=0, tail=8: dict(job_id=jid, state="running", wait=wait))
    r = _call("find_hotspots", location="Testville", wait_seconds=500)
    assert r == dict(job_id="job-1", state="running", wait=mcp_server.MAX_WAIT)
    assert started[0] == ("hotspots", dict(location="Testville", max_walk_miles=1.0, blocks=3))
    _call("find_hotspots", bbox=[1.0, 2.0, 3.0, 4.0], wind_from_deg=270)
    assert started[1] == ("hotspots", dict(bbox=[1.0, 2.0, 3.0, 4.0], max_walk_miles=1.0, wind_from_deg=270, blocks=3))
    assert _call("job_status", job_id="job-1")["state"] == "running"
    monkeypatch.setattr(jobs, "list_jobs", lambda limit=10: [dict(job_id="job-1", state="running")])
    assert _call("job_status") == dict(jobs=[dict(job_id="job-1", state="running")])


def _kml_file(analyzed: dict[str, Any]) -> Path:
    p: Path = analyzed["dir"].parent / "areas.kml"
    p.write_text(KML)
    return p


def test_cli_commands(analyzed: dict[str, Any], area: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """(area: a copy of the analyzed area, which repick rewrites.)"""
    runner = CliRunner()
    c = analyzed["candidates"][0]

    def run(*args: str) -> Any:
        res = runner.invoke(app, list(args))
        assert res.exit_code == 0, (args, res.output)
        return json.loads(res.stdout)

    assert run("explain", "--", area, str(c["lat"]), str(c["lon"]))["best_nearby"]["reasons"]
    assert run("repick", area, "--n", "2")["summary"]["n_candidates"] <= 2
    assert run("validate", area)["human_picks"] is None
    assert run("import-kml", str(_kml_file(analyzed)))["areas"]
    cam = run(
        "log-camera",
        "--name",
        "M1",
        "--arm",
        "model",
        "--zone",
        "z1",
        "--start",
        "2026-10-01",
        "--lure",
        "--",
        "1.0",
        "2.0",
    )
    assert cam["deployment"]["lure"] is True
    assert (
        run("log-camera", "--deployment", "M1", "--trail-type", "game-trail")["deployment"]["trail_type"]
        == "game-trail"
    )
    chk = run(
        "log-check", "M1", "--date", "2026-10-11", "--event", "2026-10-03T05:00 cougar 2", "--event", "2026-10-04T05:00"
    )
    assert chk["detections"]["cougar"] == 2 and chk["camera_nights"] == 10
    gpx = Path(analyzed["dir"].parent / "t.gpx")
    gpx.write_text(
        '<gpx><trk><trkseg><trkpt lat="1" lon="2"/><trkpt lat="1.001" lon="2"/></trkseg></trk>'
        '<wpt lat="1.0005" lon="2"><name>lion</name></wpt></gpx>'
    )
    assert run("log-track", str(gpx), "--date", "2026-12-01", "--confidence", "certain")["lines"] == 1
    assert run("log-transect", "R1", str(gpx), "--date", "2026-12-01")["crossings"] == 1
    assert len(run("field-log")["transects"]) == 1
    res = runner.invoke(app, ["log-check", "M1", "--event", " "])
    assert res.exit_code != 0
    OBSERVATIONS_FILE.unlink()
    assert run("jobs") == [] or isinstance(run("jobs"), list)
    calls: list[tuple[Any, ...]] = []
    for fn in ("find_hotspots", "analyze_area"):
        monkeypatch.setattr(api, fn, toys.recorder(calls, dict(ok=True)))
    monkeypatch.setattr(jobs, "start", lambda kind, params: "job-9")
    assert run("hotspots", "Testville", "--max-walk-miles", "0.5") == dict(ok=True)
    assert calls[-1][:6] == ("Testville", None, None, None, None, 0.5)
    assert run("hotspots", "Testville", "--background")["job_id"] == "job-9"
    assert run("analyze", "--bbox=1,2,3,4", "--n", "4") == dict(ok=True) and calls[-1][4] == [1.0, 2.0, 3.0, 4.0]
    res = runner.invoke(app, ["playbook"])
    assert res.exit_code == 0 and "find_hotspots" in res.stdout


def test_cli_setup_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from cougarmap import setup_harnesses

    monkeypatch.setattr(
        setup_harnesses,
        "run",
        lambda only, all_, dry, remove: dict(
            harnesses={"Cursor": ["would update x"], "Gemini CLI": "not found"},
            shared_skill="skill -> y",
            server_command=["cougarmap-mcp"],
            data_folder="/tmp/cm",
        ),
    )
    res = CliRunner().invoke(app, ["setup", "--dry-run"])
    assert res.exit_code == 0 and "- Cursor:" in res.stdout and "would update x" in res.stdout
    assert "Restart your AI apps" not in res.stdout


def test_cli_joins_typed_coordinates() -> None:
    from cougarmap.cli import _join_coords

    assert _join_coords(["hotspots", "47.4,", "-116.1", "--radius-km", "30"]) == [
        "hotspots",
        "47.4,-116.1",
        "--radius-km",
        "30",
    ]
    assert _join_coords(["hotspots", "--", "47.4,", "116.1", "--month", "10"]) == [
        "hotspots",
        "47.4,116.1",
        "--month",
        "10",
    ]
    assert _join_coords(["hotspots", "--month", "10", "47.4", "-116"]) == ["hotspots", "--month", "10", "47.4,-116"]
    assert _join_coords(["analyze", "--near", "47.4", "-116"]) == ["analyze", "--near", "47.4,-116"]
    assert _join_coords(["explain", "a", "47.1", "-117.2"]) == ["explain", "a", "47.1", "-117.2"]  # lat lon args


def test_cli_main_errors_and_map_link(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cougarmap import cli

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("no elevation data for this area")

    monkeypatch.setattr(api, "find_hotspots", boom)
    monkeypatch.setattr(sys, "argv", ["cougarmap", "hotspots", "47.4,", "116.1"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    err = capsys.readouterr().err
    assert e.value.code == 1 and "error: no elevation data" in err and "Traceback" not in err
    assert "using -116.1" in err  # the longitude note
    monkeypatch.setenv("COUGARMAP_DEBUG", "1")
    with pytest.raises(RuntimeError):
        cli.main()

    opened: list[str] = []
    monkeypatch.setattr(api, "open_file", opened.append)
    cli._map(dict(kmz="/tmp/x.kmz"), open_it=True)  # not a terminal under pytest: link only
    assert "map: file:///" in capsys.readouterr().err and opened == []
    cli._map(dict(summary=dict(outputs={})), open_it=True)
    assert capsys.readouterr().err == ""
