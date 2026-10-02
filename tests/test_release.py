"""What a release publishes stays consistent: the package version, the MCP Registry entry (server.json) and the
README's ownership line, and the command the registry tells apps to run."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cougarmap import cli, mcp_server

ROOT = Path(__file__).resolve().parents[1]


def _server() -> dict[str, Any]:
    out: dict[str, Any] = json.loads((ROOT / "server.json").read_text())
    return out


def test_versions_and_names_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    s = _server()
    pkg = s["packages"][0]
    assert s["version"] == pkg["version"] == project["version"]
    assert pkg["registryType"] == "pypi" and pkg["identifier"] == project["name"]
    assert f"<!-- mcp-name: {s['name']} -->" in (ROOT / "README.md").read_text()  # PyPI ownership check
    assert len(s["description"]) <= 100


def test_the_registry_command_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apps run `uvx cougarmap <packageArguments>`: that must be the MCP server."""
    started: list[bool] = []
    monkeypatch.setattr(mcp_server, "main", lambda: started.append(True))
    args = [a["value"] for a in _server()["packages"][0]["packageArguments"]]
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0 and started == [True]
