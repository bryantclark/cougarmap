"""`cougarmap setup`: connect CougarMap to every AI agent app found on this computer.

For each harness it registers the MCP server (the tools) and installs the playbook as a skill where the harness
supports skills. Existing config files are backed up (*.cougarmap-bak) before the first edit, and re-running is
safe. `--uninstall` removes what setup added.

Harness facts (paths, formats, timeouts) were checked against each tool's docs in Sep 2026.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from .config import HOME as DATA_HOME

NAME = "cougarmap"
H = Path.home()
APPS = Path("/Applications")  # macOS app bundles (detection only)
WIN = os.name == "nt"
MAC = sys.platform == "darwin"

SKILL_DESCRIPTION = (
    "Find mountain lion (cougar) hotspots and trail-camera spots with the cougarmap tools. Use when "
    "the user asks to find cougar/lion areas near a place or coordinate, scan a property, analyze a "
    "Google Earth area, explain a spot, check the wind, or log what a trail camera caught."
)


def server_command() -> list[str]:
    """Absolute command that starts the MCP server (works no matter which folder the harness starts in)."""
    exe = shutil.which("cougarmap-mcp")
    if not exe:
        cand = Path(sys.argv[0]).resolve().parent / ("cougarmap-mcp.exe" if WIN else "cougarmap-mcp")
        if cand.exists():
            exe = str(cand)
    if exe:
        return [str(Path(exe).resolve())]
    return [sys.executable, "-m", "cougarmap.mcp_server"]


def skill_text() -> str:
    body = files("cougarmap").joinpath("PLAYBOOK.md").read_text()
    return f"---\nname: {NAME}\ndescription: {SKILL_DESCRIPTION}\n---\n\n{body}"


# ---- file helpers --------------------------------------------------------------------------------------------


Apply = Callable[[list[str], bool, bool], list[str]]  # (server command, dry run, remove) -> messages


def _backup(p: Path) -> None:
    b = p.with_name(p.name + ".cougarmap-bak")
    if p.exists() and not b.exists():
        shutil.copy2(p, b)


def _json_merge(p: Path, entry: dict[str, Any] | None, dry: bool, key: str = "mcpServers") -> str:
    data: dict[str, Any] = {}
    if p.exists():
        txt = p.read_text().strip()
        if txt:
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                return f"skipped {p} (not plain JSON - add the server by hand)"
    servers = data.setdefault(key, {})
    if entry is None:
        if NAME not in servers:
            return f"nothing to remove in {p}"
        servers.pop(NAME)
    else:
        if servers.get(NAME) == entry:
            return f"already set in {p}"
        servers[NAME] = entry
    if not dry:
        p.parent.mkdir(parents=True, exist_ok=True)
        _backup(p)
        p.write_text(json.dumps(data, indent=2) + "\n")
    return f"{'would update' if dry else ('removed from' if entry is None else 'registered in')} {p}"


def _toml_block(cmd: list[str]) -> str:
    q = json.dumps  # TOML basic strings share JSON escaping for our purposes
    return (
        f"[mcp_servers.{NAME}]\ncommand = {q(cmd[0])}\nargs = [{', '.join(q(a) for a in cmd[1:])}]\n"
        f"startup_timeout_sec = 60\ntool_timeout_sec = 1200\n"
        f"# CougarMap only downloads public map data and writes to its own folder: don't ask before each call\n"
        f'default_tools_approval_mode = "approve"\n'
    )


def _toml_set(p: Path, block: str | None, dry: bool) -> str:
    txt = p.read_text() if p.exists() else ""
    pat = re.compile(rf"^\[mcp_servers\.{NAME}\]\n(?:(?!^\[).*\n?)*", re.M)
    if block is None:
        if not pat.search(txt):
            return f"nothing to remove in {p}"
        new = pat.sub("", txt)
    elif pat.search(txt):
        new = pat.sub(block + "\n", txt)
        if new == txt:
            return f"already set in {p}"
    else:
        new = txt + ("\n" if txt and not txt.endswith("\n") else "") + ("\n" if txt else "") + block
    if not dry:
        p.parent.mkdir(parents=True, exist_ok=True)
        _backup(p)
        p.write_text(new)
    return f"{'would update' if dry else ('removed from' if block is None else 'registered in')} {p}"


def _skill(dirpath: Path, dry: bool, remove: bool) -> str:
    d = dirpath / NAME
    if remove:
        if d.exists() and not dry:
            shutil.rmtree(d)
        return f"removed skill {d}" if d.exists() or dry else f"no skill at {d}"
    if not dry:
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(skill_text())
    return f"skill -> {d / 'SKILL.md'}"


def _claude_allow(dry: bool, remove: bool) -> str:
    """Let Claude Code call the cougarmap tools without asking each time."""
    p = H / ".claude" / "settings.json"
    rule = f"mcp__{NAME}"
    data: dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text() or "{}")
        except json.JSONDecodeError:
            return f"skipped {p} (not plain JSON)"
    allow = data.setdefault("permissions", {}).setdefault("allow", [])
    if remove:
        if rule not in allow:
            return "no Claude Code permission to remove"
        allow.remove(rule)
    elif rule in allow:
        return "Claude Code already allows cougarmap tools"
    else:
        allow.append(rule)
    if not dry:
        p.parent.mkdir(parents=True, exist_ok=True)
        _backup(p)
        p.write_text(json.dumps(data, indent=2) + "\n")
    return f"{'would update' if dry else 'updated'} {p} (allow cougarmap tools)"


# ---- harnesses ---------------------------------------------------------------------------------------------


def _claude_desktop_dir() -> Path:
    if MAC:
        return H / "Library" / "Application Support" / "Claude"
    if WIN:
        return Path(os.environ.get("APPDATA", H / "AppData" / "Roaming")) / "Claude"
    return H / ".config" / "Claude"


def _entry(cmd: list[str]) -> dict[str, Any]:
    return {"command": cmd[0], "args": cmd[1:]}


def _found(command: str | None, *paths: Path) -> bool:
    """Is an app installed: its command on PATH, or any of its files or folders present?"""
    return bool(command and shutil.which(command)) or any(p.exists() for p in paths)


# ---- what setup does for each app: (server command, dry run, remove) -> messages ------------------------------


def _claude_code(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    """Through the claude CLI when it is installed (user scope), else straight into ~/.claude.json."""
    out = []
    claude = shutil.which("claude")
    if claude:
        if remove:
            args = [claude, "mcp", "remove", "--scope", "user", NAME]
        else:
            # drop any stale registration first; failure just means there was none
            subprocess.run([claude, "mcp", "remove", "--scope", "user", NAME], capture_output=True, check=False)
            args = [claude, "mcp", "add", "--scope", "user", "--transport", "stdio", NAME, "--", *cmd]
        if not dry:
            r = subprocess.run(args, capture_output=True, text=True, check=False)
            out.append(
                ("ok: " if r.returncode == 0 else "failed: ")
                + " ".join(args[1:4])
                + (r.stderr.strip()[:200] if r.returncode else "")
            )
        else:
            out.append("would run: " + " ".join(args))
    else:
        out.append(_json_merge(H / ".claude.json", None if remove else dict(type="stdio", **_entry(cmd)), dry))
    out.append(_skill(H / ".claude" / "skills", dry, remove))
    out.append(_claude_allow(dry, remove))
    return out


def _claude_desktop(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    return [_json_merge(_claude_desktop_dir() / "claude_desktop_config.json", None if remove else _entry(cmd), dry)]


def _cowork(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    z = DATA_HOME / "cougarmap-cowork-plugin.zip"
    if remove:
        if z.exists() and not dry:
            z.unlink()
        return [f"removed {z} (also remove the plugin in Claude -> Customize -> Plugins)"]
    if not dry:
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as f:
            f.writestr(
                ".claude-plugin/plugin.json",
                json.dumps(
                    dict(
                        name=NAME,
                        version="0.1.0",
                        description="Find mountain lion hotspots and trail-camera spots.",
                        author=dict(name="CougarMap"),
                    ),
                    indent=2,
                ),
            )
            f.writestr(".mcp.json", json.dumps({"mcpServers": {NAME: _entry(cmd)}}, indent=2))
            f.writestr(f"skills/{NAME}/SKILL.md", skill_text())
    return [f"Cowork plugin -> {z}  (in the Claude app: Customize -> Plugins -> upload this file)"]


def _codex(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    home = Path(os.environ.get("CODEX_HOME", H / ".codex"))
    return [
        _toml_set(home / "config.toml", None if remove else _toml_block(cmd), dry),
        _skill(H / ".agents" / "skills", dry, remove),
    ]


def _gemini(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    e = None if remove else dict(**_entry(cmd), timeout=1_200_000)
    return [_json_merge(H / ".gemini" / "settings.json", e, dry), _skill(H / ".gemini" / "skills", dry, remove)]


def _antigravity(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    return [
        _json_merge(H / ".gemini" / "config" / "mcp_config.json", None if remove else _entry(cmd), dry),
        _skill(H / ".gemini" / "config" / "skills", dry, remove),
    ]


def _cursor(cmd: list[str], dry: bool, remove: bool) -> list[str]:
    e = None if remove else dict(type="stdio", **_entry(cmd))
    return [_json_merge(H / ".cursor" / "mcp.json", e, dry), _skill(H / ".cursor" / "skills", dry, remove)]


def harnesses() -> list[tuple[str, bool, Apply]]:
    """(name, detected?, apply(cmd, dry, remove) -> [messages])"""
    local = Path(os.environ.get("LOCALAPPDATA", H / "AppData" / "Local"))
    return [
        ("Claude Code", _found("claude", H / ".claude"), _claude_code),
        ("Claude Desktop (chat)", _claude_desktop_dir().exists(), _claude_desktop),
        ("Claude Cowork", _claude_desktop_dir().exists(), _cowork),
        ("Codex (CLI / app / IDE)", _found("codex", H / ".codex", APPS / "Codex.app"), _codex),
        ("Gemini CLI", _found("gemini", H / ".gemini" / "settings.json"), _gemini),
        (
            "Antigravity",
            _found(
                None,
                APPS / "Antigravity.app",
                H / ".gemini" / "config",
                H / ".gemini" / "antigravity",
                local / "Programs" / "Antigravity",
            ),
            _antigravity,
        ),
        ("Cursor", _found("cursor", H / ".cursor", APPS / "Cursor.app"), _cursor),
    ]


def run(only: list[str] | None = None, all_: bool = False, dry: bool = False, remove: bool = False) -> dict[str, Any]:
    cmd = server_command()
    found_in: dict[str, str | list[str]] = {}
    report: dict[str, Any] = dict(server_command=cmd, data_folder=str(DATA_HOME), harnesses=found_in)
    for name, found, fn in harnesses():
        wanted = (only and any(o.lower() in name.lower() for o in only)) or (not only and (found or all_))
        if not wanted:
            found_in[name] = "not found (use --all to set up anyway)"
            continue
        try:
            found_in[name] = fn(cmd, dry, remove)
        except Exception as e:
            found_in[name] = [f"error: {e}"]
    # a generic skill location several tools share (Codex, Gemini, Cursor read ~/.agents/skills)
    report["shared_skill"] = _skill(H / ".agents" / "skills", dry, remove)
    if not remove and not dry:
        DATA_HOME.mkdir(parents=True, exist_ok=True)
    return report
