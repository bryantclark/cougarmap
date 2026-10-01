import json
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

from cougarmap import setup_harnesses as S


def test_toml_merge_idempotent_and_uninstall(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    orig = 'model = "x"\n\n[mcp_servers.other]\ncommand = "o"\n\n[profiles.a]\nmodel = "y"\n'
    p.write_text(orig)
    block = S._toml_block(["/bin/cougarmap-mcp"])
    S._toml_set(p, block, dry=False)
    S._toml_set(p, block, dry=False)
    d = tomllib.loads(p.read_text())
    assert set(d["mcp_servers"]) == {"other", "cougarmap"}
    assert d["mcp_servers"]["cougarmap"]["tool_timeout_sec"] == 1200
    assert d["mcp_servers"]["cougarmap"]["default_tools_approval_mode"] == "approve"
    assert d["profiles"]["a"]["model"] == "y"
    assert p.read_text().count("[mcp_servers.cougarmap]") == 1
    S._toml_set(p, None, dry=False)
    assert "cougarmap" not in tomllib.loads(p.read_text())["mcp_servers"]
    assert (tmp_path / "config.toml.cougarmap-bak").read_text() == orig


def test_json_merge_keeps_other_settings(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "o"}}}))
    S._json_merge(p, {"command": "/bin/c", "args": []}, dry=False)
    d = json.loads(p.read_text())
    assert d["theme"] == "dark" and set(d["mcpServers"]) == {"other", "cougarmap"}
    S._json_merge(p, None, dry=False)
    assert set(json.loads(p.read_text())["mcpServers"]) == {"other"}


def test_skill_has_frontmatter() -> None:
    t = S.skill_text()
    assert t.startswith("---\nname: cougarmap\ndescription: ")
    assert "find_hotspots" in t


# ---- the whole setup, into a throwaway home folder -------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(S, "H", tmp_path)
    monkeypatch.setattr(S, "APPS", tmp_path / "Applications")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(S, "DATA_HOME", tmp_path / "CougarMap")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return tmp_path


def test_setup_every_app_then_uninstall(home: Path) -> None:
    dry = S.run(all_=True, dry=True)
    assert set(dry["harnesses"]) >= {"Claude Code", "Codex (CLI / app / IDE)", "Cursor", "Gemini CLI"}
    assert not (home / ".cursor").exists() and not (home / "CougarMap").exists()  # a dry run writes nothing

    r = S.run(all_=True)
    assert all(isinstance(m, list) for m in r["harnesses"].values()), r["harnesses"]
    cmd = r["server_command"]
    assert json.loads((home / ".cursor" / "mcp.json").read_text())["mcpServers"]["cougarmap"]["command"] == cmd[0]
    assert json.loads((home / ".claude.json").read_text())["mcpServers"]["cougarmap"]["type"] == "stdio"
    assert "mcp__cougarmap" in json.loads((home / ".claude" / "settings.json").read_text())["permissions"]["allow"]
    assert tomllib.loads((home / ".codex" / "config.toml").read_text())["mcp_servers"]["cougarmap"]
    gemini = json.loads((home / ".gemini" / "settings.json").read_text())["mcpServers"]["cougarmap"]
    assert gemini["timeout"] == 1_200_000
    for skills in (".claude/skills", ".agents/skills", ".gemini/skills", ".cursor/skills", ".gemini/config/skills"):
        assert (home / skills / "cougarmap" / "SKILL.md").read_text().startswith("---\nname: cougarmap")
    with zipfile.ZipFile(home / "CougarMap" / "cougarmap-cowork-plugin.zip") as z:
        assert {"skills/cougarmap/SKILL.md", ".mcp.json", ".claude-plugin/plugin.json"} <= set(z.namelist())
    again = S.run(all_=True)
    assert "already set" in " ".join(again["harnesses"]["Cursor"])

    S.run(all_=True, remove=True)
    assert "cougarmap" not in json.loads((home / ".cursor" / "mcp.json").read_text())["mcpServers"]
    assert not (home / ".claude" / "skills" / "cougarmap").exists()
    assert "mcp__cougarmap" not in json.loads((home / ".claude" / "settings.json").read_text())["permissions"]["allow"]
    assert not (home / "CougarMap" / "cougarmap-cowork-plugin.zip").exists()


def test_setup_only_detected_or_named_apps(home: Path) -> None:
    r = S.run()
    assert all(v == "not found (use --all to set up anyway)" for v in r["harnesses"].values())
    (home / ".cursor").mkdir()
    assert isinstance(S.run()["harnesses"]["Cursor"], list)
    only = S.run(only=["gemini"])["harnesses"]
    assert isinstance(only["Gemini CLI"], list) and only["Cursor"] == "not found (use --all to set up anyway)"


def test_setup_registers_with_the_claude_cli(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        ran.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    msgs = S.run(only=["claude code"])["harnesses"]["Claude Code"]
    assert (
        msgs[0].startswith("ok: mcp add --scope") and ran[0][1:3] == ["mcp", "remove"] and ran[1][1:3] == ["mcp", "add"]
    )
    assert (
        "would run: /usr/bin/claude mcp remove"
        in S.run(only=["claude code"], dry=True, remove=True)["harnesses"]["Claude Code"][0]
    )


def test_unreadable_config_is_left_alone(home: Path) -> None:
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text("{ not json")
    assert S.run(only=["cursor"])["harnesses"]["Cursor"][0].startswith("skipped")
    assert S._toml_set(home / "none.toml", None, dry=False).startswith("nothing to remove")
    assert S._json_merge(home / "none.json", None, dry=False).startswith("nothing to remove")
    assert S.server_command()[-1].endswith(("cougarmap-mcp", "cougarmap.mcp_server"))
