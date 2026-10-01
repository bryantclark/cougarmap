"""Background jobs: start, status (with waiting), listing, success and failure, in-process and in a real child."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from cougarmap import api, jobs


@pytest.fixture
def inline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Jobs run in this process, right away, instead of a detached child. Returns the started job ids."""
    started: list[str] = []

    def popen(args: list[str], **_: Any) -> None:
        started.append(args[-1])
        jobs._run(args[-1])

    monkeypatch.setattr(subprocess, "Popen", popen)
    return started


def test_job_runs_to_done_with_progress(inline: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_analyze(**k: Any) -> dict[str, Any]:
        k["log"]("step one")
        return dict(summary=dict(area=k["area_name"]))

    monkeypatch.setattr(api, "analyze_area", fake_analyze)
    jid = jobs.start("analyze", dict(area_name="Test"))
    st = jobs.status(jid, tail=5)
    assert st["state"] == "done" and st["result"] == dict(summary=dict(area="Test")) and st["progress"] == ["step one"]
    assert st["started"] and st["finished"] and inline == [jid]
    listed = jobs.list_jobs()
    assert listed[0] == dict(job_id=jid, kind="analyze", state="done", params=dict(area_name="Test"))


def test_failed_job_reports_the_error(inline: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: Any) -> dict[str, Any]:
        raise ValueError("no such area")

    monkeypatch.setattr(api, "scout_region", boom)
    st = jobs.status(jobs.start("scout", dict(location="x")))
    assert st["state"] == "failed" and st["error"] == "ValueError: no such area"


def test_running_job_and_unknown_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)  # started, never runs
    jid = jobs.start("hotspots", dict(location="x"))
    assert jobs.status(jid)["state"] == "queued"
    d = jobs._dir(jid)
    st = json.loads((d / "status.json").read_text())
    st.update(state="running", started="now", started_ts=0.0)
    (d / "status.json").write_text(json.dumps(st))
    running = jobs.status(jid, wait_seconds=0)
    assert running["state"] == "running" and running["elapsed_s"] > 0 and "still running" in running["hint"]
    assert jobs.status("20000101-000000-abcdef") == dict(
        job_id="20000101-000000-abcdef", state="unknown", error="no such job"
    )
    (jobs.JOBS / "stray").mkdir()
    assert all(j["job_id"] != "stray" for j in jobs.list_jobs(50))


def test_a_real_detached_job_fails_cleanly() -> None:
    """The child process (python -m cougarmap.jobs run <id>) runs and records its outcome."""
    jid = jobs.start("bogus", {})
    st = jobs.status(jid, wait_seconds=60)
    assert st["state"] == "failed" and st["error"] == "KeyError: 'bogus'"
    assert "Traceback" in (jobs._dir(jid) / "log.txt").read_text()
