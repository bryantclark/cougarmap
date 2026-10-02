"""Background jobs, so long runs never hit an agent harness's tool-call timeout.

A job is a detached `python -m cougarmap.jobs run <id>` process. Its folder (out/jobs/<id>/) holds spec.json,
status.json (state, progress lines), log.txt and result.json. Any process - the MCP server, the CLI, a different
agent session - can check on it by id.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import OUT_DIR

JOBS = OUT_DIR / "jobs"
JSON = dict[str, Any]


def _dir(job_id: str) -> Path:
    return JOBS / job_id


def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def start(kind: str, params: JSON) -> str:
    """Launch a job in a detached process; returns its id."""
    job_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = _dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    _write(d / "spec.json", dict(kind=kind, params=params))
    _write(d / "status.json", dict(job_id=job_id, kind=kind, state="queued", started=None, progress=[]))
    windows = os.name == "nt"
    # The child inherits its own copy of the log handle, so ours can be closed straight away.
    with (d / "log.txt").open("a") as log:
        subprocess.Popen(
            [sys.executable, "-m", "cougarmap.jobs", "run", job_id],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
            # detach: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP on Windows, a new session elsewhere
            creationflags=0x00000008 | 0x00000200 if windows else 0,
            start_new_session=not windows,
        )
    return job_id


def status(job_id: str, wait_seconds: float = 0, tail: int = 8) -> JSON:
    """Current state of a job; waits up to wait_seconds for it to finish. Includes the result when done."""
    d = _dir(job_id)
    if not d.exists():
        return dict(job_id=job_id, state="unknown", error="no such job")
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        st = json.loads((d / "status.json").read_text())
        if st["state"] in ("done", "failed") or time.time() >= deadline:
            break
        time.sleep(2)
    out = dict(
        job_id=job_id,
        kind=st["kind"],
        state=st["state"],
        started=st.get("started"),
        finished=st.get("finished"),
        progress=st.get("progress", [])[-tail:],
    )
    if st["state"] == "running" and st.get("started"):
        out["elapsed_s"] = round(time.time() - st["started_ts"])
        out["hint"] = "still running - call job_status again (large areas take several minutes the first time)"
    if st["state"] == "done" and (d / "result.json").exists():
        out["result"] = json.loads((d / "result.json").read_text())
    if st["state"] == "failed":
        out["error"] = st.get("error")
    return out


def list_jobs(limit: int = 10) -> list[JSON]:
    if not JOBS.exists():
        return []
    out = []
    for d in sorted(JOBS.iterdir(), reverse=True)[:limit]:
        try:
            st = json.loads((d / "status.json").read_text())
            spec = json.loads((d / "spec.json").read_text())
        except (OSError, ValueError):  # a job folder being created right now, or a stray file
            continue
        out.append(dict(job_id=d.name, kind=st["kind"], state=st["state"], params=spec["params"]))
    return out


def _run(job_id: str) -> None:
    from . import api

    d = _dir(job_id)
    spec = json.loads((d / "spec.json").read_text())
    st = json.loads((d / "status.json").read_text())
    st.update(state="running", started=dt.datetime.now().isoformat(timespec="seconds"), started_ts=time.time())
    _write(d / "status.json", st)

    lock = threading.Lock()  # downloads run on threads and may report from several at once

    def log(msg: str) -> None:
        msg = str(msg)
        with lock:
            print(msg, flush=True)
            st["progress"].append(msg)
            st["progress"] = st["progress"][-200:]
            _write(d / "status.json", st)

    try:
        runners: dict[str, Callable[..., JSON]] = dict(hotspots=api.find_hotspots)
        fn = runners[spec["kind"]]
        result = fn(**spec["params"], log=log)
        _write(d / "result.json", result)
        st.update(state="done")
    except Exception as e:
        st.update(state="failed", error=f"{type(e).__name__}: {e}")
        print(traceback.format_exc(), flush=True)
    st["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    _write(d / "status.json", st)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _run(sys.argv[2])
    else:
        print("usage: python -m cougarmap.jobs run <job_id>", file=sys.stderr)
        sys.exit(2)
