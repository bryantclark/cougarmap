"""Shared fixtures. Every test runs against a throwaway CougarMap home (results, cache, private folder), never the
real one: the environment is set here, before any cougarmap module is imported."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

_HOME = Path(tempfile.mkdtemp(prefix="cougarmap-tests-"))
os.environ["COUGARMAP_HOME"] = str(_HOME)
os.environ["COUGARMAP_OUT"] = str(_HOME / "results")
os.environ["COUGARMAP_CACHE"] = str(_HOME / "cache")
os.environ["COUGARMAP_PRIVATE"] = str(_HOME / "my-data")

import synthetic  # noqa: E402  (after the environment is set)
from cougarmap import analyze, api  # noqa: E402
from cougarmap.analyze import slug  # noqa: E402
from cougarmap.config import OUT_DIR, PLACEMENT  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _no_alternate_floor() -> Iterator[None]:
    """The synthetic and toy areas' spots score ~10-50: the trail alternate's absolute floor (PLACEMENT.min_score)
    is off in tests, so their roads and trails still yield alternates to check. test_model tests the floor."""
    with mock.patch.object(analyze, "PLACEMENT", dataclasses.replace(PLACEMENT, min_score=0.0)):
        yield


@pytest.fixture(scope="session")
def analyzed() -> Iterator[dict[str, Any]]:
    """The synthetic area analyzed once (public land only, October, a fast run: no worn trails), as
    api.analyze_area returns it, plus "dir": the folder its outputs and state.pkl were written to."""
    with synthetic.offline():
        r = api.analyze_area(bbox=synthetic.bbox(), month=10, log=lambda *_: None, fast=True)
    yield dict(r, dir=Path(r["summary"]["outputs"]["state"]).parent)
    api._STATES.clear()


@pytest.fixture
def area(analyzed: dict[str, Any], tmp_path: Path) -> Iterator[str]:
    """The name of a private copy of the analyzed area (in the results folder) that a test may re-pick."""
    dst = OUT_DIR / slug(f"copy {tmp_path.name}")
    shutil.copytree(analyzed["dir"], dst)
    yield dst.name
    api._STATES.clear()
    shutil.rmtree(dst, ignore_errors=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_HOME, ignore_errors=True)
