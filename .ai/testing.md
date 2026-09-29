---
paths:
  - 'tests/**'
  - 'scripts/**'
---

# Testing Rules

## Test isolation

- `tests/conftest.py` points `COUGARMAP_HOME`/`OUT`/`CACHE`/`PRIVATE` at a throwaway temp folder before any
  `cougarmap` import. Tests never touch the real results, cache, or private data.

## Where tests live

- Top-level `tests/`, one file per module or concern (`test_factors.py`, `test_truth.py`…).

## Factories / helpers

- `tests/synthetic.py` serves a made-up 1.2 km landscape in place of every data source, so the whole pipeline
  (download, factors, picking, KMZ, state, repick, explain, CLI, MCP tools, jobs) runs offline.
- `tests/toys.py` has small hand-built arrays for unit-level factor checks.

## What to test

- The default run (~15 s) needs no network and no private data; coverage must stay at 90% or more.
- Live services are marked `network`; saved states on this machine or the GPS baseline are marked `slow`. Both are
  opt-in. Tests never depend on private data.
- Each numba kernel is checked against the scipy/numpy code it replaced (`tests/test_kernels.py`).
- Model quality is measured by the eval scripts (`scripts/eval_picks.py`, `scripts/gps_check.py`), not unit tests.
