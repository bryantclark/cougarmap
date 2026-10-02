# Developing CougarMap

## Setup and the gate

```bash
uv sync                      # Python 3.13 (.python-version; >=3.12 supported); dev tools are in the default group
./scripts/check.sh           # the gate: ruff format --check, ruff check, mypy strict, pytest (+90% coverage floor)
uv run ruff format . && uv run ruff check --fix .   # fix formatting and auto-fixable lint
uv run pre-commit install -t pre-commit -t pre-push # optional: ruff + mypy on commit, pytest on push
```

`./scripts/check.sh` must pass before every commit. CI (`.github/workflows/ci.yml`) runs it on Linux, runs the
tests on macOS and Windows with Python 3.12 (the installer's version), and builds the wheel and runs it the way
the installer does. `main` is protected: changes land through pull requests with passing CI. Actions are pinned to
commit SHAs, and Dependabot proposes updates to them and to the locked dependencies weekly. Config lives in `pyproject.toml`.
mypy runs in strict mode over `src`, `tests` and `scripts`. The only `type: ignore`s are for untyped third-party
APIs (scikit-image, numba).

The default test run (~25 s) needs no network and no private data. `tests/synthetic.py` serves a made-up 1.2 km
landscape in place of every data source, so the whole pipeline runs offline: download, factors, picking, KMZ,
saved state, repick, explain, CLI, MCP tools and jobs. Coverage must stay at 90% or more. `network` tests (live
services) and `slow` tests (saved states on this machine, the GPS baseline) are opt-in:
`uv run pytest -m network` / `-m slow`.

From a checkout, prefix commands with `uv run`, or install the checkout as a global command once (editable, so it
follows your code; rerun it after changing `[project.scripts]`):

```bash
uv tool install --editable --force --python 3.12 .
```

In this repo, Claude Code picks up `.mcp.json` and `.claude/skills/cougarmap` automatically. `uv run cougarmap-mcp`
runs the MCP server; `scripts/mcp_e2e.py` drives it end to end over the network.

## Code map

- `context.py` downloads everything an area needs (`Context`). `factors.py` computes the layers from it
  (`compute_*`, each built from pure array functions). `analyze.py` scores (`combine`), applies the land and
  access rules (`apply_masks`), picks spots and writes the reasons. `export.py` writes the KMZ and JSON. `api.py`
  is what the CLI (`cli.py`), the MCP server (`mcp_server.py`) and background jobs (`jobs.py`) call; keep tool
  keyword arguments in sync across them.
- `sources/` holds one module per data source (elevation, canopy, vector layers, buildings, weather, snow), all
  cached under `~/.cache/cougarmap` by `net.py`.
- `state.py` is the one schema of the model's layers (`Layers`, a TypedDict). Each layer's annotation says whether
  and how `state.pkl` keeps it (float16, float32 or as computed), so adding a layer is one line there. `ModelState`
  is what picking, explaining and export need, and `load_state` reads any saved state. States are versioned
  (`STATE_VERSION`): `migrate` fills in the layers older versions lack with neutral values and lists the ones an
  older model never computed in `ModelState.unmodeled`, so explanations leave them out. There are no "if this
  layer exists" checks in the model code.
- `fieldlog.py` is the field log (`observations.jsonl` in the private folder): camera deployments (arm, zone,
  effort), checks and detection events, snow tracks and crossing-transect surveys, as version-2 records. Old
  records (from the old one-off `log_result`) are migrated to unpaired deployments; the file is rewritten once and the original kept as
  `observations.v1.bak`. GPX/KML/KMZ tracks and routes are read here.
- `truth.py` tests the model against that lion truth: camera rates by arm against a published base rate, a paired
  permutation test within zones, snow tracks against rotated and shifted copies, and the AUC of crossings along
  transects. `api.validate` reports it per area, along with human-pick ranks from `evaluate.py`. Its settings
  are `config.Truth`.
- `evaluate.py` is the human-pick check (setup in `data/private/eval.toml`; see
  [VALIDATION.md](VALIDATION.md)). `gps.py` is the falsification check on open puma GPS collar data
  (`scripts/gps_check.py all` downloads the data into the cache folder, tiles it, analyzes the tiles and writes
  the report). It can veto a change, never tune one.
- `explain`, `repick` and `validate` only read states inside the results folder, because a state file is a pickle.
- `setup_harnesses.py` is `cougarmap setup`. `PLAYBOOK.md` and `INSTRUCTIONS.md` are what agents read: keep the
  server instructions under 2 KB.

Helper scripts: `scripts/debug_panel.py` and `scripts/overview.py` render layers from a saved state;
`scripts/make_installer.sh` builds the double-click installer zip.

## Speed

Cached reruns on a 10-core laptop: a 20 km2 area takes ~7 s, a 228 km2 area ~21 s, a cached `find_hotspots` ~9 s,
`repick` ~3 s and `explain_point` ~1 s. A first `find_hotspots` in a new region takes ~1.5 min, mostly downloads.
Large areas (200+ km2) peak at about 9 GB of RAM.

- Hot loops are numba kernels in `terrain.py`, most of them parallel (the `edt` distance transform, geomorphons,
  labeled min/max, 3x3 median, D8 receivers). Parallel kernels are called from one thread at a time (numba's
  fallback threading layer can't run them concurrently). Threads are used around scipy, numpy, GDAL and zlib,
  which release the GIL (banded `terrain.gaussian`/`binary_opening`, the two walking routes, the circuit solves,
  warps).
- Vector features are projected once per run (`Feature.xy`), and shared masks are memoized per context.
- `state.pkl` (`statefile.py`) stores arrays as byte-shuffled zlib chunks written and read on threads, with its
  index at the end so `repick` rewrites only the options, never the arrays. Older gzip states still load. The MCP
  server keeps the last loaded state in memory for follow-up `explain_point`/`repick`/`validate` calls.
- Every download in `context.build`, the scout and the wind climatology runs in parallel (at most two Overpass
  requests in flight). `find_hotspots` fetches all its blocks while analyzing the first.

**Speedups must not change results.** `tests/test_kernels.py` checks each kernel against the scipy/numpy code it
replaced, and a full rerun of the validation areas should give bit-identical layers and candidates. A speedup that
changes numbers is a model change and goes through the eval.

## Releasing

1. In a pull request, set the new version in `pyproject.toml` and in both places in `server.json` (the
   `tests/test_release.py` check keeps them together), and merge it.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git tag v0.2.0 && git push public v0.2.0
   ```

The `Release` workflow (`.github/workflows/release.yml`) checks that the tag matches the versions and is on `main`,
publishes to PyPI by trusted publishing (the `pypi` environment, no token), creates the GitHub release with the
installer zip, and publishes `server.json` to the MCP Registry, logging in with GitHub OIDC. Versions follow
semver: a change to a tool's arguments, a CLI option or the saved-state format is a minor bump before 1.0.

## Changing the model

Model changes are measured, not argued. Rerun the test areas (`./scripts/rerun_eval.sh`), compare the
pin-free numbers with the current baseline in [VALIDATION.md](VALIDATION.md), run the GPS check
(`uv run python scripts/gps_check.py all`), and write up what the change did, including negative results, in
[experiments/](experiments/).
