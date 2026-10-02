---
# Technical factsheet. Keep it accurate; tools and agents read this header.

type: python-pkg
package_manager: uv
test_runner: pytest
db: none
deploy: pypi
language: python
---

# Stack

## Why these choices

- **uv** manages Python (3.13 via `.python-version`, `>=3.12` supported) and the lockfile; the installer installs
  uv so end users never install Python themselves.
- **numpy / scipy / numba** for the raster model. Hot loops are numba kernels in `terrain.py`; each is checked
  against the scipy/numpy code it replaced (`tests/test_kernels.py`), and speedups must not change results.
- **rasterio / pyproj / shapely / scikit-image** for geodata; **simplekml** writes the Google Earth KMZ.
- **React + MapLibre + Vite** (`explore-app/`, node 22, npm) for the opt-in weights page, built into one HTML
  file each release attaches; installs download it on the first `--interactive` run, so it adds nothing to the
  Python package.
- **typer** for the CLI (`cougarmap`), **mcp** for the agent server (`cougarmap-mcp`). Agents drive the tools;
  long operations run as background jobs so every tool call returns within ~40 s.
- No database: state is files. `state.pkl` per analyzed area, the field log is `observations.jsonl` in the
  private folder, downloads are cached in `~/.cache/cougarmap`.
- Distribution: PyPI (`uv tool install cougarmap`), a GitHub release with a double-click installer zip, and the
  MCP Registry (`server.json`). A `v*` tag on `main` publishes all three (`.github/workflows/release.yml`).

## Local services

None. Every data source is a free public web service (USGS, OSM Overpass, Open-Meteo, NSIDC, WDFW, iNaturalist),
fetched and cached on demand. The default test run needs no network.

## Common commands

- **Setup**: `uv sync`
- **Gate (run before every commit)**: `./scripts/check.sh` (ruff format --check, ruff check, mypy strict, pytest + 90% coverage floor)
- **Test**: `uv run pytest` (opt-in: `-m network`, `-m slow`)
- **Format / lint fix**: `uv run ruff format . && uv run ruff check --fix .`
- **Typecheck**: `uv run mypy`
- **Weights app**: `cd explore-app && npm ci && npm run build` (one file: `explore-app/dist/index.html`)
- **Model eval**: `./scripts/rerun_eval.sh <kml> "<Area>" ...`, or `uv run python scripts/eval_picks.py --by-cam` with `data/private/eval.toml`
- **GPS falsification check**: `uv run python scripts/gps_check.py all`
- **Run the MCP server**: `uv run cougarmap-mcp` (picked up automatically from `.mcp.json`)
- **Build the installer zip**: `./scripts/make_installer.sh`
- **Release**: bump the version in `pyproject.toml` and `server.json`, merge, then `git tag vX.Y.Z && git push public vX.Y.Z`
