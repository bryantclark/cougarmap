---
paths:
  - '**/*.py'
---

# Code Style Rules

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`); `./scripts/check.sh` enforces it.

## Type safety

- mypy runs `strict` over `src`, `tests`, and `scripts`. Every `type: ignore` carries an error code and is only for
  untyped third-party APIs (scikit-image, numba, rasterio).
- New model layers are declared once in `state.Layers` (the TypedDict annotation says how `state.pkl` keeps it);
  no "if this layer exists" checks in model code — `migrate` fills old states with neutral values.

## Naming

- Math-style names for grids and arrays (`H`, `W`, `A`, `R`, `T`) are fine (N803/N806 ignored).
- Domain names follow `CONTEXT.md` (Area, Spot, Zone, Arm, Check…).

## Error handling

- Fail loudly when correctness matters; no guessed fallbacks in the model.
- `explain`, `repick`, and `validate` only read states inside the results folder (a state file is a pickle).

## Imports

- Heavy or optional imports are deferred on purpose to keep CLI and MCP start-up fast (PLC0415 ignored).

## File & module organization

- Factors are built from pure array functions (`factors.py`); hot loops are numba kernels in `terrain.py`.
- Tunable weights and thresholds live in `config.py`, documented inline next to the factor that uses them.
- `api.py` is the one surface the CLI, MCP server, and jobs call; keep tool keyword arguments in sync across them.
- Speedups must not change results — a kernel ships with a test against the scipy/numpy code it replaces.

## Formatting

- `ruff format`, line length 120. `dict(...)` is the house style for JSON result payloads.
