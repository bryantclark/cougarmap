#!/bin/sh
# The project gate: formatting, lint, types, tests (+coverage floor). Run before every commit.
#   ./scripts/check.sh
# Network tests and the saved-state eval are opt-in: uv run pytest -m network / uv run pytest -m slow
set -e
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
