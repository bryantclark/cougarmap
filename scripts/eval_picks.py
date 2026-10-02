"""The pick-level camera-placement evaluation (cougarmap.evaluate): run it, or import evaluate from here.

uv run python scripts/eval_picks.py [--raw] [--by-cam]          # the setup in data/private/eval.toml
uv run python scripts/eval_picks.py --kml picks.kml --area KEY=SLUG [--area ...] [--check KEY]
"""

from __future__ import annotations

from cougarmap.evaluate import EvalConfig, evaluate, main, production_score, raw_score, report

__all__ = ["EvalConfig", "evaluate", "production_score", "raw_score", "report"]

if __name__ == "__main__":
    main()
