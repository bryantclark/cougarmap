---
name: 2026-10-01-human-picks-primary-metric
date: 2026-10-01
description: Pin-free agreement with human camera picks is the primary model metric; open GPS collar data may only veto clear harm
tags: [model, validation]
---

# Human picks are the primary model metric; GPS data only vetoes

Camera spots people picked by hand from the same factors are the only camera-scale reference we have in quantity,
so model changes are
judged first by how human picks rank on pin-free states (`scripts/eval_picks.py`, setup in
`data/private/eval.toml`). Out-of-region GPS collar data (`scripts/gps_check.py`) can block a change that clearly
hurts, never tune one. Lion truth from the field log outranks both once there is enough of it. Rules out demoting
the method's untested rules (wind, cold-air drainage, downwind edges) on literature or GPS data alone.
