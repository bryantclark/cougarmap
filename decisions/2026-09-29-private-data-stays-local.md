---
name: 2026-09-29-private-data-stays-local
date: 2026-09-29
description: Users' KML files, camera locations, the field log and the eval setup never enter git or any external service
tags: [privacy]
---

# Private data never leaves the machine

`data/private/` and `out/` are git-ignored; installed copies keep them in `~/Documents/CougarMap/my-data`.
The human-pick setup (`data/private/eval.toml`) is private too. Issues, PRs, commits and published docs never
carry coordinates, private area or camera names, or people's names; private validation results are published in
aggregate only. Rules out committing fixtures built from real camera data — tests use `tests/synthetic.py`.
