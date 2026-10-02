---
name: 2026-10-02-weights-page-opt-in-local
date: 2026-10-02
description: The interactive weights page (explore.html) is opt-in, a local file in the results folder; its app is downloaded on first use, not shipped in the Python package
tags: [outputs, privacy, interface]
---

# The weights page is opt-in, local only, and its app is downloaded on first use

`--interactive` / `interactive=True` (analyze, hotspots, repick) writes `explore.html` next to the KMZ; without it a
run writes exactly what it did before. The page embeds the area's model layers, so it is private like the rest of
the results folder: never uploaded, attached to issues or published. The page is a React + MapLibre app
(`explore-app/`) built into one HTML file that each release attaches; the first `--interactive` run downloads the
installed version's copy into the cache and fills it with the area's data, so the package and its Python
dependencies stay as they were. Only map tiles and fonts load from the web. Its kernel
(`explore-app/src/kernel.js`) must stay a faithful port of `analyze.combine`, the site penalties and
`pick_candidates` (`tests/test_explore.py`).
