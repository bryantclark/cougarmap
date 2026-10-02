---
name: 2026-10-02-weights-page-opt-in-local
date: 2026-10-02
description: The interactive weights page (explore.html) is opt-in, a local file in the results folder, and needs no extra Python package
tags: [outputs, privacy, interface]
---

# The weights page is opt-in and local only

`--interactive` / `interactive=True` (analyze, hotspots, repick) writes `explore.html` next to the KMZ; without it a
run writes exactly what it did before. The page embeds the area's model layers, so it is private like the rest of
the results folder: never uploaded, attached to issues or published. It is built with numpy and the standard
library only; Leaflet and the imagery load from a CDN when it is opened. Its JS kernel must stay a faithful port
of `analyze.combine`, the site penalties and `pick_candidates` (`tests/test_explore.py`).
