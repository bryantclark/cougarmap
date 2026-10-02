---
name: 2026-10-02-water-bodies-pinch-barriers
date: 2026-10-02
description: Ponds and lakes are barriers in the pinch factor (the squeeze beside them and the ends of small ponds); kept for realism although it measured neutral
tags: [model, pinch]
---

# Water bodies are pinch barriers (owner's realism call; neutral on every benchmark)

`compute_pinch` has a seventh component, `pinch_water` (`factors.water_pinch`, settings `config.WaterPinch`): land
between a pond or lake and a cliff, steep ground, an opening or another pond, by the width of the gap, and the
inlets and outlets of ponds under 5 ha. Animals walk around water, so the owner chose it for realism, knowing it
is neutral on the human picks, the SNAPSHOT USA cameras and the collar data (docs/experiments/13-water-barriers.md).
Don't drop it as "no gain", and don't read its reasons as validated: it is a modelling choice, not a measured win.
