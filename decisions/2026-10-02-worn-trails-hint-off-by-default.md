---
name: 2026-10-02-worn-trails-hint-off-by-default
date: 2026-10-02
description: Worn trails from 1 m lidar are a hidden KMZ layer and a per-spot hint, off by default, and never change a score or a pick
tags: [model, placement, performance]
---

# Worn trails are a hint, off by default, and never change a score

Worn trails detected in 1 m 3DEP lidar (`worn.py`, `Options.worn_trails`) give a hidden KMZ layer and each spot's
`worn_trail` hint, offered beside the spot like the trail alternate. As a score factor they did not improve the
human-pick ranks, so scores and picks must stay byte-identical with the option on or off. It is off by default:
on a new ~3 km radius area it downloads about 110 MB of lidar, adds about 40 s per area (three areas per
`find_hotspots`) and about 2 GB of peak memory; `--worn-trails` / `worn_trails=True` turns it on.
