---
name: 2026-10-02-worn-trails-on-unless-fast
date: 2026-10-02
description: Worn trails from 1 m lidar are a hidden KMZ layer and a per-spot hint, on by default (--fast skips them), and never change a score or a pick
tags: [model, placement, performance]
---

# Worn trails are a hint, on unless the run is fast, and never change a score

Worn trails detected in 1 m 3DEP lidar (`worn.py`, `Options.worn_trails`) give a hidden KMZ layer and each spot's
`worn_trail` hint, offered beside the spot like the trail alternate. As a score factor they did not improve the
human-pick ranks, so scores and picks must stay byte-identical with them on or off. The owner chose slow-but-complete
by default: a new ~3 km area downloads about 110 MB of lidar and takes about 40 s longer (about 7 s once cached) and
about 2 GB more peak memory. `--fast` / `fast=True` skips them (and any later slow extra); states saved before
version 6 keep them off.
