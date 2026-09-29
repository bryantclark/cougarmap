---
name: 2026-09-30-speedups-preserve-results
date: 2026-09-30
description: Performance work must leave layers and candidates bit-identical
tags: [performance, model]
---

# Speedups must not change results

Every numba kernel is tested against the scipy/numpy code it replaced (`tests/test_kernels.py`), and a full
rerun of the validation areas should give bit-identical layers and candidates. A speedup that changes numbers is
a model change and goes through the eval. Rules out approximate fast paths slipped in as optimizations.
