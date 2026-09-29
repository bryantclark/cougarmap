---
name: 2026-09-29-public-land-default
date: 2026-09-29
description: Spots are public land by default; private land is still analyzed and returned as separate private candidates
tags: [access, land]
---

# Public land only by default

`public_only` is on by default because most users scout ground they don't own. Private land is never dropped from
the analysis: every run also returns `private_candidates` (P1, P2…) and hidden KMZ layers, so switching to
private land (own property, permission) needs `repick`, not a rerun. Rules out filtering private land before
scoring.
