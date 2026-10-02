---
name: 2026-09-29-public-land-default
date: 2026-09-29
description: Ranked spots are public land; private land is still analyzed and returned as separate private candidates
tags: [access, land]
---

# Ranked spots are public land; private land is a separate list

Most users scout ground they don't own, so the ranked spots are on open-access public land. Private land is never
dropped from the analysis: every run also returns `private_candidates` (P1, P2…) and hidden KMZ layers, which is
how someone scans their own property. There is no option to mix the two lists (removed as unused). Rules out
filtering private land before scoring.
