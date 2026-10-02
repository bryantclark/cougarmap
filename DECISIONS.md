# Decisions

<!--
Index of durable technical decisions — the cross-cutting constraints an agent or contributor
must not contradict silently. One file per decision under `decisions/<YYYY-MM-DD>-<slug>.md`;
this file is the scan surface. Keep index lines to one line each.

Scope: code style → .ai/code-style.md · testing → .ai/testing.md · stack → STACK.md ·
vocabulary → CONTEXT.md · per-feature designs → the GitHub issue · model measurements →
docs/VALIDATION.md. An entry belongs here only if violating it from an unrelated change
would be a bug.

LIVE entries only: when a decision is superseded, delete its file and its index line — git
history is the journal.
-->

## Access and land

- [Ranked spots are public land; private land is a separate list](decisions/2026-09-29-public-land-default.md) — private land is still analyzed and returned as separate private candidates
- [Within 1 mile of walking from an open road](decisions/2026-09-29-one-mile-open-road.md) — walking distance from roads open to vehicles that month

## Model and validation

- [Every house within 500 m costs a little](decisions/2026-10-02-every-house-costs-a-little.md) — score x (1 + houses)^-0.34 on top of the populated-area cut; homesteads are no longer free
- [Human picks are the primary model metric; GPS data only vetoes](decisions/2026-10-01-human-picks-primary-metric.md) — pin-free human-pick ranks judge changes; out-of-region collar data can only block clear harm
- [Speedups must not change results](decisions/2026-09-30-speedups-preserve-results.md) — kernels tested against the code they replace; reruns stay bit-identical
- [Worn trails are a hint, on unless the run is fast, and never change a score](decisions/2026-10-02-worn-trails-on-unless-fast.md) — 1 m lidar trails: a hidden KMZ layer and a per-spot hint; `--fast` skips them; picks byte-identical on or off

## Agent interface

- [The weights page is opt-in and local only](decisions/2026-10-02-weights-page-opt-in-local.md) — `--interactive` writes explore.html beside the KMZ; default runs unchanged; its app downloads on first use
- [Tool calls return within 40 s; long work is a job](decisions/2026-09-30-background-jobs-40s.md) — background jobs keep every app under its tool timeout

## Privacy

- [Private data never leaves the machine](decisions/2026-09-29-private-data-stays-local.md) — KML files, camera locations, the field log and the eval setup stay out of git and external services
