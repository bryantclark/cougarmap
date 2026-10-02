# Experiments

This folder records what we tried on CougarMap, how we measured it, and what we kept, including the things that
didn't work. The current model is described in [../HOW_IT_WORKS.md](../HOW_IT_WORKS.md), and the tests in
[../VALIDATION.md](../VALIDATION.md).

| # | Round | Dates | Headline |
|---|---|---|---|
| 01 | [The first model](01-first-model.md) | 2026-09-29 | Seven free data sources, the 850 hPa wind, the cliff-base fix, and a scoring ablation that kept simple weights |
| 02 | [Edges and an honest metric](02-edges-and-metric.md) | 2026-09-30 to 10-01 | The hunting edge; the cell-percentile metric was lenient, so it was replaced by pick rank (86% vs random 99%) |
| 03 | [Model v2](03-model-v2.md) | 2026-10-01 | Three parallel designs and a judge: spot x habitat over a 20 m zone, a terrain travel line, a paved-road penalty. Median rank 50% |
| 04 | [Lion truth and the GPS check](04-lion-truth-and-gps.md) | 2026-10-01 | No open lion data at camera scale, so a field-log harness and a veto-only check on open puma GPS data |
| 05 | [Model v3](05-model-v3.md) | 2026-10-01 | Travel split, hunting edge into the opening, two air currents, a winter module. Median 46%, GPS better in 36/51 animals |
| 06 | [Wind audit and v3.1](06-wind-audit.md) | 2026-10-01 to 10-02 | The prevailing wind against stations, reanalysis and WindNinja. Shipped hillside drainage (median 43.5%) and honest wind labels |
| 07 | [Speed and reliability](07-speed-and-reliability.md) | 2026-09-29 to 10-01 | 4-5x faster reruns with bit-identical results |
| 08 | [Agents and usability](08-agents-and-usability.md) | 2026-09-30 to 10-02 | Background jobs, seven agent apps, the private-land toggle, the legend, and a CLI that takes coordinates as typed |
| 09 | [Open camera and collar data](09-open-camera-and-collar-data.md) | 2026-10-02 | SNAPSHOT USA cameras and 12 collar groups: production predicts detections within arrays (0.62), not yet in the inland Northwest (0.51); placement at the spot is worth ~3x; refits, 27 features, a CNN, a stronger winter and collar blends all failed |
| 10 | [Every house costs a little](10-house-cost.md) | 2026-10-02 | x(1 + houses within 500 m)^-0.34: camera concordance +0.010 out of region, day beds up in 12/12 collar groups, median 43.5% -> 36.1% |

## How we experiment

- **One question per experiment.** Write down what should change before measuring it.
- **Re-score saved states first.** Every analyzed area saves its layers (`state.pkl`), so most ideas are tried by
  recombining saved layers in seconds, with no downloads, on frozen copies of the states. Only a change that
  survives that gets a full rerun.
- **Read every number against its null.** Human-pick ranks against random points with the same treatment.
  GPS fixes against the animal's own available steps. A new terrain term against the same term built from a
  shifted DEM (a placebo). New smoothing against the same smoothing without the new term.
- **The human picks are a tripwire, not proof.** It has 25 sites in three areas, every area has shaped some
  choice, and per-site changes are rarely significant. A change ships if it keeps the human-pick numbers within
  the shipping rule and the GPS check doesn't veto it. Lion truth from the field outranks both once there is
  enough of it.
- **Keep the negative results.** A clear "no" saves the next person a week.
- **Independent review.** Big rounds used several independent designs and a reviewer that re-ran every claim
  before anything shipped. Reviews caught pin leakage, two robustness bugs and a missing smoothing control.

## Terms

- **Human picks:** a private set of 25 camera sites a person picked by hand from the same factors, in three areas: one small (~20 km2) and two
  large (~230 and ~310 km2). Locations are not published.
- **Rank:** where a site lands among all distinct candidate spots in its area (lower is better).
- **vsR:** the chance a site outranks a random point (0.5 = chance).
- **Fixed-K:** the share of sites within 50/100/150 m of one of the top 3 spots per km2.
- **Pin-free:** states analyzed without the water pins of the same file, which sat near some human-picked sites.
