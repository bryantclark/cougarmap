# 03. Model v2 (2026-10-01)

## Three designs and a judge

Three designs were built in parallel on frozen saved states, then a judge re-ran every claim. Baseline: median
rank 86% (random 99%), vsR 0.73.

**Design 1, infrastructure.**
- A paved-road penalty (x0.3 beside pavement, back to x1 at 800 m) replaced the blanket near-any-road penalty:
  median 86% -> 75%, smooth across settings (reach 500/800/1200 m: 80/75/75%). **Kept.**
- A quiet-travel-line factor (closed tracks, two-tracks, trails, weighted by quietness) lifted the few roadside
  sites and cost the most-held-out area 2-5 points in every variant. 19 of 25 sites are 45-1,000 m from any mapped
  line. **Rejected** as an overfitting risk; kept only as an explanation.
- Detectors for unmapped lines (narrow canopy strips, lidar benches, junctions, dead ends, crossings) showed no
  enrichment at the sites.

**Design 2, terrain travel lines.**
- The line is the topographic position index at 300 m on the 10 m DEM, percentile-ranked and squared: drainage
  bottoms and ridge spines. Sites are bimodal in it (14 of 25 in the extreme 20%, ~5 expected).
- As a gate on a 100 m smoothed score it reached 41%. The smoothing-only control scored 65%, and a placebo line
  from a misaligned DEM (flipped, or shifted 1.5 km) scored 80-94%, so the terrain signal is real.
- Least-cost-path betweenness (77-89%) and circuit-theory current (80-87%) both lost to the simple line.
- The judge rejected the gate form: it mostly thinned peaks, tied the baseline on fixed-K, and fell to 16% within
  100 m. **Kept the line as an added term**, not a gate.

**Design 3, the combination.**
- spot x habitat context (edge and water density at a 250 m Gaussian), averaged over a 20 m camera zone, with a
  soft stacking ramp (up to 1.25x per factor, from 0.35 to 0.65) instead of the hard x1.5 at 0.5. Median 53%.
- Weights, stacking, a cliff cap, canopy-contrast edges, a cover-neck corridor term and percentile-normalized
  factors each moved it by about 3 points or less.
- A fitted presence-vs-available logistic model (18 features, leave-one-area-out) scored 99-100% on held-out
  areas, with coefficients that flipped sign between folds. **25 sites are far too few to fit**; the model stays
  mechanistic.

**The judge's combination:** design 3 + the design 2 line as a fifth added term (weight 0.3, not stacked) + the
design 1 paved penalty.

| Remove | Overall median |
|---|---|
| nothing | 50% |
| habitat context | 67% |
| 20 m zone | 63% |
| paved-road penalty | 56% |

Robustness: paved reach 500/800/1200 m 49/46/46; context sigma 150/250/400 m 50/46/47; TPI radius 200/300/500 m
54/50/54; travel weight 0.1-0.5 a flat plateau of 48-53. None of the chosen values sits on a knife edge. A 30 m
zone scored 38%, but mostly by thinning peaks (random 85%); 20 m is the camera's physical scale.

## Results, and what a review changed

A review found that water pins in the human-pick file sat within ~80-135 m of three sites and fed the habitat
window, so **every headline number since is pin-free**. It also asked for a smoothing-only control (the old score
blurred over the same 20 m zone).

| | Median rank (random) | vsR | Top 25% | Fixed-K within 150 m |
|---|---|---|---|---|
| Old score | 86% (99%) | 0.73 | 8% | 48% |
| Old score blurred to 20 m (control) | 85% (98%) | 0.72 | 12% | 56% |
| **v2** | **50% (88%)** | **0.80** | 16% | 44% |

Paired per site: 21 better and 2 worse than the old score (p < 0.001); 19 and 4 against the control (p = 0.003).

**What it means:** rank and vsR improved a lot. Fixed-K didn't: the model rates the hand-picked spots much higher,
but its top picks are no closer to them, and the blurred old score did better on fixed-K.

Bugs the reviewers found, all fixed: an absolute peak threshold returned zero spots in dry or solid-timber areas;
in-place state updates weren't atomic (222 of 556 concurrent loads failed in a race test, 0 after the fix); canopy
was truncated on repick instead of rounded. The codebase went to ruff, mypy strict, and an offline test suite on a
synthetic landscape.

## Populated areas instead of buildings

One human-picked site ranked low pin-free partly because of a building penalty from a single
~28 m2 footprint beside it that may not exist. Lions use barns and homesteads, so what should matter is density,
not single buildings. **Change:** count footprints of 50 m2 or more within 500 m, with no cut up to 15 houses,
fading to x0.3. For scale: a town core had ~480 houses within 500 m, a small-town center ~220, a town edge ~40, a
rural homestead 2. Overall it changed nothing (23 of 25 sites have no house within 500 m). **Kept.** In v3 the
full cut moved to 60 houses (~76 per km2; Maletzke et al. 2017).

## A drinking-spot water model (negative result)

**Hypothesis:** what matters is where prey and lions drink: water reachable from cover, gentle banks, creek
sections at a timber edge, coves. The water pins all sat where timber reaches water; one was a small
timber-ringed pond the code had called a "lake" (over 0.5 ha), and one was an unmapped seep.

| Water layer | Median rank |
|---|---|
| current | 50% |
| drinking spots (cover + gentle banks) | 50% |
| + creek timber-edge and crossing bonuses | 46%, but the largest area worse and fewer sites in the top 25% |
| current + small-pond fix | 51% |
| no scarcity bonus | 51% |

Nothing beat the current layer beyond noise. Timber touches nearly every creek in this country, and the real
drinking features (pools, game-trail banks, seeps) aren't in any public map. **Not shipped**, except the
small-pond fix (ponds under 5 ha count as drinking water, in v3). The lesson: water a user pins carries
information the maps don't.
