# 12. Reshaped factors, a prey layer and cross-validated tuning (2026-10-02)

**Question.** The heuristic beat every learned model in round 09, but it is only modestly better than chance:
human picks at a median rank of 36%, within-array camera concordance about 0.62, and chance in the four
inland-Northwest arrays. Three ideas for doing better were tested here, and none of them ships:

- reshaping the factors to match what the open data agreed on;
- adding a prey (deer and elk) layer;
- tuning every weight and definition at once, under cross-validation.

**Answer.** The factors and weights are close to as good as these layers allow. The largest real gains this round
were a couple of points on the human picks and about +0.01 of camera concordance, inside the noise of the
held-out checks. The next gain needs different information (local field results, or where on a spot to hang
the camera; see [11](11-worn-trails.md)), not more re-tuning.

## How it was measured

Every variant was scored on full rasters through the 20 m zone and the site penalties, on three benchmarks,
against production with 95% intervals (array bootstrap for cameras, animal bootstrap for collars). The
inland-Northwest lock-box was opened once per round, for the final candidates only.

- **Human picks:** pin-free states, median rank, vsR, top 25% and per-site better/worse counts.
- **Cameras:** SNAPSHOT USA within-array concordance, outside the inland Northwest.
- **Collars:** fine-scale travel paths and kill sites, leaving one dataset out wherever anything was fitted.

The baseline is v3.2 (the per-house cost, [10](10-house-cost.md)): median 36.1%, vsR 0.821, top 25% 28%.
Experiments scored with v3.1 code report 43.5% for the same states; that gap is the house cost, not stale
states (rebuilt with v3.2 code, every layer was byte-identical).

## 1. Reshaped factors (no parameters fitted)

| Variant | Picks median (vsR), better/worse | Cameras Δ | Paths Δ | Kills Δ |
|---|---|---|---|---|
| Production v3.2 | 36.1% (0.821) | 0.625 | 0.634 | 0.586 |
| **A**: signed canopy-edge multiplier, pinch 0.15, stack 1.10 | 34.3% (0.818), 14 / 4 | +0.007 [-0.001, +0.014] | -0.001 | +0.007 [+0.001, +0.012] |
| **C**: as A, with a bench bonus in place of the edge multiplier | 34.8% (0.818), 12 / 4 | -0.000 | +0.003 | +0.000 |
| Fine ruggedness (vrm30^0.1) | 40.6% (0.815) | -0.005, significant | 0.000 | +0.007 |
| Local-relative score (P / (blur 250 m + 1)^0.25) | 39.1% (0.824), 5 / 16 | -0.008 | +0.009 | +0.002 |
| Rock as edge cover (slope >= 45 deg) | 37.5% (0.821) | -0.003 | -0.009 | -0.001 |
| Small incised draws as travel lines (4-12 m deep) | 36.9% (0.820), 1 / 9 | -0.001 | -0.006 | +0.001 |

- **A** is the edge multiplier M = (1 + |d|)^-0.10 x exp(0.05 asinh(d / 20)), where d is the signed distance to the
  canopy edge (cover = CHM >= 4 m, positive inside cover), applied to spot x habitat x season before the zone.
  It is the only variant that was non-negative on all three benchmarks, but the camera gain is not significant,
  and it would need `score_scale` 3.0 -> about 2.7 to keep the 0-100 scale. Lock-box: -0.003, neutral.
- **C** adds 1 + 0.25 x B to the score, where B = (1 - ramp(slope, 10 deg, 20 deg)) x ramp(uphill relief within
  100 m, 15 m, 60 m). This is a flat bench below big relief. It was the only change that lifted under-ranked
  cliff-base sites, but it was even on the human picks overall and neutral elsewhere.
- **Ruggedness:** collared lions bed in it, but cameras and human picks reject it.
- **Local-relative scoring** (lions select what is scarce) helped travel paths only.
- **Rock as cover and incised draws** came from reviewing aerial imagery at the human picks: several sat on open
  benches under cliffs or in narrow gullies the model rated low. Collar paths rejected both clearly, at every
  threshold tried.
- **No consistent effect:** opening-rim weight 0.5, edge floor 0.5 or no edge habitat term, wind weight 0.2,
  travel weight 0.5, and cliff or bench as a pinch component.

## 2. A prey layer

Prey is the method's first factor, and the model only infers it (edges, water, winter range). SNAPSHOT USA
records deer and elk at every camera: 21,677 deer events (11,078 mule deer, 10,053 white-tailed deer, 546 unknown)
and 2,140 elk events in the western arrays, against about 400 puma events.

- **Prey models:** conditional Poisson with array x year fixed effects, a log-nights offset, camera feature type as
  a nuisance term, and ridge or elastic net on 8 terrain and cover features.
  - Prey goes up with cover at 100 m, openness at 300 m, being just inside the cover edge, distance from roads,
    and nearness to water.
  - With the whole region held out, concordance was only 0.54 for deer and 0.55 for elk; feature type alone gives
    0.533.
  - With only the camera's own array held out, it was 0.589 for deer and 0.586 for elk.
- **The ceiling, from observed prey (no model):** within arrays outside the inland Northwest, puma detections do
  not follow observed deer and elk rates. The negative binomial slope on log prey rate is +0.02 [-0.13, +0.17],
  and multiplying production by observed prey^0.5 changes concordance by -0.001. Deer traffic varies a lot
  between nearby cameras (in forested inland-Northwest arrays, deer reached every camera, with a within-array
  CV of 0.77 and a 90th/10th percentile ratio of about 6), but lion detections don't track it.
- **As a factor** (measured on v3.1, cross-fitted so a camera's own array never trained its prey layer):
  - every positive prey multiplier cost the human picks 3-16 points of median rank;
  - production x prey^0.5 was neutral on cameras and fell in the lock-box, -0.018 [-0.026, -0.016];
  - production x elk^0.5 helped cameras (+0.009) and paths (+0.017), but the path gain came from a study area with
    no elk, so the layer stood in for canopy, road distance and water, which the model already has;
  - blurred prey density at 250 m and 500 m was flat.

In this country deer are everywhere, so prey density can't tell one spot from another. What sets a spot apart is
the ground that funnels and hides a hunting lion.

## 3. Cross-validated tuning

This tuned about 30 parameters at once:

- the five weights, the stack multiplier and its thresholds;
- the habitat floors, saturations and context window, and the zone;
- the travel split (drainage vs spine) and the spine floor;
- the pinch component weights, the edge definition's thresholds and radii, and the bench and edge multipliers.

The search was a coordinate search on a rank-normalized sum of the three benchmarks. The tuning itself was
cross-validated over six folds:

- human picks with one area held out, rotating;
- cameras with half the regions held out;
- collars with one dataset held out.

Only the held-out numbers count.

| | Picks median (vsR), better/worse | Cameras Δ | Collars Δ |
|---|---|---|---|
| Production v3.2 | 36.1% (0.821) | 0.624 | 0.596 |
| Full search, 30 parameters | 38.2% (0.814), 9 / 15 | +0.012 [-0.000, +0.031] | +0.006 [-0.004, +0.015] |
| Weights and stack only, 6 parameters | 38.7% (0.819), 14 / 9 | +0.010 [-0.002, +0.024] | -0.001 [-0.006, +0.004] |

Neither passes the shipping rule. Tuned on all the data (in-sample), the picks reach 34.6%, but vsR falls to
0.810, breaking the rule, and the lock-box moves -0.001 [-0.032, +0.019].

- **Directions that held in most folds:** travel 0.30 -> 0.43 (5 of 6), stack 1.25 -> 1.10 (4 of 6), edges ->
  0.20-0.25 (4 of 6), pinch -> 0.15. Together these give about +0.01 on cameras, not significant, with no held-out
  gain on the picks and nothing in the lock-box. They are a hypothesis for local field data, not a change.
- **Definitions were not stable:** no edge definition was picked consistently, and the cliff-base weight landed
  at 0, 0.5 or 1.5 depending on the fold. The context window, zone, floors and saturations were as inconsistent.
- **Approximations:** cameras and collars were scored on cached windows around each point (zone blur cut at
  2.5 sigma, collar windows every 6 m, half the path strata). The baseline reproduces within 0.001 on cameras and
  0.003 on kill ranks. Pick numbers are exact. Edge-definition variants were applied as differences from a
  reproduced baseline, because the saved canopy height is rounded to whole metres (about 1.6 points of noise).

## Caveats

- Every benchmark here is weak on its own. The human picks are 25 sites in three areas. Cameras in the inland
  Northwest are four arrays. Collar fixes carry 10-20 m of GPS error, which blurs camera-scale effects. A real
  +0.01 can hide in that noise, and so can a real loss.
- Gains that fit one benchmark and lose another (ruggedness, local-relative scoring, elk) are the signature of
  each source measuring something different: beds, travel and camera passes are different behaviours.
