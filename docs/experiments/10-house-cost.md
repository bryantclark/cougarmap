# 10. Every house costs a little (2026-10-02)

Until now only populated areas cost anything: no cut up to 15 houses within 500 m, the full x0.3 at 60. A
homestead or a few neighbours were free, on the reasoning that lions use barns and homesteads (round 03). The
open-data round tested that against real lions and found the opposite lean: a small cost for every house is the
one score change that improved predictions out of region from all three sources. This reverses the earlier
choice, so it is recorded as a decision the owner can accept or reject.

**Change:** score x (1 + houses within 500 m)^-0.34 on top of the populated-area ramp (`Options.houses_exponent`,
0 turns it off). 1 house x0.79, 3 x0.62, 5 x0.54, 15 x0.39, 30 x0.31, 60 x0.25 (x0.074 with the ramp). A spot
with houses near it says so in its reasons. Saved states pick it up on repick, with no rerun.

## Evidence

The exponents checked (0.15, 0.34, 0.56, 0.8) were fixed before the check, each with placebo reads, and the
inland-Northwest lock-box was read once, at the end.

- **Cameras** (SNAPSHOT USA 2019-2024, CC0; Rooney et al. 2025, doi:10.1111/geb.13941; 1,197 cameras in the 30
  western arrays with a puma), within-array concordance (0.5 = chance) with array-bootstrap intervals:
  - search regions (23 arrays): 0.645 -> 0.655, +0.010 [+0.003, +0.023]; nested leave-one-region-out +0.012;
  - the 3 arrays inside collar study areas: +0.005; the 4 inland-Northwest lock-box arrays: 0.507 -> 0.512,
    neutral (only 5% of those cameras have 15+ houses nearby);
  - placebo (pin moved 1.5-2.5 km): -0.004;
  - the gain is in ranking only: the negative binomial count model with array fixed effects is unchanged
    (rate ratio per SD 1.86 -> 1.85), and it comes mostly from California.
- **Collars** (12 groups, open datasets listed in VALIDATION.md and the open-data round): night steps unchanged
  (-0.004 to +0.005); day beds up in all 12 groups (+0.000 to +0.022), beyond the placebo in 8 of 12; kill sites
  mixed (2 groups up, 2 down).
- **Human picks** and the **GPS check** below.

## Measured on this tree

**Human picks** (pin-free states, 25 human-picked sites in three areas):

| | Median rank (random) | vsR | Top 25% (random) | Fixed-K within 50 / 100 / 150 m | Areas |
|---|---|---|---|---|---|
| v3.1 | 43.5% (88.8%) | 0.813 | 20% (4%) | 12 / 40 / 52% | 40.8, 43.5, 44.1% |
| **houses_exponent 0.34** | **36.1%** (87.0%) | **0.821** | 28% (5%) | 12 / 40 / 52% | 27.7, 36.9, 41.6% |

22 sites rank better, 2 worse, 1 the same; every area improves, the held-out one least (43.5% -> 41.6%).
Random points move up too (88.8% -> 87.0%), so read the gain against vsR (+0.008) as much as the median. With the
water pins in (not the honest numbers): 43.5% -> 35.6%, vsR 0.815 -> 0.822, fixed-K unchanged. Within the
shipping rule (no area worse, vsR up).

**GPS check** (the 28 v3 tile states, three original collar datasets): pooled rank at 20 m 0.514 -> 0.514, at
~500 m 0.506 -> 0.506; animal by animal 23 better, 16 worse (p = 0.34) at 20 m, 24 / 15 at ~500 m (p = 0.20).
The shifted-map null stays at 0.50. No veto.

## Not chosen

- **A steeper exponent, 0.563**: slightly better camera concordance (+0.015) and median rank 35.7%, but fixed-K
  within 150 m fell to 48%. 0.34 is the safer choice.

## Caveats

- Small: +1 point of camera concordance, in ranking only, and nothing in the inland-Northwest arrays, where few
  cameras have houses near them.
- Part of the human-pick gain may reflect how the picks were made (away from people and theft), which is
  exactly what a site penalty is for, but it means the picks are not independent evidence here.
