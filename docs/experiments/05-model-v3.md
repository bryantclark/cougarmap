# 05. Model v3 (2026-10-01)

**Shipping rule:** on pin-free human-pick states a change may not lose more than ~1 point of overall median rank
or 0.01 vsR, and no area may get worse by more than ~3 points. The GPS check can only veto.

## What each change did

Each candidate was rebuilt from saved layers (missing inputs from the disk cache only) and recombined with the
production weights. "GPS" is the pooled 20 m rank and, in brackets, the animals where the candidate ranks the
fixes higher than v2.

| Change | Median | vsR | Fixed-K 50/100/150 m | GPS 20 m |
|---|---|---|---|---|
| v2 | 50.3% | 0.805 | 12 / 32 / 48% | 0.508 |
| Travel split alone (spines x0.5 off crossings, gentle-grade bottoms) | 50.7% | 0.801 | 20 / 44 / 56% | 0.509 (27/51) |
| Hunting edge into the opening alone | 49.3% | 0.810 | 16 / 32 / 52% | 0.511 (28/47) |
| Downwind ends per air current alone | 50.7% | 0.803 | 16 / 40 / 52% | 0.508 (15/34) |
| **The three together** | 45.6% | 0.812 | 16 / 44 / 60% | 0.512 (32/51) |
| Populated areas: full cut at 60 houses | 50% | 0.806 | 12 / 32 / 48% | better in 7/9 |
| Recreation sites (x0.7 within 50 m, gone by 400 m) | 50% | 0.806 | 12 / 32 / 48% | 9/15 |
| Ponds under 5 ha are drinking water | 50.6% | 0.805 | 12 / 32 / 48% | 9/16 |
| Winter module (Nov-Apr) | identical at month 10 | | | January tiles: 31/44 (p = 0.010) |
| **v3 (all of it)** | **45.6%** | **0.812** | **16 / 44 / 60%** | **0.514 (36/51, p = 0.005)** |

The hunting edge into the opening shipped only together with the travel split. Alone it cost the small area 11
points, past the 3-point limit; with the split, that area stayed within a point. Per site, v3 against v2 is 14
better and 9 worse (p = 0.40), so not significant.

## Placebos: what is and isn't supported

- **Travel split.** The same spine mass spread evenly, or the crossing gates moved 1.5 km, scored about as well as
  the real gates, on the picks and on GPS. What the data support is that **ridge spines count about half as much
  as drainage bottoms**, not *where* the gates sit. The gates are kept on the method's saddle and winter-thermal
  rules and on the movement science; snow tracks can test them.
- **Hunting edge into the opening.** Pushing the band 25 m deeper into cover instead was worse on both tests
  (51.6%, GPS 21/50 animals): the opening side is what helps.
- **Downwind ends per air current.** With the daytime wind reversed, the combination scores the same median and a
  slightly lower vsR (0.804 vs 0.812). The picks barely tell the real wind from a reversed one. It shipped as a fix
  to how we read our own method (two air currents, where v2 blended them into one), not on the picks' support.
- **Winter module.** Shifted 1.5 km it is better than v2 in 24/37 and 23/40 animals; the real module in 31/44. By
  component: sun-facing slopes alone 34/45 (p < 0.001), low ground 22/34, snow 13/18. Aspect carries most of it.
  Its strength was set before any test and was not raised on GPS.

## Negative results

- **Wind convergence read against the daytime wind:** 51.1%, and it cost the largest area 7 points inside the
  combination. Not shipped; worth revisiting with lion truth.
- **The windward side from the daytime wind:** 51.5%. No.
- **Bottoms weighted by cold-air drainage strength:** 59.4%. Drainage is already in the wind factor.
- **Ridge spines fully gated** (zero away from crossings): 41.3% median but vsR 0.771, and the ridge sites dropped
  15-34 points. A floor of 0.3 or 0.7, or bottoms only, was worse than 0.5.
- **Fence lines as an edge:** no change at all. OpenStreetMap fences cover under 0.05% of the ground here. That
  means no data, not a rejection.
- **Prey-valued openings** (hay and pasture, 3-15-year-old forest loss, 2-35-year-old burns, as a multiplier on
  the edge): no better than the same values shuffled across openings. A bug found on the way: one crop layer's
  "Pasture, Forest" class is forest grazing allotments, not pasture.
- **Water as point sources** (short seasonal reaches and their heads, plus OSM springs and watering places): 58%,
  and clearly worse on GPS (10/46 animals, p = 0.0002).
- **Water uniqueness per source** instead of per connected component: no effect.
- **Water density out of the habitat multiplier:** one area 4 points worse, past the limit. A near miss.
- **No scarcity bonus:** 51%.

## Calibration

The hunting edge reaching into openings roughly doubled edge density, so habitat scored ~1.6x higher and every
area's top five clipped at 100. The display scale went from 5 to 3, so 60+ still means strong. The median rank was
45.63% at both scales: the scale only sets the display.

## The alternate on the trail

Cameras on dirt roads and trails catch more of the lions that pass, but 19 of the 25 human-picked sites sit 45-1,000 m
off mapped lines. So each spot can carry an optional alternate beside a quiet road or trail, and it never replaces
the spot. Measured on the three areas (15 spots each), 8 of 45 spots got one. All 8 were on or beside a road open
in October: on this ground the mapped "quiet" two-tracks are mostly part of the drivable network (92-100% of their
camera-band cells lie within 20 m of an open road). Seasonal forest roads now count as quiet only while they are
closed, and the reason says so when an alternate sits on the open network. Picks and scores were unchanged.
Whether on-trail cameras catch more lions here is a question for the field protocol's on-feature cameras.

## Caveats

- v3 was judged on the same three areas the shipping rule reads, so none of them is a check for it.
- Two sites rank at the bottom under every variant tried. Nothing in the layers explains them; they were likely
  picked for trail detail, prey sign or local knowledge.
- The travel line may partly track access, because roads follow bottoms and ridges. Re-measured on v3, the 17 sites
  more than 100 m from an open road rank slightly *better* than all 25 (43.7% vs 45.6%).
- Pin precision is tens of metres, a large share of the remaining gap.
