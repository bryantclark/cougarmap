# 04. Lion truth and the GPS check (2026-10-01)

## Is there real lion data to test against?

- No open dataset tests the model at camera scale (20 m) in our region. The closest study, the Washington
  Predator-Prey Project (Bassing et al. 2023, Ecol. Appl.: random on-trail cameras plus GPS collars), withholds
  camera and fix locations as sensitive. It does publish detection rates, which became our camera base rate.
- Open GPS collar data exists in other ecosystems: Fishlake National Forest (Utah), the Olympic Cougar Project and
  UDWR (Utah), with fixes 1-8 hours apart. SNAPSHOT USA is open but has to be downloaded by hand. iNaturalist is
  too coarse.
- From the literature: on-trail cameras detect far more lions; drainage bottoms are used more than ridgelines for
  travel (Dickson & Beier 2007); kills cluster at edges, including the open side (Laundré & Hernández 2003;
  Holmes & Laundré 2006). No study has tested wind rules for cougars.
- **A realistic target.** With a few lions per 100 km2 and home ranges of 240-750 km2, "perfect" isn't achievable
  at 20 m. A measurable goal is 2-3x the detection rate of a random trail camera.

**Decision:** build a lion-truth harness for field data, use the open GPS data only to falsify, and keep the
human picks as the main gate.

## The lion-truth harness

A field log (schema v2) for camera deployments by arm (model, human, control, on-feature, off-feature), checks
with downtime, detection events (30-minute independence), snow tracks and crossing transects. A survey with zero
crossings counts as an absence. The tests:

- **Cameras:** detections per 100 camera-nights against the published base rate (0.87 in summer, 0.37 in winter),
  and a rate ratio with a sign-flip permutation test across paired zones.
- **Snow tracks:** each track against 200 rotated and shifted copies of itself.
- **Transects:** the AUC of crossings along their own route.

Synthetic tests plant a signal and recover it (a planted track at the 100th percentile, crossings AUC above 0.9,
3x cameras over 30 zones at p < 0.01). With no signal they stay at ~0.5. **Power:** about 15 paired zones to see
a 3x gain (about 40 for 2x), or 18-37 snow tracks. That is why [the field protocol](../FIELD_PROTOCOL.md) puts snow
tracks and crossing routes first.

## The GPS falsification check

Data: 52 pumas and about 16,000 night and twilight fixes, on 14 tiles of 8 km over Fishlake, Olympic and UDWR.
Each used fix is ranked against 15 available points drawn from the same animal's own step lengths and turning
angles (iSSF-style), at 20 m and at ~500 m, with a Boyce index. A map-shift null lands at 0.50 everywhere, as it
should. A DEM-shift placebo moves the terrain 1.5 km.

**The rule: veto only, never tune.** The data are other ecosystems, and fixes hours apart lean toward beds and
kills.

| | Pooled rank, 20 m | ~500 m | Animals above chance, 20 m |
|---|---|---|---|
| v2 | 0.508 | 0.500 | 31/51 |
| v3 | **0.514** | 0.506 | 32/52 |

- **Fishlake is the one clear pass:** 0.543, with 7 of 8 animals above chance, and the model beats the DEM-shift
  placebo there.
- Olympic and UDWR sit near chance, and the placebo scores as well there, so overall the model fails the placebo
  rule. A pass would be weak support anyway; a fail is a warning.
- Ablations: the hunting edge carries signal (33/49 animals, p = 0.02); the winter module helps on January tiles
  (30/39, p = 0.001); wind, pinch points and the site penalties are neutral. Wind, cold-air drainage and downwind
  edges can't be tested by fixes hours apart, so none of this demotes them.

The full tables are in [../VALIDATION.md](../VALIDATION.md#gps-falsification-check-open-collar-data-other-ecosystems).
