# 02. Edges and an honest metric (2026-09-30 to 10-01)

## The hunting edge

**Question:** the edge layer looked random on the map, not like meadow edges. Why?

**Diagnosis:** in a large saved run, 93% of strong edge cells were valley bottoms and ridgelines, weighted 0.65
and 0.5, above a real meadow edge (capped at 0.45). The canopy input was fine: the 1 m canopy height map lined up
with the openings on NAIP aerial imagery. Sentinel-2 greenness also separated grass from timber, but at 10 m it
was too blurry to use.

**Change:** an edge is cover (trees 4 m or taller within 8 m), backed by a stand, looking onto an opening of
0.15 ha or more within 60 m, with road and trail cuts filtered out. The most downwind end of each opening ranks
highest. Valley, ridge, water and trail lines became tie-breakers. A sweep of ~20 variants, re-scored from saved
runs in seconds, found the best version: drop the "solid stand" rule (open pine parkland counts as cover) and
give full credit 35 m back, fading out by 75 m.

**Kept.** The layer traces real timber edges on aerial imagery. The share of sites in the top 10% of the held-out
area rose from 38% to 54%.

## The metric was lenient

The map didn't look like the human picks, yet the reported agreement was high. Ranking every candidate spot
in each area and placing each site in that ranking showed why. Most ground scores near zero, so random points
averaged the ~70th cell percentile. **On the pick ranking, the typical site was beaten by 86% of the area's
candidate spots (random: 99%)**, and only 4% landed in the top 10%. Within 1 km of a site, a median of 16
candidate spots beat it, against 17 for a random point.

**Replaced the metric** with the pick-rank harness (`cougarmap/evaluate.py`) and corrected the write-up. The
held-out area had been used while tuning edges, so it was reclassified as a third tuning area.

## Why our picks differed

Factor values at the human-picked sites, at our top-15 picks per area, and at 900 random points:

- "Pinch" was effectively cliff base. Cliff bases cover 1-2.5% of the ground but sat under 76% of our top
  picks, against 8% of the sites (random 7%). The hard x1.5 stacking rule caused it.
- Strong wind sat under 44% of our picks and 16% of the sites, pulling picks into valley bottoms.
- 8 of the 25 sites had no factor at 0.5 or more.
- 24% of the sites are within 40 m of an open road (random 10%).

## Houses

Some picks sat next to houses. In a rural test area OpenStreetMap had 1 building where Microsoft's building
footprints had 639. A penalty of 1 / (1 + 0.3 per building within 150 m) removed homestead picks without moving
the metric. *Replaced in round 03 by a populated-area penalty.*

## Water

- Scarcity had halved nearly every water source. It became a bonus (0.75 + 0.25 x scarcity), with full strength
  kept within 40 m of the water. Neutral to slightly positive, and strong water could finally count.
- The predicted "likely water" layer from round 01 was the strongest water source on ~15% of one area, and no
  human-picked site drew its water score from it. **Removed**, with the Sentinel-2 download.

## Weights alone don't fix it (negative result)

Ten recombinations of saved layers (water weight 0.30, a cliff-base cap, a stack multiplier of 1.2, a soft stack,
equal weights and combinations) moved the typical site at best from the bottom 14% to the bottom 24% of our list.
Water at 0.30 was the most useful single change. The hypothesis we carried into v2: **the model scored where a
lion hunts, but a camera needs where a lion walks past.**
