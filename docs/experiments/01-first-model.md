# 01. The first model (2026-09-29)

The first version stacked the four factors (wind, edges, pinch points, limited water) with weights of 0.35,
0.30, 0.20 and 0.15, and a x1.5 bonus for each extra factor at 0.5 or more. It was measured with a
**cell-percentile** metric: for each human-picked site, the best score within a small window of the pin, as a
percentile of the area. Random points got the same window. (Round 02 found this metric lenient and replaced it.)

## Can every factor come from free, keyless data?

We probed USGS 3DEP 1 m lidar (windowed COG reads, ~5.5 s for a 1 km window; the 3DEP ImageServer export
returned 504s), the 1/3" and 1" DEMs (1-2 s), Meta/WRI canopy height v2 (~4.6 s; v1 was ~18 s), the NHD water
MapServer, OSM Overpass, USFS MVUM road seasons, PAD-US and Open-Meteo. **All worked.** NHD timed out on large
boxes but answered small tiles in ~1.6 s, so NHD queries are tiled and run in parallel.

## Which wind is "the prevailing wind"?

Ten years of ERA5 wind at 100 m gave a N/NNE dawn wind in every month (consistency 0.6-0.73). That is valley
drainage, not the regional wind. The 850 hPa wind above the ridgetops, on high-pressure dawn and dusk hours, was
WSW-SW in most months (consistency 0.2-0.43), which matched local field knowledge. **Kept:** the prevailing
wind is the 850 hPa high-pressure dawn/dusk wind. The surface wind is reported for reference, and users can
override the direction. (Round 06 revisits this.)

## First run, and the cliff-base fix

A 20 km2 area at 3 m took 30 s. Human-picked sites averaged the 83rd percentile against 78 for random points. One
site scored *below* random, with every factor at zero on the pin. A layer panel showed
why: the site sits ~120 m below a rock face, at the foot of a talus and escarpment band, and the model measured
"cliff base" from the face.

**Change:** a barrier is a connected band of steep ground (35 degrees or more, 12 m or more of relief). Bands with
rock (50 degrees or more) count fully and plain steep bands 70%. The base is gentle ground within 60 m of the foot
of the band. **Result:** that area went from 83 to 95 (random 82), and that site moved up sharply.
**Kept**, as a modeling fix rather than a weight tweak.

## Smaller changes, all kept

- **Likely unmapped water** from land shape (wetness index at channel heads, slope feet, pits) plus late-summer
  Sentinel-2 greenness. It barely moved the metric (94.8 -> 95.1) and gave better reasons. *Removed in round 02:
  no human-picked site drew its water from it.*
- **Resolution cap** raised from 4M to 12M cells, so a 300 km2 area runs at ~5 m and still uses the lidar.
- **Cold-air drainage spread across the valley floor** (tens of metres wide instead of a 10 m channel), wider
  valley edges, and trails under canopy as travel edges. The gap to random held, and the result was more
  realistic.
- **Windward ridges** limited to ~100 m below crests; terrain water only at channel heads, slope feet and pits.
- **Wind channeled along valleys** in proportion to how well the valley lines up with the wind. It was neutral
  on the metric (82.9 -> 82.7 in the held-out area) and kept because it is physically right and makes the reasons
  accurate.

## Tightening the eval window

With a 30 m window, 46% of random points reached the top 10%. A 15 m window, about the accuracy of a Google
Earth pin, fixed that and became the standard.

## Scoring variants: a negative result

Would another way of combining the factors beat the weights? Per-factor percentile normalization, equal weights,
both together, and "max" variants that credit one strong factor were all scored from saved states with no
recompute. In one tuning area every variant was within a point or two. In the other, every rank-normalized
variant lost ~7 points and halved the top-10% share; equal weights were a wash. **Kept the weights.**

## The first held-out test

On a 228 km2 area with 13 sites, never used for tuning: mean 82.9 vs random 66.8 (p = 0.009), top-10% share
0.38 vs 0.22. Across the three areas:

| Area | Sites | Site mean percentile | Random | p |
|---|---|---|---|---|
| Tuning A | 6 | 95.2 | 69.5 | 0.0008 |
| Tuning B | 6 | 73.7 | 62.5 | 0.18 |
| Held out | 13 | 82.9 | 66.8 | 0.009 |

The misses were sites picked on things the model can't see: fine trail detail, prey sign, local knowledge.

## Spreading the picks, and the scout

The 15 picks in one 87 km2 area clustered in a single canyon (median pair distance 3.4 km). A cap of 3 spots per
800 m hotspot zone, plus a `repick` tool, spread them across 7-12 zones per area. In a 40 km regional scout the
wind factor was zero for every block. Spreading drainage at scout scale and changing its aggregate fixed that
(block wind 0.44-1.0). The top blocks near a city were public parks with reported sightings. That raised a
caveat that is now in every report: city and county parks often restrict trail cameras.
