# How CougarMap scores a spot

CougarMap encodes our field method for placing lion cameras: four factors, stacked, with the most weight on wind
and edges. On top of that it adds habitat context, natural travel lines, a winter module and penalties for
people. This page describes the current model (v3.1). [VALIDATION.md](VALIDATION.md) covers how it is tested, and
[experiments/](experiments/) covers how it got here.

```
score = scale x zone20( spot x habitat x season ) x site penalties
```

(`analyze.combine`, `analyze.site_penalty`; every weight and threshold lives in `config.py`.)

## The pipeline

1. **Scout** (`scout.py`, regional searches only): a coarse screen on 30 m data ranks ~3 km blocks of a large
   circle by terrain, public land, road access and iNaturalist cougar sightings. The best blocks get the
   detailed run.
2. **Context** (`context.py`): download everything one area needs (lidar elevation, canopy height, water, roads,
   land, buildings, wind climatology and, in winter, snow) onto a fine grid (1-5 m, by area size).
3. **Factors** (`factors.py`, `terrain.py`, `corridor.py`): compute each layer from pure array functions.
4. **Score and rules** (`analyze.py`): combine, apply the land and walking rules, pick spots, write the reasons.
5. **Export** (`export.py`): the KMZ, `summary.json`, `candidates.geojson` and the saved `state.pkl`.

## The spot

The four factors, weighted wind 0.35, edges 0.30, pinch points 0.20 and water 0.15, plus the terrain travel line
(0.30). Each factor also multiplies the spot by a smooth stacking bonus of up to 1.25x, ramping in from 0.35 to
0.65, so "more factors is better" holds without a jump at any threshold.

- **Wind.** Lions hunt into the wind at dawn and dusk. Two air currents matter. The first is the cold air
  draining down hillsides and valleys: its direction comes from the DEM smoothed over 60 m, and its strength from
  the multiple-flow-direction catchment per metre of contour width (`terrain.specific_catchment`), so open
  hillsides drain as well as channels, at any cell size. It is spread across the valley floor. The second is the
  prevailing wind on high-pressure days: HRRR's 850 hPa wind above the ridgetops at dawn and dusk, on hours with
  pressure at or above that month's median, not falling, and dry for 24 hours (`sources/weather.py`). The best
  ground is a valley where both flow the same way, with the wind channeled along the valley in proportion to how
  well they line up. The windward side within 100 m below a crest also scores. The direction terms are scaled by
  0.7 + 0.3 x the wind's consistency, because the high-pressure twilight wind is often light and unsteady, and
  reports give that consistency and the most common direction next to the mean. You can override the wind
  direction (`wind_from_deg`), for example with a nearby weather station's.
- **Edges: the hunting edge.** Timber cover (patchy counts) 0-35 m back from an opening, fading out by 75 m,
  plus the opening's own rim out to 15-25 m from real cover, where kills cluster (Laundré & Hernández 2003;
  Holmes & Laundré 2006). The most downwind end of each opening counts up to 0.25 more. It is scored for each
  air current, the evening cold-air drainage and the daytime high-pressure wind, so an end downwind in both is
  best. Openings come from the 1 m canopy height map; road and trail cuts don't count.
- **Pinch points.** Saddles (from geomorphons), the base of cliffs and escarpments (measured from the foot of the
  whole steep band, not the rock face), stream and lake banks, fences, and funnels from a circuit-theory
  current map.
- **Limited water.** Springs, seeps, intermittent and ephemeral streams, and ponds and marshes under 5 ha. Water
  counts more when it is scarce: the only water within a mile gets a bonus, and big lakes (5 ha or more) make
  nearby water less scarce. The 5 ha size is our model choice. Water you pin in Google Earth counts at full
  strength.
- **Travel line.** Drainage bottoms and ridge spines, from the percentile of the topographic position index at
  300 m, squared, with a 3 m relief floor so flat ground makes no lines. Bottoms count all year, more on gentle
  grades (full up to 8 degrees, half at 25 and steeper; Dickson et al. 2005, Dunford et al. 2020). Ridge spines
  count half, except where lions cross them: within 150 m of a saddle, within 100 m of a junction of ridges,
  and from November to March above big south- and southeast-facing slopes. Lions cross ridges at saddles, and
  use ridgelines less for travel than canyon bottoms (Dickson & Beier 2007).

## Habitat and season

- **Habitat.** `(0.2 + edge density / 0.3) x (0.5 + water density / 0.3)`, with both densities taken over a 250 m
  Gaussian (about 500 m across). A lion spends its time in a mosaic of timber and openings with water around.
- **Season** (November-April only; 1 otherwise). `0.6 + 0.4 x W`, where W is shallow snow x low ground x
  sun-facing slopes: snow counts fully up to 25 cm and drops to 0.2 at 60 cm (the SNODAS median on the 15th of
  the month over the last five winters); low ground is the position in the elevation range within 2.5 km; sun
  exposure comes from aspect and steepness. In Washington, from December to March, WDFW deer and elk winter
  range also counts (1 inside, feathered over 400 m to 0.8 outside). Season applies to the habitat around a spot,
  never the spot itself.

## The camera zone

The score is averaged over a 20 m Gaussian. That is the camera's view and about the error of a GPS or Google Earth
pin.

## Site penalties

- **Paved roads:** x0.3 within 20 m, fading to x1 at 800 m (traffic, people, theft). Gravel roads, forest roads,
  tracks and trails cost nothing.
- **Populated areas:** houses (footprints of 50 m2 or more) within 500 m. Up to 15 houses there is no cut, fading
  to x0.3 at 60 houses, about 76 per km2. That is the density below which 99% of eastern Washington cougar use
  falls (Maletzke et al. 2017). On top of that every house costs a little: x(1 + houses)^-0.34, so 1 house x0.79,
  5 x0.54, 15 x0.39, 60 x0.25 (x0.074 with the populated-area cut). Chosen on open camera and collar data and the
  human picks ([experiments/10](experiments/10-house-cost.md)).
- **Recreation sites** (OpenStreetMap trailheads, campgrounds, picnic sites, parking, toilets, shelters): x0.7
  within 50 m, fading to x1 at 400 m.

## Rules and picking

- **Public land** (PAD-US public access) for the ranked spots. Private land is scored the same way and its spots
  are returned separately as private candidates (P1, P2...) and hidden KMZ layers; walks to them may cross
  private land. A regional search chooses its blocks by public land, so private land far from public ground is
  only analyzed when you name the area (a small radius or a KML outline).
- **Within 1 mile of walking** from a road open to vehicles that month. Walking is a least-cost route over slope
  and terrain (about 30 minutes), not a straight line. Road seasons come from the USFS Motor Vehicle Use Map.
- Lakes, and cells too close to a paved road, can't hold a camera.
- **Picking:** local peaks of the score at least 150 m apart, at most 3 per hotspot zone (800 m), best first.
  Each spot gets plain-English reasons from the layers at that spot. Each reason names the wind it means.
- **Alternate on the trail** (no score change): the best usable cell within 150 m beside a quiet road or trail
  (a closed or gated forest road, a forest road in a month it is closed, a two-track, a path), at least 100 m from
  a recreation site, scoring at least 30 and at least half the spot's own score. Cameras on dirt roads and trails
  catch more of the lions that pass (Kolowski & Forrester 2017; Bassing et al. 2023). But good spots mostly sit
  off mapped lines, so this is offered as an option and never replaces the spot.
- **Worn trails** (no score change; skipped with `--fast`): game trails, cattle trails and old two-tracks
  found in 1 m lidar (`worn.py`): oriented trough filters across the 1 m DEM, lines that traverse a slope rather
  than run down it, a bench (flat tread between a cut and a fill) on sidehills, and no lines running along a
  creek or channel. They go in a hidden KMZ layer, and a spot within 30 m of one on no map gets a hint on where
  to face the camera. Human picks sit beside such lines far more often than control points, but as a score
  factor they did not help ([experiments/11](experiments/11-worn-trails.md)).

## The weights page

`--interactive` (`explore.py`) writes `explore.html`, a page that reruns the score and the picking in the browser
as the weights change. Only the spot term depends on the weights, so the page carries, per cell of a coarser grid
(a whole number of analysis cells, about 10 m; coarser past a million cells): the five weighted layers, the
habitat x season multiplier, the three site-penalty multipliers and the usable ground (public and private), as
block means quantized to uint8 (habitat uint16), deflated and base64-encoded. In a Web Worker it repeats `combine`
exactly on that grid (the stacking ramps, the 20 m camera-zone Gaussian, the normalization by the weights and the
0-100 clip), multiplies in each penalty raised to its slider's strength (0 drops it, 1 is the model, 2 squares the
multiplier), and picks as `pick_candidates` does (the 6 m smoothing, `peak_local_max`'s local maxima and greedy
spacing, the snap, at most 3 per 800 m zone). Grid cells map to lon/lat by a quadratic fit good to a few
centimetres. The KMZ's other layers (the hunting edge, land, walking range, worn trails, routes, air flow,
saddles) come along for the page's layer menu.

The factor sliders are shares that add up to 100%: `combine` divides by the sum of the weights, so only their
ratios ever mattered, and raising one share lowers the rest in proportion. Dragging stays fast because the score
is taken apart once: with the stacking multiplier fixed, the blurred spot score is a weighted sum of five blurred
layers, and the stacking bonus is a polynomial in the multiplier, so 25 blurred layers make that slider a weighted
sum too. A slider move then costs the sum and the peak picking (a van Herk max filter): about 30-45 ms on a 3 km
area, down from about a second.

What it approximates: it scores block means rather than each analysis cell (the stacking is not linear, so a
block of mixed ground scores a little differently), the peak spacing rounds 150 m to whole page cells, a block
is usable when half its cells are, and a spot sits at a page-cell centre. On a 3 km area (9 m cells) all 10 of
the model's spots and all 15 of its private-land spots came back within 30 m, and the page's score correlated
with the model's at r = 0.994; on a 130 km2 area (22 m cells) 13 of 15. Reasons, walks and land names stay in the
KMZ and summary.json.

## Scale

`Habitat.score_scale` (3.0) calibrates the display so that strong real spots land at 60-100. It changes only the
0-100 number and its clip at 100, never the ranking.
