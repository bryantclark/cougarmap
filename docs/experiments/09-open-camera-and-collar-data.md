# 09. Open camera and collar data (2026-10-02)

Until this round the model was tested against human picks, a few collar datasets and almost no lion detections.
This round brought in every open dataset we could find with real lions in it: a national camera-trap survey and
twelve groups of GPS-collared cougars, kill sites and fine-scale travel paths. It asked two questions: how well
does the current score (production: score x site penalties) predict where lions are, and can anything we tried
do better out of region? Three independent approaches (reshaped features, a patch CNN, and a differentiable
refit of every parameter) and a judge that re-ran the decisive numbers with its own code.

Short answer: production predicts lion detections within camera arrays in most of the West, not yet in the
inland Northwest; nothing we tried beats it out of region except a small cost per nearby house (proposed as its
own change, since it reverses an earlier decision); and where a camera is strapped at its spot matters about as
much as the spot.

## Open data used

None of it goes in git; it is downloaded into the cache or a private folder and read there.

**Cameras.** SNAPSHOT USA, the coordinated national camera-trap survey: 2019-2023 (Rooney et al. 2025, Global
Ecology and Biogeography, doi:10.1111/geb.13941; data Dryad doi:10.5061/dryad.k0p2ngfhn, CC0) and 2024 (Dryad
doi:10.5061/dryad.bnzs7h4qf, CC0). Unbaited cameras about 50 cm high, mostly September-October, in arrays of 7+
cameras 200 m to 5 km apart. In the western US (urban arrays dropped; none had a puma): 2,377 deployments in 70
arrays, 101,246 camera-nights, 386 independent puma detections (30-minute rule) at 209 cameras. The 30 arrays
with at least one puma (1,197 cameras) carry the tests. Each array was analyzed by the production pipeline at
its survey month, 201 areas, no pins, private land allowed.

**Collars** (night steps, day beds and night fixes against the animal's own available points, as in
[04](04-lion-truth-and-gps.md) and VALIDATION.md):

| Dataset | Source | License |
|---|---|---|
| Fishlake NF, Utah (USU coyote and puma study) | Movebank study 1720694224, doi:10.5441/001/1.7d8301h2 | CC0 |
| Olympic Cougar Project (Elbroch & Sager-Fradkin 2026) | Movebank Data Repository, doi:10.5441/001/1.716 | CC BY-NC 4.0 |
| Utah DWR cougars 2019-2020 | Movebank Data Repository, doi:10.5441/001/1.712 | CC0 |
| Southern BC Cougar Project (collars and kill sites) | Borealis, doi:10.5683/SP3/K1EOPQ | CC BY-NC 4.0 |
| West-central Alberta cougar survival | Borealis, doi:10.5683/SP4/39NWNV | CC0 |
| West-central Alberta (Beale & Widmeyer; collars and kill clusters) | Borealis, doi:10.7939/DVN/QPVCWU | CC0 |
| SW Alberta telemetry and roads (Banfield) | Borealis, doi:10.7939/DVN/ZRBYCN | CC0 |
| Teton / Greater Yellowstone (Elbroch 2016) | Zenodo 5003887 (Dryad vf85j) | CC0 |
| Modoc, California, juvenile dispersal | Dryad doi:10.5061/dryad.hdr7sqvrw | CC0 |
| UC Davis Southern California, 5/15-min fixes | figshare 4983095 | CC BY 4.0 |
| Santa Cruz Mountains (Wang; Dunford) | Zenodo 5005666 and 3972242 | CC0 |
| USGS northern Arizona / southern Utah 2003-13 | ScienceBase 5d113038e4b0941bde55058e | USGS release, no license stated |
| Yellowstone northern range cougar kills 2016-22 | Dryad doi:10.5061/dryad.sxksn03c5 | CC0 |
| West Cascades, Washington, cougar kill sites (Robins et al. 2024) | Dryad doi:10.5061/dryad.dv41ns249 | CC0 |

Thanks to every team that published these. The CC BY-NC sets are fine for this open, non-commercial tool; check
the licenses before any commercial use. In total the collar tiles grew from 14 to 173 (8 km squares, each run
by the production pipeline in October and January): 40,140 valid night strata in the original three datasets,
about 110,000 more in the new ones. Canadian tiles run on a 30 m DEM with no water term, so they are read
separately.

## Methods

- **Within-array camera tests.** Cameras are only compared with cameras of the same array, so region, year,
  observer and camera model cancel out. Concordance: of the camera pairs in one array with different puma rates,
  the share where the higher-scoring camera has the higher rate (0.5 = chance), with an array bootstrap. Also a
  within-array detection AUC and a negative binomial model with array fixed effects, the recorded camera
  placement and a log camera-nights offset. A camera is read as production reads a pin: the smoothed score,
  best cell within 15 m.
- **Regions, fixed in code before any result.** Search: California (9 arrays), Pacific Northwest (4),
  Southwest (5), Rockies (5). Leak, reported on its own: the 3 arrays inside collar study areas. Lock-box: the 4
  inland-Northwest arrays, the region closest to our own test areas, sealed until the end and read once.
- **Leave-one-region-out and leave-one-dataset-out.** Anything chosen was chosen without the region or collar
  dataset it is scored on.
- **Placebos.** Every new layer or map was also read with the pin (or the collar points) moved 1.5-2.5 km, and
  the refits were repeated on permuted labels. A gain the placebo also gets is texture, not signal.
- **Human picks.** The private set of 25 human-picked sites in three areas, pin-free, as the tripwire.

## Results: how well production predicts real lions

**Cameras, within arrays.**

| Arrays | Concordance | Note |
|---|---|---|
| All 26 outside the lock-box | 0.62 | AUC 0.64 |
| Search regions (23) | 0.645 | NB rate ratio 1.86 per SD of score |
| Rockies | 0.74 | |
| Southwest | 0.65 | |
| California | 0.41 | below chance |
| Pacific Northwest coast | 0.37 | below chance |
| Inland Northwest, lock-box (4) | **0.51** | chance; NB rate ratio 1.02 |

So the map ranks cameras the right way round in the interior West, the wrong way on the coast, and not at all
in the inland Northwest arrays. That last one is the honest headline for our own ground: no camera evidence yet
that the map predicts lions there.

**Collars.** In all 12 collar groups production ranks a lion's next night position above its available
alternatives a little better than chance (0.52-0.56), and day beds better (0.54-0.64), each against a map-shift
placebo near 0.50. On the BC collars south of the border, the nearest habitat to our test areas, the placebo is
noisy (8 animals, mostly one per tile): real minus placebo is +0.02 for night steps and +0.04 for day beds, 5 of
8 animals, no clear evidence either way.

**Real travel paths and kill sites.** The two 5/15-minute datasets show actual paths: the path's mean score
against the same path rotated about its start beats 0.68 of the copies in Southern California (689 paths, 8 of 8
animals; placebo 0.48) and 0.61 in the Santa Cruz Mountains (631, 5 of 5; placebo 0.53). Kill sites against
the animal's home range: 0.54 (BC), 0.55 (west-central Alberta), 0.56 (West Cascades, 6 of 6 animals) and 0.65
(Yellowstone, 6 of 6), placebos 0.48-0.52.

## Results: placement at the spot

In the same array, against cameras with no recorded placement, cameras on dirt roads detected pumas 3.75x as
often [2.18, 6.46] and cameras on trails 3.34x [2.14, 5.23] (negative binomial, array fixed effects, score in
the model); in the inland Northwest 2.7-2.8x. Against cameras recorded as off-trail it is about 2x, with an
interval that includes 1. Raw rates: 2.76 detections per 100 nights on roads, 1.16 on trails, 0.63 at other
recorded placements, 0.26 unrecorded.

The map cannot find those trails. A mapped quiet line (closed or forest road, two-track, OSM path) within 15 m
of the camera gave no gain once placement was known (rate ratio 0.83 [0.50, 1.39]); only 13% of road/trail
cameras sat that close to a mapped line, against 8% of the others; and the within-array AUC for picking the
road/trail camera from mapped lines was 0.50. So the gain has to come from the person setting the camera.

**Shipped (no score change):**

- every reported spot gets a placement line: beside the game trail, old two-track or closed road through it
  (within ~50 m), knee-to-waist high, a few metres off the line, facing along it (PLAYBOOK, agent
  instructions, field protocol);
- the trail alternate stays an option, never the default, and its config says why (mapped lines gave no gain);
- `log_camera` warns while `trail_type` is missing, and agents are told to always ask for it;
- `validate` compares like with like: only on-trail cameras against the on-trail base rate (an off-trail
  camera would look about 3x worse than it is), off-trail and unrecorded ones reported as not comparable, and
  a warning on any zone whose model and control cameras weren't placed alike.

## Results: the one score change that survived

A small cost for every nearby house, score x (1 + houses within 500 m)^-0.34 on top of the populated-area cut.
It was the only change that agreed across all three sources: camera concordance +0.010 [+0.003, +0.023] in the
search regions (+0.012 leave-one-region-out; placebo -0.004; lock-box +0.005, neutral, because few inland
cameras have houses near them), collar day beds up in all 12 groups (night steps unchanged), and the human picks'
median rank 43.5% -> 36.1% with fixed-K unchanged. It reverses an owner decision (homesteads were free), so it is
proposed as a separate change for the owner to accept or not. The gain is ranking only: the count model sees no
change.

## Negative results

- **Refitting the parameters.** All 23 combine and site-penalty parameters, plus a winter exponent, tuned by
  gradient descent on cameras (and optionally collars and picks), with a prior toward the current values. Best
  out-of-region gain +0.008 concordance, interval spanning zero; permuted-label runs gained as much; the
  lock-box went down (-0.020) and the picks got worse on every learned set. The cameras consistently wanted a
  wider 20 m zone and more travel weight, but that held up neither out of sample nor on the picks. Refitting the
  five factor weights on collars also gained nothing out of region: the current weights are as good as any
  fitted set.
- **27 reshaped features** (bench, ruggedness, 50 m edges, edge contrast, opening size, water geometry, quiet
  lines and more), each as a multiplier on production. Every fully nested search lost to production on the
  held-out regions (region mean -0.02 to -0.06). Signals are regional: down-weighting edges helps the coast,
  where production is reversed, and costs about 0.17 in the Rockies. Bench agreed with cameras and collars but
  sent the picks to 78%; ruggedness pointed opposite ways for cameras and collars.
- **A patch CNN** (480 m patches of 17 layers, about 13k parameters). On collars the convolutional layers added
  about +0.001 over a linear model on the same inputs; trained on cameras it memorized the training arrays
  (0.79-0.86 in-sample against 0.54-0.60 held out) and lost to production; every patch model was far worse on
  the picks (vsR 0.36-0.73 against 0.81). One camera-trained linear model scored 0.70 in the lock-box but worse
  everywhere else; the lock-box has now been opened, so that is a lead for new inland camera data, not a result.
- **A stronger winter module** (season^2.7 to ^6). The earlier collars all chose it, so it was tested a priori
  on the new collars. It failed on the BC collars south of the border, the nearest to our test areas (day beds
  -0.012, 6 of 7 animals down), and elsewhere the placebo moved as much as the real change. Cameras and picks
  can't test it (all autumn). The winter module stays as it is until winter field data say otherwise.
- **Collar-trained layers blended in.** The best collar model of the previous round (an elastic-net conditional
  logit with a functional response, the clear winner on held-out collars) as a "where lions travel" layer:
  production x travel^0.5 or ^1 left camera concordance unchanged (+0.001 [-0.020, +0.016]) and beat the
  map-shift placebo only weakly; on its own the collar layer ranked cameras worse than production
  (0.54 against 0.60). On the picks it was neutral (median within about 5 points, vsR flat). Not shipped.

## Caveats

- Few arrays: 30 with pumas, 3-9 per region; single-region intervals are wide. One array holds most of the
  lock-box's detections.
- SNAPSHOT cameras run in September-October, so nothing here tests the winter module, and placement labels are
  coarse ("unknown" for most cameras).
- The lock-box has been read by every approach; it is spent. The next blind test is new inland cameras, logged
  with their placement.
- Collar fixes 1-16 h apart mostly record beds and kills, not the 20 m camera spot; the travel-path test is the
  only one at walking scale, and only in two California datasets.
