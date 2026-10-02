# Validation

Three tests, each answering a different question:

1. **Human picks.** Would the tool pick the camera spots a person chose by hand from the same factors? This is the gate on
   every model change, because it is the only test at the scale of a camera (20 m) that we have plenty of.
2. **Lion truth.** Do the map's scores line up with where lions actually went: camera detections, snow tracks
   and crossings on fixed routes? This is the real test, and it needs field data (see
   [FIELD_PROTOCOL.md](FIELD_PROTOCOL.md)).
3. **Open GPS collar data.** Do pumas in other ecosystems go where the map scores high? This can only veto a
   change, never tune one.

## Human picks

`cougarmap/evaluate.py` (run by `scripts/eval_picks.py`) ranks a set of known-good camera pins among every
candidate spot the model would consider.

- **Rank:** every distinct candidate spot in an area (smoothed local score peaks at least 150 m apart, the
  picker's own spacing) is ranked. Each pin gets the best cell within 15 m of it (pin accuracy) and is placed in
  that ranking: 10% means only a tenth of the area's candidate spots beat it. Lower is better.
- **Random:** 300 random points per area get exactly the same treatment. Smoother maps have fewer peaks, which
  pulls random points up too, so always read a result against its random baseline.
- **vsR:** the chance a pin ranks better than a random point (0.5 = chance).
- **Fixed-K:** take the top 3 spots per km2 (how someone would actually place cameras) and count the pins within
  50/100/150 m of one. It doesn't depend on how many peaks a map has, so it keeps smoothing effects honest.
- **Default ranking:** the production ranking (score x site penalties), as the picker uses it before the land
  and walk rules. `--raw` scores the bare score.

**Setting it up with your own picks.** Put camera pins named `Cam01`, `Cam02`... in a Google Earth file, analyze
the areas that hold them, and describe the setup in `data/private/eval.toml` (git-ignored):

```toml
kml = "my-areas.kml"          # relative to this file
states = "../../out/no-pins"  # optional; default out/
check = "b"                   # optional: an area held out of tuning, reported on its own
[areas]
a = "area-a"                  # key = the analyzed area's slug
b = "area-b"
```

Then `uv run python scripts/eval_picks.py --by-cam`. `./scripts/rerun_eval.sh <kml> "<Area>" ...` re-analyzes the
areas with and without the KML's water pins and evaluates both. `validate(area)` also reports how a KML's CamNN
pins rank, alongside the lion truth.

**Pin leakage.** If the person who chose the cameras also pinned water or sign next to them, those pins lift the
score around the cameras: right for using the tool, wrong for testing it. Report the numbers from states analyzed
with `--no-pins`.

**The shipping rule** we use: on the pin-free human-pick states, a change must not lose more than about 1 point of
overall median rank or 0.01 vsR, and no area may get worse by more than about 3 points. The GPS check can veto.

**Our numbers.** We test against a private set of 25 human-picked camera sites in three areas. The
locations stay private; the aggregate results are:

| | Median rank (random) | vsR | Top 25% (random) | Fixed-K within 150 m (random) |
|---|---|---|---|---|
| v1 score | 86% (99%) | 0.73 | 8% (1%) | 48% (20%) |
| v2 | 50% (91%) | 0.81 | 12% (3%) | 48% (20%) |
| v3 | 46% (89%) | 0.81 | 16% (4%) | 60% (21%) |
| v3.1 | 43.5% (89%) | 0.81 | 20% (4%) | 52% (21%) |
| v3.2: every house costs a little | 36.1% (87%) | 0.82 | 28% (5%) | 52% (20%) |
| v3.3: ponds and lakes as pinch barriers | 36.3% (87%) | 0.82 | 28% (5%) | 56% (20%) |
| **v3.4: approaches to water and meadows (current)** | **35.7% (87%)** | **0.82** | 28% (5%) | 56% (20%) |

All three areas were used in tuning, so none of them is a blind test. Per site, v3 against v2 is 14 better, 9
worse and 2 tied (p = 0.40), and v3.1 against v3 is 11 better and 9 worse: 25 sites are a tripwire, not a test
with power.
v3.2 (score x (1 + houses within 500 m)^-0.34, [experiments/10](experiments/10-house-cost.md)) against v3.1: 22
better, 2 worse, 1 tied; every area better; GPS pooled rank unchanged at 0.514 (23 animals better, 16 worse). It also
lifted within-array camera concordance in open SNAPSHOT USA data out of region (+0.010) and collar day beds in all
12 groups.
Not kept after v3.2 ([experiments/12](experiments/12-reshape-prey-tuning.md)): factor reshapes (fine ruggedness,
rock as edge cover, incised draws, local-relative scoring), a deer and elk prey layer, and a cross-validated
tuning of about 30 weights and definitions. None passed the held-out checks; the best (a signed canopy-edge
multiplier) gained 2 points on the picks and +0.007 camera concordance, inside the noise.
v3.3 (ponds and lakes as pinch barriers, [experiments/13](experiments/13-water-barriers.md)) against v3.2: neutral
on every check (median 36.1% -> 36.3%, vsR 0.821 -> 0.821, fixed-K within 150 m 52% -> 56%; one site better and
three worse by half a point or more; GPS pooled rank unchanged at 0.514, 2 animals better and 1 worse; cameras and
collars unchanged). It ships as the owner's realism call, not as a measured gain. Stream confluences, tried in the
same round, cost a little on the picks and cameras and were not kept.
v3.4 (destination approaches in the travel line, [experiments/14](experiments/14-water-approaches.md)) against
v3.3: median 36.3% -> 35.7%, vsR 0.821 -> 0.822, 13 sites better by half a point or more and none worse; areas
27.9 / 36.9 / 41.7% -> 27.9 / 35.8 / 41.0% (against v3.2: 36.1% -> 35.7%). Cameras and collars unchanged (+0.000;
-0.000 [-0.001, +0.002]); the inland-Northwest lock-box 0.512 -> 0.513; GPS pooled rank
unchanged at 0.514 (14 animals better, 17 worse). The same corridors added to
the water factor instead hurt two areas, and lidar trails to water were neutral; neither was kept.
[experiments/](experiments/) has every change, ablation, placebo and negative result behind these numbers.

### Controls to rerun when smoothing or terrain terms change

- **Smoothing-only control:** smoothing alone moves the rank metric (the v1 score at a 100 m Gaussian went
  86% -> 65%, random 99% -> 92%). Compare any new smoothing against the same smoothing without the new term, and
  against fixed-K: `evaluate(lambda st: terrain.gaussian(st.layers["score"], 20 / st.fine.res))`.
- **Pin leakage control:** report from states analyzed with `--no-pins`.
- **Placebo control:** rebuild a new terrain term from a misaligned DEM (flipped, or shifted 1.5 km). The v2 travel
  line's placebo scored 80-94%, against 50% for the real line, so its gain is terrain signal, not field
  structure. Repeat this for any new terrain term.

### Worn trails (lidar): a hint, not a factor

The worn-trail layer (`worn.py`, 1 m 3DEP lidar) recovers 58% of the mapped OpenStreetMap tracks and paths
in the human-pick areas, and about 75% of what it finds is on no map. 32% of the human-picked sites sit within
15 m of an unmapped worn line, against 11-16% for matched control points: people picking by hand put cameras by
these lines. Added to the score in any of the forms tried (a travel term, a multiplier near any or only unmapped
lines), it did not improve the human-pick ranks (median 43.3-47.3% against 43.5%, vsR flat, per-site changes
even or worse), so it changes no score and only offers each spot a `worn_trail` hint on where to face the camera.
With it on, the spots and scores are byte-identical to a run with it off (`tests/test_worn.py` checks this on the
synthetic area). The production creek-bank mask removed a quarter of the raw detections on the small human-pick
area, at a cost of 5 points of mapped-track recall; the pick-proximity numbers predate it.
[experiments/11-worn-trails.md](experiments/11-worn-trails.md) has the details.

## Lion truth (field data)

The human-pick numbers measure whether the tool agrees with people picking by hand. What they can't show
is where lions actually walk. `validate(area)` (`truth.area_truth`) reports every field test the log allows, each against its
own null:

- **Cameras.** Cougar detections (30-minute independence) per 100 camera-nights, net of downtime, by arm and
  by placement: on a trail or dirt road (`trail_type` game-trail, hiking-trail, closed-road or open-dirt), off
  one (none, paved), or not recorded. Only the on-feature cameras are compared with random on-trail cameras of
  the Washington Predator-Prey Project in NE Washington and the Okanogan: 0.87 in summer (Jul-Sep, 159 /
  18,377), 0.37 in winter (Dec-Feb, 72 / 19,262), and the midpoint in between (Bassing et al. 2023, Ecol. Appl.
  33:e2745), with a one-sided Poisson p-value. In SNAPSHOT USA arrays trail and dirt-road cameras caught about
  3x more lions than other cameras of the same array ([experiments/09](experiments/09-open-camera-and-collar-data.md)),
  so an off-trail camera against that base rate would look like a miss it isn't: off-feature and unrecorded
  cameras are reported as not comparable (`not_compared`). Arms with fewer than 30 on-feature camera-nights
  aren't compared (the raw `by_arm` output holds back vs_base and its p-value then too, with
  `too_few_nights`). Where a model (or human) camera and a control share a zone, the report gives the pooled
  rate ratio and a paired sign-flip permutation test on the per-zone rate differences (exact up to 16 zones),
  and a `placement_warning` counting the zones whose two cameras weren't placed alike (or whose placement isn't
  logged). An
  on-feature camera (a pick's "alternate on the trail") is compared the same way with the pick's own camera in
  its zone (arm model or human), so that zone stays in the model-vs-control test, or with an off-feature camera.
  Power, from a simulation with gamma site heterogeneity (CV 0.75) and 180 nights per camera: about 15 zones for
  an 80% chance to see a 3x gain and about 40 for 2x at the summer rate; at the winter rate, about 20 and 50.
- **Snow tracks.** The mean production score every 5 m along a followed track, against 200 copies of the
  same shape rotated at random about its centre and shifted 100-1,500 m (each keeping 90% of its points in
  the area). The percentile is the share of copies the track beats, so 0.5 is chance. Across tracks, a
  one-sided sign test. If 70% of tracks beat their copies, about 37 tracks show it; at 80%, about 18. The
  copies are moved at random, not kept at the track's distance from roads, so this only partly corrects for
  where people find tracks: a track found along a road in a drainage bottom can beat its copies because of
  where people drive. The report splits the sign test by where each track starts (`by_start`: its first point,
  where it was found, within 100 m of an open road or not); the tracks found away from roads are the cleaner
  test.
- **Crossing transects.** The score at each crossing against every point of its own route, as an AUC (0.5 =
  chance), with a p-value from crossings placed at random along the same routes. The report also gives
  crossings per 10 km on the top and bottom third of route scores, which absence surveys feed.

The synthetic tests (`tests/test_truth.py`) plant a signal and recover it: a track along a high-scoring
valley, crossings at high-scoring points, and model cameras at 3x the control rate. With no signal they land at
about 0.5 with no excess false alarms.

## GPS falsification check (open collar data, other ecosystems)

Open GPS collar data can only try to **falsify** the model: it can veto a change that clearly hurts there, never
overrule the method's rules, and no weight is ever tuned on it. The human-pick check stays the primary
check. `gps.py` holds the evaluator (settings `config.Gps`); `scripts/gps_check.py all` reproduces everything below.

**Data** (downloaded by `scripts/gps_check.py fetch` into the cache folder, `~/.cache/cougarmap/gps`, never into
git; nothing of ours is sent anywhere):

| Dataset | Where | Pumas / fixes | Usual fix interval | Source | License |
|---|---|---|---|---|---|
| Fishlake | Fishlake NF, central Utah: pinyon-juniper, sagebrush, aspen-conifer | 12 / 17,939 | 3-8 h (one 16 h) | USU coyote and puma study, Movebank study 1720694224 (public JSON API), doi:10.5441/001/1.7d8301h2 | CC0 1.0 |
| Olympic | Olympic Peninsula, Washington: wet coastal conifer, clearcuts, farmland | 47 / 105,438 | 1-3 h | Elbroch & Sager-Fradkin 2026, "Olympic Cougar Project", Movebank Data Repository, doi:10.5441/001/1.716 | CC BY-NC 4.0 |
| UDWR | northern and central Utah (Wasatch, Book Cliffs) | 40 / 47,774 | 2 h | "GPS tracking of cougars in Utah by UDWR (2019-2020)", Movebank Data Repository, doi:10.5441/001/1.712 | CC0 1.0 |

All three downloaded without a login from public endpoints. The Olympic data are non-commercial (CC BY-NC):
fine for this open, non-commercial tool; check the license before any commercial use.

The model inputs added in v3 are open data too, fetched like any other layer into the cache folder (never git):
NSIDC SNODAS snow depth (G02158, masked daily, NOAA/NSIDC, public domain; winter months only), WDFW Priority
Habitats and Species on the Web, public layer (deer and elk winter range as mapped there; Washington only,
December-March), and OpenStreetMap trailheads, campgrounds and parking (ODbL, like the roads).

**Design** (iSSF-style, per used fix):

- Used fixes: night and twilight fixes (sun below +6 degrees) ending a step at the animal's usual fix interval
  (within 25%, intervals up to 8 h). 16,000 strata from 52 animals fall inside the analyzed tiles.
- Available: 15 points from the same animal's own night step lengths and turning angles, around its previous
  fix (a random heading where the step before was irregular). Each used fix is ranked only among its own
  available points: **rank** = the share it beats (0.5 = chance), so an animal is compared only with where it
  could have gone next.
- Two scales: the production score itself (already averaged over the 20 m camera zone) and the score blurred at
  a 250 m Gaussian (~500 m across, the habitat scale).
- **Boyce index**: Spearman correlation of the used/available ratio with score bin (10 equal-count bins of the
  available scores; 1 = more use the higher the score). Pooled across animals, so it mixes their ranges.
- **Map-shift null** (design check): the same strata on the score map shifted 1.5 km on a torus, four directions
  averaged. It should land at 0.5, and does (0.499-0.503 in every dataset).
- **DEM-shift placebo**: the three terms built from the DEM alone (wind, pinch, travel) moved 1.5 km east or north
  before recombining, so the terrain no longer lines up with the canopy, water and the lions.
- **Ablations**: the production ranking recombined from the saved layers ("recombined", identical to production
  to three decimals) without each term in turn. "Ref. better" counts the animals (20+ strata) where the full
  model ranks the lions' fixes higher than the ablation, i.e. where that term helps.
- Tiles: 14 squares of 8 km (64 km2, 3 m analysis) over the densest fixes, chosen greedily so each animal adds
  at most 300 fixes to a tile's value (4 Fishlake, 6 Olympic, 4 UDWR). The production pipeline ran on each at
  month 10 (scored on fixes from May-November) and month 1 (December-April), private land allowed. Lidar
  covered 100% of the Utah tiles and 32-100% of the Olympic ones (the rest is the 10 m 3DEP DEM). The wind
  climatology is Open-Meteo's for each tile, like any run.

**Results (v3 production, 2026-10-01), with the v2 baseline.** Rank 0.5 = chance. All 28 tile states were
rerun with v3 at its final calibration. The score
scale change (5 -> 3) left every pooled number and animal count here unchanged except three ~500 m counts.

| | Strata | Animals | Rank 20 m | Rank ~500 m | Boyce 20 m | Boyce ~500 m | Map-shift null 20 m / 500 m | Moving steps (>= 100 m), rank 20 m | Animals above chance, 20 m | ~500 m |
|---|---|---|---|---|---|---|---|---|---|---|
| Fishlake | 1,867 | 8 | **0.543** | 0.520 | 0.72 | 0.35 | 0.501 / 0.500 | 0.525 | 7/8 (p=0.035) | 6/8 (p=0.14) |
| Olympic | 9,768 | 28 | 0.510 | 0.505 | 0.38 | 0.49 | 0.502 / 0.499 | 0.519 | 15/28 (p=0.43) | 16/28 (p=0.29) |
| UDWR | 4,360 | 16 | 0.509 | 0.503 | 0.27 | 0.27 | 0.501 / 0.499 | 0.498 | 10/16 (p=0.23) | 7/16 (p=0.77) |
| **All** | 15,995 | 52 | **0.514** | 0.506 | 0.60 | 0.26 | 0.502 / 0.499 | 0.514 | 32/52 (p=0.06) | 29/52 (p=0.24) |
| All, v2 baseline | 15,995 | 52 | 0.508 | 0.500 | 0.68 | 0.06 | 0.502 / 0.500 | 0.507 | 31/51 (p=0.08) | 26/52 (p=0.56) |

v2 per dataset (20 m / ~500 m): Fishlake 0.543 / 0.517 (Boyce 0.82 / 0.27), Olympic 0.503 / 0.498, UDWR 0.503 /
0.498. **v3 against v2, animal by animal:** v3 ranks the fixes higher in 36 of 51 animals at 20 m (p=0.005):
Fishlake 4/8 (flat), Olympic 19/28 (p=0.09), UDWR 13/15 (p=0.007); at ~500 m 29/51 (p=0.40). Across-animal
median rank at 20 m: Fishlake 0.524, Olympic 0.504, UDWR 0.526 (one-sided sign tests in the table; animals with
fewer than 20 strata get no verdict of their own). `scripts/gps_check.py report` writes the per-animal table.

**v3.1** (hillside cold-air drainage), recomputed on the same 28 tile states: pooled rank 0.515 against 0.514 at
20 m, with 33 of 52 animals better and 16 worse (p = 0.02). The effect is tiny, but it is the only wind change the
GPS data lean toward. The tables here are v3.

Ablations and the placebo, v3 (pooled rank at 20 m / ~500 m; in brackets, animals where the full model ranks
higher at 20 m):

| Score | Fishlake | Olympic | UDWR | All | Full model better, all, 20 m | ~500 m |
|---|---|---|---|---|---|---|
| production (recombined) | 0.543 / 0.520 | 0.510 / 0.505 | 0.509 / 0.503 | 0.514 / 0.506 | | |
| no travel line | 0.541 / 0.512 (4/7) | 0.513 / 0.504 (10/27) | 0.515 / 0.496 (7/16) | 0.517 / 0.503 | 21/50 (p=0.32) | 36/50 (p=0.003) |
| no edges | 0.539 / 0.522 (4/7) | 0.501 / 0.503 (18/27) | 0.501 / 0.511 (11/15) | 0.505 / 0.507 | 33/49 (p=0.02) | 27/52 (p=0.89) |
| no pinch | 0.542 / 0.520 (4/7) | 0.509 / 0.505 (12/26) | 0.510 / 0.504 (4/13) | 0.513 / 0.506 | 20/46 (p=0.46) | 17/44 (p=0.17) |
| no water | 0.543 / 0.526 (3/8) | 0.511 / 0.508 (11/27) | 0.511 / 0.501 (4/14) | 0.515 / 0.508 | 18/49 (p=0.09) | 23/51 (p=0.58) |
| no wind | 0.539 / 0.520 (5/7) | 0.513 / 0.504 (11/27) | 0.511 / 0.505 (8/16) | 0.515 / 0.506 | 24/50 (p=0.89) | 19/50 (p=0.12) |
| no habitat context | 0.538 / 0.522 (7/8) | 0.506 / 0.504 (16/28) | 0.519 / 0.519 (6/16) | 0.514 / 0.510 | 29/52 (p=0.49) | 22/51 (p=0.40) |
| no 20 m zone | 0.547 / 0.521 (4/8) | 0.516 / 0.505 (5/25) | 0.513 / 0.502 (7/14) | 0.518 / 0.506 | 16/47 (p=0.04) | 14/27 (p=1.0) |
| no site penalties | 0.543 / 0.519 (2/4) | 0.511 / 0.508 (11/23) | 0.510 / 0.505 (9/13) | 0.514 / 0.508 | 22/40 (p=0.64) | 25/45 (p=0.55) |
| no winter module | 0.542 / 0.521 (3/5) | 0.509 / 0.503 (14/19) | 0.506 / 0.497 (13/15) | 0.512 / 0.504 | 30/39 (p=0.001) | 27/43 (p=0.13) |
| DEM-shift placebo, east | 0.510 / 0.507 (7/8) | 0.515 / 0.505 (14/28) | 0.521 / 0.494 (8/16) | 0.516 / 0.502 | 29/52 (p=0.49) | 32/52 (p=0.13) |
| DEM-shift placebo, north | 0.523 / 0.514 (5/8) | 0.522 / 0.511 (10/27) | 0.511 / 0.499 (8/16) | 0.519 / 0.508 | 23/51 (p=0.58) | 25/52 (p=0.89) |

The winter module is off in the month-10 tiles, so "no winter module" differs only on the month-1 tiles.

**What it says.**

- The design is sound: the shifted-map null sits at 0.50 everywhere, and the synthetic tests recover a planted
  signal (`tests/test_gps.py`).
- **v3 is not vetoed anywhere, and it is better than v2** in 36 of 51 animals (Olympic and UDWR), with Fishlake
  flat at 0.543. Fishlake's Boyce index at 20 m fell from 0.82 to 0.72 (its ~500 m Boyce rose from 0.27 to 0.35).
- **Fishlake is still the one clear pass.** At 20 m the night fixes outrank their own available points in 7 of
  8 animals, and the model beats the DEM-shift placebo east in 7/8 animals, north in 5/8 (v2: 7/7).
- **Olympic and UDWR are still near chance** (0.510 / 0.509), and the DEM-shift placebo still scores as well
  (0.511-0.522): the model passes the placebo rule in neither, so overall it still fails it (29/52 and 23/51
  animals; v2 30/51 and 25/50).
- **Which terms carry the signal.** The hunting edge still does (without it the full model loses ground in
  33/49 animals, p=0.02). The travel line no longer hurts at 20 m (v2: removing it helped in 34/51 animals,
  p=0.02; v3: 29/50, p=0.32) and now helps at the ~500 m habitat scale (36/50, p=0.003): halving the ridge spines
  off the crossings did what the v2 ablation pointed to. The winter module helps on the month-1 tiles (30/39,
  p=0.001). Water at the habitat scale is now neutral (v2 leaned negative, 18/52). The 20 m zone lowers the rank
  at 20 m because fixes and available points are compared cell by cell and the zone blurs that contrast; it is a
  camera-view choice, so this says little about it. Wind, pinch points and the site penalties are neutral.
- None of this demotes the method's rules. Wind, cold-air drainage and downwind edges cannot be tested by
  1-3 h fixes in other mountains, and nothing is tuned here: every v3 value was chosen on the human picks or
  set a priori, and the GPS check only had a veto.

**Caveats.**

- Other ecosystems: Utah pinyon-juniper and aspen, Olympic rainforest and clearcuts, not inland Northwest dry
  conifer. A pass here is weak support and a fail is a warning, nothing more.
- Coarse fixes (1-8 h) are biased toward beds and kills and say nothing about the 20 m camera spot; straight-line
  steps hours apart do not show a path. "Moving steps" only partly removes the bed/kill bias.
- Consecutive fixes are not independent, so the per-animal p-values are optimistic; the across-animal sign tests
  are the ones to read.
- The tiles cover only the densest fixes (8 km squares), so long-ranging steps near tile edges drop out, and a
  few Olympic animals with many fixes weigh heavily in the pooled numbers.
- Lidar covers only part of three Olympic tiles; the rest runs on the 10 m DEM, which weakens the terrain terms
  there.

Reproduce: `uv run python scripts/gps_check.py all` (fetch, tiles, run, report; the run step resumes and takes
about 45 minutes the first time, mostly downloads; the report about 3 minutes). `report` writes
`out/gps/report.json`, and `tables` prints the tables again from it. The slow tripwire
`uv run pytest -m slow tests/test_gps.py` checks the null and this baseline (all 0.514, Fishlake 0.543 at 20 m).
The month-1 runs also download the winter module's inputs (SNODAS, five dates of ~23 MB, cached per date; WDFW
PHS for the Washington tiles).
