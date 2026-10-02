# 14. Approaches to water and meadows (2026-10-02)

**Where the idea came from.** The owner's method: animals bed in timber and travel to destinations, water to drink
and meadows to feed, and a hunting lion takes the covered approach. What matters for a camera is the trail they
take to get there, not the water's edge itself. The model's water factor only counts nearness to water (full
within 40 m, fading by 300 m), and its travel line only knows terrain (drainage bottoms and ridge crossings).

**Answer.** Five forms were fixed in advance and measured; nothing was fitted, so in-sample is held-out. One
ships: the approaches as part of the **travel line** (V4). It improves the human picks (13 sites better, none
worse on this tree) and moves nothing else. Adding the same corridors to the **water** factor hurt two of the
three areas, and lidar trails to water were neutral.

## The variants

Every variant routes over the same cost and scores where routes converge (all fixed before measuring):

- **Cost per metre:** x(1 + 2(1 - c8)), with c8 the share of canopy at least 4 m tall (median over ~5 m) within
  8 m; x(1 + 3 ramp(slope, 20-40 degrees)); x(1 - 0.3 t) on a travel line or a mapped two-track or path (t = the
  larger); x10 on cliffs; 50 on a lake (0.5 ha or more) that is not a destination.
- **Routes:** least-cost (8-neighbour Dijkstra, up to 2,000 cost-weighted metres) from every cell to the nearest
  destination. Origins send their area down the tree to the destination; **convergence** = log10 of the flow
  through a cell over an even flow across a 150 m front, clipped 0-1 (1 = ten times an even flow), 3x3 max.
- **Under cover:** the covered share (within 5 m of cover) of the route's last 60 m into the destination, times the
  cell's own nearness to cover (1 up to 10 m, 0 at 25 m), square-rooted.
- **Downwind:** 0.5-1 by how far the cell lies downwind of its destination in the dawn/dusk air flow.

| Variant | Destinations and origins | How it enters the score |
|---|---|---|
| V1 | the production water sources (weighted as the water factor); origins in cover (60% within 30 m) 150-300 m from limited water | **replaces the water factor**: source weight x scarcity x convergence x cover x a taper (full to 100 m, half at 300 m) |
| V2 | as V1 | V1 x downwind |
| V3 | as V1 | max(V2, the production water factor) |
| **V4 (shipped)** | **limited water and meadows (the openings layer); origins in bedding cover (70% within 30 m) 150-300 m from a destination** | **travel = max(travel line, convergence x cover x downwind x a taper: full to 75 m, gone at 150 m)** |
| V5 | worn trails from 1 m lidar that reach within 30 m of limited water (lidar areas only) | max(water factor, source weight x scarcity x trail credit (full to 150 m along the trail, gone at 300 m) x cover x downwind) |

## Results (the prototype, on frozen fields of the v3.2 states)

Human picks on the pin-free states (25 sites in three areas); per site: sites at least half a point better /
worse. Cameras are SNAPSHOT USA within-array concordance outside the inland Northwest; collars are the six collar
groups of [09](09-open-camera-and-collar-data.md) (fine-scale paths and kill sites), with 95% intervals (array and
animal bootstraps).

| Variant | Picks median (vsR) | Per site | Areas | Cameras Δ | Collars Δ |
|---|---|---|---|---|---|
| Production v3.2 | 36.1% (0.821) | | 27.7, 36.9, 41.5% | (0.624) | (0.596) |
| V1 corridors as water | 36.0% (0.830) | 15 / 7 | 35.6, 34.2, 43.2% | +0.001 [-0.005, +0.008] | +0.003 [+0.001, +0.006] |
| V2 V1 x downwind | 36.1% (0.830) | 15 / 7 | 35.1, 34.2, 42.8% | +0.002 [-0.005, +0.008] | +0.003 [+0.001, +0.006] |
| V3 max(V2, water) | 36.2% (0.821) | 2 / 0 | 27.7, 35.9, 41.6% | +0.000 | -0.000 |
| **V4 approaches in the travel line** | **35.5% (0.821)** | **14 / 0** | **27.7, 35.7, 40.8%** | **+0.000 [+0.000, +0.000]** | **-0.000 [-0.001, +0.002]** |
| V5 lidar trails to water | 36.0% (0.821) | 0 / 0 | 27.7, 37.0, 41.5% | (lidar areas only) | |

- **Corridors as water (V1, V2)** moved the median nowhere and split the sites: 15 better, 7 worse. The small
  area lost 8 points and the largest 1.7, and the one human pick that sits beside a pond fell far down the
  ranking: taking the plain nearness to water away costs exactly the spots people put beside water. The collar
  data leaned their way (+0.003, kill sites +0.005 [+0.002, +0.008]), but out-of-region collars only veto, and
  the picks break the shipping rule. Not kept.
- **V3** (keep the water factor, add the corridor beside it) is nearly a no-op: the corridors mostly sit where
  water already scores.
- **V4** lifts 14 sites by half a point or more and lowers none, with every other benchmark unchanged (collar
  paths and kills ±0.002). The inland-Northwest lock-box, opened once for V4 only: 0.512 -> 0.513
  (+0.001 [+0.000, +0.001]).
- **Lidar trails to water (V5)** moved no site by half a point: where it found a trail to water, the water factor
  already scored the ground.

## Measured on this tree (V4 as shipped)

`factors.destination_approach` (kernels `terrain._least_cost_tree`, `terrain._tree_flow`) reproduces the
prototype's field exactly: on the saved v3.2 states, the largest difference anywhere is one float16 step (under
5e-4, at under 30 cells per area). A fresh run reads the canopy in whole metres (as `state.pkl` keeps it), so the
layer is the same whether it comes from a run or from a saved state.

**Human picks** (rebuilt pin-free states on top of v3.3, the pond barriers of [13](13-water-barriers.md); the same
states with the approaches left out of the travel line reproduce v3.3 exactly):

| | Median rank (random) | vsR | Top 25% (random) | Fixed-K within 50 / 100 / 150 m | Areas |
|---|---|---|---|---|---|
| v3.2 | 36.1% (87.0%) | 0.821 | 28% (5%) | 12 / 40 / 52% | 27.7, 36.9, 41.6% |
| v3.3: ponds as barriers | 36.3% (87.3%) | 0.821 | 28% (5%) | 12 / 40 / 56% | 27.9, 36.9, 41.7% |
| **v3.4: destination approaches** | **35.7%** (86.7%) | **0.822** | 28% (5%) | 12 / 40 / 56% | 27.9, 35.8, 41.0% |

Against v3.3: 13 sites better by half a point or more, none worse. The change is the prototype's (-0.6 points of
median; areas 0, -1.1 and -0.7 against 0, -1.2 and -0.7); the absolute numbers sit 0.2 points higher because v3.3
already moved the baseline from 36.1% to 36.3%, and one site that the prototype moved by just over half a point
moves by just under it here. Within the shipping rule, and better.

**GPS check** (the 28 v3 tile states, three original collar datasets, the approach rebuilt on each tile from its
saved layers): pooled rank at 20 m 0.514 -> 0.514, at ~500 m 0.506 -> 0.507; animal by animal 14 better and 17
worse at 20 m (p = 0.72), 12 / 13 at ~500 m. The shifted-map null stays at 0.50. No veto.

**Runtime** (the approach step of a full run, downloads cached): 1.4 s on a ~20 km2 area at 3 m (the whole run
10 s), 1.7 s on a 3 km-radius area at 3 m, and 9.5-10 s on the 230 and 310 km2 areas at 4.5-5.5 m (whole runs
about 40 s). Most of it is the least-cost tree over every cell: a 4-ary heap with the start cells settled
outside it runs it twice as fast as a binary heap, with identical routes. A coarser grid would be faster again
but would change the result, so it was not done.

## Caveats

- 13 better and none worse is the clearest per-site result since the house cost, but the 25 sites are still a
  tripwire, and every area shaped some earlier choice.
- The cameras and collars are blind to it, as for any term this local: an approach covers 1-5% of an area.
- Destinations are mapped water (NHD) and the canopy's openings; small unmapped water makes no approach.
