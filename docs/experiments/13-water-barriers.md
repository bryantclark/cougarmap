# 13. Ponds as barriers, and stream confluences (2026-10-02)

**Where the ideas came from.** LLM agents picking camera spots blind (from the maps and imagery, without the
model's scores) kept giving two reasons the model has no term for: a stream confluence, and the strip of ground
between a pond and a slope that animals walking around the water have to use. Both were turned into fixed
definitions and measured. Nothing was fitted, so in-sample is held-out.

**Answer.** Neither idea moves any benchmark beyond noise. Confluences cost a little on the human picks and on
cameras; ponds as barriers are neutral everywhere. The owner kept ponds as barriers anyway, for realism (water is
a barrier animals walk around), as a decision rather than a measured gain
([decision](../../decisions/2026-10-02-water-bodies-pinch-barriers.md)).

## The variants

Every variant is a new pinch component, max-combined with the others (keeping the +0.1 per extra component above
0.3), except C3.

- **Confluences** (D8 on the 10 m DEM smoothed 20 m, as the travel and wind terms use): a junction is a land cell
  with at least two D8 donors that each drain at least A; the component is 0.8 x (1 - d / 45 m).
  - **C1**: A = 2 ha. **C2**: A = 10 ha. **C3**: A = 2 ha, added to the travel line instead of the pinch.
- **Ponds and lakes as barriers.** Water: NHD lakes, ponds and reservoirs (ftype 390/436) of at least 0.02 ha,
  plus the lakes the model already masks.
  - **W1, the squeeze:** land within 45 m of a shore whose nearest other barrier lies on the far side of it
    (cosine of the angle between the directions to the water and to the barrier <= -0.3). Other barriers: cliffs,
    ground of 35 degrees or more on the DEM smoothed 6 m (opened 2 cells, so no single cells), openings, and other
    water bodies; barrier cells within 9 m of the shore are the bank itself and are ignored. Value 0.8 x a ramp of
    the gap (shore distance + barrier distance): full at 40 m, zero at 100 m.
  - **W2, squeeze and ends (shipped):** W1, plus the ends of ponds under 5 ha: NHD flowlines crossing the shore,
    and D8 channels of at least 2 ha entering or leaving; 0.8 x (1 - d / 45 m).
  - **W3, bank and ends:** every pond as a bank (the lake bank, 0.7 x (1 - d / 30 m), extended below 0.5 ha), plus
    the ends.
  - **W4, post hoc:** W2 with a wider squeeze (zero at 150 m) and ends reaching 60 m, tried after seeing W2.

## Results

Baseline v3.2: human picks 36.1% (vsR 0.821), cameras 0.624, collars 0.596, paths 0.636, kills 0.576. Human
picks on the pin-free states; cameras are SNAPSHOT USA within-array concordance outside the inland Northwest; collars,
paths and kills as in [12](12-reshape-prey-tuning.md), with 95% intervals (array and animal bootstraps). Per site:
sites that moved at least half a point better / worse.

| Variant | Picks median (vsR) | Fixed-K 150 m | Per site | Areas | Cameras Δ | Collars Δ |
|---|---|---|---|---|---|---|
| Production v3.2 | 36.1% (0.821) | 52% | | 27.7, 36.9, 41.5% | | |
| C1 confluence 2 ha | 37.6% (0.818) | 44% | 6 / 12 | 27.1, 37.3, 38.6% | -0.003 [-0.008, +0.005] | +0.000 |
| C2 confluence 10 ha | 36.3% (0.821) | 48% | 5 / 10 | 26.7, 37.3, 42.2% | -0.003 [-0.007, +0.001] | -0.000 |
| C3 confluence 2 ha, travel | 36.3% (0.819) | 48% | 4 / 4 | 27.7, 35.8, 42.4% | -0.001 [-0.004, +0.004] | -0.001; paths -0.002 [-0.004, -0.000] |
| W1 pond squeeze | 36.2% (0.822) | 56% | 0 / 2 | 27.9, 36.9, 41.6% | -0.000 [-0.000, +0.000] | -0.000 |
| **W2 squeeze and ends** | **36.2% (0.822)** | 56% | 1 / 3 | 27.9, 36.9, 41.6% | -0.000 [-0.000, +0.000] | -0.000 [-0.001, +0.000] |
| W3 pond bank and ends | 36.2% (0.821) | 52% | 1 / 2 | 27.2, 36.9, 41.6% | -0.000 | -0.000 |
| W4 post-hoc wider W2 | 36.3% (0.822) | 56% | 1 / 3 | 27.9, 36.7, 41.6% | +0.000 [-0.001, +0.001] | -0.000 |

(This table is from the prototype on frozen fields; its baseline reads 41.5% for the largest area where the eval
reads 41.6%.)

- **Confluences** pull the ranking toward stream junctions the human picks mostly don't sit on: C1 loses 1.5
  points of median and 8 points of fixed-K, and every variant leans negative on cameras. C3, the travel version,
  also cost collar paths a little (-0.002 [-0.004, -0.000]). None kept.
- **Ponds** touch little ground (under 1% of the human-pick areas is within reach of a squeeze or an end), so
  they can only be neutral on these benchmarks: cameras and collar fixes hardly ever sit beside a pond. They lift
  fixed-K within 150 m from 52% to 56% (one more site), and one human pick on a pond strip moved from 23% to 22%;
  three others lost about a point, and the rest moved less than half a point.
- W3 (every pond a bank) is the weaker form; W4, chosen after seeing W2, was no better, so W2 shipped as fixed
  in advance.

## Measured on this tree (W2 as shipped)

The prototype counted the squeeze and the ends as two pinch components; production max-combines them into one
(`pinch_water`), so a cell where both are above 0.3 no longer gets the +0.1. The component itself is identical to
the prototype's on all three human-pick areas (every cell).

**Human picks** (rebuilt pin-free states, 25 human-picked sites in three areas):

| | Median rank (random) | vsR | Top 25% (random) | Fixed-K within 50 / 100 / 150 m | Areas |
|---|---|---|---|---|---|
| v3.2 | 36.1% (87.0%) | 0.821 | 28% (5%) | 12 / 40 / 52% | 27.7, 36.9, 41.6% |
| **v3.3: ponds as barriers** | **36.3%** (87.3%) | **0.821** | 28% (5%) | 12 / 40 / 56% | 27.9, 36.9, 41.7% |

Separate components (the prototype's combine) give 36.2%, vsR 0.822 and 27.9, 36.9, 41.6% on the same states.
1 site better and 3 worse by half a point or more, the rest within half a point. With the water pinch taken out,
the rebuilt states reproduce v3.2 (every other layer is unchanged). Within the shipping rule.

**GPS check** (the 28 v3 tile states, three original collar datasets, the water pinch computed on each tile):
pooled rank at 20 m 0.514 -> 0.514, at ~500 m 0.506 -> 0.506; animal by animal 2 better, 1 worse at 20 m, 4 / 4
at ~500 m. The shifted-map null stays at 0.50. No veto.

**Runtime:** about 0.6 s on a 3 km-radius area at 3 m (a full run there takes about 45 s) and 1.6 s on a 230 km2
area at 4.5 m.
The D8 receivers it needs come from the drainage computation the run already does.

## Caveats

- Neutral is the expected result for a term this local: it decides between nearby spots beside ponds, and none of
  the benchmarks has many samples there. It is untested, not shown to be wrong.
- Small ponds are often missing from NHD, and the squeeze only sees mapped water.
