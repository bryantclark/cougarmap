# 11. Worn trails from 1 m lidar (2026-10-02)

**Question.** Game trails, cattle trails and old two-tracks are where cameras catch passing lions, and most of
them are on no map. Can 1 m bare-earth lidar (USGS 3DEP) find them, and do they help pick or place cameras?

## The detector

Oriented trough filters on the 1 m DEM: the second derivative across an anisotropic Gaussian (1 and 2 m across,
6 m along) in 12 directions, each scale divided by its robust spread over the area (the lidar noise). On ground
steeper than 8 deg only directions at least 40 deg off the fall line count (drainages run down the fall line,
trails traverse), and the ground just below the line must be a bench: flatter than both the cut above and the
fill below. Hysteresis (4 and 7 noise units), a 1-cell skeleton, lines of at least 40 m. Code: `worn.py`,
settings `config.WornTrails`.

## What it finds (the prototype, on the human-pick areas)

- It recovers **58%** of the mapped OpenStreetMap tracks and paths, and about **75%** of what it finds is on no map.
- **Human picks sit near unmapped worn lines.** 32% of the human-picked camera sites are within 15 m of an
  unmapped detection, against 11-16% for matched control points.
- **As a score factor it does not help (negative result).** Added to the score as a travel term (max with the
  travel line, at 0.5 and 1.0) or as a multiplier (x1.25 to x2, any detection or unmapped only, within 15 or
  30 m), the median human-pick rank went from 43.5% to between 43.3% and 47.3% (lower is better), vsR stayed
  at 0.81 (0.806-0.819), and per-site changes split about evenly or worse (5-14 sites better, 10-16 worse).
  Some variants helped one area and hurt another. Nothing passed the shipping rule. **Decision:** it changes no score; it is a hidden KMZ layer and a per-spot placement
  hint, offered beside the spot like the trail alternate
  ([decision](../../decisions/2026-10-02-worn-trails-hint-off-by-default.md)).

## Production fixes (2026-10-02)

- **Creek banks.** A creek bank or channel edge looks like a bench: a steep side, a flat floor below. Lines
  within 10 m of a mapped flowline or waterbody shore (NHD), or of a channel the 3 m DEM drains at least 2 ha
  into, and within 30 deg of parallel to it, are dropped (a line crossing a channel is kept). On the ~20 km2
  human-pick area this removed **22 km of 85 km (25%)**; lines within 10 m of a channel fell from 21 km to 2 km.
  Recall of mapped tracks and paths there went from 66% to 61% (some tracks do follow creeks). The pick-proximity
  numbers above were measured before this mask.
- **Faint flat-meadow two-tracks** (two shallow ruts across smooth open ground) mostly stay under the noise floor.
  Not solved; noted in the docs.
- Edges are padded by odd reflection so a plain slope stays a plane (an even mirror made a false trough along the
  downhill edge).

## Cost and default

Measured on a ~3 km radius area (28 km2, 37 million 1 m cells, 10-core laptop): about **110 MB** of 1 m lidar
downloaded (cached as a ~70 MB GeoTIFF), about 30 s to download it and about 7 s to detect (FFT filters on
threads). With every other source cached, the run went from 10 s to 48 s the first time and 17 s once the lidar
was cached, and peak memory from 2.7 GB to 4.6-5.0 GB. On the ~20 km2 human-pick area: 12 s to 42 s.
`find_hotspots` analyzes three such blocks. So it is **off by default** (`--worn-trails`, `worn_trails=True`);
areas over 60 km2 get no layer. With it on, the spots and scores of both areas were byte-identical to the runs
with it off (every spot field except the new hint, and the summary).
