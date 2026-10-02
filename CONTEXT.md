# Context

## What this project is

CougarMap finds mountain lion (cougar) areas and trail-camera spots by stacking four factors — wind, edges,
pinch points, and limited water — over free public map data. It works anywhere in the US and was built and
tuned in the dry conifer mountains of the inland Northwest. It runs from a terminal or is driven by an AI agent:
ask "find cougar areas near Missoula" and the agent scouts the region, analyzes the best blocks, and hands back a
Google Earth KMZ with ranked camera pins and the reasons for each. The method is our own encoding of where lions hunt
and travel, tested against human camera picks, open GPS collar data, and real lion detections.

## Core entities

- **Area** — a block analyzed at full resolution (a bbox, a point + radius, or a polygon from a KML). Has one
  saved **State** and many **Spots**. Named by slug (`my-area`).
- **Region** — the large circle a **Scout** screens on a coarse grid to choose which Areas deserve analysis.
- **Factor** — one of the four method layers (wind, edges, pinch points, limited water), computed per Area.
- **Layer** — any raster the model computes for an Area (factors, terrain, access, land, season). The one schema
  is `state.Layers`.
- **State** — `state.pkl`: an Area's layers and options, saved so **Repick**, **Explain**, and **Validate** run
  without recomputing. Versioned and migrated.
- **Spot** (candidate) — a ranked camera location with a score (0–100, 60+ strong), plain-English reasons, walk
  distance/time, land owner, an optional **Trail alternate** and an optional **Worn-trail hint**. Private-land spots are **Private candidates**
  (`P1`, `P2`…).
- **Zone** — a group of nearby camera sites used to pair a model camera with a control camera.
- **Camera deployment** — a camera put out at a site, with an **Arm** (`model`, `human`, `control`), a zone, and
  effort (camera-nights). Has many **Checks**.
- **Check** — a visit to a camera: detection events, downtime.
- **Snow track** / **Transect** — lion truth from tracking: a followed track (GPX/KML), or a fixed route surveyed
  for lion crossings (zero crossings count).
- **Field log** — `observations.jsonl` in the private folder; holds deployments, checks, tracks, transects.
- **Shared field log** — someone's own field-log records as one file (`share-results`), added to another log with
  `import-results` under the sender's name (`sam/M1`); a newer file from the same sender replaces the older one.
- **Weights page** — `explore.html`, written with `--interactive` / `interactive=True`: a local page where the
  factor weights are sliders and the top Spots move live. It reruns the score on a coarser grid; it never changes
  the saved State or the KMZ.
- **Job** — a background run (analysis, hotspots) the agent polls with `job_status`.

## Relationships

```
Region ──scout──< Area ── State ──< Spot (─ Trail alternate, ─ Worn-trail hint)
                   │
                   └──< Zone ──< Camera deployment ──< Check
Field log ──< Camera deployment | Snow track | Transect
```

## Vocabulary that matters

- **Pinch point** — terrain that funnels travel: saddles, cliff bases, banks, fences. *Not* a road chokepoint.
- **Limited water** — water that is scarce nearby (springs, seeps, seasonal water, ponds < 5 ha); counts more when
  it is the only water within a mile. *Not* rivers or big lakes.
- **Edge** — the hunting edge: timber cover within ~35 m of an opening, plus the opening's rim; best at the most
  downwind end. *Not* any land-cover boundary.
- **Prevailing wind** — the dawn/dusk high-pressure wind; **daytime wind** is the daytime high-pressure wind used
  for downwind ends of openings. Both are the method's; they are not a contradiction.
- **Open road** — a road open to vehicles in the month analyzed (USFS MVUM seasons). The access rule is within
  1 mile of *walking* from one.
- **Public land** — the land rule for the ranked spots. Private land is still analyzed and returned as Private candidates
  (a separate list and hidden KMZ layers); there is no option to mix them.
- **Human picks** — camera spots a person chose by hand from the same factors, as `CamNN` pins in a KML
  (`evaluate.py`, set up in `data/private/eval.toml`). A reference, *not* ground truth. The field-log arm for such a
  camera is `human`.
- **Lion truth** — actual detections: camera hits, snow tracks, transect crossings. The only real ground truth.
- **Pins** — water/sign points the user marks in Google Earth. Pin-free eval (`--no-pins`) is the honest one
  when the person who chose the human-picked cameras also placed the pins.
- **Trail alternate** — an optional nearby spot on a quiet road or trail; offered as an option, never instead of
  the Spot.
- **Placement** — where a camera is strapped at its spot: **on-feature** (beside a game trail, two-track, closed
  or dirt road: `trail_type` game-trail, hiking-trail, closed-road, open-dirt), off-feature (none, paved), or
  unrecorded. A field call, not a map one; cameras are compared like with like by placement. *Not* the Trail
  alternate, which is a different spot on a mapped line.
- **Worn trail** — a trail tread found in 1 m bare-earth lidar (a game trail, cattle trail or old two-track),
  mapped or not; on unless the run is fast (`--fast`). A **Worn-trail hint** (`worn_trail`) names the nearest one on no map
  within 30 m of a Spot, as where to face the camera. It changes no score and never moves the Spot.

## Examples / canonical dialogues

> "Find cougar hotspots near Missoula for November." The agent calls `find_hotspots`, polls `job_status`, then
> reports the best spot per zone with score, reasons, walk time, land name, the prevailing wind, and the KMZ path.

> "Why is spot 3 good?" The agent calls `explain_point` on the saved State — no rerun.

> "Camera M1 caught a cougar at 5:40 on Nov 3." The agent calls `log_check` for M1; later `validate(area)`
> compares model vs control arms within zones and reports sample-size caveats honestly.

## Ambiguities / open questions

- Very little lion truth exists yet; most evidence is still agreement with human picks, not detections.
- Prey is not modeled; small water is often unmapped.
