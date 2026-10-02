---
name: cougarmap
description: Find mountain lion (cougar) areas and trail-camera spots with the CougarMap tools. Use when the user asks to find cougar/lion areas "near <place>" or a coordinate, scan a property, analyze an area from a Google Earth KML, explain why a spot is good, check the prevailing wind, or log what a trail camera caught.
---

# CougarMap

CougarMap turns our field method for placing lion cameras into math over public map data. Four factors, stacked; the more of
them at one spot, the better (activity goes up roughly exponentially with each extra factor), assuming prey:

1. **Wind** (most important). Lions hunt into the wind at dawn and dusk. Two air currents matter: cold air
   draining down hillsides and valleys at dawn/dusk, and the prevailing wind on *high-pressure* days. Best:
   valleys where both flow the same way (consistent air ~23 h/day). Also the windward side of ridges.
2. **Edges** (second). Meadow/timber edges (in the timber and on the opening's rim), valley bottoms, ridgelines,
   water edges. What matters most: the downwind end of an opening's edge is the single best place.
3. **Topographic pinch points.** Saddles, the base of cliffs/escarpments, stream and lake banks, fences, and
   natural funnels where movement concentrates.
4. **Limited water.** Springs, seeps, seasonal water, and (a model choice, not from the method) ponds under
   5 ha - best when it's the only water within a mile. Small water is often unmapped: the user's own water pins count at full strength.

Hard rules: within **1 mile of walking** (along the easiest route) from a road that's **open that month**;
**public land only** by default. Spots very close to a road are penalized (people, theft).

## Tools (MCP server `cougarmap`)

| Tool | Use |
|---|---|
| `find_hotspots(location / kml+area_name / bbox, radius_km, month, max_walk_miles, wind_from_deg)` | ranked camera spots + KMZ (a background job) |
| `job_status(job_id)` | poll a running job until `state` is `done`; no id = recent jobs |
| `repick(area, n_candidates, per_zone, max_walk_miles)` | re-select spots from a saved run in seconds |
| `explain_point(area, lat, lon)` | why a spot scores the way it does |
| `wind_summary(location, month)` | prevailing high-pressure wind |
| `import_kml(path)` | use their Google Earth file (its water/sign pins count) and list its areas |
| `open_file(path)` | open the KMZ in Google Earth |
| `log_camera(lat, lon, name, arm, zone, ...)` | a camera put out (arm model / human / control / ...; zone pairs them) |
| `log_check(deployment, date, events, downtime_nights, removed)` | a camera visit and what it caught (nothing counts) |
| `log_track(file, snow_age_h, confidence)` | a lion track followed in snow (GPX/KML/KMZ) |
| `log_transect(route, file)` | a fixed-route survey after snow: waypoints = crossings (none counts) |
| `field_log()` | what's logged: camera ids, effort, detections, tracks, surveys |
| `validate(area)` | test the area against the field log, plus how human camera picks rank |
| `playbook()` | the full guidance |

## How to handle requests

- **"Find cougar areas near X"**: `find_hotspots(location="X")` (default radius 25 km: it scouts the region and
  analyzes the best blocks), then `job_status` until done. Month = the current month unless the user names one.
- **"Scan my property at ..."/"include private land"**: a small radius (about 1.5 km) or their KML outline, then
  report the result's `private_candidates`. For an area already analyzed, don't rerun: its `private_candidates`
  are already there.
- **An area in their Google Earth file**: `import_kml(path)` lists its areas, then `find_hotspots(kml=...,
  area_name=...)`. Big areas (100+ km2) take a few minutes the first time (downloads); tell the user before
  starting.
- **"Assume the wind is from the west"**: `wind_from_deg=270`. The model's default wind is the high-pressure
  dawn/dusk 850 hPa wind; mention when it disagrees with what the user knows.
- **"Within 2 miles"**: `max_walk_miles=2`. For an area already analyzed, `repick(area, max_walk_miles=2)` is
  instant (but it can only shrink walking reach beyond the original run's 1-mile road search padding + ~5%).
- **"More spread out" / "more spots"**: `repick(area, per_zone=1, n_candidates=20)`.
- Results are grouped into hotspot **zones** (`summary.zones`); present the best spot per zone first.

## Reporting results

Lead with the top 3-5 spots. For each: score, the plain-English reasons (they're returned), walking distance and
time, land status, and coordinates. Then give the KMZ path (`summary.outputs.kmz`) - it opens in Google Earth with
ranked pins, walking routes, air-flow arrows, saddles, and toggleable factor layers. A spot may carry a
`trail_alternate` (the best cell within 150 m beside a quiet road or trail): give it in one line as an option
("or on the two-track 80 m east, score 52"), never instead of the spot; if `open_to_vehicles`, say it is on a
road open that month (traffic, theft). Reasons name their wind: the dawn/dusk high-pressure wind (drainage
convergence, windward ridges) or the daytime one (downwind ends of openings); both are the method's. Mention caveats briefly:
wind is modeled not measured (test it in the field), prey isn't modeled, and small water is often missing from maps.

Practical notes to pass on when relevant: city/county parks and some state land restrict or prohibit trail
cameras (check the managing agency - the land name is in each result); busy trail parks mean theft risk.
Runtimes (downloads are cached after the first run): find_hotspots ~1.5 min the first time in a region, ~10 s
after; analyze a few seconds for a 3 km block, ~30 s for a 200-300 km2 area (plus its first download); repick and
explain_point a few seconds.

Keep the private data folder private: it holds the user's camera locations and notes. Never publish or share it.
