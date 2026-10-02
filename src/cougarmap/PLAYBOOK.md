# CougarMap playbook (for any AI agent using the cougarmap tools)

CougarMap finds mountain lion (cougar) hotspots and trail-camera spots. It stacks four factors over free
public map data. The more factors at one spot, the better, assuming there's prey around:

1. **Wind** (most important). Lions hunt into the wind at dawn and dusk. Two things matter: cold air draining
   down hillsides and valleys, and the prevailing wind on high-pressure days. The best places are valleys where
   both flow the same way, plus the windward side just below ridgelines.
2. **Edges** (second). The hunting edge: in timber cover (even patchy), within about 35 m of a meadow, clearcut, burn or other
   opening, so a lion can watch prey out in the open without being seen, and the opening's own rim, out to about
   25 m from real cover, where kills cluster. The most downwind end of an opening is the single best place:
   downwind for the evening cold air draining downhill and for the daytime high-pressure wind (best when both
   agree). Ridgelines, valley bottoms, water and trails through timber add to an edge but don't make one.
3. **Pinch points.** Saddles, the base of cliffs, stream and lake banks, fences, and natural travel funnels.
4. **Limited water.** Springs, seeps, seasonal water, and (a model choice, not from the field method) ponds under
   5 ha. It counts for more when it's the only water within a mile. Small water is often unmapped:
   when the user finds some, have them pin it in Google Earth (it counts at full strength).

How spots are rated: over the camera's ~20 m view, by the habitat within ~500 m (hunting edge and water
around), with extra credit on natural travel lines: drainage bottoms (most on gentle grades) and ridge spines
where lions cross them (saddles, junctions of ridges, and in winter ridges above big south/southeast-facing
slopes), and the covered approaches from bedding timber to water and meadows; other ridge spines get half. In winter (November-April) the habitat also counts low, sun-facing
ground with shallow snow, where deer winter, and (December-March, Washington) mapped deer/elk winter range.
Paved roads cut the score out to 800 m (traffic, people, theft), and so do populated areas (more than 15
houses within 500 m; about 60 gets the full cut) and trailheads, campgrounds and parking areas (out to 400 m).
A few houses cost a little too (people, dogs, camera theft): 1 house within 500 m x0.79, 5 x0.54, 15 x0.39; the
spot's reasons say so ("3 houses within 500 m ... score x0.62"). Gravel and forest roads don't count.

**Alternate on the trail.** Each spot may carry a `trail_alternate`: the best cell within 150 m beside a quiet
road or trail (a closed forest road, a forest road closed that month, a two-track, a path), away from
trailheads, scoring 30+ and at least half the spot. Cameras on dirt roads and trails catch more of the lions
passing, but the method's own spots mostly sit off mapped lines, and in SNAPSHOT USA arrays a mapped line within
15 m of a camera gave no detection gain (the trails that matter are mostly unmapped game trails, found on the
ground: see the placement line below), so the spot stays the pick: mention the alternate as an option ("or put it on the two-track 80 m east"), never in place of the spot. When
`open_to_vehicles` is true it sits on or beside a road open that month: say so (more traffic, theft risk). In
the KMZ they are a hidden folder ("Alternate spots on a trail/two-track").

**Worn trails (lidar).** Unless the run is fast (`fast=True`, `--fast`), it also finds worn trails in 1 m
lidar where the USGS has it: game trails, cattle trails and old two-tracks show as narrow troughs and benches,
most of them on no map. They go in a hidden KMZ folder ("Worn trails (lidar)": ones on no map bright, ones on a
mapped road or trail grey), and each spot within 30 m of one on no map gets a `worn_trail` hint: where to face
the camera ("a worn line on no map 20 m N: hang the camera facing it"). It changes no score and never moves the
spot: give it in one line after the spot's reasons, as a placement tip, and say it is from lidar (check it on
the ground). A first run downloads about 110 MB more per area and takes about 40 s longer (about 7 s once cached);
use `fast=True` only when the user wants a quick look and isn't placing cameras yet. `summary.worn_trails` says whether it ran and how much it found; `summary.notes` says when there is no
1 m lidar there. Faint two-tracks across flat open meadows are often missed.

**Two winds.** Reasons name the wind they mean: the dawn/dusk high-pressure wind (`summary.wind.prevailing_from`)
for where it lines up with cold-air drainage and for the windward side of ridges, and the daytime high-pressure
wind (`daytime_from`) for the downwind ends of openings. Both are the method's: don't present them as a
contradiction. A user's own wind (`wind_from_deg`) stands in for both.

Hard rules (defaults): **public land only**, and **within 1 mile of walking** from a road that is open that month.
Private land is still analyzed: every run also returns `private_candidates` ("P1", "P2"...) and puts them in the KMZ
as hidden layers ("Private land spots" folder, "Lion score on private land", "Private land"). Mention them only if
the user asks or the public spots are weak; they need landowner permission.

## Which tool to use

| The user says | Call |
|---|---|
| "find cougar hotspots near X", "where should I put cameras around X" | `find_hotspots(location="X")` |
| "... within 10 miles of X" | `find_hotspots(location="X", radius_km=16)` |
| "scan my property at LAT,LON", "include private land" | `find_hotspots(location="LAT,LON", radius_km=1.5)`, then report its `private_candidates` |
| an area in their Google Earth file | `import_kml(path)` (lists its areas), then `find_hotspots(kml=..., area_name=...)` |
| "use this KML" / gives a file path | `import_kml(path)` first, so their water/sign pins are used |
| "why is spot #3 good?" | `explain_point(area, lat, lon)` |
| "more spread out", "more spots", "within 2 miles" (same area) | `repick(area, ...)` (seconds, no re-download) |
| "where are the game trails?", "exactly where do I hang it?" | the spot's `worn_trail` hint and the "Worn trails (lidar)" layer (rerun without `fast` if it was a fast run) |
| "what about private land there?" (same area) | read `private_candidates` from the last result (no rerun) |
| "I put a camera out at LAT,LON" | `log_camera(lat, lon, name, arm, zone, start, ...)` (see Field log below) |
| "checked camera X: a cougar on Nov 3 at 5:40" | `log_check(deployment="X", events=[...])` |
| "log this lion track" + a GPX/KML file | `log_track(file, snow_age_h, confidence)` |
| "surveyed Route 1, two crossings / nothing" | `log_transect(route, file or crossings)` |
| "what have I logged?", "which cameras are out?" | `field_log()` |
| "send my results to Sam", "share my camera log" | `share_results(name=<the user's name>)`: the file has camera locations, so tell them to send it privately |
| "how is the map doing in AREA?" | `validate(area)` |
| "my camera at LAT,LON got a lion" (a one-off, not a test) | `log_camera(lat, lon, name, start=..., end=...)` (arm unpaired), then `log_check(deployment, events=[...])` |
| "open it" | `open_file(kmz_path)` (opens Google Earth) |

`area` (explain_point, repick) is the analyzed area's name, as in the result's `summary.area`, or its folder in
the results folder; saved analyses anywhere else are not read.

**Long runs:** `find_hotspots` starts a background job. It waits up to
`wait_seconds` (default 30, max 40) and return either the finished result or a `job_id` with `state: running`. In that
case, call `job_status(job_id, wait_seconds=30)` again and keep checking until `state` is `done`. Tell the user
it's working. First runs in a new region take 1-5 minutes (downloading elevation, canopy and map data); repeat
runs take seconds to a minute. Never start the same job twice. If you lose track, `job_status()` with no id lists the recent jobs.

Month: use the current month unless the user names a season or month. The wind changes with the season.

## How to report results

- Lead with the **top 3-5 spots** (the best spot in each hotspot zone first). For each, give: the score (0-100;
  60+ is strong), the plain-English reasons (returned with every spot, so use them), walking distance and time
  from the road, the land name, and coordinates.
- Say what the prevailing wind is and that it's modeled from weather history (HRRR 850 hPa, the wind above the
  ridges), with its consistency R: under 0.35 it is unsteady (it holds that direction only about a third of the
  time; `most_common_from` can differ), so say so. `ground_level` is the 10 m wind, for reference only: at dawn
  and dusk it mostly shows cold air draining down the main valley, and it often disagrees with the 850 hPa one.
  If the user knows the local wind (or trusts a nearby weather station), they can say "assume the wind is from
  the west" -> `wind_from_deg=270` (degrees it blows FROM).
- Give the **KMZ path** and offer to open it. In Google Earth it shows ranked pins (click one for its reasons),
  walking routes, dawn/dusk air-flow arrows, saddles, and toggleable layers for each factor.
- Give every spot one **placement line**: on the ground, set the camera beside the game trail, old two-track or
  closed road that crosses the spot (look within ~50 m), about knee-to-waist high, a few metres off the line and
  facing along it. In SNAPSHOT USA camera arrays, cameras on trails and dirt roads caught about 3x more lions
  than other cameras in the same array. Maps don't show most game trails, so this is a field call: the spot says
  where, the trail through it says exactly where.
- When a spot has a `trail_alternate`, add it in one line after the spot's reasons, as an option. A
  `worn_trail` hint (worn trails on) goes in one line too, as where to face the camera.
- Brief caveats, once: the wind is modeled rather than measured (check it in the field), prey isn't modeled, and
  small water sources are often missing from maps. City/county parks and some state land restrict trail
  cameras, so check with whoever manages the land (the land name is in each result).
- Pass on `summary.notes`. When every spot scores low (no mapped water, solid timber or open flats), the spots
  are the best of weak ground and the note says so: tell the user, and ask whether they know of springs, ponds
  or guzzlers the maps miss (pins in their KML count as water).
- Don't dump raw JSON. Keep it conversational, the way you'd tell a friend who photographs wildlife.

## Field log: testing the map against lions

The tool encodes our field method; what tests it against lions is the user's own field data, logged with
the tools below (the field protocol in the project's docs/FIELD_PROTOCOL.md explains it for the user). Misses count
as much as hits: always log a check that caught nothing and a route survey with no crossings.

- **Cameras.** `log_camera` when one goes out: `arm` is why it's there: `model` (one of this tool's picks),
  `human` (a spot a person picked by hand), `control` (a nearby spot chosen without the map: the yardstick), `on-feature` /
  `off-feature`, or `unpaired` (not part of a test). Cameras compared with each other share a `zone` (put out
  the same day, 150-500 m apart, set up alike). Always ask for `trail_type` (what the camera watches: game-trail, hiking-trail,
  closed-road, open-dirt, paved, or none): on-trail cameras catch about 3x more lions, so `validate` compares
  only like with like, and a camera with no `trail_type` can't be compared with the on-trail base rate (the
  result carries a `warning` until it is set: `log_camera(deployment=id, trail_type=...)`). Also ask for
  `height_m`, `facing_deg`, `lure`.
  The result's `deployment.id` is what later calls use. `log_check(deployment, date, events, downtime_nights,
  removed)` at each visit: one event per visit (photos of one animal within 30 min are one), with species
  cougar / deer / elk / other; downtime = nights it wasn't recording; event times with a zone are converted to
  local time. `log_camera(deployment=id, end=...)` corrects or ends one; `end=""` reopens one the user says is
  still out (a check after a camera's end is refused until then). If the user doesn't know an id, `field_log()`
  lists them. A camera on a pick's trail alternate is `on-feature` in the pick's zone, and the pick's own camera
  stays `model` (or `human`): the report pairs them.
- **Snow tracks.** `log_track(file=<GPX/KML/KMZ>, snow_age_h, snow_depth_cm, confidence)`; confidence
  certain / probable / possible (possible tracks are kept but not tested). Points `[[lat, lon], ...]` work too.
- **Crossing routes.** `log_transect(route, file)`: the file's track is the route and its waypoints named "lion"
  the crossings; other waypoints come back as `ignored_waypoints`: read them to the user and ask whether any was
  a crossing. `crossings=[[lat, lon], ...]` adds more. A repeat survey of a known route needs only `route` and
  `date`; zero crossings is a real result.
- **The report.** `validate(area)`: cougar detections per 100 camera-nights by arm and placement; only on-trail
  cameras are compared with random on-trail cameras in NE Washington (about 0.9 per 100 in summer, 0.4 in
  winter), and off-trail or unrecorded ones are reported as not comparable, not as misses. Model vs control
  within zones (rate ratio and a paired permutation p-value) carries a `placement_warning` when the two cameras
  of a zone weren't placed alike: say so, since placement alone can make a 3x difference. Then each snow
  track's percentile vs copies of it rotated and shifted 100-1,500 m (0.5 = chance; `by_start` splits road-found tracks, which the test only partly corrects, from the rest), the
  crossing routes' AUC (0.5 = chance), sample-size advice, and where any human camera picks (CamNN pins in their KML) rank. Below 30
  on-trail camera-nights an arm's vs_base is withheld (`too_few_nights`): don't quote a rate ratio for it. Read the `summary` lines to the user in plain words, and be honest about sample size: a few
  tracks or one season of cameras is not a verdict (about 15-20 paired zones over 6 months to see a 3x gain).

## Privacy

The user's Google Earth files, camera locations, camera results, tracks and routes live in the private data folder. Never
upload, publish or share them, and don't paste camera coordinates anywhere outside this conversation.
