# Field protocol: testing the map against real lions

CougarMap encodes our field method for placing lion cameras, and it can be checked against camera spots people
picked by hand. What it can't check on its own is where lions actually walked. That has to come from the field, and the misses count as much
as the hits. "Nobody crossed here" or "the camera got nothing" is a result. Please log it.

There are three ways to collect this, best first:

1. **Snow tracks.** Follow a lion's trail and record it with your phone's GPS. One good track says more than a
   season of cameras.
2. **Crossing routes.** Drive or walk the same 2-3 routes after each fresh snow, and mark every place a lion
   crossed.
3. **Paired cameras.** In each spot you test, run one camera on the map's pick and one on a "control" spot
   nearby, and log every check.

Everything you log stays on this computer, in the private data folder, and never leaves it.

## What you need

- A phone with a GPS app that records a track and exports it as **GPX or KML**. Gaia GPS, onX, Avenza and most
  hunting and hiking GPS apps do this. Before the first trip, check that export works.
- A ruler, or a glove for scale in photos.
- Your trail cameras. Use the same model, height and settings at both cameras of a pair.

## 1. Snow tracks

**When:** 6-48 hours after a fresh snowfall, so you know the track is recent. Write down roughly how many hours
ago the snow stopped, and how deep it is.

**Is it a lion?** Check the prints:

- Round, with 4 toes and no claw marks.
- About 3.5-4 inches wide for an adult.
- A big heel pad with 3 lobes at the back.
- Often the hind foot lands in the front print.
- Sometimes a tail drag.

Photograph a few prints next to the ruler. Then rate it:

- **certain**: clear prints and several of the signs above.
- **probable**: most likely a lion.
- **possible**: can't rule out a dog or a bobcat. Log it anyway. It's kept, but left out of the test.

**Record it:**

1. Start the GPS track recording in your app.
2. Walk beside the trail, 1-2 m to the side, not on it.
3. Follow it **backwards** (the way the lion came from). That's safer, and you won't push the lion off a kill.
4. Go as far as you reasonably can. 0.5-2 km is great. Even 200 m helps.
5. If you find a bed, a kill or a scrape, drop a waypoint and name it.
6. Stop the recording and export it as GPX or KML.

Tracks you find from a road still count, but the test only partly corrects for them. It compares each track
with copies of itself moved around the area, and those copies don't stay near roads. So a track found along a
road in a creek bottom can beat its copies just because roads follow bottoms. Follow the track away from the
road as far as it holds. The report shows tracks found near a road separately from the rest; the ones found
away from roads are the cleaner test.

Safety: don't follow toward a fresh kill, carry bear spray, and tell someone where you are.

**Log it.** Tell the assistant something like:

> Log this lion track: ~/Downloads/track-dec3.gpx. Snow stopped about 20 hours ago, 15 cm deep, certain.

(Command line: `cougarmap log-track ~/Downloads/track-dec3.gpx --snow-age-h 20 --snow-depth-cm 15 --confidence certain`)

## 2. Crossing routes (transects)

**Set up once per area.** Choose **2-3 fixed routes, each 5-10 km long**: forest roads, two-tracks or a long
trail. Pick routes that cross different ground: creek bottoms, saddles, ridges and meadow edges. Give each
route a name, like "Route 1". Use the same routes all winter. That is what makes the surveys comparable.

**After each fresh snow** (or when a dirt road is muddy or dusty enough to hold prints):

1. Wait a night so animals have moved. Survey within 1-2 days.
2. Start the GPS track recording.
3. Drive slowly (under 10 mph) or walk the whole route.
4. At **every place a lion trail crosses the route**, drop a waypoint named "lion". If the trail walks along the
   road for a while, mark where it got on and where it got off.
5. Export the GPX or KML. Only waypoints with "lion" in the name count as crossings. Any others the app adds
   (parking, start, a photo spot) are left out, and the assistant tells you which ones it skipped. If it skips
   one that really was a lion crossing, rename it "lion" and log the survey again.

**Log every survey, including the ones with no crossings:**

> Log a transect on Route 1: ~/Downloads/route1-dec4.gpx, snow, fell about a day ago.
>
> Surveyed Route 1 again today after the new snow: no lion crossings.

A repeat survey with no crossings doesn't need a file, because the route is remembered. On the command line:
`cougarmap log-transect "Route 1" ~/Downloads/route1-dec4.gpx --snow-age-h 24`, or just
`cougarmap log-transect "Route 1" --date 2026-12-11` for a survey with nothing crossed.

## 3. Paired cameras

A **zone** is one test: two cameras put out on the same day, 150-500 m apart, and set up the same way.

- **Model camera** (arm `model`): on one of the map's top picks.
- **Control camera** (arm `control`): a spot in the same zone that nobody chose from the map. A simple way: in
  Google Earth, step 200-400 m from the model pick in a direction picked by a coin flip or the clock's second
  hand. Then use the nearest usable tree or game trail there.
- If someone picked a spot in that zone by hand, a third camera can go there (arm `human`).
- Some picks come with an "alternate on the trail": the best-scoring spot beside a quiet road or trail within
  150 m. To test whether on-trail cameras catch more lions on this ground, as they did in other studies, put an
  extra camera there with arm `on-feature`, in the same zone. The camera at the pick itself stays the `model`
  camera (or `human`, if a person picked it). The report compares the trail camera with the pick's camera in
  that zone, and the pick's camera with the control as usual, so nothing is lost. For example:

  > Put out a camera, on-feature arm, zone "A", at the trail alternate for that spot, on the two-track.

  If the alternate says it is on a road open to vehicles, expect more people and a higher theft risk there.

Keep everything else the same at both cameras: the camera model, the height, no lure (or the same lure at
both), and a similar facing. North-facing avoids sun glare. Write down the height, the facing and the kind of
trail at each camera (paved, open dirt road, closed road, hiking trail, game trail, none).

**Run them** for at least 2-3 months. Check both cameras of a zone on the same trip, every 4-6 weeks. At each
check, log:

- the date;
- any nights the camera wasn't working (dead battery, full card, knocked over: count from the last photo);
- every visit: its date and time, the species (cougar, deer, elk or other) and how many animals. Photos of the
  same animal within 30 minutes are one visit. Log deer and elk too: they show where the prey is.

When a camera comes down, say so. If you said a camera came down and it is actually still out, say that too
("M-A1 is still out"), or the next check can't be logged. Examples:

> Put out a camera, model arm, zone "A", at 47.3712, -116.1029, 1 m high facing north, on a game trail.
>
> Checked camera M-A1 today: a cougar on Nov 3 at 5:40 am, 3 deer on Nov 10 at 7 pm. No downtime.
>
> Checked the zone A control camera: nothing. The battery died about 10 days before I got there.
>
> Took down both zone A cameras today.

(Command line: `cougarmap log-camera --name M-A1 --arm model --zone "A" --height-m 1 --facing-deg 0 --trail-type game-trail -- 47.3712 -116.1029`,
then `cougarmap log-check M-A1 --event "2026-11-03T05:40 cougar 1" --event "2026-11-10T19:00 deer 3"`.)

**How many?** Lions are rare: a random trail camera in northeast Washington gets about 1 cougar visit per 100 nights in summer and
1 per 270 in winter. To show that the map's picks do 3 times better than controls, it takes about **15-20 zones
running about 6 months each**. To show 2 times better, about 40-50. Cameras that run mostly in winter need the
higher numbers. So cameras are a multi-season project, and that's why tracks and crossing routes come first
in the first winter.

## Sharing your results

If you're working with someone, for example one of you runs cameras and the other studies the results, send them
your whole log. Tell the assistant:

> Send my results to Sam.

(Command line: `cougarmap share-results --name <your first name>`.) It writes one file, and you email or message it
to them. **The file has your camera locations, so send it only to people you trust, and never post it.** They add it
with `cougarmap import-results <the file>`. Your cameras show up in their log under your name (`yourname/M1`), and
their `validate` counts them. Send a new file whenever you've logged more: it replaces the last one, it doesn't
double up.

## Checking the results

Ask the assistant: **"How is the map doing in My Area?"** (it runs `validate`). Or ask: **"What have I
logged?"** (`field_log`).

The report shows:

- **Cameras:** cougar visits per 100 camera-nights for each arm, compared with a random trail camera in NE
  Washington, and model vs control within each zone.
- **Snow tracks:** for each track, the share of its moved copies that it beat. Around 50% is chance. Clearly
  above 50% on most tracks means the map finds where lions walk. Tracks found near a road are also shown on
  their own.
- **Crossing routes:** whether the crossings fall on the parts of each route the map rates highly. 0.5 is
  chance.
- **Human picks:** if your KML has camera pins named CamNN, where they rank on the map.

Small numbers say little. A handful of tracks or one season of cameras is a start, not a verdict.
