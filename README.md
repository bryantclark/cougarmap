# CougarMap

CougarMap finds where to put a trail camera for mountain lions (cougars). Give it a place and it reads the
terrain from free public map data. It ranks camera spots you can walk to and explains why each one is good. The
results come as a Google Earth map.

It works anywhere in the United States. It was built and tuned in the dry conifer mountains of the inland
Northwest.

## Quick start

**1. Install.** You need [uv](https://docs.astral.sh/uv/), which brings its own Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv tool install --python 3.12 git+https://github.com/bryantclark/cougarmap
```

On Windows, install uv with `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` and then run the same
`uv tool install` line.

**2. Find camera spots around a coordinate.** Give latitude and longitude the way Google Maps copies them. West
longitudes are negative. This analyzes everything within 3 km of the point in detail:

```bash
cougarmap analyze --near "47.3712, -116.1029" --radius-km 3
```

The first run in a new place takes a few minutes, mostly downloading elevation and map data. Later runs take
under a minute. When it finishes, it prints the best spots and a link to the map, and opens the map in
[Google Earth Pro](https://www.google.com/earth/about/versions/) (free).

To search a whole region around a town instead, use `hotspots`. It screens a 25 km circle and analyzes the
most promising blocks:

```bash
cougarmap hotspots "Missoula, MT"
```

**3. (Optional) Use it from your AI assistant.**

```bash
cougarmap setup
```

This connects CougarMap to the AI apps it finds on your computer: Claude, Codex, Gemini CLI, Antigravity or
Cursor. Restart the app, then ask *"find me cougar camera spots near Missoula, MT"*.

## What you get

For each spot:

- a **score** from 0 to 100 (60 or more is strong);
- the **reasons** in plain English, for example "downwind end of a meadow edge, where cold air draining down the
  valley meets the prevailing wind";
- the **walk** from the nearest road that is open that month, with distance and time;
- the **land** it sits on (national forest, state land and so on), and its coordinates.

The Google Earth map has the ranked pins (click one for its reasons), walking routes, dawn and dusk air-flow
arrows, saddles, and a layer for each factor you can switch on and off.

The ranked spots are on **public land** within **1 mile of walking** from a road open that month. Private land is
scored the same way and its best spots are found too, but kept apart: they're listed separately (P1, P2...) and
sit in map layers that start switched off. Tick "Private land spots" in Google Earth to see them, for example to
scan your own property, and get landowner permission before using one. `--max-walk-miles` and `--month` change
the other defaults.

## How it picks spots

CougarMap stacks four things lions use. The more of them at one spot, the better the spot:

1. **Wind.** Lions hunt into the wind at dawn and dusk. The best ground is where cold air draining down a valley
   flows the same way as the prevailing wind on high-pressure days.
2. **Edges.** Timber cover next to a meadow, clearcut or burn, where a lion can watch prey in the open. The most
   downwind end of an opening is best.
3. **Pinch points.** Saddles, the base of cliffs, stream banks and natural funnels where travel concentrates.
4. **Limited water.** Springs, seeps, seasonal water and small ponds. Water counts most when it is the only water
   for a mile.

It also favors natural travel lines such as drainage bottoms and ridge crossings. In winter it adds low,
sun-facing ground with shallow snow, where deer spend the winter. It avoids paved roads, towns, trailheads and
campgrounds. [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) has the details.

**What it can't see:** the wind is modeled from weather history, not measured on the ground. Prey isn't modeled.
Small springs and seeps are often missing from maps. If you know of water the maps miss, pin it in Google Earth
and import the file (`cougarmap import-kml my-pins.kml`): your pins count as water. Some parks and state land
restrict trail cameras, so check with whoever manages the land.

## Testing it against real lions

The map is only as good as what it catches. CougarMap keeps a field log of cameras, camera checks, snow tracks
and road surveys. `cougarmap validate <area>` tests the map against what you logged, with honest sample sizes.
[docs/FIELD_PROTOCOL.md](docs/FIELD_PROTOCOL.md) explains what to record in plain language.

## Privacy

Everything runs on your computer. CougarMap only downloads public map data (USGS, the Forest Service,
OpenStreetMap, weather and snow data). Your Google Earth files, camera locations and field log stay in
`~/Documents/CougarMap/my-data` (set `COUGARMAP_HOME` to move it) and are never uploaded.

## More

- [docs/USAGE.md](docs/USAGE.md): every command, the AI app setup, output files, and the data sources.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the scoring model, term by term.
- [docs/VALIDATION.md](docs/VALIDATION.md): how the model is tested and what the tests show.
- [docs/experiments/](docs/experiments/): what we tried, what worked, and what didn't.
- [docs/DEVELOPING.md](docs/DEVELOPING.md): working on the code.

## License

MIT; see [LICENSE](LICENSE). Data downloaded at run time keeps its own terms (OpenStreetMap is ODbL; the
Olympic Cougar Project GPS data used only by the validation check is CC BY-NC 4.0).
