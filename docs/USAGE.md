# Using CougarMap

## With an AI assistant

`cougarmap setup` registers the CougarMap MCP server, and installs the playbook as a skill where the app supports
skills:

| App | Tools (MCP) | Playbook |
|---|---|---|
| Claude Code | `claude mcp add --scope user` + an allow rule | `~/.claude/skills/cougarmap` |
| Claude Desktop (chat) | `claude_desktop_config.json` | server instructions |
| Claude Cowork | plugin zip in `~/Documents/CougarMap/` (upload via Customize -> Plugins) | in the plugin |
| Codex (CLI, app, IDE) | `~/.codex/config.toml` (20 min tool timeout, pre-approved) | `~/.agents/skills/cougarmap` |
| Gemini CLI | `~/.gemini/settings.json` (20 min timeout) | `~/.gemini/skills/cougarmap` |
| Antigravity | `~/.gemini/config/mcp_config.json` | `~/.gemini/config/skills/cougarmap` |
| Cursor | `~/.cursor/mcp.json` | `~/.cursor/skills/cougarmap` |

Options: `--dry-run` shows what it would change, `--only codex` sets up one app, `--all` includes apps it didn't
detect, and `--uninstall` removes everything it added. Before its first edit to a config file it saves a
`*.cougarmap-bak` copy.

Long operations run as background jobs. Each tool call waits at most 40 seconds, then returns a `job_id` that
the assistant checks with `job_status`. That keeps CougarMap under every app's tool timeout (Cursor's is about
60 s). The playbook reaches the model three ways: as short MCP server instructions (under 2 KB, because Claude
Code truncates longer ones), as a `playbook` tool, and as a skill.

Tested end to end with the Codex CLI and a Claude Code MCP client. The Gemini, Antigravity, Cursor and Cowork
configs follow each app's documentation but have not been run live.

Things to ask:

- "Find cougar camera spots near Missoula, MT for November."
- "Scan my property at 47.3712, -116.1029." (its private-land spots come back separately)
- "Import my Google Earth file at ~/Downloads/my-areas.kml and look at the My Area area."
- "Why is spot 3 good?" / "Give me more spread-out spots." / "Only within half a mile of the road."
- "Assume the wind is from the west."
- "Where are the game trails near those spots?" (worn trails from 1 m lidar: slower on a first run)
- "I put a camera out at ..." / "Checked camera M1: a cougar on Nov 3 at 5:40 am." / "How is the map doing?"

## Command line

```bash
cougarmap analyze --near "47.3712, -116.1029" --radius-km 3   # everything around a spot, in detail
cougarmap hotspots "Missoula, MT"                             # a region: screens 25 km, analyzes the best blocks
cougarmap hotspots "Missoula, MT" --radius-km 40 --background && cougarmap jobs
cougarmap analyze --bbox=-116.13,47.35,-116.08,47.39        # one area in detail
cougarmap analyze --near "47.3712, -116.1029" --radius-km 1.5   # your property: see its P1, P2... spots
cougarmap analyze --near "47.3712, -116.1029" --radius-km 1.5 --fast   # skip the slow extras (worn trails)
cougarmap import-kml ~/Downloads/my-areas.kml               # your outlines and water pins
cougarmap analyze --kml ~/Documents/CougarMap/my-data/my-areas.kml --area "My Area"
cougarmap repick my-area --n 20 --per-zone 1
cougarmap explain -- my-area 47.3712 -116.1029
cougarmap open <the .kmz path it prints>
```

`hotspots` and `analyze --near` take coordinates as typed (`47.37, -116.10`, quoted or not). A
longitude missing its minus sign is made west, with a note. `explain` and `log-camera` take separate
latitude and longitude numbers: put `--` before them so a negative longitude isn't read as an option. When
`hotspots` or `analyze` finishes it prints a `file://` link to the map, and at a terminal it opens the map in
Google Earth (`--no-open` skips that). Errors print one line; `COUGARMAP_DEBUG=1` shows the traceback.
`cougarmap <command> --help` lists every option.

### Field log

```bash
cougarmap log-camera --name M1 --arm model --zone z1 --trail-type game-trail -- 47.3712 -116.1029
cougarmap log-check M1 --event "2026-11-03T05:40 cougar 1" --event "2026-11-10T19:00 deer 3"
cougarmap log-track ~/Downloads/track.gpx --snow-age-h 20 --confidence certain
cougarmap log-transect "Route 1" ~/Downloads/ridge.gpx     # waypoints named "lion" are crossings
cougarmap field-log
cougarmap validate my-area
cougarmap share-results --name Sam                  # your log as one file, to send privately
cougarmap import-results ~/Downloads/cougarmap-field-log-sam-2026-12-01.json   # add someone's to yours
```

[FIELD_PROTOCOL.md](FIELD_PROTOCOL.md) explains what to record and why.

## Where files go

| What | Installed | In a source checkout with `data/private/` |
|---|---|---|
| Results (`<area>/cougarmap.kmz`, `summary.json`, `candidates.geojson`, `state.pkl`) | `~/Documents/CougarMap/results` | `out/` |
| Your KML files and the field log (`observations.jsonl`) | `~/Documents/CougarMap/my-data` | `data/private/` |
| Downloaded map data (cache) | `~/.cache/cougarmap` | `~/.cache/cougarmap` |

`COUGARMAP_HOME` moves the first two; `COUGARMAP_OUT`, `COUGARMAP_PRIVATE` and `COUGARMAP_CACHE` move one each.
`state.pkl` holds the computed layers, so `explain`, `repick` and `validate` don't rerun the analysis. It is a
pickle, so CougarMap only reads states inside its results folder.

## Data sources (all free, no keys)

| Layer | Source |
|---|---|
| Elevation | USGS 3DEP 1 m lidar (TNM API + S3 COGs), 1/3 and 1 arc-second fallback |
| Worn trails (skipped with `--fast`) | USGS 3DEP 1 m lidar at full resolution over the area (about 110 MB for a 3 km radius, cached) |
| Canopy height | Meta/WRI global canopy height v2 (1 m, AWS open data) |
| Water | USGS NHD (springs, streams by permanence, waterbodies) |
| Roads, trails, fences | OpenStreetMap (Overpass), ODbL |
| National forest road seasons | USFS Motor Vehicle Use Map |
| Public land | USGS PAD-US (public access layer) |
| Buildings (populated-area penalty) | Microsoft Global ML Building Footprints |
| Wind | Open-Meteo historical forecasts (HRRR): 850 hPa (prevailing, scored), 10 m (ground level, reference) |
| Trailheads, campgrounds, parking | OpenStreetMap (Overpass), ODbL |
| Snow depth (winter months) | NSIDC SNODAS (G02158, masked daily; NOAA, public domain) |
| Deer/elk winter range (Dec-Mar, Washington only) | WDFW Priority Habitats and Species on the Web, public layer |
| Cougar sightings (scout prior) | iNaturalist |

## A double-click installer

For someone who won't use a terminal, every [GitHub release](https://github.com/bryantclark/cougarmap/releases)
has `CougarMap-installer.zip`. Unzip it and double-click `Install CougarMap.command` (Mac), or right-click
`install.ps1` -> Run with PowerShell (Windows). It installs uv, installs CougarMap from PyPI, and runs
`cougarmap setup`. `READ-ME-FIRST.txt` explains it in plain English. Running it again updates CougarMap.

## Updating

```bash
uv tool upgrade cougarmap
```

## In the MCP Registry

CougarMap is listed in the [MCP Registry](https://registry.modelcontextprotocol.io) as
`io.github.bryantclark/cougarmap`. Apps that install servers from the registry run it with `uvx cougarmap mcp`.
The first start downloads its dependencies (about 150 MB) and can take a minute, longer than some apps wait.
`uv tool install cougarmap` followed by `cougarmap setup` avoids that: the server then starts in a second.

## Uninstall

```bash
cougarmap setup --uninstall
```

```bash
uv tool uninstall cougarmap
```
