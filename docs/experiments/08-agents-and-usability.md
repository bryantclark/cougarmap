# 08. Agents and usability (2026-09-30 to 10-02)

## Running in any agent app

- **Background jobs.** A large first run takes minutes, and agent apps time out tool calls (Cursor after ~60 s).
  Long runs became detached jobs with spec, status, log and result files, and no tool call waits more than 40 s.
  The agent polls `job_status`.
- **One call for the common request.** `find_hotspots` does the scout, the detailed blocks and one merged KMZ.
- **Instructions where the model sees them.** Short MCP server instructions (under 2 KB, because Claude Code
  truncates longer ones), a `playbook` tool and skills.
- **`cougarmap setup`** registers the server with seven apps and backs up each config it touches. A fresh-home
  install rehearsal installed uv and the tool, registered every detected app, kept unrelated config, was
  idempotent, and uninstalled cleanly.

**Codex end to end.** Without the skill or instructions, Codex ignored the tool and searched the web. With them,
but with approvals disallowed, it refused ("tool requires approval"). With auto-approval for this server it called
`find_hotspots`, polled `job_status` and gave a correct ranked answer. **Decision:** `setup` writes the approval
and timeout settings for Codex.

**Not yet tested live:** Windows, Antigravity, Cursor, Cowork and Gemini CLI. One headless agent smoke test never
reached a tool call because its login had expired.

## Private land as a toggle, not a rerun

The access step now computes two sets of walking routes in one pass: one that never crosses private land, and one
that may go anywhere. Private spots come back separately (P1, P2...) and as hidden KMZ layers, and
`repick(public_only=False)` switches without a rerun. A fresh run returned 15 public and 15 private spots, and an
old state repicked in 5.8 s. **Cost:** saved states grew ~11%. Kept.

## The map legend

A user didn't know what the map's colors meant. A rendered legend (heat ramp, blue ramp, pins, routes, air-flow
arrows, saddles) is now a Google Earth screen overlay that is on by default. It was checked by eye; there has been
no user study. **Open:** with private land included, the "top 30%" shading covers much more ground and looks busy.

## Street addresses (negative result)

Three sessions asked for a map around a rural street address. OpenStreetMap's Nominatim found nothing each time;
the US Census geocoder matched it. Rural addresses are a main use case (scanning your own property), so a Census
fallback is a good next step. It isn't built yet.

## The command line, from real terminal use

Running the CLI by hand turned up:

- `lat, lon` without quotes (two shell words) was rejected;
- a negative longitude was read as an option flag;
- a longitude typed without its minus sign landed in Asia and surfaced as a full traceback;
- a noisy "not georeferenced" warning appeared in normal output.

The same test caught a real bug: `find_hotspots` reported every block's land, reasons and bounding box from the
*last* block.

**Changes:** coordinates are accepted as typed (quoted or not, `47.37N 116.10W` style, `--near` on `hotspots`).
A US-latitude point with a positive 60-170 longitude is flipped west, with a note. Errors print one line, and the
warning is silenced. `hotspots` and `analyze` print a link to the map and open it in Google Earth at a terminal.
The installer prints the next command to type. The previously failing commands now work, and the test suite
passed at 97.8% coverage.

**Side finding:** where results go depends on which install the `cougarmap` command resolves to (a source checkout
or the installed tool). A visible "results go to ..." line would help.

## MCP vs CLI on the same failure

Through the MCP tool, a failed geocode came back fast as a clear job error, and the agent recovered by geocoding
elsewhere and retrying with coordinates. The retry finished inside one 40 s wait. Through the CLI, the same failure
was a full traceback (fixed above).
