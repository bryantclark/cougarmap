#!/bin/sh
# Re-analyze the test areas of a KML and rank its camera pins (CamNN) among each area's spots.
#
#   ./scripts/rerun_eval.sh <areas.kml> "<Area name>" ["<Area name>" ...]
#
# Each area runs twice, at month 10: as the tool runs it (out/<slug>, with the KML's water/sign pins), and
# without those pins (out/no-pins/<slug>). When the person who chose the cameras also placed the pins, the
# pin-free numbers are the honest ones. Large areas (200+ km2) peak at ~9 GB of RAM, so areas run one at a time.
# Logs go to out/debug/.
set -e
[ $# -ge 2 ] || { sed -n '4p' "$0" | cut -c3-; exit 2; }
cd "$(dirname "$0")/.."
KML="$1"; shift
mkdir -p out/debug out/no-pins
AREAS=""
for a in "$@"; do
  s="$(echo "$a" | tr ' A-Z' '-a-z')"
  AREAS="$AREAS --area $s=$s"
  uv run cougarmap analyze --kml "$KML" --area "$a" --month 10 --no-open \
    > /dev/null 2> "out/debug/$s.log"
  COUGARMAP_OUT=out/no-pins uv run cougarmap analyze --kml "$KML" --area "$a" --month 10 \
    --no-pins --no-open > /dev/null 2> "out/debug/$s-no-pins.log"
done
echo "== with the KML pins (out/)"
# shellcheck disable=SC2086
uv run python scripts/eval_picks.py --kml "$KML" $AREAS --states out --by-cam
echo "== without them (out/no-pins/)"
# shellcheck disable=SC2086
uv run python scripts/eval_picks.py --kml "$KML" $AREAS --states out/no-pins --by-cam
