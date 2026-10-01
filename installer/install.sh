#!/bin/sh
# CougarMap installer (macOS / Linux). Double-click "Install CougarMap.command" on a Mac, or run: sh install.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "== Installing CougarMap =="
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "-- installing uv (Python tool manager, from astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
WHEEL="$(ls "$HERE"/cougarmap-*.whl | sort | tail -1)"
echo "-- installing $(basename "$WHEEL")"
"$UV" tool install --force --python 3.12 "$WHEEL"
"$UV" tool update-shell >/dev/null 2>&1 || true
BIN="$("$UV" tool dir --bin)"
echo "-- connecting CougarMap to your AI tools"
"$BIN/cougarmap" setup
if [ "$(uname)" = Darwin ] && ! ls -d /Applications/Google\ Earth* >/dev/null 2>&1; then
  echo
  echo "-- Google Earth Pro isn't installed; the maps open in it. Free download: https://www.google.com/earth/about/versions/"
fi
echo
echo "Done. Restart your AI app (Claude, Codex, Gemini, Antigravity, Cursor...) and ask:"
echo '   "Find me the cougar hotspots near Missoula, MT"'
echo "Or open a new Terminal window and type:"
echo '   cougarmap hotspots "Missoula, MT"'
