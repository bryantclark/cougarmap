#!/bin/sh
# Build dist/CougarMap-installer.zip: the wheel + double-click installers + read-me.
set -e
cd "$(dirname "$0")/.."
rm -rf dist && uv build --wheel -q
STAGE=dist/CougarMap-installer && mkdir -p "$STAGE"
cp dist/cougarmap-*.whl installer/install.sh "installer/Install CougarMap.command" installer/install.ps1 installer/READ-ME-FIRST.txt "$STAGE/"
chmod +x "$STAGE/install.sh" "$STAGE/Install CougarMap.command"
(cd dist && zip -qr -X CougarMap-installer.zip CougarMap-installer)
ls -la dist/CougarMap-installer.zip
