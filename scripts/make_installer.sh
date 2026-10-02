#!/bin/sh
# Build dist/CougarMap-installer.zip: the double-click installers and the read-me. They install CougarMap from PyPI,
# so the zip never goes stale; the release workflow attaches it to each GitHub release.
set -e
cd "$(dirname "$0")/.."
STAGE=dist/CougarMap-installer
rm -rf "$STAGE" dist/CougarMap-installer.zip && mkdir -p "$STAGE"
cp installer/install.sh "installer/Install CougarMap.command" installer/install.ps1 installer/READ-ME-FIRST.txt "$STAGE/"
chmod +x "$STAGE/install.sh" "$STAGE/Install CougarMap.command"
(cd dist && zip -qr -X CougarMap-installer.zip CougarMap-installer)
ls -la dist/CougarMap-installer.zip
