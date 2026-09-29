"""Building footprints from Microsoft's Global ML Building Footprints (public, no account), mapped from aerial
imagery so rural houses, barns and sheds are covered far better than in OpenStreetMap. Tiled by level-9 Bing
quadkeys; each tile is a gzipped GeoJSON-lines file of a few MB."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math

from ..net import cached, get
from .canopy import _tile_xy, quadkey
from .dem import LonLatBounds

INDEX = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
LEVEL = 9


def _index() -> dict[str, list[str]]:
    def build() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for row in csv.DictReader(io.StringIO(get(INDEX, timeout=120).text)):
            out.setdefault(row["QuadKey"], []).append(row["Url"])
        return out

    out: dict[str, list[str]] = cached("buildings-index", {"v": 1}, build)
    return out


Building = tuple[float, float, float]  # lon, lat (centroid), footprint area in m2


def _area_m2(ring: list[list[float]], lat: float) -> float:
    """Shoelace area of a lon/lat ring, on a local flat projection (fine for building-sized polygons)."""
    kx, ky = 111_320.0 * math.cos(math.radians(lat)), 110_540.0
    xs = [p[0] * kx for p in ring]
    ys = [p[1] * ky for p in ring]
    return 0.5 * abs(sum(xs[i - 1] * ys[i] - xs[i] * ys[i - 1] for i in range(len(ring))))


def fetch_buildings(lonlat_bounds: LonLatBounds) -> list[Building]:
    """Centroid and footprint area of every building footprint inside the bounds."""
    minx, miny, maxx, maxy = lonlat_bounds

    def build() -> list[Building]:
        x0, y0 = _tile_xy(minx, maxy, LEVEL)
        x1, y1 = _tile_xy(maxx, miny, LEVEL)
        idx = _index()
        out: list[Building] = []
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for url in idx.get(quadkey(x, y, LEVEL), []):
                    for line in gzip.decompress(get(url, timeout=180).content).splitlines():
                        ring = json.loads(line)["geometry"]["coordinates"][0]
                        lon = sum(p[0] for p in ring) / len(ring)
                        lat = sum(p[1] for p in ring) / len(ring)
                        if minx <= lon <= maxx and miny <= lat <= maxy:
                            out.append((lon, lat, _area_m2(ring, lat)))
        return out

    pts: list[Building] = cached("buildings", [*(round(v, 5) for v in lonlat_bounds), "v2"], build)
    return pts
