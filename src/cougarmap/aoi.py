"""Areas of interest: from a KML/KMZ file, a point + radius, a bounding box, or a place name."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from .grid import utm_epsg
from .net import cached, get_json

KML_NS = "{http://www.opengis.net/kml/2.2}"


class UserPoint(TypedDict):
    """A pin from the user's Google Earth file (lon/lat)."""

    name: str
    lat: float
    lon: float
    kind: str | None  # water | seasonal_water | sign | camera (see point_kind), None = not used
    note: str


@dataclass
class Placemark:
    name: str
    geom: BaseGeometry  # lon/lat
    folder: str = ""
    description: str = ""


@dataclass
class AOI:
    name: str
    geom: BaseGeometry  # lon/lat polygon
    user_points: list[UserPoint] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.geom.centroid
        return c.y, c.x

    def area_km2(self) -> float:
        c = self.geom.centroid
        t = Transformer.from_crs(4326, utm_epsg(c.x, c.y), always_xy=True).transform
        return float(shp_transform(t, self.geom).area) / 1e6


def _coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def read_kml(path: str | Path) -> list[Placemark]:
    path = Path(path)
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
            root = ET.fromstring(z.read(name))
    else:
        root = ET.parse(path).getroot()
    out: list[Placemark] = []

    def walk(el: ET.Element, folder: str) -> None:
        for ch in el:
            tag = ch.tag.replace(KML_NS, "")
            if tag in ("Folder", "Document"):
                n = ch.find(f"{KML_NS}name")
                walk(ch, n.text if n is not None and n.text else folder)
            elif tag == "Placemark":
                n = ch.find(f"{KML_NS}name")
                d = ch.find(f"{KML_NS}description")
                name = (n.text or "").strip() if n is not None else ""
                desc = re.sub(r"<[^>]+>", " ", d.text or "").strip() if d is not None else ""
                for g in ch.iter():
                    gt = g.tag.replace(KML_NS, "")
                    if gt in ("Point", "LineString", "Polygon"):
                        c = g.find(f".//{KML_NS}coordinates")
                        if c is None or not c.text:
                            continue
                        pts = _coords(c.text)
                        geom: BaseGeometry
                        if gt == "Point":
                            geom = Point(pts[0])
                        elif gt == "Polygon" and len(pts) >= 3:
                            geom = Polygon(pts)
                        elif gt == "LineString" and len(pts) >= 2:
                            # Google Earth "paths" drawn around an area are nearly closed rings
                            closed = (
                                len(pts) > 20 and Point(pts[0]).distance(Point(pts[-1])) < 0.01 * LineString(pts).length
                            )
                            geom = Polygon(pts) if closed else LineString(pts)
                        else:
                            continue
                        out.append(Placemark(name, geom, folder, desc))
                        break

    walk(root, "")
    return out


def point_kind(name: str) -> str | None:
    n = name.lower()
    if "seasonal" in n and "water" in n:
        return "seasonal_water"
    if "water" in n or "spring" in n or "guzzler" in n or "wallow" in n or "seep" in n:
        return "water"
    if "scrape" in n or "kill" in n or "track" in n or "sign" in n:
        return "sign"
    if re.match(r"cam\s*\d+", n):
        return "camera"
    return None


def user_points_from_kml(path: str | Path) -> list[UserPoint]:
    return [
        UserPoint(name=p.name, lat=p.geom.y, lon=p.geom.x, kind=point_kind(p.name), note=p.description)
        for p in read_kml(path)
        if isinstance(p.geom, Point)
    ]


def areas_in_kml(path: str | Path) -> list[Placemark]:
    seen: set[tuple[str, float]] = set()
    out: list[Placemark] = []
    for p in read_kml(path):
        if isinstance(p.geom, Polygon) and p.name and p.name != "Untitled Path":
            key = (p.name, round(p.geom.area, 8))
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def from_kml(path: str | Path, name: str | None = None) -> AOI:
    areas = areas_in_kml(path)
    if not areas:
        raise ValueError(f"no named polygons in {path}")
    if name:
        sel = [a for a in areas if a.name.lower() == name.lower()] or [
            a for a in areas if name.lower() in a.name.lower()
        ]
        if not sel:
            raise ValueError(f"no area named {name!r}; available: {[a.name for a in areas]}")
        areas = sel[:1]
    geom = unary_union([a.geom.buffer(0) for a in areas])
    pts = [p for p in user_points_from_kml(path) if geom.buffer(0.02).contains(Point(p["lon"], p["lat"]))]
    return AOI(areas[0].name if len(areas) == 1 else Path(path).stem, geom, pts)


def circle(lat: float, lon: float, radius_km: float, name: str | None = None) -> AOI:
    epsg = utm_epsg(lon, lat)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    x, y = fwd(lon, lat)
    geom = shp_transform(inv, Point(x, y).buffer(radius_km * 1000, 64))
    return AOI(name or f"{lat:.4f},{lon:.4f} r{radius_km:g}km", geom)


def bbox(west: float, south: float, east: float, north: float, name: str | None = None) -> AOI:
    return AOI(name or f"bbox {west:.3f},{south:.3f},{east:.3f},{north:.3f}", box(west, south, east, north))


def geocode(place: str) -> dict[str, Any]:
    """Place name -> {lat, lon, display_name} using OpenStreetMap Nominatim."""

    def fetch() -> dict[str, Any]:
        j = get_json(
            "https://nominatim.openstreetmap.org/search",
            params=dict(q=place, format="json", limit=1, countrycodes="us"),
            timeout=30,
        )
        if not j:
            raise ValueError(f"could not find {place!r}")
        return dict(lat=float(j[0]["lat"]), lon=float(j[0]["lon"]), display_name=j[0]["display_name"])

    out: dict[str, Any] = cached("geocode", place.strip().lower(), fetch)
    return out


_COORD = r"(-?\d+(?:\.\d+)?)\s*°?\s*([NSEWnsew])?"
_COORDS = re.compile(rf"^\s*{_COORD}\s*[, ]\s*{_COORD}\s*$")


def parse_coords(text: str) -> tuple[float, float, str | None] | None:
    """'47.92,-117.58', '47.92 -117.58' or '47.92N 117.58W' -> (lat, lon, note); None if it isn't coordinates.
    The data only covers the US, so a longitude that came without its minus sign (e.g. pasted '47.37, 116.10':
    western China) is made west, and the note says so."""
    m = _COORDS.match(text)
    if not m:
        return None
    lat, ns, lon, ew = float(m[1]), (m[2] or "").upper(), float(m[3]), (m[4] or "").upper()
    lat = -abs(lat) if ns == "S" else lat
    lon = -abs(lon) if ew == "W" else lon
    note = None
    if not ew and 18 <= lat <= 72 and 60 <= lon <= 170:
        note = f"longitude {lon} is in Asia; CougarMap covers the US, so using {-lon} (west longitudes are negative)"
        lon = -lon
    return lat, lon, note


def parse_location(text: str) -> tuple[float, float, str]:
    """'47.92,-117.58' (see parse_coords) or a place name -> (lat, lon, label)."""
    c = parse_coords(text)
    if c:
        return c[0], c[1], f"{c[0]},{c[1]}"
    g = geocode(text)
    return g["lat"], g["lon"], g["display_name"]
