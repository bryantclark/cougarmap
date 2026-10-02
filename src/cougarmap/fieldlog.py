"""The field log: what the user found on the ground, kept in OBSERVATIONS_FILE (one JSON record per line), the
lion truth the model is tested against (truth.py).

Five record types (schema version 2; every record carries "type" and "v"):

- `deployment`: one camera at one spot for one stretch of time. Its `arm` says why it is there: `model` (the tool's
  pick), `human` (a spot a person picked by hand from the same factors), `control` (a nearby spot picked without the
  model, the yardstick), `on-feature` / `off-feature` (a pinch, edge or trail vs a spot beside it), or `unpaired`.
  Cameras that share a `zone` are compared with each other. A later deployment record with the same id updates it (e.g.
  its end; an end of "" reopens it: still out).
- `check`: a visit to a camera: the date, nights it was down since the last visit (dead battery, full card,
  knocked over), and whether it was taken down. Effort (camera-nights) runs from the start to the end or the
  last check, minus the downtime. A check (or detection) after a camera's end is refused: it would add
  detections with no effort.
- `event`: one independent detection on a camera: when (local time; a time with a zone or "Z" is converted to
  this computer's local time when it is logged), which species (cougar / deer / elk / other), how many.
- `track`: a lion's trail followed on the ground (snow, mud) and recorded with a phone GPS: its lines, the date,
  snow age and depth, and how sure the user is it was a lion.
- `transect`: one survey of a fixed route after fresh snow (or on a muddy/dusty road): the route's lines and
  every place a lion trail crossed it. No crossings is a valid result (an absence).

Coordinates are [lat, lon] pairs, the same order as every lat/lon argument of the tools.

Version 1 (the original log_result, before this schema) wrote one record per camera result with lion_seen and a
detection count. `migrate` turns each into an unpaired deployment plus its cougar events; the file is rewritten
in version 2 (the original kept beside it as observations.v1.bak) the first time anything new is logged. A
version-1 end is the last night the result covers, not a removal: the deployment stays open (later checks add
to it) and a check the morning after holds the effort.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

from .config import OBSERVATIONS_FILE, TRUTH

SCHEMA_VERSION: Final = 2

Arm = Literal["model", "human", "control", "on-feature", "off-feature", "unpaired"]
ARMS: Final[tuple[str, ...]] = ("model", "human", "control", "on-feature", "off-feature", "unpaired")
ARM_ALIASES: Final[dict[str, str]] = {"expert": "human"}  # the human arm's earlier name, still read
TRAIL_TYPES: Final[tuple[str, ...]] = ("paved", "open-dirt", "closed-road", "hiking-trail", "game-trail", "none")
# trail types where a camera watches a line lions walk, like the on-trail cameras of the base rate (truth.py);
# "none" and "paved" are off one. On-trail cameras catch about 3x more lions (docs/experiments/09), so cameras are
# compared like with like.
ON_FEATURE_TRAILS: Final[tuple[str, ...]] = ("open-dirt", "closed-road", "hiking-trail", "game-trail")
PLACEMENTS: Final[tuple[str, ...]] = ("on-feature", "off-feature", "unrecorded")
SPECIES: Final[tuple[str, ...]] = ("cougar", "deer", "elk", "other")
CONFIDENCE: Final[tuple[str, ...]] = ("certain", "probable", "possible")  # best first
SURFACES: Final[tuple[str, ...]] = ("snow", "mud", "dust")

type LatLon = list[float]  # [lat, lon]
type Line = list[LatLon]


class Deployment(TypedDict):
    type: Literal["deployment"]
    v: int
    id: str
    name: str | None
    lat: float
    lon: float
    arm: str
    zone: str | None
    start: str | None  # date the camera was set
    end: str | None  # date it was taken down (None = still out)
    downtime_nights: float  # nights it was not recording, known when it was set up or updated
    trail_type: str | None
    height_m: float | None
    facing_deg: float | None  # compass direction the camera faces
    lure: bool
    notes: str | None
    logged: str


class Check(TypedDict):
    type: Literal["check"]
    v: int
    deployment: str
    date: str
    downtime_nights: float  # nights it was down since the previous check
    removed: bool
    notes: str | None
    logged: str


class Event(TypedDict):
    type: Literal["event"]
    v: int
    deployment: str
    datetime: str | None  # None: the time was not recorded (version-1 counts)
    species: str
    count: int  # animals in the photo/clip
    notes: str | None
    logged: str


class Track(TypedDict):
    type: Literal["track"]
    v: int
    id: str
    date: str
    lines: list[Line]
    species: str
    snow_age_h: float | None  # hours since the snow fell (how fresh the track can be)
    snow_depth_cm: float | None
    confidence: str
    source: str | None  # the GPX/KML it came from
    notes: str | None
    logged: str


class Transect(TypedDict):
    type: Literal["transect"]
    v: int
    id: str
    route: str  # the fixed route's name: surveys of one route are compared with each other
    date: str
    lines: list[Line]
    crossings: list[LatLon]  # empty = surveyed, no lion crossed
    species: str
    surface: str
    snow_age_h: float | None
    source: str | None
    notes: str | None
    logged: str


Record = Deployment | Check | Event | Track | Transect


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return dt.date.today().isoformat()


def _choice(value: str, allowed: tuple[str, ...], what: str) -> str:
    v = value.strip().lower().replace("_", "-").replace(" ", "-")
    v = ARM_ALIASES.get(v, v) if allowed is ARMS else v
    if v not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(allowed)} (got {value!r})")
    return v


def date_of(text: str | None) -> dt.date | None:
    """The date part of an ISO date or datetime string."""
    return dt.date.fromisoformat(text[:10]) if text else None


def local_time(text: str) -> dt.datetime:
    """An ISO date/time as a plain local time: one with a time zone (an offset or "Z", as phone and camera
    exports often carry) is converted to this computer's local time and the zone dropped, so every event time in
    the log compares with every other. Raises on a malformed time."""
    t = dt.datetime.fromisoformat(text.strip())
    return t.astimezone().replace(tzinfo=None) if t.tzinfo is not None else t


def normalize_time(text: str) -> str:
    """An event time in the log's one format: local, no zone, to the minute (seconds kept when given). A bare
    date stays a date."""
    text = text.strip()
    if len(text) == 10:
        date_of(text)  # raises on a malformed date
        return text
    t = local_time(text)
    return t.isoformat(timespec="minutes" if t.second == 0 and t.microsecond == 0 else "seconds")


# ---- reading, migrating, writing -------------------------------------------------------------------------------


def migrate(rec: dict[str, Any], taken: set[str]) -> list[Record]:
    """One record of any version -> version-2 records. taken: deployment ids already used (updated in place)."""
    if rec.get("v") == SCHEMA_VERSION:
        if rec.get("type") == "deployment" and rec.get("arm") in ARM_ALIASES:
            rec = {**rec, "arm": ARM_ALIASES[rec["arm"]]}
        return [cast(Record, rec)]
    if "type" in rec:
        raise ValueError(f"unknown field-log record version {rec.get('v')!r}")
    return _from_v1(rec, taken)


def _unique_id(base: str, taken: set[str]) -> str:
    i, out = 1, base
    while out in taken:
        i += 1
        out = f"{base}#{i}"
    taken.add(out)
    return out


def _from_v1(rec: dict[str, Any], taken: set[str]) -> list[Record]:
    """A version-1 camera result -> an unpaired deployment (left open: a version-1 end is "result through", not
    a removal), its cougar detections, and a check the morning after the end (never after the day it was logged)
    that holds its effort: start to end, both nights counted. A result without dates has no known effort."""
    logged = str(rec.get("logged") or _now())
    dep_id = _unique_id(str(rec.get("name") or f"obs-{logged[:10]}"), taken)
    dep = Deployment(
        type="deployment",
        v=SCHEMA_VERSION,
        id=dep_id,
        name=rec.get("name"),
        lat=float(rec["lat"]),
        lon=float(rec["lon"]),
        arm="unpaired",
        zone=None,
        start=rec.get("start"),
        end=None,
        downtime_nights=0.0,
        trail_type=None,
        height_m=None,
        facing_deg=None,
        lure=False,
        notes=rec.get("notes"),
        logged=logged,
    )
    out: list[Record] = [dep]
    start, through = date_of(rec.get("start")), date_of(rec.get("end"))
    if start and through and through >= start:
        day = min(through + dt.timedelta(days=1), max(through, date_of(logged) or through))
        out.append(
            Check(
                type="check",
                v=SCHEMA_VERSION,
                deployment=dep_id,
                date=day.isoformat(),
                downtime_nights=0.0,
                removed=False,
                notes=f"from a version-1 camera result (effort through {through.isoformat()})",
                logged=logged,
            )
        )
    if rec.get("lion_seen"):
        times: list[str | None] = [normalize_time(str(t)) for t in rec.get("times") or []]
        n = max(int(rec.get("detections") or 0), len(times), 1)
        times += [None] * (n - len(times))
        out += [
            Event(
                type="event",
                v=SCHEMA_VERSION,
                deployment=dep_id,
                datetime=t,
                species="cougar",
                count=1,
                notes="from a version-1 camera result" if t is None else None,
                logged=logged,
            )
            for t in times
        ]
    return out


def _parse(lines: Iterable[str]) -> tuple[list[Record], bool]:
    """Records (migrated) and whether any line was in an older version."""
    taken: set[str] = set()
    raw = [json.loads(line) for line in lines if line.strip()]
    for r in raw:  # version-2 deployment ids first, so migrated ones never take them
        if r.get("v") == SCHEMA_VERSION and r.get("type") == "deployment":
            taken.add(r["id"])
    out: list[Record] = []
    old = False
    for r in raw:
        old |= r.get("v") != SCHEMA_VERSION
        out += migrate(r, taken)
    return out, old


def load(path: Path | None = None) -> list[Record]:
    """Every record in the field log, in version 2 (older lines are migrated in memory; the file is unchanged)."""
    path = path or OBSERVATIONS_FILE
    if not path.exists():
        return []
    return _parse(path.read_text().splitlines())[0]


def append(records: list[Record], path: Path | None = None) -> int:
    """Add records to the field log; returns how many records it holds. A log with older-version lines is first
    rewritten in version 2 (atomically, keeping the original as <name>.v1.bak)."""
    path = path or OBSERVATIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing, old = _parse(path.read_text().splitlines()) if path.exists() else ([], False)
    if old:
        bak = path.with_suffix(".v1.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in existing))
        tmp.replace(path)
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    return len(existing) + len(records)


# ---- sharing a field log with someone you trust ------------------------------------------------------------

SHARE_FORMAT: Final = "cougarmap-field-log"
_ID_KEYS: Final = {
    "deployment": ("id", "zone"),
    "check": ("deployment",),
    "event": ("deployment",),
    "track": ("id",),
    "transect": ("id", "route"),
}


def share_tag(name: str) -> str:
    """A sender's name as the prefix their ids get on import: lowercase letters, digits and dashes."""
    tag = "-".join("".join(c if c.isalnum() else " " for c in name.lower()).split())
    if not tag:
        raise ValueError("give a name for whose results these are (e.g. your first name)")
    return tag


def share(records: list[Record], shared_by: str, cougarmap: str) -> dict[str, Any]:
    """The user's own records (none imported from someone else) as one shareable document. It holds camera
    locations and tracks: it is for people the user trusts, sent privately."""
    own = [r for r in records if "shared_by" not in r]
    return dict(
        format=SHARE_FORMAT,
        v=SCHEMA_VERSION,
        shared_by=share_tag(shared_by),
        exported=_now(),
        cougarmap=cougarmap,
        records=own,
    )


def from_share(doc: dict[str, Any], tag: str | None = None) -> list[Record]:
    """Records from a shared document, ready to add to this log: every id, zone and route is prefixed with the
    sender's tag ("sam/M1"), so they never collide with the user's own, and each record carries shared_by."""
    if doc.get("format") != SHARE_FORMAT or doc.get("v") != SCHEMA_VERSION:
        raise ValueError("not a CougarMap shared field log (cougarmap share-results makes one)")
    tag = share_tag(tag or str(doc.get("shared_by") or ""))
    out: list[Record] = []
    for rec in doc["records"]:
        r = dict(rec)
        for key in _ID_KEYS.get(r.get("type", ""), ()):
            if r.get(key):
                r[key] = f"{tag}/{r[key]}"
        r["shared_by"] = tag
        out.append(migrate(r, set())[0])
    return out


def replace_shared(tag: str, records: list[Record], path: Path | None = None) -> tuple[int, int]:
    """Swap in a sender's records: the ones imported from them before are dropped (a newer file from the same
    person replaces the older one). Rewrites the log atomically. Returns (records dropped, records in the log)."""
    path = path or OBSERVATIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing, old = _parse(path.read_text().splitlines()) if path.exists() else ([], False)
    if old and not path.with_suffix(".v1.bak").exists():
        shutil.copy2(path, path.with_suffix(".v1.bak"))  # as append does: keep the version-1 original once
    keep = [r for r in existing if r.get("shared_by") != tag]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in [*keep, *records]))
    tmp.replace(path)
    return len(existing) - len(keep), len(keep) + len(records)


# ---- making records ---------------------------------------------------------------------------------------------


def deployments(records: list[Record]) -> dict[str, Deployment]:
    """Deployments by id, with later records' set fields overriding earlier ones."""
    out: dict[str, Deployment] = {}
    for r in records:
        if r["type"] == "deployment":
            d = r
            if d["id"] in out:
                merged = dict(out[d["id"]])
                merged.update({k: v for k, v in d.items() if v is not None})
                if merged.get("end") == "":  # reopened: still out
                    merged["end"] = None
                out[d["id"]] = cast(Deployment, merged)
            else:
                out[d["id"]] = d
    return out


def ended(dep: Deployment, checks: list[Check]) -> str | None:
    """The date a camera came down: its end, else its first removal check; None while it is still out."""
    return dep["end"] or next((c["date"] for c in sorted(checks, key=lambda c: c["date"]) if c["removed"]), None)


def placement_of(dep: Deployment) -> str:
    """on-feature (the camera watches a trail, two-track or dirt road), off-feature, or unrecorded. The trail_type
    decides; without one, an on-feature / off-feature arm says it."""
    t = dep["trail_type"]
    if t:
        return "on-feature" if t in ON_FEATURE_TRAILS else "off-feature"
    return dep["arm"] if dep["arm"] in ("on-feature", "off-feature") else "unrecorded"


def new_deployment(
    records: list[Record],
    lat: float | None,
    lon: float | None,
    name: str | None = None,
    arm: str | None = None,
    zone: str | None = None,
    start: str | None = None,
    end: str | None = None,
    downtime_nights: float | None = None,
    trail_type: str | None = None,
    height_m: float | None = None,
    facing_deg: float | None = None,
    lure: bool | None = None,
    notes: str | None = None,
    deployment: str | None = None,
) -> Deployment:
    """A new deployment, or (deployment = an existing id) an update holding only the fields given. end="" on an
    update reopens a deployment (still out)."""
    known = deployments(records)
    arm_v = _choice(arm, ARMS, "arm") if arm else None
    trail_v = _choice(trail_type, TRAIL_TYPES, "trail_type") if trail_type else None
    for d in (start, end):
        date_of(d)  # raises on a malformed date
    if deployment is not None:
        if deployment not in known:
            raise ValueError(f"no camera deployment {deployment!r}; known: {sorted(known)}")
        upd: dict[str, Any] = dict(
            type="deployment",
            v=SCHEMA_VERSION,
            id=deployment,
            logged=_now(),
            name=name,
            lat=lat,
            lon=lon,
            arm=arm_v,
            zone=zone,
            start=start,
            end=end,
            downtime_nights=downtime_nights,
            trail_type=trail_v,
            height_m=height_m,
            facing_deg=facing_deg,
            lure=lure,
            notes=notes,
        )
        return cast(Deployment, upd)
    if lat is None or lon is None:
        raise ValueError("a new camera deployment needs lat and lon (or deployment=<id> to update one)")
    return Deployment(
        type="deployment",
        v=SCHEMA_VERSION,
        id=_unique_id(name or f"cam-{(start or _today())[:10]}", set(known)),
        name=name,
        lat=lat,
        lon=lon,
        arm=arm_v or "unpaired",
        zone=zone,
        start=start or _today(),
        end=end or None,
        downtime_nights=downtime_nights or 0.0,
        trail_type=trail_v,
        height_m=height_m,
        facing_deg=facing_deg,
        lure=bool(lure),
        notes=notes,
        logged=_now(),
    )


def new_check(
    records: list[Record],
    deployment: str,
    date: str | None = None,
    downtime_nights: float = 0.0,
    removed: bool = False,
    events: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> list[Record]:
    """A camera check and the detections found on it. events: [{datetime, species, count}]; times are stored in
    local time without a zone (normalize_time). A check after the camera's end is refused, since it would add
    detections with no effort."""
    known = deployments(records)
    if deployment not in known:
        raise ValueError(f"no camera deployment {deployment!r}; known: {sorted(known)}")
    day = (date or _today())[:10]
    date_of(day)
    end = ended(known[deployment], [r for r in records if r["type"] == "check" and r["deployment"] == deployment])
    if end and day > end[:10]:
        raise ValueError(
            f"camera {deployment!r} came down on {end[:10]}, before this check ({day}). If this is the visit when "
            f"it came down, log the check on {end[:10]}. If it is still out, reopen it first: "
            f"log_camera(deployment={deployment!r}, end='') (command line: log-camera --deployment "
            f"{deployment} --end ''), or give its real end date the same way."
        )
    out: list[Record] = [
        Check(
            type="check",
            v=SCHEMA_VERSION,
            deployment=deployment,
            date=day,
            downtime_nights=float(downtime_nights),
            removed=removed,
            notes=notes,
            logged=_now(),
        )
    ]
    for e in events or []:
        raw = e.get("datetime") or e.get("time")
        when = normalize_time(str(raw)) if raw else None  # raises on a malformed time
        if when:
            seen = local_time(when).date()
            start = date_of(known[deployment]["start"])
            if (start and seen < start) or seen > dt.date.fromisoformat(day):
                raise ValueError(
                    f"event at {when} is outside camera {deployment!r}'s time out "
                    f"({known[deployment]['start']} to this check, {day}): check the date and year"
                )
        out.append(
            Event(
                type="event",
                v=SCHEMA_VERSION,
                deployment=deployment,
                datetime=when,
                species=_choice(str(e.get("species", "cougar")), SPECIES, "species"),
                count=int(e.get("count", 1)),
                notes=e.get("notes"),
                logged=_now(),
            )
        )
    return out


def _lines(points: list[LatLon] | None, file: str | None) -> tuple[list[Line], Geo, str | None]:
    """(lines, the file's contents, first time) from given points and/or a GPX/KML/KMZ file."""
    g = read_geo(file) if file else Geo()
    lines: list[Line] = list(g.lines)
    if points:
        lines.append([[float(a), float(b)] for a, b in points])
    return lines, g, g.first_time


def is_crossing(name: str | None) -> bool:
    """Whether a waypoint's name marks a lion crossing (holds one of TRUTH.crossing_words, any case)."""
    low = (name or "").lower()
    return any(w in low for w in TRUTH.crossing_words)


def new_track(
    records: list[Record],
    file: str | None = None,
    points: list[LatLon] | None = None,
    date: str | None = None,
    snow_age_h: float | None = None,
    snow_depth_cm: float | None = None,
    confidence: str = "probable",
    species: str = "cougar",
    name: str | None = None,
    notes: str | None = None,
) -> Track:
    """A followed track from a GPX/KML/KMZ file (its lines, or its waypoints in order) and/or [[lat, lon], ...]."""
    lines, g, first = _lines(points, file)
    if not lines and len(g.points) >= 2:
        lines = [g.points]
    if not lines:
        raise ValueError("a track needs a line: a GPX/KML/KMZ file with a track, or points [[lat, lon], ...]")
    day = (date or first or _today())[:10]
    date_of(day)
    taken = {r["id"] for r in records if r["type"] == "track"}
    return Track(
        type="track",
        v=SCHEMA_VERSION,
        id=_unique_id(name or f"track-{day}", taken),
        date=day,
        lines=lines,
        species=_choice(species, SPECIES, "species"),
        snow_age_h=snow_age_h,
        snow_depth_cm=snow_depth_cm,
        confidence=_choice(confidence, CONFIDENCE, "confidence"),
        source=Path(file).name if file else None,
        notes=notes,
        logged=_now(),
    )


def new_transect(
    records: list[Record],
    route: str,
    file: str | None = None,
    line: list[LatLon] | None = None,
    crossings: list[LatLon] | None = None,
    date: str | None = None,
    surface: str = "snow",
    snow_age_h: float | None = None,
    species: str = "cougar",
    notes: str | None = None,
) -> tuple[Transect, list[str]]:
    """One survey of a fixed route, and the names of the file's waypoints left out. The route is the file's lines
    (or line, [[lat, lon], ...]); crossings are the file's waypoints named as one (is_crossing: "lion", ...)
    plus any given. Other waypoints (parking, start, ...) are left out, never counted. No crossings = surveyed,
    nothing crossed. A route surveyed before can omit its line: the last survey's line is reused."""
    lines, g, first = _lines(line, file)
    keep = [p for p, n in zip(g.points, g.point_names, strict=True) if is_crossing(n)]
    ignored = [n or "(no name)" for n in g.point_names if not is_crossing(n)]
    if not lines:
        prev = [r for r in records if r["type"] == "transect" and r["route"] == route]
        if not prev:
            raise ValueError(f"route {route!r} has no line yet: give a GPX/KML/KMZ file or line [[lat, lon], ...]")
        lines = prev[-1]["lines"]
    day = (date or first or _today())[:10]
    date_of(day)
    taken = {r["id"] for r in records if r["type"] == "transect"}
    t = Transect(
        type="transect",
        v=SCHEMA_VERSION,
        id=_unique_id(f"{route} {day}", taken),
        route=route,
        date=day,
        lines=lines,
        crossings=[*keep, *([[float(a), float(b)] for a, b in crossings or []])],
        species=_choice(species, SPECIES, "species"),
        surface=_choice(surface, SURFACES, "surface"),
        snow_age_h=snow_age_h,
        source=Path(file).name if file else None,
        notes=notes,
        logged=_now(),
    )
    return t, ignored


# ---- camera effort ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """A deployment with its effort and detections folded in from its checks and events."""

    dep: Deployment
    nights: float  # camera-nights: start to end (or the last check), minus downtime
    nights_by_month: dict[int, float]  # the same split by calendar month
    detections: dict[str, int]  # independent detections per species
    last_check: str | None
    times: list[str] = field(default_factory=list)  # cougar detection times


def independent(times: list[str | None], min_gap_min: float) -> int:
    """Detections of one species at one camera, counting photos less than min_gap_min apart as one. Untimed
    events each count once."""
    known = sorted(local_time(t) for t in times if t)  # local_time: logs written before times were normalized
    n = len(times) - len(known)
    last: dt.datetime | None = None
    for t in known:
        if last is None or (t - last).total_seconds() / 60 >= min_gap_min:
            n += 1
        last = t
    return n


def cameras(records: list[Record], min_gap_min: float) -> list[Camera]:
    """Every deployment's effort and detections."""
    deps = deployments(records)
    checks: dict[str, list[Check]] = {k: [] for k in deps}
    events: dict[str, list[Event]] = {k: [] for k in deps}
    for r in records:
        if r["type"] == "check" and r["deployment"] in deps:
            checks[r["deployment"]].append(r)
        elif r["type"] == "event" and r["deployment"] in deps:
            events[r["deployment"]].append(r)
    out = []
    for k, d in deps.items():
        cs = sorted(checks[k], key=lambda c: c["date"])
        end = ended(d, cs) or (cs[-1]["date"] if cs else None)
        start = date_of(d["start"])
        stop = date_of(end)
        down = d["downtime_nights"] + sum(c["downtime_nights"] for c in cs)
        nights = [start + dt.timedelta(days=i) for i in range((stop - start).days)] if start and stop else []
        keep = max(0.0, len(nights) - down) / len(nights) if nights else 0.0
        by_month: dict[int, float] = {}
        for night in nights:
            by_month[night.month] = by_month.get(night.month, 0.0) + keep
        ev = events[k]
        det = {s: independent([e["datetime"] for e in ev if e["species"] == s], min_gap_min) for s in SPECIES}
        times = sorted((e["datetime"] for e in ev if e["species"] == "cougar" and e["datetime"]), key=local_time)
        out.append(Camera(d, len(nights) * keep, by_month, det, cs[-1]["date"] if cs else None, times))
    return out


# ---- GPX / KML / KMZ --------------------------------------------------------------------------------------------


@dataclass
class Geo:
    """The lines and waypoints of a GPS file, as [lat, lon], with each waypoint's name (None: unnamed)."""

    lines: list[Line] = field(default_factory=list)
    points: list[LatLon] = field(default_factory=list)
    point_names: list[str | None] = field(default_factory=list)
    first_time: str | None = None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el: ET.Element, name: str) -> str | None:
    for ch in el:
        if _local(ch.tag) == name and ch.text:
            return ch.text.strip()
    return None


def _gpx(root: ET.Element) -> Geo:
    g = Geo()
    times: list[str] = []

    def pt(el: ET.Element) -> LatLon:
        t = _child_text(el, "time")
        if t:
            times.append(t)
        return [float(el.attrib["lat"]), float(el.attrib["lon"])]

    for el in root.iter():
        tag = _local(el.tag)
        if tag in ("trkseg", "rte"):
            line = [pt(p) for p in el if _local(p.tag) in ("trkpt", "rtept")]
            if len(line) >= 2:
                g.lines.append(line)
        elif tag == "wpt":
            g.points.append(pt(el))
            g.point_names.append(_child_text(el, "name"))
    g.first_time = min(times) if times else None
    return g


def _kml_coords(text: str) -> Line:
    out = []
    for tok in text.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            out.append([float(parts[1]), float(parts[0])])
    return out


def _placemark_name(el: ET.Element, parent: dict[ET.Element, ET.Element]) -> str | None:
    """The name of the KML Placemark holding el (None: unnamed or not in one)."""
    up = parent.get(el)
    while up is not None and _local(up.tag) != "Placemark":
        up = parent.get(up)
    return _child_text(up, "name") if up is not None else None


def _kml(root: ET.Element) -> Geo:
    g = Geo()
    times: list[str] = []
    parent = {ch: el for el in root.iter() for ch in el}
    for el in root.iter():
        tag = _local(el.tag)
        if tag in ("LineString", "LinearRing", "Point"):
            c = _child_text(el, "coordinates")
            pts = _kml_coords(c) if c else []
            if tag == "Point" and pts:
                g.points.append(pts[0])
                g.point_names.append(_placemark_name(el, parent))
            elif len(pts) >= 2:
                g.lines.append(pts)
        elif tag == "Track":  # gx:Track: <when> times and "lon lat alt" <gx:coord>s
            line = []
            for ch in el:
                if _local(ch.tag) == "coord" and ch.text:
                    lon, lat = ch.text.split()[:2]
                    line.append([float(lat), float(lon)])
                elif _local(ch.tag) == "when" and ch.text:
                    times.append(ch.text.strip())
            if len(line) >= 2:
                g.lines.append(line)
        elif tag == "TimeStamp":
            t = _child_text(el, "when")
            if t:
                times.append(t)
    g.first_time = min(times) if times else None
    return g


def read_geo(path: str | Path) -> Geo:
    """Lines (tracks, routes, paths) and waypoints from a GPX, KML or KMZ file (phone GPS apps and Google Earth
    export these)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".kmz":
        with zipfile.ZipFile(p) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if name is None:
                raise ValueError(f"no .kml inside {p.name}")
            root = ET.fromstring(z.read(name))
    else:
        root = ET.parse(p).getroot()
    tag = _local(root.tag).lower()
    if tag == "gpx":
        return _gpx(root)
    if tag == "kml":
        return _kml(root)
    raise ValueError(f"{p.name} is not a GPX or KML file")


def line_length_m(line: Line) -> float:
    """Length of a [lat, lon] line, metres (equirectangular: fine at track scale)."""
    total = 0.0
    k = 111_320.0  # metres per degree of latitude
    for (a1, o1), (a2, o2) in itertools.pairwise(line):
        dx = (o2 - o1) * k * math.cos(math.radians((a1 + a2) / 2))
        dy = (a2 - a1) * k
        total += (dx * dx + dy * dy) ** 0.5
    return total
