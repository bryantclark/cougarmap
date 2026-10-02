"""Open GPS collar data as a falsification check: where real pumas went at night, against where they could have
gone. Never a tuning target, and never a reason to overrule the method's rules (evaluate.py stays the primary
check): it can only veto a change that clearly hurts here.

The data are other people's and other ecosystems (central and northern Utah, the Olympic Peninsula), with fixes
2-8 h apart that lean toward beds and kills, so they say nothing about a 20 m camera spot. What they can test is
whether the score, at the camera scale (20 m) and the habitat scale (~500 m), is higher where lions actually were
than where they could have been on the same step.

The design is iSSF-style (integrated step selection): for every night or twilight fix (the sun below
`Gps.max_sun_deg`) that ends a step at the animal's usual fix interval, `Gps.n_available` available points are
drawn from that animal's own night step lengths and turning angles around the previous fix. Each used fix is
ranked among its own available points (0.5 = chance), so an animal is only ever compared with where it could
have gone next, never with ground far out of its range. Reported per animal and per dataset:

- **rank** at the 20 m score and at the habitat scale (the score blurred at `Gps.context_m`);
- **Boyce index**: the Spearman correlation of the used/available ratio with score bin (equal-count bins of the
  available scores; 1 = more use the higher the score, 0 = no relation);
- **map-shift null**: the same strata scored on the score map shifted 1.5 km (toroidally, four directions
  averaged), a check on the design itself: it should land at about 0.5;
- the moving steps alone (used step at least `Gps.moving_min_m`), which are less biased to beds and kills.

`gps_check(score_fn)` runs all of it for any score function of a loaded state (the same `score_fn` interface as
evaluate.evaluate), on the tiles scripts/gps_check.py analyzed; `ablations()` and `dem_shift()` build the model
ablations and the DEM-shift placebo (the DEM-only terms moved 1.5 km before recombining). Settings are
`config.Gps`.

The data stay out of git: `fetch` downloads them into the cache folder (sources and licenses in DATASETS and
docs/VALIDATION.md).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import gc
import json
import math
import os
import zlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import stats

from . import aoi as aoi_mod
from . import net
from . import terrain as T
from .analyze import combine, site_penalty
from .arrays import Floats, Mask
from .config import CACHE_DIR, GPS, OUT_DIR, Gps, Habitat, Options, Weights
from .context import Log
from .evaluate import production_score
from .state import Layers, ModelState, load_state
from .truth import sign_test, values_at

JSON = dict[str, Any]
type ScoreFn = Callable[[ModelState], Floats]
type Strs = npt.NDArray[np.str_]

GPS_DIR = Path(os.environ.get("COUGARMAP_GPS_DIR", OUT_DIR / "gps"))  # tiles.json and <dataset>/<tile>/state.pkl
DATA_DIR = CACHE_DIR / "gps"  # the downloaded collar data (not ours: kept out of git)
PUMA = "Puma concolor"
EARTH_R = 6_371_008.8


@dataclass(frozen=True)
class Dataset:
    """One open GPS collar dataset."""

    key: str
    title: str
    region: str
    url: str
    fmt: Literal["csv", "json"]  # a Movebank repository CSV, or the Movebank public JSON API
    doi: str
    license: str
    n_tiles: int


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in (
        Dataset(
            "fishlake",
            "USU coyote and puma, Fishlake National Forest (Movebank study 1720694224)",
            "central Utah: pinyon-juniper, sagebrush and aspen-conifer, 1,600-3,400 m",
            "https://www.movebank.org/movebank/service/public/json?study_id=1720694224&sensor_type=gps",
            "json",
            "10.5441/001/1.7d8301h2",
            "CC0 1.0",
            4,
        ),
        Dataset(
            "olympic",
            'Elbroch & Sager-Fradkin 2026, "Olympic Cougar Project", Movebank Data Repository',
            "Olympic Peninsula, Washington: wet coastal conifer forest, clearcuts and farmland",
            "https://datarepository.movebank.org/server/api/core/bitstreams/73378d1e-eea8-4616-a989-2648b33e9768/content",
            "csv",
            "10.5441/001/1.716",
            "CC BY-NC 4.0",
            6,
        ),
        Dataset(
            "udwr",
            '"GPS tracking of cougars in Utah by UDWR (2019-2020)", Movebank Data Repository',
            "northern and central Utah: Wasatch and Book Cliffs, mountain shrub, aspen and conifer",
            "https://datarepository.movebank.org/server/api/core/bitstreams/704656c2-04c9-4199-a6fe-1bdc318cf84d/content",
            "csv",
            "10.5441/001/1.712",
            "CC0 1.0",
            4,
        ),
    )
}


# ---- the collar data --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Track:
    """One animal's fixes, sorted by time (unix seconds, UTC)."""

    animal: str
    t: Floats
    lon: Floats
    lat: Floats

    @classmethod
    def of(cls, animal: str, rows: list[tuple[float, float, float]]) -> Track:
        a = np.array(sorted(set(rows)), float).reshape(-1, 3)
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0]) if len(a) else np.zeros(0, bool)  # one fix per time
        a = a[keep]
        return cls(animal, a[:, 0], a[:, 1], a[:, 2])


def data_path(key: str) -> Path:
    return DATA_DIR / f"{key}.{DATASETS[key].fmt}"


def fetch(key: str, log: Log = print) -> Path:
    """Download a dataset into the cache folder once (public, no login); the path to it."""
    p = data_path(key)
    if not p.exists():
        d = DATASETS[key]
        log(f"downloading {d.title} ({d.license}, doi:{d.doi})")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(net.get(d.url, timeout=600).content)
        tmp.replace(p)
    return p


def _utc(s: str) -> float:
    return dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC).timestamp()


def read_movebank_csv(path: Path) -> dict[str, Track]:
    """A Movebank repository CSV: the visible puma fixes with coordinates, by animal."""
    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("individual-taxon-canonical-name", PUMA) != PUMA or r.get("visible", "true") == "false":
                continue
            if r.get("manually-marked-outlier") == "true" or not r["location-long"] or not r["location-lat"]:
                continue
            rows[r["individual-local-identifier"]].append(
                (_utc(r["timestamp"]), float(r["location-long"]), float(r["location-lat"]))
            )
    return {a: Track.of(a, v) for a, v in rows.items()}


def read_movebank_json(path: Path) -> dict[str, Track]:
    """The Movebank public JSON API's study export: the puma fixes, by animal."""
    out: dict[str, Track] = {}
    for ind in json.loads(path.read_text())["individuals"]:
        if ind.get("individual_taxon_canonical_name") != PUMA:
            continue
        rows = [
            (loc["timestamp"] / 1000, loc["location_long"], loc["location_lat"])
            for loc in ind["locations"]
            if loc.get("location_long") is not None and loc.get("location_lat") is not None
        ]
        if rows:
            out[ind["individual_local_identifier"]] = Track.of(ind["individual_local_identifier"], rows)
    return out


def load(key: str) -> dict[str, Track]:
    p = data_path(key)
    return read_movebank_json(p) if DATASETS[key].fmt == "json" else read_movebank_csv(p)


def sun_elevation(t: Floats, lon: Floats, lat: Floats) -> Floats:
    """Solar elevation (degrees) at unix time t (UTC) and lon/lat: the low-precision almanac formulas (about
    0.1 degree, plenty for telling night from day)."""
    d = np.asarray(t, float) / 86400 - 10957.5  # days since J2000.0
    g = np.radians(357.529 + 0.98560028 * d)
    q = 280.459 + 0.98564736 * d
    ecl_lon = np.radians(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.00000036 * d)
    ra = np.arctan2(np.cos(eps) * np.sin(ecl_lon), np.cos(ecl_lon))
    dec = np.arcsin(np.sin(eps) * np.sin(ecl_lon))
    gmst = np.radians((280.46061837 + 360.98564736629 * d) % 360)
    ha = gmst + np.radians(lon) - ra
    la = np.radians(lat)
    out: Floats = np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))
    return out


# ---- the iSSF design --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Strata:
    """Used fixes and their available points (lon/lat), one row per stratum."""

    dataset: str
    animal: Strs  # (n,)
    t: Floats  # (n,) time of the used fix
    used: Floats  # (n, 2) lon, lat
    avail: Floats  # (n, K, 2) lon, lat
    step_m: Floats  # (n,) length of the used step

    def __len__(self) -> int:
        return len(self.t)

    def take(self, keep: Mask) -> Strata:
        return Strata(
            self.dataset, self.animal[keep], self.t[keep], self.used[keep], self.avail[keep], self.step_m[keep]
        )

    @property
    def months(self) -> npt.NDArray[np.int_]:
        return np.array([dt.datetime.fromtimestamp(x, dt.UTC).month for x in self.t], int)


def usual_interval(t: Floats) -> float:
    """The animal's usual fix interval (s): the median gap between fixes."""
    return float(np.median(np.diff(t))) if len(t) > 1 else math.inf


def _local(lon: Floats, lat: Floats, lat0: float) -> tuple[Floats, Floats]:
    """lon/lat -> metres east/north on a local equirectangular plane (an animal's range is tens of km)."""
    return np.radians(lon) * EARTH_R * math.cos(math.radians(lat0)), np.radians(lat) * EARTH_R


def _lonlat(x: Floats, y: Floats, lat0: float) -> tuple[Floats, Floats]:
    return np.degrees(x / (EARTH_R * math.cos(math.radians(lat0)))), np.degrees(y / EARTH_R)


def animal_strata(tr: Track, dataset: str, cfg: Gps = GPS) -> Strata | None:
    """The animal's night strata: each night step at its usual interval, with cfg.n_available available points
    from its own night step lengths and turning angles (from a random heading when the step before it isn't at the
    usual interval). Seeded by the animal, so a stratum never depends on which other animals were loaded."""
    usual = usual_interval(tr.t)
    if len(tr.t) < 3 or usual > cfg.max_interval_h * 3600:
        return None
    lat0 = float(np.mean(tr.lat))
    x, y = _local(tr.lon, tr.lat, lat0)
    dtime = np.diff(tr.t)
    reg = np.abs(dtime - usual) <= cfg.interval_tol * usual  # step i: fix i -> fix i + 1
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    heading = np.arctan2(dx, dy)  # from north, clockwise
    moved = length > 1.0
    has_prev = np.concatenate([[False], reg[:-1] & moved[:-1]])  # a heading to turn from
    turn = np.angle(np.exp(1j * (heading - np.roll(heading, 1))))
    night = sun_elevation(tr.t[1:], tr.lon[1:], tr.lat[1:]) < cfg.max_sun_deg  # at the step's end (the used fix)
    use = reg & night
    if not use.any():
        return None
    pool = use if use.sum() >= cfg.min_dist_steps else reg
    lengths = length[pool]
    turns_ok = pool & has_prev & moved
    turns = turn[turns_ok] if turns_ok.sum() >= cfg.min_dist_steps else turn[reg & has_prev & moved]
    rng = np.random.default_rng([cfg.seed, zlib.crc32(f"{dataset}/{tr.animal}".encode())])
    idx = np.nonzero(use)[0]
    n, k = len(idx), cfg.n_available
    ln = rng.choice(lengths, (n, k))
    tn = rng.choice(turns, (n, k)) if len(turns) else rng.uniform(-math.pi, math.pi, (n, k))
    prev_heading = np.where(has_prev[idx], np.roll(heading, 1)[idx], np.nan)
    free = np.isnan(prev_heading)
    h = np.where(free[:, None], rng.uniform(-math.pi, math.pi, (n, k)), prev_heading[:, None] + tn)
    ax, ay = x[idx][:, None] + ln * np.sin(h), y[idx][:, None] + ln * np.cos(h)
    alon, alat = _lonlat(ax, ay, lat0)
    return Strata(
        dataset,
        np.full(n, tr.animal),
        tr.t[idx + 1],
        np.stack([tr.lon[idx + 1], tr.lat[idx + 1]], axis=1),
        np.stack([alon, alat], axis=2),
        length[idx],
    )


def concat(dataset: str, parts: list[Strata], k: int) -> Strata:
    if not parts:
        return Strata(dataset, np.zeros(0, str), np.zeros(0), np.zeros((0, 2)), np.zeros((0, k, 2)), np.zeros(0))
    return Strata(
        dataset,
        np.concatenate([p.animal for p in parts]),
        np.concatenate([p.t for p in parts]),
        np.concatenate([p.used for p in parts]),
        np.concatenate([p.avail for p in parts]),
        np.concatenate([p.step_m for p in parts]),
    )


def build_strata(dataset: str, tracks: Mapping[str, Track], cfg: Gps = GPS) -> Strata:
    parts = [s for tr in tracks.values() if (s := animal_strata(tr, dataset, cfg)) is not None]
    return concat(dataset, parts, cfg.n_available)


# ---- tiles over the fixes ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """One analysis area over a cluster of fixes: a tile_km square in lon/lat."""

    dataset: str
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north

    def dir(self, month: int, root: Path | None = None) -> Path:
        return (root or GPS_DIR) / self.dataset / f"{self.name}-m{month:02d}"

    def aoi(self) -> aoi_mod.AOI:
        return aoi_mod.bbox(*self.bbox, name=f"gps {self.name}")

    def contains(self, lon: Floats, lat: Floats) -> Mask:
        w, s, e, n = self.bbox
        out: Mask = (lon >= w) & (lon < e) & (lat >= s) & (lat < n)
        return out


def choose_tiles(strata: Strata, n_tiles: int, cfg: Gps = GPS) -> list[Tile]:
    """Up to n_tiles tile_km squares (on a lon/lat lattice) holding the most used fixes, picked greedily: each
    animal adds at most its remaining tile_need fixes to a square's value, so the tiles spread over animals."""
    if not len(strata):
        return []
    lat0 = float(np.mean(strata.used[:, 1]))
    dlat = cfg.tile_km * 1000 / (EARTH_R * math.pi / 180)
    dlon = dlat / math.cos(math.radians(lat0))
    ix = np.floor(strata.used[:, 0] / dlon).astype(int)
    iy = np.floor(strata.used[:, 1] / dlat).astype(int)
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, i, j in zip(strata.animal, ix, iy, strict=True):
        counts[(int(i), int(j))][str(a)] += 1
    need: dict[str, int] = defaultdict(lambda: cfg.tile_need)
    tiles: list[Tile] = []
    while len(tiles) < n_tiles and counts:
        cell = max(sorted(counts), key=lambda c: sum(min(v, need[a]) for a, v in counts[c].items()))
        if sum(min(v, need[a]) for a, v in counts[cell].items()) == 0:
            break
        for a, v in counts.pop(cell).items():
            need[a] = max(0, need[a] - v)
        i, j = cell
        bbox = (round(i * dlon, 6), round(j * dlat, 6), round((i + 1) * dlon, 6), round((j + 1) * dlat, 6))
        tiles.append(Tile(strata.dataset, f"{strata.dataset}-t{len(tiles) + 1}", bbox))
    return tiles


def write_tiles(tiles: list[Tile], root: Path | None = None) -> Path:
    p = (root or GPS_DIR) / "tiles.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([dataclasses.asdict(t) for t in tiles], indent=1))
    return p


def read_tiles(root: Path | None = None) -> list[Tile]:
    p = (root or GPS_DIR) / "tiles.json"
    return [Tile(d["dataset"], d["name"], tuple(d["bbox"])) for d in json.loads(p.read_text())]


def analyze_tiles(tiles: list[Tile], root: Path | None = None, cfg: Gps = GPS, log: Log = print) -> None:
    """Run the production pipeline on every tile at each season's month (private land allowed), one at a time;
    tiles already analyzed are kept."""
    from .analyze import run

    for t in tiles:
        for month, _ in cfg.seasons:
            out = t.dir(month, root)
            if (out / "state.pkl").exists():
                continue
            log(f"== {t.name} month {month}")
            run(t.aoi(), Options(month=month, public_only=False), log=log, out_dir=out)
            gc.collect()


# ---- scoring the strata -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    """One score map at a set of strata: values at the used fix and its available points (NaN = outside)."""

    used: Floats  # (n,)
    avail: Floats  # (n, K)

    def valid(self, cfg: Gps = GPS) -> Mask:
        out: Mask = ~np.isnan(self.used) & (np.sum(~np.isnan(self.avail), axis=1) >= cfg.min_available)
        return out

    def ranks(self) -> Floats:
        """Each used fix's rank among its available points: the share it beats, ties half (0.5 = chance)."""
        ok = ~np.isnan(self.avail)
        u = self.used[:, None]
        beat = np.sum((self.avail < u) & ok, axis=1) + 0.5 * np.sum((self.avail == u) & ok, axis=1)
        out: Floats = beat / np.maximum(ok.sum(axis=1), 1)
        return out


def _cells(st: ModelState, lonlat: Floats) -> Floats:
    """lon/lat points (..., 2) -> fractional (row, col) cells of the fine grid, flattened to (m, 2)."""
    flat = lonlat.reshape(-1, 2)
    x, y = st.fine.from_lonlat(flat[:, 0], flat[:, 1])
    g = st.fine
    return np.stack([(g.y0 - np.asarray(y)) / g.res, (np.asarray(x) - g.x0) / g.res], axis=1)


def _shifted_values(score: Floats, mask: Mask, pts: Floats, dr: int, dc: int) -> Floats:
    """The score read (dr, dc) cells away on a torus, at the points inside the area (others NaN)."""
    _, ok = values_at(score, mask, pts)
    v = np.full(len(pts), np.nan)
    H, W = score.shape
    r = (np.floor(pts[ok, 0]).astype(int) + dr) % H
    c = (np.floor(pts[ok, 1]).astype(int) + dc) % W
    v[ok] = score[r, c]
    return v


def score_strata(st: ModelState, score: Floats, s: Strata, cfg: Gps = GPS) -> dict[str, Scored]:
    """The strata scored on one map: at 20 m ("20m"), at the habitat scale ("ctx"), and for each on the map
    shifted cfg.shift_m east, south, west and north on a torus ("null20m-0".."null20m-3", "nullctx-0".."-3": the
    design check, whose ranks are averaged)."""
    n, k = len(s), cfg.n_available
    pts = _cells(st, np.concatenate([s.used[:, None, :], s.avail], axis=1))
    ctx = T.blur(score, cfg.context_m, st.fine.res)
    m = st.aoi_mask
    out: dict[str, Scored] = {}
    for name, sc in (("20m", score), ("ctx", ctx)):
        v = values_at(sc, m, pts)[0].reshape(n, k + 1)
        out[name] = Scored(v[:, 0], v[:, 1:])
        d = round(cfg.shift_m / st.fine.res)
        for i, (dr, dc) in enumerate(((0, d), (d, 0), (0, -d), (-d, 0))):
            nv = _shifted_values(sc, m, pts, dr, dc).reshape(n, k + 1)
            out[f"null{name}-{i}"] = Scored(nv[:, 0], nv[:, 1:])
    return out


def boyce(used: Floats, avail: Floats, bins: int = GPS.boyce_bins) -> float | None:
    """Boyce index: Spearman correlation between score bin and the used/available ratio in it, over equal-count
    bins of the available scores (merged where scores tie). 1: use rises with score; 0: no relation."""
    used, avail = used[~np.isnan(used)], avail[~np.isnan(avail)]
    if len(used) < 5 or len(avail) < bins:
        return None
    edges = np.unique(np.quantile(avail, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:  # fewer than 3 bins
        return None
    inner = edges[1:-1]
    pu = np.bincount(np.searchsorted(inner, used, side="right"), minlength=len(edges) - 1) / len(used)
    pa = np.bincount(np.searchsorted(inner, avail, side="right"), minlength=len(edges) - 1) / len(avail)
    ratio = np.where(pa > 0, pu / np.maximum(pa, 1e-12), np.nan)
    ok = ~np.isnan(ratio)
    if ok.sum() < 3 or np.ptp(ratio[ok]) == 0:
        return None
    return round(float(stats.spearmanr(np.arange(len(ratio))[ok], ratio[ok]).statistic), 3)


def _p_above_half(ranks: Floats, k: Floats) -> float | None:
    """One-sided p that the mean rank is above chance (normal approximation; the rank among k available points
    is uniform on 0, 1/k, ..., 1 under chance, variance (k+2)/(12k)). Consecutive fixes are not independent, so
    this is optimistic: the across-animal sign test is the one to trust."""
    if not len(ranks):
        return None
    var = float(np.sum((k + 2) / (12 * k))) / len(ranks) ** 2
    return round(float(stats.norm.sf((float(np.mean(ranks)) - 0.5) / math.sqrt(var))), 4)


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


@dataclass
class _Pool:
    """Everything one score function gave, stratum by stratum, across tiles (chunks of arrays, one per add)."""

    chunks: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
    _all: dict[str, np.ndarray] | None = None

    def add(self, s: Strata, sc: dict[str, Scored], cfg: Gps) -> None:
        ok = sc["20m"].valid(cfg) & sc["ctx"].valid(cfg)  # the shifted maps read the same points
        c: dict[str, np.ndarray] = dict(
            animal=np.array([f"{s.dataset}:{a}" for a in s.animal[ok]], dtype=str),
            dataset=np.full(int(ok.sum()), s.dataset),
            moving=s.step_m[ok] >= cfg.moving_min_m,
            k=np.sum(~np.isnan(sc["20m"].avail[ok]), axis=1).astype(float),
        )
        for key, name in (("20m", "20"), ("ctx", "ctx")):
            c[f"rank{name}"] = sc[key].ranks()[ok]
            c[f"null{name}"] = np.mean([sc[f"null{key}-{i}"].ranks()[ok] for i in range(4)], axis=0)
            c[f"used{name}"] = sc[key].used[ok]
            c[f"avail{name}"] = sc[key].avail[ok]
        self.chunks.append(c)
        self._all = None

    @property
    def all(self) -> dict[str, np.ndarray]:
        if self._all is None:
            keys = ("animal", "dataset", "moving", "k", "rank20", "null20", "used20", "rankctx", "nullctx", "usedctx")
            out = {k: np.concatenate([c[k] for c in self.chunks]) if self.chunks else np.zeros(0) for k in keys}
            for k in ("avail20", "availctx"):
                out[k] = np.concatenate([c[k] for c in self.chunks]) if self.chunks else np.zeros((0, 1))
            self._all = out
        return self._all

    def summary(self, sel: Mask, cfg: Gps) -> JSON:
        """The stats for the selected strata."""
        if not sel.any():
            return dict(n=0)
        a = {k: v[sel] for k, v in self.all.items() if k not in ("animal", "dataset")}
        mv = a["moving"].astype(bool)
        return dict(
            n=int(sel.sum()),
            rank20=_r(float(np.mean(a["rank20"]))),
            rank_ctx=_r(float(np.mean(a["rankctx"]))),
            p20=_p_above_half(a["rank20"], a["k"]),
            p_ctx=_p_above_half(a["rankctx"], a["k"]),
            boyce20=boyce(a["used20"], a["avail20"].ravel(), cfg.boyce_bins),
            boyce_ctx=boyce(a["usedctx"], a["availctx"].ravel(), cfg.boyce_bins),
            null20=_r(float(np.mean(a["null20"]))),
            null_ctx=_r(float(np.mean(a["nullctx"]))),
            n_moving=int(mv.sum()),
            moving_rank20=_r(float(np.mean(a["rank20"][mv]))) if mv.any() else None,
            moving_rank_ctx=_r(float(np.mean(a["rankctx"][mv]))) if mv.any() else None,
        )


def _across(animals: dict[str, JSON], cfg: Gps) -> JSON:
    """Across animals with at least cfg.min_strata strata: the median rank, how many beat chance (a sign test),
    and the same for the null."""
    ok = {a: v for a, v in animals.items() if v["n"] >= cfg.min_strata}
    out: JSON = dict(n_animals=len(ok))
    for key in ("rank20", "rank_ctx", "null20", "null_ctx", "moving_rank20"):
        vals = [v[key] for v in ok.values() if v.get(key) is not None]
        out[f"median_{key}"] = _r(float(np.median(vals))) if vals else None
        if key.startswith("rank"):
            out[f"sign_{key}"] = sign_test(vals)
    return out


def _results(pool: _Pool, cfg: Gps) -> JSON:
    animal, dataset = pool.all["animal"], pool.all["dataset"]
    animals = {a: dict(pool.summary(animal == a, cfg), dataset=a.split(":")[0]) for a in sorted(set(animal))}
    datasets: JSON = {}
    for d in sorted(set(dataset)):
        own = {a: v for a, v in animals.items() if v["dataset"] == d}
        datasets[d] = dict(pooled=pool.summary(dataset == d, cfg), across_animals=_across(own, cfg))
    overall = dict(pooled=pool.summary(np.ones(len(animal), bool), cfg), across_animals=_across(animals, cfg))
    return dict(animals=animals, datasets=datasets, overall=overall)


def season_months(month: int, cfg: Gps = GPS) -> tuple[int, ...]:
    return dict(cfg.seasons)[month]


_STRATA: dict[tuple[str, Gps], Strata] = {}


def strata_for(dataset: str, cfg: Gps = GPS) -> Strata:
    """The dataset's strata (built once per process from the cached data)."""
    key = (dataset, cfg)
    if key not in _STRATA:
        _STRATA[key] = build_strata(dataset, load(dataset), cfg)
    return _STRATA[key]


def gps_check_many(
    fns: Mapping[str, ScoreFn],
    root: Path | None = None,
    datasets: list[str] | None = None,
    cfg: Gps = GPS,
    log: Log = print,
) -> dict[str, JSON]:
    """Every score function on every analyzed tile (each state loaded once): {name: results} as gps_check
    returns them."""
    tiles = [t for t in read_tiles(root) if datasets is None or t.dataset in datasets]
    pools = {name: _Pool() for name in fns}
    for t in tiles:
        s_all = strata_for(t.dataset, cfg)
        inside = t.contains(s_all.used[:, 0], s_all.used[:, 1])
        for month, _ in cfg.seasons:
            path = t.dir(month, root) / "state.pkl"
            if not path.exists():
                log(f"  {t.name} month {month}: not analyzed, skipped")
                continue
            s = s_all.take(inside & np.isin(s_all.months, season_months(month, cfg)))
            if not len(s):
                continue
            st = load_state(path)
            for name, fn in fns.items():
                pools[name].add(s, score_strata(st, fn(st), s, cfg), cfg)
            log(f"  {t.name} month {month}: {len(s)} strata")
            del st
            gc.collect()
    return {name: dict(_results(p, cfg), score=name) for name, p in pools.items()}


def gps_check(
    score_fn: ScoreFn = production_score,
    root: Path | None = None,
    datasets: list[str] | None = None,
    cfg: Gps = GPS,
    log: Log = print,
) -> JSON:
    """The GPS falsification check for one score function of a loaded state (default: the production ranking):
    {"animals": {"<dataset>:<animal>": stats}, "datasets": {d: {"pooled", "across_animals"}}, "overall": ...}.
    Stats: n strata, rank20/rank_ctx (mean used-vs-available rank, 0.5 = chance), p20/p_ctx, boyce20/boyce_ctx,
    null20/null_ctx (the map-shift design check, ~0.5), and the moving steps alone."""
    return gps_check_many({"score": score_fn}, root, datasets, cfg, log)["score"]


def compare(a: JSON, b: JSON, key: str = "rank20", cfg: Gps = GPS) -> JSON:
    """Does score a beat score b animal by animal? Per dataset and overall: animals (with cfg.min_strata strata)
    where a's rank is higher, lower, the mean difference, and a two-sided sign test."""

    def one(names: list[str]) -> JSON:
        pairs = [
            (a["animals"][n][key], b["animals"][n][key])
            for n in names
            if n in b["animals"] and a["animals"][n]["n"] >= cfg.min_strata and a["animals"][n].get(key) is not None
        ]
        better = sum(x > y for x, y in pairs)
        worse = sum(x < y for x, y in pairs)
        p = float(stats.binomtest(better, better + worse, 0.5).pvalue) if better + worse else None
        diff = float(np.mean([x - y for x, y in pairs])) if pairs else None
        return dict(n=len(pairs), better=better, worse=worse, mean_diff=_r(diff), p_value=_r(p, 4))

    names = list(a["animals"])
    out = {d: one([n for n in names if a["animals"][n]["dataset"] == d]) for d in a["datasets"]}
    out["overall"] = one(names)
    return out


# ---- ablations and the placebo ----------------------------------------------------------------------------------


def recombine(
    st: ModelState, layers: Mapping[str, Floats] | None = None, w: Weights | None = None, penalties: bool = True
) -> Floats:
    """The production ranking recomputed from a state's saved layers (analyze.combine x site penalties), with
    some layers swapped (layers) or other weights (w)."""
    A = cast(Layers, {**st.layers, **(layers or {})})
    score, _ = combine(A, w or st.opts.weights, st.fine.res)
    out: Floats = score * site_penalty(A, st.opts) if penalties else score
    return out


def _zero(name: str) -> ScoreFn:
    return lambda st: recombine(st, {name: np.zeros_like(st.layers[name])})  # type: ignore[literal-required]


def _habitat(**kw: float) -> ScoreFn:
    def fn(st: ModelState) -> Floats:
        w = st.opts.weights
        h: Habitat = dataclasses.replace(w.habitat, **cast("dict[str, Any]", kw))
        return recombine(st, w=dataclasses.replace(w, habitat=h))

    return fn


def ablations() -> dict[str, ScoreFn]:
    """The production ranking recombined from the saved layers ("recombined", the reference for the others), and
    without each model term in turn: a factor (layer set to 0; water also leaves the habitat window), the habitat
    context (made constant), the 20 m zone, the site penalties, the winter module (season made 1)."""
    return {
        "recombined": recombine,
        "no travel": _zero("travel"),
        "no edges": _zero("edges"),
        "no pinch": _zero("pinch"),
        "no water": _zero("water"),
        "no wind": _zero("wind"),
        "no habitat": _habitat(edge_sat=math.inf, water_sat=math.inf),
        "no zone": _habitat(zone_m=0.0),
        "no penalties": lambda st: recombine(st, penalties=False),
        "no winter module": lambda st: recombine(st, {"season": np.ones_like(st.layers["season"])}),
    }


DEM_TERMS = ("wind", "pinch", "travel")  # the terms built from the elevation model alone


def dem_shift(direction: Literal["E", "N"] = "E", cfg: Gps = GPS) -> ScoreFn:
    """The DEM-shift placebo: the terms built from the elevation model alone (DEM_TERMS) moved cfg.shift_m
    (toroidally) before recombining, so the terrain no longer lines up with the canopy, water and the lions."""

    def fn(st: ModelState) -> Floats:
        d = round(cfg.shift_m / st.fine.res)
        axis = 1 if direction == "E" else 0
        return recombine(st, {k: np.roll(st.layers[k], d, axis=axis) for k in DEM_TERMS})  # type: ignore[literal-required]

    return fn
