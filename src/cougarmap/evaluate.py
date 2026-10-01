"""Pick-level evaluation: would the tool pick the camera spots a person picked by hand from the same factors? The
primary check on any model change when you have such picks (it measures agreement with the person who chose the
spots; truth.py tests against lion field data).

The human picks are camera pins named CamNN in a KML/KMZ, and the areas are analyzed areas whose saved states
(<states_dir>/<slug>/state.pkl) cover them. Every distinct candidate spot in an area (local score peaks at least
150 m apart, the same spacing the picker uses) is ranked; each camera pin is scored by the best cell within 15 m
(pin accuracy) and placed in that ranking. Random points get the same treatment, so compare against the random
baseline.

A second, cruder check (fixed-K): take the top 3 spots per km2, the way someone would actually place cameras, and
count how many camera pins fall within 50/100/150 m of one. It keeps smoothing effects visible: a smoother map has
fewer, broader peaks, which helps rank metrics without putting a pick closer to the pin.

Usage (scripts/eval_picks.py runs this module):
    uv run python scripts/eval_picks.py --kml picks.kml --area north=north-ridge --area south=south-fork
    uv run python scripts/eval_picks.py ... --raw         # the score alone (no site penalties)
    uv run python scripts/eval_picks.py ... --by-cam      # plus per-camera ranks
    uv run python scripts/eval_picks.py ... --check south # report one area's fixed-K on its own (a held-out area)

With no arguments the setup comes from data/private/eval.toml (EvalConfig; git-ignored, so the human picks
and area names never enter the repo). COUGARMAP_EVAL_DIR overrides the states folder (default out/); point it at
states analyzed with --no-pins when the person who chose the cameras also placed the water pins, or the pins leak
into the numbers.

From code (to try a scoring idea without a rerun):
    from cougarmap.evaluate import evaluate
    evaluate(lambda st: my_score(st.layers), areas={...}, kml=...)  # any function of a ModelState -> score array
"""

from __future__ import annotations

import argparse
import gc
import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

from . import aoi
from .analyze import site_penalty
from .arrays import Floats, Ints
from .config import PRIVATE_DIR, PROJECT_ROOT
from .state import ModelState, load_state

CONFIG = PRIVATE_DIR / "eval.toml"  # the private human-pick setup (git-ignored, like everything in PRIVATE_DIR)
STATES_DIR = Path(os.environ.get("COUGARMAP_EVAL_DIR", PROJECT_ROOT / "out"))  # where <slug>/state.pkl live


@dataclass(frozen=True)
class EvalConfig:
    """Which human picks to rank, and where. Read from CONFIG:

    kml = "my-areas.kml"          # the CamNN pins; relative paths are relative to the config file
    states = "../../out/no-pins"  # optional: the folder of <slug>/state.pkl (default STATES_DIR)
    check = "south"               # optional: the held-out area's key
    [areas]                       # key = analyzed area slug
    north = "north-ridge"
    south = "south-fork"
    """

    kml: Path | None = None
    areas: dict[str, str] = field(default_factory=dict)
    states: Path | None = None
    check: str | None = None

    @classmethod
    def load(cls, path: Path = CONFIG) -> EvalConfig:
        if not path.exists():
            return cls()
        raw = tomllib.loads(path.read_text())
        here = path.parent
        return cls(
            kml=here / raw["kml"] if "kml" in raw else None,
            areas={str(k): str(v) for k, v in raw.get("areas", {}).items()},
            states=here / raw["states"] if "states" in raw else None,
            check=raw.get("check"),
        )


SPACING_M, PIN_M, LOCAL_M = 150.0, 15.0, 1000.0
FIXED_K_PER_KM2 = 3.0  # fixed-K check: this many top spots per km2
FIXED_K_RADII_M = (50.0, 100.0, 150.0)


def cameras(kml: Path) -> list[dict[str, Any]]:
    """The camera pins (CamNN) in a KML/KMZ: the human picks. Coordinates rounded as list_areas gives them."""
    return [
        dict(name=p["name"], lat=round(p["lat"], 6), lon=round(p["lon"], 6))
        for p in aoi.user_points_from_kml(kml)
        if p["kind"] == "camera"
    ]


@dataclass(frozen=True)
class Spots:
    """The distinct candidate spots of one score map: smoothed local peaks at least SPACING_M apart."""

    peaks: Ints  # (n, 2) row/col of each spot
    values: Floats  # smoothed score at each spot
    order: Floats  # values, best first
    pin_max: Floats  # best smoothed score within PIN_M of every cell
    res: float

    @classmethod
    def from_score(cls, score: Floats, res: float) -> Spots:
        sm = ndimage.gaussian_filter(score, max(0.5, 6 / res))
        pk = peak_local_max(  # type: ignore[no-untyped-call]  # scikit-image is untyped
            sm, min_distance=max(1, int(SPACING_M / res)), threshold_abs=1e-6, exclude_border=False
        )
        pv = sm[pk[:, 0], pk[:, 1]]
        mx = ndimage.maximum_filter(sm, size=2 * max(1, int(PIN_M / res)) + 1)
        return cls(peaks=pk, values=pv, order=np.sort(pv)[::-1], pin_max=mx, res=res)

    def rank(self, row: int, col: int) -> tuple[float, int]:
        """(fraction of spots that score at least as well, spots within LOCAL_M that beat this cell)."""
        v = self.pin_max[row, col]
        frac = min(1.0, (int((self.order > v).sum()) + 1) / len(self.order))
        d = np.hypot(self.peaks[:, 0] - row, self.peaks[:, 1] - col) * self.res
        near = (d <= LOCAL_M) & (d > 2 * PIN_M)
        return frac, int((self.values[near] > v).sum())

    def top_k_distance(self, points: Ints, k: int) -> Floats:
        """Distance (m) from each (row, col) point to the nearest of the k best spots."""
        top = self.peaks[np.argsort(-self.values, kind="stable")[:k]]
        if len(top) == 0 or len(points) == 0:
            return np.full(len(points), np.inf)
        d: Floats = np.hypot(top[None, :, 0] - points[:, None, 0], top[None, :, 1] - points[:, None, 1])
        return d.min(axis=1) * self.res


@dataclass(frozen=True)
class AreaPins:
    """Pins (the human camera picks) ranked among one area's spots, and random points the same way."""

    spots: Spots
    names: list[str]
    ranks: list[float]  # Spots.rank fraction per pin inside the area
    locs: list[int]  # spots within LOCAL_M that beat each pin
    pins: list[tuple[int, int]]
    random: list[tuple[float, int]]
    random_points: Ints

    @classmethod
    def rank(cls, st: ModelState, score: Floats, pins: list[dict[str, Any]], n_random: int = 300) -> AreaPins:
        """score: the area's score array (fine grid); pins: [{name, lat, lon}] (those outside are skipped)."""
        g, m = st.fine, st.aoi_mask
        spots = Spots.from_score(np.where(m, score, 0).astype("float32"), g.res)
        names: list[str] = []
        ranks: list[float] = []
        locs: list[int] = []
        rcs: list[tuple[int, int]] = []
        for p in pins:
            a, b = g.rowcol(*g.from_lonlat(p["lon"], p["lat"]))
            if not (g.contains_rc(a, b) and m[int(a), int(b)]):
                continue
            fr, lo = spots.rank(int(a), int(b))
            names.append(p["name"])
            ranks.append(fr)
            locs.append(lo)
            rcs.append((int(a), int(b)))
        rr, cc = np.nonzero(m)
        idx = np.random.default_rng(0).choice(len(rr), min(n_random, len(rr)), replace=False)
        rnd_pts = np.stack([rr[idx], cc[idx]], axis=1)
        rnd = [spots.rank(int(a), int(b)) for a, b in rnd_pts]
        return cls(spots, names, ranks, locs, rcs, rnd, rnd_pts)


def production_score(st: ModelState) -> Floats:
    """What the picker ranks (before the land/walk rules): the score times the paved-road and building penalties."""
    return st.layers["score"] * site_penalty(st.layers, st.opts)


def raw_score(st: ModelState) -> Floats:
    return st.layers["score"]


def vs_random(ranks: Floats, random_ranks: Floats) -> float:
    """Probability a camera ranks better than a random point (ties count half). 0.5 = no better than chance."""
    r = np.asarray(random_ranks)
    return float(np.mean([np.mean(r > x) + 0.5 * np.mean(r == x) for x in ranks]))


def evaluate(
    score_fn: Callable[[ModelState], Floats] = production_score,
    areas: dict[str, str] | None = None,
    n_random: int = 300,
    verbose: bool = True,
    states_dir: Path | None = None,
    kml: Path | None = None,
    check_area: str | None = None,
) -> dict[str, Any]:
    """score_fn(state) -> score array on state.fine (default: the production ranking), for the states in
    states_dir/<slug>/state.pkl of areas ({key: slug}) and the camera pins in kml, each defaulting to
    EvalConfig.load() (states: COUGARMAP_EVAL_DIR first). check_area is the key whose fixed-K result is also
    reported on its own. Returns metrics per area and overall, plus the fixed-K check."""
    cfg = EvalConfig.load()
    areas = areas if areas is not None else cfg.areas
    kml = kml or cfg.kml
    if kml is None or not areas:
        raise ValueError(f"no human picks: pass kml and areas, or set them in {CONFIG}")
    check_area = check_area if check_area is not None else cfg.check
    cams = cameras(kml)
    states_dir = states_dir or (STATES_DIR if "COUGARMAP_EVAL_DIR" in os.environ else cfg.states) or STATES_DIR
    out: dict[str, Any] = {"areas": {}, "cams": {}}
    all_rank: list[float] = []
    all_local: list[int] = []
    rnd_rank: list[float] = []
    rnd_local: list[int] = []
    fk_cam: list[float] = []
    fk_rnd: list[float] = []
    fk_check: list[float] = []
    for key, slug in areas.items():
        st = load_state(states_dir / slug / "state.pkl")
        g, m = st.fine, st.aoi_mask
        a = AreaPins.rank(st, score_fn(st), cams, n_random)
        spots, ranks, locs, pins, rnd, rnd_pts = a.spots, a.ranks, a.locs, a.pins, a.random, a.random_points
        for name, fr, lo in zip(a.names, ranks, locs, strict=True):
            out["cams"][name] = dict(area=key, rank_frac=round(fr, 3), beaten_within_1km=lo)
        k = max(3, round(FIXED_K_PER_KM2 * float(m.sum()) * g.res * g.res / 1e6))
        dc = spots.top_k_distance(np.array(pins).reshape(-1, 2), k)
        fk_cam += list(dc)
        fk_rnd += list(spots.top_k_distance(rnd_pts, k))
        if key == check_area:
            fk_check += list(dc)
        out["areas"][key] = dict(
            n_cams=len(ranks),
            n_spots=len(spots.order),
            median_rank=float(np.median(ranks)),
            top10=float(np.mean(np.array(ranks) <= 0.10)),
            top25=float(np.mean(np.array(ranks) <= 0.25)),
            median_beaten_1km=float(np.median(locs)),
            random_median_rank=float(np.median([x[0] for x in rnd])),
            random_median_beaten_1km=float(np.median([x[1] for x in rnd])),
            fixed_k=k,
            fixed_k_within_150m=float(np.mean(dc <= 150)),
        )
        all_rank += ranks
        all_local += locs
        rnd_rank += [x[0] for x in rnd]
        rnd_local += [x[1] for x in rnd]
        del st, a, spots
        gc.collect()
    ranks_all, locs_all = np.array(all_rank), np.array(all_local)
    cam_d, rnd_d, chk_d = np.array(fk_cam), np.array(fk_rnd), np.array(fk_check)
    out["overall"] = dict(
        median_rank=float(np.median(ranks_all)),
        mean_rank=float(np.mean(ranks_all)),
        top10=float(np.mean(ranks_all <= 0.10)),
        top25=float(np.mean(ranks_all <= 0.25)),
        median_beaten_1km=float(np.median(locs_all)),
        local_best_or_2nd=float(np.mean(locs_all <= 1)),
        random_median_rank=float(np.median(rnd_rank)),
        random_top25=float(np.mean(np.array(rnd_rank) <= 0.25)),
        random_median_beaten_1km=float(np.median(rnd_local)),
        vs_random=vs_random(ranks_all, np.array(rnd_rank)),
    )
    out["fixed_k"] = dict(
        per_km2=FIXED_K_PER_KM2,
        cams_within={f"{r:g}": float(np.mean(cam_d <= r)) for r in FIXED_K_RADII_M},
        random_within={f"{r:g}": float(np.mean(rnd_d <= r)) for r in FIXED_K_RADII_M},
        check_area_within={f"{r:g}": float(np.mean(chk_d <= r)) for r in FIXED_K_RADII_M} if chk_d.size else {},
        median_distance_m=float(np.median(cam_d)),
        check_area=check_area,
    )
    if verbose:
        report(out)
    return out


def report(out: dict[str, Any], by_cam: bool = False) -> None:
    o = out["overall"]
    print(
        f"OVERALL  median rank {o['median_rank']:.0%} (random {o['random_median_rank']:.0%}, "
        f"vsR {o['vs_random']:.2f})  mean {o['mean_rank']:.0%}  top10% {o['top10']:.0%}  "
        f"top25% {o['top25']:.0%} (random {o['random_top25']:.0%})  "
        f"beaten within 1 km: median {o['median_beaten_1km']:.0f} (random {o['random_median_beaten_1km']:.0f}), "
        f"local best/2nd {o['local_best_or_2nd']:.0%}"
    )
    for k, v in out["areas"].items():
        print(
            f"  {k:8s} {v['n_cams']:2d} cams / {v['n_spots']:5d} spots  median rank {v['median_rank']:.0%} "
            f"(random {v['random_median_rank']:.0%})  top25% {v['top25']:.0%}  "
            f"beaten 1 km {v['median_beaten_1km']:.0f} (random {v['random_median_beaten_1km']:.0f})  "
            f"fixed-K {v['fixed_k']} spots: {v['fixed_k_within_150m']:.0%} within 150 m"
        )
    fk = out["fixed_k"]
    within = "  ".join(
        f"<={r} m {fk['cams_within'][r]:.0%} (random {fk['random_within'][r]:.0%})" for r in fk["cams_within"]
    )
    chk = "  ".join(f"<={r} m {v:.0%}" for r, v in fk["check_area_within"].items())
    held_out = f" | {fk['check_area']}: {chk}" if fk["check_area_within"] else ""
    print(f"FIXED-K  top {fk['per_km2']:g}/km2: {within}  median {fk['median_distance_m']:.0f} m{held_out}")
    if by_cam:
        print("  " + " ".join(f"{n}:{c['rank_frac']:.0%}/{c['beaten_within_1km']}" for n, c in out["cams"].items()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="eval_picks", description="Rank human camera picks among an area's spots.")
    ap.add_argument("--kml", type=Path, default=None, help="KML/KMZ with the CamNN pins")
    ap.add_argument(
        "--area",
        action="append",
        default=[],
        metavar="KEY=SLUG",
        help="an analyzed area (repeat)",
    )
    ap.add_argument("--states", type=Path, default=None, help="folder of <slug>/state.pkl")
    ap.add_argument("--check", default=None, help="key of the held-out area")
    ap.add_argument("--raw", action="store_true", help="the score alone, without the site penalties")
    ap.add_argument("--by-cam", action="store_true", help="also print each camera's rank")
    a = ap.parse_args(argv)
    areas = dict(s.partition("=")[::2] for s in a.area) if a.area else None
    try:
        res = evaluate(
            raw_score if a.raw else production_score,
            areas=areas,
            verbose=False,
            states_dir=a.states,
            kml=a.kml,
            check_area=a.check,
        )
    except ValueError as e:
        ap.error(str(e))
    report(res, by_cam=a.by_cam)


if __name__ == "__main__":
    main()
