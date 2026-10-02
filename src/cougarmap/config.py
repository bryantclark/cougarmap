"""Defaults and paths. Every tunable number in the model lives here so it can be adjusted in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aoi import UserPoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Where results and private files live. A dev checkout keeps them in the repo; an install keeps them somewhere
# easy to find (~/Documents/CougarMap). Override with COUGARMAP_HOME.
_default_home = (
    PROJECT_ROOT if (PROJECT_ROOT / "data" / "private").exists() else Path.home() / "Documents" / "CougarMap"
)
HOME = Path(os.environ.get("COUGARMAP_HOME", _default_home))
CACHE_DIR = Path(os.environ.get("COUGARMAP_CACHE", Path.home() / ".cache" / "cougarmap"))
OUT_DIR = Path(os.environ.get("COUGARMAP_OUT", HOME / "out" if HOME == PROJECT_ROOT else HOME / "results"))
PRIVATE_DIR = Path(
    os.environ.get("COUGARMAP_PRIVATE", HOME / "data" / "private" if HOME == PROJECT_ROOT else HOME / "my-data")
)
OBSERVATIONS_FILE = PRIVATE_DIR / "observations.jsonl"

USER_AGENT = "cougarmap/0.1 (personal wildlife-camera planning tool)"

MILE_M = 1609.344

# Worker threads for the parallel parts outside numba (GDAL warps, image encoding, state compression, downloads).
THREADS = min(8, os.cpu_count() or 4)


@dataclass
class Habitat:
    """How a spot's surroundings and the camera's view shape its score (see analyze.combine)."""

    context_m: float = 250.0  # Gaussian sigma of the surrounding-habitat window (~500 m across)
    edge_floor: float = 0.2  # habitat multiplier part with no hunting edge around (solid timber, open flats)
    edge_sat: float = 0.3  # hunting-edge density around a spot that counts as a full timber/opening mosaic
    water_floor: float = 0.5  # habitat multiplier part with no water around
    water_sat: float = 0.3  # water density around a spot that counts as fully watered
    zone_m: float = 20.0  # camera detection zone and pin error (Gaussian sigma, m); 0 = rate single cells
    travel_tpi_m: float = 300.0  # terrain scale of the drainage-bottom / ridge-spine skeleton
    travel_power: float = 2.0  # sharpens travel lines: only this landscape's real bottoms and spines count fully
    travel_relief_m: float = 3.0  # no travel line out of DEM noise on flat ground (TPI below this fades out)
    # Bottoms count all year; a ridge spine is a crossing only at a saddle or a junction of spines, or in winter
    # above big sun-facing slopes (our method's winter-thermal rule). Elsewhere a spine keeps this share of
    # its line (the human picks and the GPS check both put spines below bottoms; neither tests where the gates
    # sit).
    travel_spine_floor: float = 0.5
    travel_saddle_m: float = 150.0  # the gate fades out this far from a saddle
    travel_junction_m: float = 100.0  # ... and this far from a junction of the spine skeleton
    travel_junction_pos: float = 0.5  # the spine skeleton: cells above this travel position
    # bottoms on gentle grades count fully, steep ones down to travel_gentle_min (Dickson 2005, Dunford 2020)
    travel_gentle_deg: tuple[float, float] = (8.0, 25.0)
    travel_gentle_min: float = 0.5
    # winter thermals: in these months a spine with big S/SE-facing slopes around it is a crossing too
    thermal_months: tuple[int, ...] = (11, 12, 1, 2, 3)
    thermal_aspect_deg: tuple[float, float] = (100.0, 215.0)  # the slopes' facing (compass degrees)
    thermal_min_slope_deg: float = 10.0
    thermal_radius_m: float = 300.0
    thermal_frac: tuple[float, float] = (0.2, 0.45)  # gate ramps 0 -> 1 over this share of such slopes in the disc
    # Calibrated on the validation areas so strong real spots land at 60-100 (v3 top-5 picks run about 55-90).
    # Only the 0-100 display and its clip at 100 depend on it: 5.0 before v3, when the hunting edge stopped at the
    # timber and habitat scored ~1.6x lower.
    score_scale: float = 3.0


@dataclass
class EdgeBand:
    """The hunting edge (factors.hunting_edge). In cover it runs 0-35 m back from an opening, gone by 75 m; it also
    reaches into the opening itself, where kills cluster (Laundre & Hernandez 2003, Holmes & Laundre 2006): full
    to open_full_m from cover, gone at open_reach_m, where enough of the ground around is cover."""

    open_full_m: float = 15.0
    open_reach_m: float = 25.0
    open_cover_m: float = 30.0  # radius of the disc the cover share is measured in
    open_cover_frac: float = 0.25  # cover share of that disc that counts as real cover (not a lone tree)
    open_weight: float = 1.0
    # the downwind end of an opening counts up to this much more (x 1 - bonus + bonus x Q, Q 0-1)
    downwind_bonus: float = 0.25


@dataclass
class Weights:
    """How much each term counts. Wind first, edges second (per our method)."""

    wind: float = 0.35
    edges: float = 0.30
    pinch: float = 0.20
    water: float = 0.15
    travel: float = 0.30  # drainage bottoms / ridge spines: where a lion walks (added in, no stacking bonus)
    # Stacking ("more factors is better"): each of the four factors multiplies the spot by up to stack_multiplier,
    # ramping in smoothly from stack_from to stack_to so there is no jump at any one value.
    stack_multiplier: float = 1.25
    stack_from: float = 0.35
    stack_to: float = 0.65
    stack_threshold: float = 0.5  # a factor at or above this counts as "on" in the per-spot factors_on report
    habitat: Habitat = field(default_factory=Habitat)
    edge_band: EdgeBand = field(default_factory=EdgeBand)


@dataclass(frozen=True)
class WaterRules:
    """What counts as limited water (factors.water_sources, water_scarcity): small ponds and marshes are drinking
    water, and only big lakes are the permanent water that makes other sources less scarce. The 5 ha sizes are a
    model choice: the method's limited-water examples are springs, wallows, guzzlers and short surfacing streams,
    with no pond size. It is neutral on the human-pick and GPS checks (docs/experiments/)."""

    pond_max_ha: float = 5.0  # ponds and marshes under this size are a limited drinking-water source
    permanent_min_ha: float = 5.0  # only lakes at least this big count as permanent water in the scarcity bonus


WATER = WaterRules()


@dataclass(frozen=True)
class Winter:
    """The winter module (factors.compute_season): in winter, a multiplier on the habitat around a spot (never the
    spot itself) for where deer winter and lions follow them: low ground (Cooley et al. 2008: winter kills ~180 m
    lower; Bassing et al. 2023: winter camera use falls with elevation), sun-facing slopes (Elbroch et al. 2013)
    and shallow snow (Sullender et al. 2025), times mapped deer/elk winter range as a soft prior. Set a priori;
    never strengthened on the GPS check (docs/VALIDATION.md)."""

    months: tuple[int, ...] = (11, 12, 1, 2, 3, 4)
    floor: float = 0.6  # multiplier = floor + (1 - floor) x W, W 0-1
    relief_radius_m: float = 2500.0  # "low" = position in the elevation range within this radius
    relief_smooth_m: float = 125.0  # that range's min/max are smoothed over this (Gaussian sigma)
    min_relief_m: float = 30.0  # flatter country than this has no low/high
    south_steep_tan: float = 0.25  # steepness (rise/run) from which a slope's aspect counts fully
    # snow depth (NSIDC SNODAS, the median on snodas_day of the month over the last snodas_years winters): full
    # value up to snow_full_cm, falling linearly to snow_min at snow_zero_cm and deeper
    snow_full_cm: float = 25.0
    snow_zero_cm: float = 60.0
    snow_min: float = 0.2
    snodas_years: int = 5
    snodas_day: int = 15
    # WDFW Priority Habitats and Species deer/elk winter range (Washington only): 1 inside, feathered over
    # phs_feather_m, phs_outside elsewhere, in the months deer are on it
    phs_months: tuple[int, ...] = (12, 1, 2, 3)
    phs_outside: float = 0.8
    phs_feather_m: float = 400.0


WINTER = Winter()


@dataclass(frozen=True)
class Placement:
    """The optional "alternate on the trail" for each spot (analyze.trail_alternate): the best cell beside a quiet
    linear feature nearby. Cameras on dirt roads and game trails detect far more lions passing (Kolowski &
    Forrester 2017; Bassing et al. 2023 in NE Washington), but spots picked by hand on these factors mostly sit off
    mapped lines, so this is an extra suggestion, never the default pick. It changes no score.

    In SNAPSHOT USA camera arrays (2019-2024, docs/experiments/09-open-camera-and-collar-data.md) trail and
    dirt-road cameras caught about 3x more lions than other cameras in the same array, but a mapped quiet line
    within on_m of a camera gave no detection gain (rate ratio 0.83 [0.50, 1.39] given the camera's recorded
    placement), and only 13% of the on-trail cameras sat that close to a mapped line: the trails that matter are
    mostly unmapped game trails. So the gain comes from where the camera is strapped at the spot (the playbook's
    placement line), not from moving the spot onto a mapped line or giving mapped lines a score bonus."""

    search_m: float = 150.0  # how far from the spot to look
    on_m: float = 15.0  # a camera this close to the line watches it (strapped to a tree beside it)
    min_rec_m: float = 100.0  # not this close to a trailhead, campground or parking area (people)
    min_score_frac: float = 0.5  # only offered when it scores at least this share of the spot itself
    min_score: float = 30.0  # ... and at least this score (on the 0-100 scale; 60+ = strong)
    # a cell this close to a road open to vehicles that month (the access model's road_dist) is on the drivable
    # network: more traffic and theft. Quiet cells (closed roads, trails, two-tracks off the open network) are
    # offered first; an open one only when no quiet one qualifies, and the reason says it is open.
    open_road_m: float = 20.0


PLACEMENT = Placement()


@dataclass
class Options:
    """Per-run options. The agent sets these from the user's request."""

    month: int | None = None  # 1-12; None = current month
    wind_from_deg: float | None = None  # override the prevailing wind (direction it blows FROM, degrees)
    max_walk_miles: float = 1.0  # hard limit, measured along the easiest walking route from an open road
    paved_penalty: float = 0.7  # score is cut by up to this much beside a paved road: traffic, people, theft
    paved_full_m: float = 20.0  # the full paved-road cut applies this close (road, shoulder, pull-outs)
    paved_reach_m: float = 800.0  # the cut fades out linearly by this distance from pavement
    # people around a spot: houses (footprints >= house_min_m2) are counted within houses_radius_m. Every house
    # costs a little (people, dogs, camera theft: houses_exponent below), and towns and subdivisions cost a lot:
    # edge-of-town density starts the populated-area cut, town density gets it all. Rural ground is no longer
    # free, but a homestead or a few neighbours cost far less than a town.
    house_min_m2: float = 50.0  # smaller footprints (sheds, blinds, trailers, false detections) are ignored
    houses_radius_m: float = 500.0
    houses_from: float = 15.0  # no cut up to this many houses within houses_radius_m (~19 per km2)
    # the full cut from this many (~76 houses/km2: Maletzke et al. 2017, 99% of eastern-WA cougar use at
    # <= 76.5 residences/km2)
    houses_full: float = 60.0
    houses_penalty: float = 0.7  # score is cut by up to this much in a populated area
    # every house within houses_radius_m costs a little: score x (1 + houses) ** -houses_exponent, on top of the
    # populated-area cut (1 house x0.79, 5 x0.54, 15 x0.39). Chosen on SNAPSHOT USA cameras (within-array, out of
    # region), collar day beds and the human picks (docs/experiments/10-house-cost.md). 0 turns it off.
    houses_exponent: float = 0.34
    # recreation sites (trailheads, campgrounds, picnic sites, parking, toilets, shelters: OpenStreetMap): people
    # and dogs at the spot, and cameras get found. Set before any test (docs/VALIDATION.md, Site penalties).
    rec_penalty: float = 0.3  # score is cut by up to this much beside one
    rec_full_m: float = 50.0  # the full cut applies this close
    rec_reach_m: float = 400.0  # the cut fades out linearly by this distance
    n_candidates: int = 15
    candidate_spacing_m: float = 150.0
    per_zone: int = 3  # at most this many spots within zone_radius_m of each other (spread picks across hotspots)
    zone_radius_m: float = 800.0
    max_cells: int = 12_000_000  # caps analysis resolution for big areas (~5 m for 300 km2)
    min_res_m: float = 3.0
    weights: Weights = field(default_factory=Weights)
    user_points: list[UserPoint] = field(default_factory=list)  # pins: kind water | seasonal_water | sign


@dataclass(frozen=True)
class Truth:
    """How field results (cameras, snow tracks, crossing transects) are scored against the model (truth.py).
    These set the tests, not the model: changing them never changes a pick."""

    # Cougar detections per 100 camera-nights at random on-trail cameras in the Washington Predator-Prey Project
    # (NE Washington and Okanogan, Bassing et al. 2023, Ecol. Appl. 33:e2745): summer (Jul-Sep) 159 / 18,377,
    # winter (Dec-Feb) 72 / 19,262. Months in between get the midpoint.
    base_summer_per_100: float = 0.87
    base_winter_per_100: float = 0.37
    summer_months: tuple[int, ...] = (7, 8, 9)
    winter_months: tuple[int, ...] = (12, 1, 2)
    min_nights: float = 30.0  # an arm's rate is compared with the base rate only past this many camera-nights
    independent_min: float = 30.0  # photos of one species at one camera this close together are one detection
    # track-path test: the same track shape rotated and shifted this far inside the area is the null
    shift_min_m: float = 100.0
    shift_max_m: float = 1500.0
    n_shifted: int = 200
    sample_m: float = 5.0  # tracks and transect routes are scored every this many metres
    min_inside: float = 0.9  # a shifted copy must keep this share of its points inside the area
    min_confidence: str = "probable"  # tracks less sure than this (certain > probable > possible) aren't tested
    # a track whose first point (where it was found: tracks are followed backwards) is this close to an open road
    # was "found from a road"; the report splits the track test by it, since the shifted copies don't keep
    # their distance to roads (so road-found tracks are only partly corrected for where people go)
    near_road_m: float = 100.0
    # a transect file's waypoints count as lion crossings only when their name holds one of these words
    # (phone apps also export parking, start and other waypoints with the track)
    crossing_words: tuple[str, ...] = ("lion", "cougar", "puma")
    n_permutations: int = 10_000  # Monte Carlo draws for the camera and transect p-values
    # sample-size helper (camera arms): site-to-site spread of the true rate (gamma CV; neighbouring cameras
    # differ a lot, Kolowski 2021) and the default effort
    site_cv: float = 0.75
    nights_per_camera: int = 180
    alpha: float = 0.05
    power_target: float = 0.8


TRUTH = Truth()


@dataclass(frozen=True)
class Gps:
    """How open GPS collar data from other ecosystems is used to try to falsify the model (gps.py). A test, never
    a tuning target: changing these never changes a pick, and no weight is ever fitted on these data."""

    # used fixes: night and twilight (the sun below this, degrees), when lions travel and hunt
    max_sun_deg: float = 6.0
    # steps: consecutive fixes at the animal's usual interval (within this share of it), up to max_interval_h
    interval_tol: float = 0.25
    max_interval_h: float = 8.0
    # iSSF-style design: for each used fix, this many available points drawn from the animal's own night step
    # lengths and turning angles around its previous fix (fewer than min_available inside the tile: dropped)
    n_available: int = 15
    min_available: int = 10
    min_dist_steps: int = 30  # fewer night steps than this: the animal's steps at any hour give the distributions
    moving_min_m: float = 100.0  # a used step at least this long is "moving" (not a bed or a kill)
    context_m: float = 250.0  # habitat scale: the score averaged at this Gaussian sigma (~500 m across)
    shift_m: float = 1500.0  # the placebo and null shifts
    boyce_bins: int = 10  # Boyce index: equal-count bins of the available scores
    min_strata: int = 20  # animals with fewer used fixes in the tiles get no verdict of their own
    # analysis tiles: squares this size over the densest fixes, chosen greedily so each animal adds at most
    # tile_need used fixes to a tile's value (spreads the tiles over animals instead of one homebody)
    tile_km: float = 8.0
    tile_need: int = 300
    # the production pipeline runs at each month here, scored on fixes from the months paired with it
    seasons: tuple[tuple[int, tuple[int, ...]], ...] = ((10, (5, 6, 7, 8, 9, 10, 11)), (1, (12, 1, 2, 3, 4)))
    seed: int = 0


GPS = Gps()
