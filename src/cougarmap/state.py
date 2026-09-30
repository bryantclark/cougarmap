"""The model state: every computed layer of an analysis, typed, and how it is saved to and loaded from state.pkl.

`Layers` is the one schema of the model's layers. Each entry's annotation says whether the layer is kept in
state.pkl and how (`Store`); entries without one are intermediate, used only during a run. Fine-grid layers have
plain names, 10 m grid layers end in `_mid`.

`ModelState` is what picking, explaining and exporting need: grids, options, wind, terrain and the layers. A run
builds a `context.Context` (a ModelState plus the downloaded inputs); `load_state` gives back a ModelState.
States saved by older versions go through `migrate`, which fills the layers they lack with neutral values
(no buildings, no lakes, ...) so the model code never checks what a state has. Layers an older model did not
compute at all are listed in `ModelState.unmodeled`, so explanations leave them out instead of reporting a zero.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, TypedDict, cast, get_args, get_origin, get_type_hints

import numpy as np
from rasterio.enums import Resampling

from . import statefile
from .aoi import AOI
from .arrays import Floats, Ints, Mask
from .config import THREADS, Options
from .grid import Grid
from .sources.dem import DemInfo
from .sources.weather import Wind
from .terrain import Saddle

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# 1: before model v2 (no travel lines, habitat context or paved-road distance; some states lack the any-route
#    walk, building counts, lakes or the along-valley wind). 2: model v2, unversioned. 3: versioned.
# 4: houses within 500 m (populated areas) replace buildings within 150 m.
# 5: recreation sites (rec_dist); ridge-spine crossing gates; downwind ends per air current; the winter module;
#    quiet roads/trails (placement suggestions).
STATE_VERSION: Final = 5

PAVED_DIST_CAP_M: Final = 2000.0  # paved-road distances are stored up to this (well past the penalty's reach)


class Store(enum.Enum):
    """How a layer is kept in state.pkl."""

    HALF = "float16 on disk, float32 in memory: smooth 0-1 layers and unit vectors"
    FULL = "float32: distances, walking times and the scores spots are ranked by"
    EXACT = "as computed: masks, labels, counts, indices and lookup lists"


class SaddlePoint(Saddle):
    x: float  # mid-grid CRS coordinates of the saddle (m)
    y: float


class Layers(TypedDict, total=False):
    """Every layer of an analysis (see the module docstring for the naming and the Store annotations)."""

    # terrain
    slope: Annotated[Floats, Store.HALF]  # degrees
    slope_mid: Floats
    landform_mid: Annotated[Ints, Store.EXACT]  # geomorphon class (terrain.LANDFORM_NAMES)
    acc_mid: Floats  # upslope contributing area, m2 (single flow path)
    sca_mid: Floats  # upslope area per metre of contour width, m2/m (multiple flow directions)
    tpi_mid: Floats  # topographic position (m above the surrounding mean)
    cliff: Annotated[Mask, Store.EXACT]
    # travel lines: drainage bottoms and ridge spines
    travel: Annotated[Floats, Store.HALF]
    travel_pos_mid: Annotated[Floats, Store.HALF]  # -1 bottom .. +1 spine
    travel_gate_mid: Annotated[Floats, Store.HALF]  # 0-1: a spine is a crossing here (saddle, spine junction)
    travel_thermal_mid: Annotated[Floats, Store.HALF]  # 0-1: ... or a ridge above sun-facing slopes (winter)
    # land status
    land_id: Annotated[Ints, Store.EXACT]  # index into land_names / land_access (0 = private / unknown)
    land_id_mid: Ints
    land_names: Annotated[list[str], Store.EXACT]
    land_access: Annotated[list[str | None], Store.EXACT]  # PAD-US pub_access code ("OA" = open access)
    public: Annotated[Mask, Store.EXACT]
    public_mid: Mask
    usfs_mid: Mask
    # wind: cold-air drainage, the prevailing wind, the air actually moving at dawn/dusk
    wind: Annotated[Floats, Store.HALF]
    wind_mid: Floats
    windx_mid: Annotated[Floats, Store.HALF]  # along-valley / exposed wind direction (unit east, north)
    windn_mid: Annotated[Floats, Store.HALF]
    conv_mid: Annotated[Floats, Store.HALF]  # drainage and wind agree
    windward_mid: Annotated[Floats, Store.HALF]
    drain_mid: Annotated[Floats, Store.HALF]
    drainx_mid: Annotated[Floats, Store.HALF]  # cold-air drainage direction (unit east, north)
    drainn_mid: Annotated[Floats, Store.HALF]
    cos_mid: Annotated[Floats, Store.HALF]
    flowx_mid: Annotated[Floats, Store.HALF]  # dawn/dusk air flow (unit east, north)
    flown_mid: Annotated[Floats, Store.HALF]
    flowx: Annotated[Floats, Store.HALF]
    flown: Annotated[Floats, Store.HALF]
    # edges: the hunting edge (timber overlooking an opening), downwind ends, secondary travel lines
    edges: Annotated[Floats, Store.HALF]
    meadow: Annotated[Mask, Store.EXACT]  # openings big enough to hunt
    meadow_label: Annotated[Ints, Store.EXACT]  # opening id (0 = none)
    meadow_ha: Annotated[Floats, Store.EXACT]  # opening size by id
    edge_q: Floats  # downwind position over the opening, both currents (1 = most downwind)
    edge_q_drain: Annotated[Floats, Store.HALF]  # ... along the evening cold-air drainage
    edge_q_wind: Annotated[Floats, Store.HALF]  # ... along the daytime high-pressure wind
    edge_downwind: Annotated[Floats, Store.HALF]
    edge_meadow: Annotated[Floats, Store.HALF]  # the hunting edge itself
    edge_ridge: Annotated[Floats, Store.HALF]
    edge_valley: Annotated[Floats, Store.HALF]
    edge_water: Annotated[Floats, Store.HALF]
    edge_route: Annotated[Floats, Store.HALF]
    # pinch points
    pinch: Annotated[Floats, Store.HALF]
    pinch_saddle: Annotated[Floats, Store.HALF]
    pinch_cliffbase: Annotated[Floats, Store.HALF]
    pinch_clifftop: Annotated[Floats, Store.HALF]
    pinch_bank: Annotated[Floats, Store.HALF]
    pinch_fence: Annotated[Floats, Store.HALF]
    pinch_funnel: Annotated[Floats, Store.HALF]
    saddle_points: Annotated[list[SaddlePoint], Store.EXACT]
    # limited water
    water: Annotated[Floats, Store.HALF]
    water_kind: Annotated[Ints, Store.EXACT]  # index into water_labels of the source that scores (-1 = none)
    water_labels: Annotated[list[str], Store.EXACT]
    water_scarcity: Annotated[Floats, Store.HALF]
    lake: Annotated[Mask, Store.EXACT]
    # access, traffic, buildings
    road_mid: Mask
    road_dist: Annotated[Floats, Store.FULL]  # m to the nearest open road
    walk_m: Annotated[Floats, Store.FULL]  # walking route length from an open road, never across private land
    walk_s: Annotated[Floats, Store.FULL]
    walk_pred_mid: Annotated[Ints, Store.EXACT]  # route predecessors (flat mid-grid cell index, -1 = start)
    walk_any_m: Annotated[Floats, Store.FULL]  # the same by any route (private land allowed)
    walk_any_s: Annotated[Floats, Store.FULL]
    walk_any_pred_mid: Annotated[Ints, Store.EXACT]
    paved_dist: Annotated[Floats, Store.HALF]  # m to pavement, capped at factors.PAVED_DIST_CAP_M
    closed_track_mid: Annotated[Mask, Store.EXACT]
    trail_kind: Annotated[Ints, Store.EXACT]  # beside a quiet road/trail (factors.TRAIL_KINDS code, 0 = none)
    houses: Annotated[Floats, Store.HALF]  # houses within houses_radius_m (how populated the area is)
    rec_dist: Annotated[Floats, Store.HALF]  # m to a trailhead/campground/parking area, capped at PAVED_DIST_CAP_M
    # the winter module: where deer winter (a multiplier on the habitat around a spot; 1 outside Nov-Apr)
    season: Annotated[Floats, Store.HALF]
    winter_mid: Annotated[Floats, Store.HALF]  # 0-1: low, sun-facing ground with shallow snow
    winter_range_mid: Annotated[Mask, Store.EXACT]  # inside mapped deer/elk winter range (WDFW PHS)
    # the camera-spot score (analyze.combine) and the land/access rules (analyze.apply_masks)
    edge_density: Annotated[Floats, Store.HALF]
    water_density: Annotated[Floats, Store.HALF]
    context: Annotated[Floats, Store.HALF]  # habitat multiplier
    score: Annotated[Floats, Store.FULL]  # 0-100, before the land/access rules and site penalties
    n_on: Annotated[Ints, Store.EXACT]  # factors at or above the stack threshold
    usable: Annotated[Mask, Store.EXACT]
    final: Annotated[Floats, Store.FULL]
    usable_private: Mask
    final_private: Floats


def _saved_layers() -> dict[str, Store]:
    out: dict[str, Store] = {}
    for name, hint in get_type_hints(Layers, include_extras=True).items():
        if get_origin(hint) is Annotated:
            stores = [m for m in get_args(hint)[1:] if isinstance(m, Store)]
            out[name] = stores[0]
    return out


SAVED: Final[dict[str, Store]] = _saved_layers()  # layer -> how state.pkl keeps it


@dataclass(kw_only=True)
class ModelState:
    """An analyzed area: what picking, explaining and exporting need (and what state.pkl holds)."""

    aoi: AOI
    opts: Options
    month: int
    fine: Grid  # analysis grid (AOI + small pad)
    mid: Grid  # 10 m-ish grid (AOI + walking pad) for landforms, drainage, walking
    wind: Wind
    dem_info: DemInfo
    z: Floats  # fine DEM (m)
    chm: Floats  # fine canopy height (m)
    aoi_mask: Mask  # fine
    layers: Layers = field(default_factory=Layers)
    notes: list[str] = field(default_factory=list)
    unmodeled: frozenset[str] = frozenset()  # layers the model version that made this state did not compute

    def up(self, a_mid: np.ndarray, resampling: Resampling = Resampling.bilinear) -> Floats:
        """Mid-grid array -> fine grid."""
        return self.fine.resample_from(a_mid, self.mid, resampling=resampling)

    def any_route(self, row: int, col: int) -> bool:
        """Is this fine cell reached by the any-route walk (private ground, or everything when private land is
        allowed) rather than the public-access walk?"""
        return not (self.opts.public_only and self.layers["public"][row, col])


# ---- saving ------------------------------------------------------------------------------------------------


def _stored(name: str, a: np.ndarray) -> np.ndarray:
    if SAVED[name] is Store.EXACT or a.dtype.kind != "f":
        return a
    return a.astype(np.float16 if SAVED[name] is Store.HALF else np.float32)


def _stored_chm(chm: Floats) -> np.ndarray:
    """Canopy height as state.pkl keeps it: whole metres, 0-255."""
    return np.rint(np.clip(chm, 0, 255)).astype(np.uint8)


def to_saved_precision(st: ModelState) -> None:
    """Round the layers state.pkl keeps at reduced precision (float16 layers, canopy height) to that precision, in
    place, so a fresh analysis picks and describes spots from the same values a re-pick of its saved state will."""
    A = cast("dict[str, Any]", st.layers)
    for name, store in SAVED.items():
        a = A.get(name)
        if store is Store.HALF and isinstance(a, np.ndarray) and a.dtype.kind == "f":
            A[name] = a.astype(np.float16).astype(np.float32)
    st.chm = _stored_chm(st.chm).astype(np.float32)


def save_state(st: ModelState, path: Path) -> None:
    """Write the saved layers (SAVED) and everything needed to pick, explain and export again to path."""
    layers: dict[str, Any] = dict(st.layers)
    arrays = {k: _stored(k, v) for k, v in layers.items() if k in SAVED and isinstance(v, np.ndarray)}
    meta = {k: v for k, v in layers.items() if k in SAVED and not isinstance(v, np.ndarray)}
    slim = dict(
        version=STATE_VERSION,
        aoi=st.aoi,
        opts=st.opts,
        month=st.month,
        fine=st.fine,
        mid=st.mid,
        wind=st.wind,
        dem_info=st.dem_info,
        notes=st.notes,
        unmodeled=sorted(st.unmodeled),
        meta=meta,
        z=st.z.astype(np.float32),
        chm=_stored_chm(st.chm),
        aoi_mask=st.aoi_mask,
    )
    # most layers are sparse or smooth: compressing takes ~2 GB down to a few hundred MB
    statefile.save(dict(slim=slim, arrays=arrays), path)


def save_state_opts(path: str | Path, opts: Options) -> bool:
    """Store new run options in a saved state without rewriting its arrays (what a re-pick changes).
    False for states in the older file formats, which need a full save_state. The saved usable/final layers keep
    the rules they were computed with; analyze.apply_masks rebuilds them from these options (explain_point and
    repick always do)."""
    if not statefile.is_v2(path):
        return False

    def swap(tree: dict[str, Any]) -> dict[str, Any]:
        return dict(tree, slim=dict(tree["slim"], opts=opts))

    statefile.update(path, swap)
    return True


# ---- loading -----------------------------------------------------------------------------------------------


def load_state(path: str | Path) -> ModelState:
    """Read a saved state (any file format and any state version)."""
    return migrate(statefile.load(path))


def _widen(a: Any) -> Any:
    return a.astype(np.float32) if isinstance(a, np.ndarray) and a.dtype == np.float16 else a


def _rebuild[T](cls: Callable[..., T], old: DataclassInstance) -> T:
    """A fresh dataclass instance from one unpickled from an older version: fields it had keep their values
    (nested dataclasses too), fields added since get their defaults, fields removed since are dropped."""
    vals = vars(old)
    kw: dict[str, Any] = {}
    for f in dataclasses.fields(old):  # the current class's fields (pickles refer to classes by name)
        if f.name in vals:
            v = vals[f.name]
            kw[f.name] = _rebuild(type(v), v) if dataclasses.is_dataclass(v) and not isinstance(v, type) else v
    return cls(**kw)


def state_version(tree: dict[str, Any]) -> int:
    slim = tree["slim"]
    if "version" in slim:
        return int(slim["version"])
    return 2 if "context" in tree["arrays"] else 1


def migrate(tree: dict[str, Any]) -> ModelState:
    """A ModelState from a loaded state.pkl tree of any version (see STATE_VERSION)."""
    version = state_version(tree)
    if version > STATE_VERSION:
        raise ValueError(f"state saved by a newer CougarMap (version {version}); update CougarMap to read it")
    slim = tree["slim"]
    with ThreadPoolExecutor(THREADS) as ex:  # float16 on disk -> float32 to compute with
        arrays = dict(zip(tree["arrays"], ex.map(_widen, tree["arrays"].values()), strict=True))
    raw: dict[str, Any] = {**arrays, **slim["meta"]}
    fine, mid = slim["fine"], slim["mid"]
    unmodeled = set(slim.get("unmodeled", ()))

    def fill(name: str, make: Callable[[], Any], not_modeled: bool = False) -> None:
        if name not in raw:
            raw[name] = make()
            if not_modeled:
                unmodeled.add(name)

    # same values as the code paths these states took before
    def same_as(name: str, other: str) -> None:
        fill(name, lambda: raw[other])

    for k in ("m", "s", "pred_mid"):  # before both route sets were kept there was one walk
        same_as(f"walk_any_{k}", f"walk_{k}")
    same_as("windx_mid", "drainx_mid")  # before the along-valley wind, reasons used the drainage direction
    same_as("windn_mid", "drainn_mid")
    fill("lake", lambda: np.zeros(fine.shape, bool))
    fill("closed_track_mid", lambda: np.zeros(mid.shape, bool))
    fill("meadow_label", lambda: np.zeros(fine.shape, np.int32))  # an area without openings
    fill("meadow_ha", lambda: np.zeros(1, np.float32))
    # not modeled by version 1 (neutral values: no travel line, no habitat bonus, no pavement nearby)
    fill("travel", lambda: np.zeros(fine.shape, np.float32), True)
    fill("travel_pos_mid", lambda: np.zeros(mid.shape, np.float32), True)
    fill("edge_density", lambda: np.zeros(fine.shape, np.float32), True)
    fill("water_density", lambda: np.zeros(fine.shape, np.float32), True)
    fill("context", lambda: np.ones(fine.shape, np.float32), True)
    fill("paved_dist", lambda: np.full(fine.shape, PAVED_DIST_CAP_M, np.float32), True)
    # not modeled before version 4 (its old 150 m building count meant something else): no populated area
    fill("houses", lambda: np.zeros(fine.shape, np.float32), True)
    # not modeled before version 5: no recreation site near; the travel line and downwind ends as they were
    # (every spine a full line, one blended air flow), so the old reasons apply
    fill("rec_dist", lambda: np.full(fine.shape, PAVED_DIST_CAP_M, np.float32), True)
    fill("travel_gate_mid", lambda: np.ones(mid.shape, np.float32), True)
    fill("travel_thermal_mid", lambda: np.zeros(mid.shape, np.float32), True)
    fill("edge_q_drain", lambda: np.zeros(fine.shape, np.float32), True)
    fill("edge_q_wind", lambda: np.zeros(fine.shape, np.float32), True)
    fill("season", lambda: np.ones(fine.shape, np.float32), True)  # no winter module
    fill("trail_kind", lambda: np.zeros(fine.shape, np.int8), True)  # no trail suggestions
    fill("winter_mid", lambda: np.zeros(mid.shape, np.float32), True)
    fill("winter_range_mid", lambda: np.zeros(mid.shape, bool), True)

    layers = cast("Layers", {k: v for k, v in raw.items() if k in SAVED})  # layers dropped since are ignored
    return ModelState(
        aoi=slim["aoi"],
        opts=_rebuild(Options, slim["opts"]),
        month=int(slim["month"]),
        fine=fine,
        mid=mid,
        wind=slim["wind"],
        dem_info=slim["dem_info"],
        z=slim["z"],
        chm=slim["chm"].astype(np.float32),
        aoi_mask=slim["aoi_mask"],
        layers=layers,
        notes=list(slim["notes"]),
        unmodeled=frozenset(unmodeled),
    )
