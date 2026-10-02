"""The interactive weights page (`explore.html`, opt-in: `--interactive`): one self-contained local HTML file next
to the KMZ where a weight slider per factor moves the top camera spots live.

The page recomputes the camera-spot score in the browser (explore.js, a line-for-line port of analyze.combine,
the site penalties and pick_candidates' peak picking) from the per-cell inputs the weights act on, embedded here
on a coarser grid:

- the five weighted layers (wind, edges, pinch, water, travel), block means of the fine grid, uint8;
- the habitat x season multiplier (analyze.combine's `habitat * season`), which no slider changes, uint16;
- the site penalties (paved road, houses, recreation) as multipliers, uint8, so each can be switched off;
- the usable ground (public land within the walk limit, the main spot list's rules): a block is usable when at
  least half its fine cells are.

Each layer is deflate-compressed and base64-encoded. Cells are about EXPLORE_RES_M (a whole number of fine
cells), coarser when the area would pass EXPLORE_MAX_CELLS. Coarse cell centres map to lon/lat (and back, for the
heat overlay) by quadratic fits valid to well under a metre over an area this size; `geo.max_error_m` records it.
"""

from __future__ import annotations

import base64
import json
import math
import zlib
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from shapely.geometry import MultiPolygon, Polygon

from .analyze import PEAK_FLOOR, PEAK_MIN, PEAK_REL, Spot, site_penalty_parts
from .arrays import Floats, Mask
from .grid import Grid
from .state import ModelState

EXPLORE_RES_M = 10.0  # target page cell size: the camera zone (20 m sigma) spans two cells
EXPLORE_MAX_CELLS = 1_000_000  # ~11 MB of raw layers before compression; bigger areas get coarser cells
FILE_NAME = "explore.html"
WEIGHTED = ("wind", "edges", "pinch", "water", "travel")
PENALTIES = ("paved", "houses", "recreation")

JSON = dict[str, Any]


def block_factor(fine: Grid) -> int:
    """How many fine cells on a side one page cell spans."""
    f = max(1, round(EXPLORE_RES_M / fine.res))
    while math.ceil(fine.height / f) * math.ceil(fine.width / f) > EXPLORE_MAX_CELLS:
        f += 1
    return f


def block_mean(a: npt.ArrayLike, f: int) -> Floats:
    """Mean over f x f blocks (the last row/column of blocks padded with the edge values)."""
    a = np.asarray(a, dtype="float64")
    if f == 1:
        return a.astype("float32")
    H, W = a.shape
    p = np.pad(a, ((0, (-H) % f), (0, (-W) % f)), mode="edge")
    out: Floats = p.reshape(p.shape[0] // f, f, p.shape[1] // f, f).mean(axis=(1, 3)).astype("float32")
    return out


def quantize(a: Floats, dtype: str) -> tuple[npt.NDArray[Any], float]:
    """a (>= 0) as unsigned integers of dtype and the scale that turns them back (value = q x scale)."""
    top = float(np.iinfo(dtype).max)
    hi = float(np.nanmax(a)) if a.size else 0.0
    scale = hi / top if hi > 0 else 1.0
    q = np.rint(np.clip(np.nan_to_num(a), 0, None) / scale).clip(0, top).astype(dtype)
    return q, scale


def encode(q: npt.NDArray[Any]) -> str:
    """Little-endian bytes, deflated (zlib), base64: what the page's DecompressionStream('deflate') reads."""
    return base64.b64encode(zlib.compress(q.astype(q.dtype.newbyteorder("<")).tobytes(), 9)).decode("ascii")


def decode(layer: JSON) -> Floats:
    """A payload layer back to float32 values (tests and checks)."""
    raw = zlib.decompress(base64.b64decode(layer["data"]))
    q = np.frombuffer(raw, dtype=np.dtype(layer["dtype"]).newbyteorder("<"))
    out: Floats = (q.astype("float64") * layer["scale"]).astype("float32").reshape(layer["shape"])
    return out


def _layer(a: Floats, dtype: str) -> JSON:
    q, scale = quantize(a, dtype)
    return dict(dtype=dtype, scale=scale, shape=list(a.shape), data=encode(q))


def _terms(u: Floats, v: Floats) -> Floats:
    out: Floats = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)
    return out


def georef(fine: Grid, f: int, shape: tuple[int, int]) -> JSON:
    """Quadratic fits between page cells and lon/lat. Page cell (row, col) has its centre at fine-grid
    x0 + (col + 0.5) x f x res, y0 - (row + 0.5) x f x res. Normalized coordinates u = col / W - 0.5,
    v = row / H - 0.5 (and the same of lon/lat over the grid's bounds) keep the fit well conditioned.
    fwd: [lon coefs, lat coefs] of (1, u, v, u2, uv, v2); inv: [col coefs, row coefs] of the same terms of the
    normalized lon/lat."""
    H, W = shape
    cell = fine.res * f
    rr, cc = np.meshgrid(np.linspace(-0.5, H - 0.5, 25), np.linspace(-0.5, W - 0.5, 25), indexing="ij")
    x = fine.x0 + (cc + 0.5) * cell
    y = fine.y0 - (rr + 0.5) * cell
    lon, lat = fine.to_lonlat(x.ravel(), y.ravel())
    lon, lat = np.asarray(lon), np.asarray(lat)
    u, v = cc.ravel() / W - 0.5, rr.ravel() / H - 0.5
    X = _terms(u, v)
    fwd = [np.linalg.lstsq(X, t, rcond=None)[0] for t in (lon, lat)]
    west, east, south, north = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    lu, lv = (lon - west) / (east - west) - 0.5, (north - lat) / (north - south) - 0.5
    L = _terms(lu, lv)
    inv = [np.linalg.lstsq(L, t, rcond=None)[0] for t in (cc.ravel(), rr.ravel())]
    # the fits' worst error, in metres, over the sample points (forward) and in page cells (inverse)
    fx, fy = fine.from_lonlat(X @ fwd[0], X @ fwd[1])
    err_fwd = float(np.max(np.hypot(np.asarray(fx) - x.ravel(), np.asarray(fy) - y.ravel())))
    err_inv = float(np.max(np.hypot(L @ inv[0] - cc.ravel(), L @ inv[1] - rr.ravel()))) * cell
    return dict(
        fwd=[c.tolist() for c in fwd],
        inv=[c.tolist() for c in inv],
        bounds=[west, south, east, north],  # lon/lat box of the sample points (the inverse fit's normalization)
        max_error_m=round(max(err_fwd, err_inv), 3),
    )


def cell_lonlat(geo: JSON, shape: tuple[int, int], row: float, col: float) -> tuple[float, float]:
    """(lon, lat) of a page cell centre from the payload's forward fit (what the page does)."""
    H, W = shape
    t = _terms(np.array([col / W - 0.5]), np.array([row / H - 0.5]))[0]
    return float(t @ np.array(geo["fwd"][0])), float(t @ np.array(geo["fwd"][1]))


def _rings(geom: Any) -> list[list[list[float]]]:
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom] if isinstance(geom, Polygon) else []
    return [[[round(y, 6), round(x, 6)] for x, y in p.exterior.coords] for p in polys]


def _spot(c: Spot) -> JSON:
    return dict(
        rank=c.get("rank"),
        name=c.get("name"),
        lat=c["lat"],
        lon=c["lon"],
        score=c["score"],
        factors=c["factors"],
        reasons=c["reasons"][:4],
    )


def payload(st: ModelState, cands: list[Spot]) -> JSON:
    """Everything the page needs: the coarse layers, the model's weights and picking rules, the georeference,
    the area outline, the user's water/sign pins and the model's own spots (the faint ghosts). st must have its
    masks applied (analyze.apply_masks)."""
    A, o = st.layers, st.opts
    w, h = o.weights, o.weights.habitat
    f = block_factor(st.fine)
    layers: dict[str, JSON] = {k: _layer(block_mean(A[k], f), "uint8") for k in WEIGHTED}  # type: ignore[literal-required]
    layers["habitat"] = _layer(block_mean(A["context"] * A["season"], f), "uint16")
    parts = site_penalty_parts(A, o)
    for name in PENALTIES:
        pen = np.ones(st.fine.shape, "float32")
        for _, part in (p for p in parts if p[0] == name):
            pen *= part
        layers[name] = _layer(block_mean(pen, f), "uint8")
    usable: Mask = block_mean(A["usable"], f) >= 0.5
    layers["usable"] = dict(dtype="uint8", scale=1.0, shape=list(usable.shape), data=encode(usable.astype("uint8")))
    shape = (int(usable.shape[0]), int(usable.shape[1]))
    pins = [
        dict(name=p["name"], kind=p["kind"], lat=round(p["lat"], 6), lon=round(p["lon"], 6))
        for p in o.user_points or st.aoi.user_points
        if p["kind"] in ("water", "seasonal_water", "sign")
    ]
    return dict(
        version=1,
        area=st.aoi.name,
        month=st.month,
        grid=dict(height=shape[0], width=shape[1], res=st.fine.res * f, fine_res=st.fine.res, block=f),
        geo=georef(st.fine, f, shape),
        weights=dict(
            wind=w.wind,
            edges=w.edges,
            pinch=w.pinch,
            water=w.water,
            travel=w.travel,
            stack_multiplier=w.stack_multiplier,
            stack_from=w.stack_from,
            stack_to=w.stack_to,
        ),
        habitat=dict(zone_m=h.zone_m, score_scale=h.score_scale, edge_floor=h.edge_floor, water_floor=h.water_floor),
        pick=dict(
            n=o.n_candidates,
            spacing_m=o.candidate_spacing_m,
            per_zone=o.per_zone,
            zone_radius_m=o.zone_radius_m,
            peak_min=PEAK_MIN,
            peak_rel=PEAK_REL,
            peak_floor=PEAK_FLOOR,
        ),
        layers=layers,
        outline=_rings(st.aoi.geom),
        pins=pins,
        model_spots=[_spot(c) for c in cands],
    )


def kernel_js() -> str:
    """The page's scoring and picking code (also run under node by the tests)."""
    return files("cougarmap").joinpath("explore.js").read_text()


def render(data: JSON) -> str:
    """The page: the template with the kernel and the payload inlined (one file, no other local files)."""
    page = files("cougarmap").joinpath("explore_page.html").read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return page.replace("/*__KERNEL__*/", kernel_js()).replace("/*__PAYLOAD__*/null", blob)


def write_page(st: ModelState, cands: list[Spot], out_dir: Path) -> Path:
    """Write explore.html for an analyzed area into its output folder and return its path."""
    path = Path(out_dir) / FILE_NAME
    path.write_text(render(payload(st, cands)), encoding="utf-8")
    return path
