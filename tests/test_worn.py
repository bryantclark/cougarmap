"""The worn-trail detector (worn.py) on small made-up 1 m DEMs: a bench trail carved across a sidehill is found,
a gully down the fall line and plain noise are not, and a creek bank is found only until the channel is known."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import synthetic
import toys
from cougarmap import analyze, api, context, worn
from cougarmap import aoi as aoi_mod
from cougarmap.arrays import Floats, Ints, Mask
from cougarmap.config import WORN, Options, WornTrails
from cougarmap.grid import Grid
from cougarmap.sources import lidar
from cougarmap.state import load_state

N = 240  # cells (m) on a side
SLOPE = math.tan(math.radians(15))  # the sidehill falls to the south (rows grow southward)
ROW = 120  # the bench trail's uphill edge (a traverse, west-east)
COLS = slice(20, 220)  # ... 200 m long


def noise(shape: tuple[int, int], sd: float = 0.02, seed: int = 0) -> Floats:
    """Lidar-like elevation noise (m)."""
    out: Floats = np.random.default_rng(seed).normal(0, sd, shape).astype(np.float32)
    return out


def sidehill(bench: bool = True) -> Floats:
    """A 15 deg slope; with bench, a trail carved across it: a 1 m cut, a 2 m flat tread, a 1 m fill."""
    rows = np.arange(N, dtype=np.float64)[:, None] * np.ones((1, N))
    z = 1000 - SLOPE * rows
    if bench:
        d = rows - ROW
        tread = 1000 - SLOPE * (ROW + 1)
        cut = (d >= -1) & (d < 0)
        flat = (d >= 0) & (d <= 2)
        fill = (d > 2) & (d <= 3)
        on = np.zeros((N, N), bool)
        on[:, COLS] = True
        z = np.where(on & cut, z[ROW - 1] + (tread - z[ROW - 1]) * (d + 1), z)
        z = np.where(on & flat, tread, z)
        z = np.where(on & fill, tread + (1000 - SLOPE * (ROW + 3) - tread) * (d - 2), z)
    out: Floats = (z + noise((N, N))).astype(np.float32)
    return out


def gully() -> Floats:
    """The same slope with a 0.5 m deep, 6 m wide gully straight down the fall line."""
    rows, cols = np.mgrid[0:N, 0:N].astype(np.float64)
    z = 1000 - SLOPE * rows - 0.5 * np.clip(1 - np.abs(cols - N / 2) / 3, 0, None)
    out: Floats = (z + noise((N, N), seed=1)).astype(np.float32)
    return out


CREEK_ROW = 120


def creek() -> Floats:
    """A creek running west-east down a gentle valley: a 3 m flat channel floor cut 1 m into the floodplain with
    45 deg banks, the valley sides rising at 12 deg beyond 15 m."""
    rows, cols = np.mgrid[0:N, 0:N].astype(np.float64)
    d = np.abs(rows - CREEK_ROW)
    z = 1000 - 0.03 * cols - np.clip(1.0 - np.clip(d - 1.5, 0, None), 0, 1.0)  # floor 1 m down, 1:1 banks
    z += np.clip(d - 15, 0, None) * math.tan(math.radians(12))
    out: Floats = (z + noise((N, N), seed=2)).astype(np.float32)
    return out


def to_creek(rows: Ints, cols: Ints) -> tuple[Floats, Floats, Floats]:
    """Distance and vector (x = column, y = row) from cells to the creek's centre line."""
    vy = (CREEK_ROW - rows).astype(np.float64)
    vx = np.zeros_like(vy)
    return np.abs(vy), vx, vy


def everywhere() -> Mask:
    return np.ones((N, N), bool)


def test_bench_trail_is_found() -> None:
    det = worn.detect(sidehill(), everywhere())
    rr, cc = np.nonzero(det.lines)
    on = (np.abs(rr - (ROW + 1)) <= 3) & (cc >= COLS.start) & (cc < COLS.stop)
    assert on.sum() >= 0.7 * (COLS.stop - COLS.start)  # most of the 200 m trail
    assert (~on).sum() <= 10  # and next to nothing else
    assert det.creek_cells == 0


def test_plain_slope_has_no_trail() -> None:
    assert worn.detect(sidehill(bench=False), everywhere()).lines.sum() == 0


def test_gully_down_the_fall_line_is_not_a_trail() -> None:
    assert worn.detect(gully(), everywhere()).lines.sum() == 0
    # ... because of the slope tests (fall line, bench): treat the slope as flat and the gully floor shows up
    loose = WornTrails(flat_deg=90.0)
    assert worn.detect(gully(), everywhere(), p=loose).lines.sum() > 50


def test_creek_bank_is_dropped_once_the_channel_is_known() -> None:
    z = creek()
    blind = worn.detect(z, everywhere())
    rr, _ = np.nonzero(blind.lines)
    along = np.abs(rr - CREEK_ROW) <= WORN.creek_m
    assert along.sum() > 100  # without the channel the banks read as trails
    seen = worn.detect(z, everywhere(), to_creek)
    rr2, _ = np.nonzero(seen.lines)
    assert (np.abs(rr2 - CREEK_ROW) <= WORN.creek_m).sum() == 0
    assert seen.creek_cells >= along.sum() * 0.8


def test_trail_crossing_a_creek_is_kept() -> None:
    """A line perpendicular to the channel is a crossing, not a bank: only its cells on the channel go."""
    theta = np.array([6, 6, 0, 0], np.int8)  # 6 of 12 = 90 deg (north-south), 0 = east-west
    d = np.array([5.0, 0.5, 5.0, 20.0])
    vx, vy = np.zeros(4), np.array([5.0, 0.5, 5.0, 20.0])  # the channel lies due south of each cell
    assert worn.along_channel(theta, (d, vx, vy)).tolist() == [False, True, True, False]


def test_no_data_and_partial_data() -> None:
    z = sidehill()
    empty = worn.detect(np.full((N, N), np.nan, np.float32), everywhere())
    assert empty.lines.sum() == 0
    z[:, :60] = np.nan  # no lidar on the west quarter: the trail is found where there is
    det = worn.detect(z, everywhere())
    _, cc = np.nonzero(det.lines)
    assert cc.size > 80 and cc.min() >= 60 + worn.HALO


def test_only_inside_cells() -> None:
    inside = np.zeros((N, N), bool)
    inside[:, : N // 2] = True
    _, cc = np.nonzero(worn.detect(sidehill(), inside).lines)
    assert cc.size > 50 and cc.max() < N // 2


def test_kernel_points_along_its_line() -> None:
    """A trough along a direction answers most to the filter of that direction."""
    rows = np.mgrid[0:61, 0:61][0].astype(np.float64)
    z = -0.2 * np.exp(-((rows - 30) ** 2) / 2)  # a trough along a row (east-west)
    best = [float((worn.kernel(1.0, 6.0, i * math.pi / 12) * z[12:49, 12:49]).sum()) for i in range(12)]
    assert int(np.argmax(best)) == 0 and best[0] > 0
    for su in (1.0, 2.0):
        assert abs(float(worn.kernel(su, 6.0, 0.3).sum())) < 1e-5


def test_robust_spread() -> None:
    assert worn.robust_spread(np.zeros(10, np.float32)) == 1.0  # no spread: no scaling
    assert worn.robust_spread(np.array([np.nan, -np.inf], np.float32)) == 1.0
    v = np.random.default_rng(0).normal(0, 2.0, 100_000).astype(np.float32)
    assert worn.robust_spread(v) == pytest.approx(2.0, rel=0.03)


def test_vectorize_and_keep_long() -> None:
    g = Grid(32611, 500_000.0, 5_000_000.0, 1.0, 100, 100)
    sk = np.zeros((100, 100), bool)
    sk[10, 10:70] = True  # 60 m east-west
    for i in range(45):  # 45 cells diagonal
        sk[40 + i, 20 + i] = True
    sk[80, 5:15] = True  # 10 m: too short to keep
    long = worn.keep_long(sk, 40)
    assert long.sum() == 60 + 45 and not long[80, 5:15].any()
    lines = worn.vectorize(long, g)
    lengths = sorted(round(ln.length) for ln in lines)
    assert lengths == [59, 62]  # 59 square steps, 44 diagonal ones
    assert all(len(ln.coords) == 2 for ln in lines)  # straight lines simplify to their ends
    assert worn.vectorize(np.zeros((5, 5), bool), g) == []
    assert not worn.keep_long(np.zeros((5, 5), bool), 3).any()


def test_fine_cells() -> None:
    one = Grid(32611, 500_000.0, 5_000_000.0, 1.0, 30, 30)
    fine = Grid(32611, 499_997.0, 5_000_003.0, 3.0, 12, 12)
    r, c, ok = worn.fine_cells(np.array([0, 29]), np.array([0, 29]), one, fine)
    assert r.tolist() == [1, 10] and c.tolist() == [1, 10] and ok.all()


# ---- in the model: the layer, the hint, and no change to any score ----------------------------------------------


def _u_span(ln: worn.WornLine) -> float:
    """How far a line strays from the synthetic benched track's u (m)."""
    us = [float(synthetic._FWD.transform(lon, lat)[0] - synthetic.CX) for lon, lat in ln["lonlat"]]
    return max(abs(u - synthetic.TRAIL_U) for u in us)


def _without_hint(spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in c.items() if k != "worn_trail"} for c in spots]


def test_worn_trails_change_no_score_or_pick(analyzed: dict[str, Any]) -> None:
    """The synthetic area again as a normal run, worn trails on: the same spots and scores as conftest.analyzed (a fast
    run, without them),
    plus the layer, the hints and a summary of them."""
    with synthetic.offline():
        r = api.analyze_area(bbox=synthetic.bbox(), month=10, area_name="worn on", log=lambda *_: None)
    assert _without_hint(r["candidates"]) == _without_hint(analyzed["candidates"])
    assert _without_hint(r["private_candidates"]) == _without_hint(analyzed["private_candidates"])
    off = analyzed["summary"]["worn_trails"]
    assert off["on"] is False and "--fast" in off["how"]
    assert all(c["worn_trail"] is None for c in analyzed["candidates"])
    s = r["summary"]["worn_trails"]
    assert s["on"] is True and 0.6 < s["km"] < 2.5 and 0.6 < s["unmapped_km"] <= s["km"]
    hints = [c["worn_trail"] for c in r["candidates"] + r["private_candidates"] if c["worn_trail"]]
    assert s["spots_with_hint"] == sum(c["worn_trail"] is not None for c in r["candidates"])
    assert hints and all(h["distance_m"] <= WORN.hint_m and "hang the camera" in h["reason"] for h in hints)
    kmz = Path(r["summary"]["outputs"]["kmz"])
    with zipfile.ZipFile(kmz) as z:
        doc = z.read("doc.kml").decode()
    assert "<name>Worn trails (lidar)</name><visibility>0</visibility>" in doc
    assert "#worn_unmapped" in doc and "Placement hint: a worn line on no map" in doc
    st = load_state(kmz.parent / "state.pkl")
    assert "worn_lines" not in st.unmodeled
    for h in hints:  # each points at a line on no map
        cell = st.fine.cell(h["lon"], h["lat"])
        assert cell is not None and st.layers["worn_unmapped"][cell]
    # the benched track is found end to end, on no map except where the mapped path and gated road cross it
    on_track = [ln for ln in st.layers["worn_lines"] if _u_span(ln) < 3]
    assert sum(ln["length_m"] for ln in on_track) > 0.9 * (synthetic.TRAIL_V[1] - synthetic.TRAIL_V[0])
    assert 0 < sum(ln["length_m"] for ln in on_track if ln["mapped"]) < 120
    # a re-pick keeps the hints without recomputing
    again = api.repick(kmz.parent.name)
    assert [c["worn_trail"] for c in again["candidates"]] == [c["worn_trail"] for c in r["candidates"]]
    api._STATES.clear()


def test_worn_hint_distances_and_wording() -> None:
    st = toys.state()  # 5 m cells
    st.layers["worn_unmapped"][100, 104] = True  # 20 m east of cell (100, 100)
    near = analyze.describe(st, 100, 100)["worn_trail"]
    assert near is not None and near["distance_m"] == 20 and near["direction"] == "E"
    assert "20 m E: hang the camera a few metres toward it" in near["reason"]
    close = analyze.describe(st, 100, 102)["worn_trail"]
    assert close is not None and close["reason"].endswith("10 m E: hang the camera facing it")
    on = analyze.describe(st, 100, 104)["worn_trail"]
    assert on is not None and "runs through this spot" in on["reason"]
    assert analyze.describe(st, 100, 90)["worn_trail"] is None  # 70 m: out of reach
    assert analyze.describe(st, 100, 97)["worn_trail"] is None  # 35 m: just out


def test_worn_notes() -> None:
    a = aoi_mod.circle(47.37, -116.10, 1.0)
    on, off = Options(), Options(worn_trails=False)
    g = lidar.lidar_grid(a.geom, 32611, 10)
    z = np.zeros((2, 2), np.float32)
    assert context._worn_notes(a, off, None) == []
    assert "could not be downloaded" in context._worn_notes(a, on, None)[0]
    assert "no 1 m lidar here" in context._worn_notes(a, on, lidar.Lidar1m(g, z, 0.0, []))[0]
    assert "(50% of the area)" in context._worn_notes(a, on, lidar.Lidar1m(g, z, 0.5, ["t"]))[0]
    assert context._worn_notes(a, on, lidar.Lidar1m(g, z, 1.0, ["t"])) == []
    big = aoi_mod.circle(47.37, -116.10, 5.0)
    assert "over 60 km2" in context._worn_notes(big, on, None)[0]


# ---- the 1 m lidar source -----------------------------------------------------------------------------------------


def _tile(path: Path, x0: float, y0: float, n: int, value: float) -> str:
    g = Grid(32611, x0, y0, 1.0, n, n)
    g.write_tif(path, np.full(g.shape, value, np.float32))
    return str(path)


def test_lidar_source_merges_tiles_newest_first_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid = Grid(32611, 500_000.0, 5_000_100.0, 1.0, 100, 100)
    new = _tile(tmp_path / "new.tif", 500_000.0, 5_000_100.0, 60, 2.0)  # the north-west 60 x 60 m
    old = _tile(tmp_path / "old.tif", 500_040.0, 5_000_100.0, 60, 1.0)  # overlaps it, further east
    calls: list[tuple[Any, ...]] = []
    tiles = [
        dict(title="new", url=new, date="2020"),
        dict(title="old", url=old, date="2010"),
        dict(title="gone", url=str(tmp_path / "missing.tif"), date="2000"),
    ]
    monkeypatch.setattr(lidar, "lidar_tiles", toys.recorder(calls, tiles))
    got = lidar.fetch_lidar_1m(grid)
    assert got.sources == ["new", "old"] and got.coverage == pytest.approx(60 * 100 / 100**2)
    assert got.z[0, 0] == 2.0 and got.z[0, 50] == 2.0 and got.z[0, 80] == 1.0 and np.isnan(got.z[80, 0])
    again = lidar.fetch_lidar_1m(grid)  # from the cache: no new query
    assert len(calls) == 1 and np.array_equal(again.z, got.z, equal_nan=True) and again.sources == got.sources
    elsewhere = Grid(32611, 600_000.0, 5_000_100.0, 1.0, 10, 10)
    monkeypatch.setattr(lidar, "lidar_tiles", toys.recorder(calls, []))
    none = lidar.fetch_lidar_1m(elsewhere)
    assert none.coverage == 0 and none.sources == []
    assert lidar.sub_grid(grid, (0.0, 0.0, 1.0, 1.0)) is None
