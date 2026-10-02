"""The interactive weights page (explore.py, explore.js): its payload, georeference, kernel parity with the Python
model (under node, when node is installed) and how each surface turns it on."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import synthetic
from cougarmap import api, context, explore, net
from cougarmap import terrain as T
from cougarmap.analyze import apply_masks, pick_all, ramp
from cougarmap.arrays import Floats
from cougarmap.cli import app
from cougarmap.grid import Grid
from cougarmap.state import ModelState, load_state

NODE = shutil.which("node")
RUNNER = Path(__file__).with_name("explore_run.mjs")
KERNEL = Path(__file__).parents[1] / "explore-app" / "src" / "kernel.js"


@pytest.fixture(scope="module")
def state(analyzed: dict[str, Any]) -> ModelState:
    st = load_state(analyzed["dir"] / "state.pkl")
    apply_masks(st)
    return st


@pytest.fixture(scope="module")
def data(state: ModelState, analyzed: dict[str, Any]) -> dict[str, Any]:
    return explore.payload(state, *pick_all(state))


def test_a_normal_run_writes_no_page(analyzed: dict[str, Any]) -> None:
    assert "explore" not in analyzed["summary"]["outputs"] and not (analyzed["dir"] / explore.FILE_NAME).exists()


def test_payload_layers_are_the_coarse_inputs(state: ModelState, data: dict[str, Any]) -> None:
    g, f = data["grid"], data["grid"]["block"]
    assert f == max(1, round(explore.EXPLORE_RES_M / state.fine.res)) and g["res"] == state.fine.res * f
    shape = [math.ceil(state.fine.height / f), math.ceil(state.fine.width / f)]
    assert [g["height"], g["width"]] == shape
    extra = set(explore.map_layers(state))
    assert {"edge_meadow", "public", "private", "reach"} <= extra
    assert set(data["layers"]) == {*explore.WEIGHTED, "habitat", *explore.PENALTIES, "usable", "usable_private", *extra}
    A = state.layers
    for k in explore.WEIGHTED:
        got, want = explore.decode(data["layers"][k]), explore.block_mean(A[k], f)  # type: ignore[literal-required]
        assert list(got.shape) == shape and data["layers"][k]["dtype"] == "uint8"
        assert np.abs(got - want).max() <= data["layers"][k]["scale"] / 2 + 1e-6  # quantization: half a step
    hab = explore.decode(data["layers"]["habitat"])
    assert (
        np.abs(hab - explore.block_mean(A["context"] * A["season"], f)).max()
        <= data["layers"]["habitat"]["scale"] / 2 + 1e-6
    )
    pen = (
        explore.decode(data["layers"]["paved"])
        * explore.decode(data["layers"]["houses"])
        * explore.decode(data["layers"]["recreation"])
    )
    assert float(pen.min()) >= 0 and float(pen.max()) <= 1.0 + 1e-6 and float(pen.min()) < 1  # roads and houses
    usable = explore.decode(data["layers"]["usable"])
    assert set(np.unique(usable)) <= {0.0, 1.0} and usable.any()
    assert data["pick"]["n"] == state.opts.n_candidates and data["weights"]["wind"] == state.opts.weights.wind
    pub, priv = pick_all(state)
    assert [s["name"] for s in data["model_spots"]] == [c["name"] for c in pub]
    assert [s["name"] for s in data["model_private_spots"]] == [c["name"] for c in priv]
    feats = data["features"]
    assert set(feats) == {"worn", "routes", "airflow", "saddles"} and feats["airflow"]
    assert all(len(pt) == 2 for r in feats["routes"] for pt in r["coords"])
    assert data["outline"] and all(len(p) == 2 for p in data["outline"][0])


def test_block_factor_caps_the_page_size() -> None:
    small = Grid(32611, 500_000.0, 5_300_000.0, 3.0, 2000, 2000)
    assert explore.block_factor(small) == 3
    big = Grid(32611, 500_000.0, 5_300_000.0, 3.0, 8000, 8000)
    f = explore.block_factor(big)
    assert math.ceil(8000 / f) ** 2 <= explore.EXPLORE_MAX_CELLS < math.ceil(8000 / (f - 1)) ** 2
    # the last block repeats the edge column
    assert explore.block_mean(np.arange(10.0).reshape(2, 5), 2).tolist() == [[3.0, 5.0, 6.5]]


@pytest.mark.parametrize("x0", [320_000.0, 500_000.0, 690_000.0])  # far west in the zone, central, far east
def test_georeference_is_good_to_a_metre(x0: float) -> None:
    """A ~6 km area: page cells to lon/lat and back, with the zone's grid convergence included."""
    fine = Grid(32611, x0, 5_310_000.0, 3.0, 2100, 2100)
    f = 3
    shape = (700, 700)
    geo = explore.georef(fine, f, shape)
    assert geo["max_error_m"] < 1.0
    for row, col in [(0, 0), (350, 120), (699, 699), (10, 650)]:
        x, y = fine.x0 + (col + 0.5) * f * fine.res, fine.y0 - (row + 0.5) * f * fine.res
        lon, lat = explore.cell_lonlat(geo, shape, row, col)
        bx, by = fine.from_lonlat(lon, lat)
        assert math.hypot(float(bx) - x, float(by) - y) < 1.0


def test_page_inlines_the_payload(data: dict[str, Any]) -> None:
    html = explore.render(data)
    assert explore.DATA_MARK not in html
    start = html.index('type="application/json">') + len('type="application/json">')
    blob = html[start : html.index("</script>", start)]
    assert "<" not in blob and json.loads(blob)["grid"] == data["grid"]  # "<" escaped: inert inside the tag


def test_app_html_comes_from_the_override_a_local_build_the_cache_or_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = f"<html>{explore.DATA_MARK}</html>"
    monkeypatch.delenv("COUGARMAP_EXPLORE_APP")
    monkeypatch.setattr(explore, "PROJECT_ROOT", tmp_path / "src")
    monkeypatch.setattr(explore, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(explore, "version", lambda _: "9.9.9")
    calls: list[str] = []

    class Resp:
        text = app

    def fake_get(url: str, **_: Any) -> Resp:
        calls.append(url)
        return Resp()

    monkeypatch.setattr(net, "get", fake_get)
    logs: list[str] = []
    assert explore.app_html(logs.append) == app  # downloaded once, from this version's release
    assert calls == ["https://github.com/bryantclark/cougarmap/releases/download/v9.9.9/cougarmap-explore.html"]
    assert logs and "first --interactive run" in logs[0]
    assert explore.app_html(logs.append) == app and len(calls) == 1  # then from the cache
    built = tmp_path / "src" / "explore-app" / "dist" / "index.html"
    built.parent.mkdir(parents=True)
    built.write_text("local " + app)
    assert explore.app_html().startswith("local")  # a source checkout's own build comes first
    monkeypatch.setenv("COUGARMAP_EXPLORE_APP", str(built))
    assert explore.app_html().startswith("local")


def test_app_download_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COUGARMAP_EXPLORE_APP")
    monkeypatch.setattr(explore, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(explore, "CACHE_DIR", tmp_path / "cache")

    def offline(url: str, **_: Any) -> Any:
        raise RuntimeError("request failed")

    monkeypatch.setattr(net, "get", offline)
    with pytest.raises(RuntimeError, match="could not download the weights app"):
        explore.app_html(lambda *_: None)

    class NotTheApp:
        text = "<html>a 404 page</html>"

    monkeypatch.setattr(net, "get", lambda url, **_: NotTheApp())
    with pytest.raises(RuntimeError, match="is not the weights app"):
        explore.app_html(lambda *_: None)
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").rglob("*.html"))
    monkeypatch.setenv("COUGARMAP_EXPLORE_APP", str(tmp_path / "broken.html"))
    (tmp_path / "broken.html").write_text("<html>no marker</html>")
    with pytest.raises(RuntimeError, match="no single place"):
        explore.render({"x": 1})


# ---- the kernel under node ----------------------------------------------------------------------------------


def _node(data: dict[str, Any], tmp_path: Path, **settings: Any) -> dict[str, Any]:
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    out = subprocess.run(
        ["node", str(RUNNER), str(KERNEL), str(tmp_path / "data.json"), str(tmp_path / "settings.json")],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result: dict[str, Any] = json.loads(out.stdout)
    return result


def _reference_final(data: dict[str, Any], w: dict[str, float], on: dict[str, Any], usable: str = "usable") -> Floats:
    """The page's score in numpy, from the decoded payload: analyze.combine and the site penalties on the page
    grid (the kernel must agree with this to float32 rounding)."""
    L = {k: explore.decode(v) for k, v in data["layers"].items()}
    h, res = data["habitat"], data["grid"]["res"]
    spot = sum(w[k] * L[k] for k in explore.WEIGHTED)
    for k in ("wind", "edges", "pinch", "water"):
        spot = spot * (1 + (w["stack_multiplier"] - 1) * ramp(L[k], w["stack_from"], w["stack_to"]))
    s = T.gaussian((spot * L["habitat"]).astype("float32"), h["zone_m"] / res)
    top = sum(w[k] for k in explore.WEIGHTED) * w["stack_multiplier"] ** 4 * (1 + h["edge_floor"])
    top *= 1 + h["water_floor"]
    score = np.clip(100 * h["score_scale"] * s / top, 0, 100)
    for p in explore.PENALTIES:
        strength = 1.0 if on[p] is True else 0.0 if on[p] is False else float(on[p])
        score = score * L[p] ** strength  # a penalty slider: the model's multiplier to its strength
    out: Floats = np.where(L[usable] > 0, score, 0).astype("float32")
    return out


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_kernel_at_the_model_weights_finds_the_model_spots(
    state: ModelState, data: dict[str, Any], tmp_path: Path
) -> None:
    on = dict(paved=True, houses=True, recreation=True)
    r = _node(data, tmp_path, final=True)
    final = np.array(r["final"], "float32").reshape(data["grid"]["height"], data["grid"]["width"])
    assert np.abs(final - _reference_final(data, data["weights"], on)).max() < 1e-3
    assert r["ms"] < 1000
    py = data["model_spots"]
    js = r["spots"]
    assert js and abs(len(js) - len(py)) <= 1

    def dist(a: dict[str, Any], b: dict[str, Any]) -> float:
        ax, ay = state.fine.from_lonlat(a["lon"], a["lat"])
        bx, by = state.fine.from_lonlat(b["lon"], b["lat"])
        return math.hypot(float(ax) - float(bx), float(ay) - float(by))

    near = [min(dist(p, j) for j in js) for p in py]
    assert sum(d <= 30 for d in near) >= 0.8 * len(py), near
    assert abs(js[0]["score"] - py[0]["score"]) < 0.1 * py[0]["score"] + 1


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_kernel_follows_the_sliders(data: dict[str, Any], tmp_path: Path) -> None:
    w = dict(data["weights"], wind=0.0, water=0.6, stack_multiplier=1.6)
    on = dict(paved=False, houses=True, recreation=False)
    r = _node(data, tmp_path, weights=w, on=on, n=5, final=True)
    final = np.array(r["final"], "float32").reshape(data["grid"]["height"], data["grid"]["width"])
    assert np.abs(final - _reference_final(data, w, on)).max() < 1e-3
    assert 0 < len(r["spots"]) <= 5 and [s["rank"] for s in r["spots"]] == list(range(1, len(r["spots"]) + 1))
    scores = [s["score"] for s in r["spots"]]
    assert scores == sorted(scores, reverse=True)
    off = _node(data, tmp_path, weights=dict.fromkeys(explore.WEIGHTED, 0.0))
    assert off["spots"] == []  # every weight at 0: nothing scores


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_only_the_weight_ratios_matter(data: dict[str, Any], tmp_path: Path) -> None:
    """The page's sliders are shares that add up to 100%: the model's score divides by the sum of the weights, so
    scaling every weight changes nothing."""
    share = {k: data["weights"][k] / sum(data["weights"][k] for k in explore.WEIGHTED) for k in explore.WEIGHTED}
    a = _node(data, tmp_path, final=True)
    b = _node(data, tmp_path, weights=dict(data["weights"], **share), final=True)
    assert np.abs(np.array(a["final"]) - np.array(b["final"])).max() < 1e-3 and a["spots"] == b["spots"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_penalty_strengths_and_the_private_list(state: ModelState, data: dict[str, Any], tmp_path: Path) -> None:
    on = dict(paved=0.5, houses=2.0, recreation=0.0)
    r = _node(data, tmp_path, on=on, final=True)
    final = np.array(r["final"], "float32").reshape(data["grid"]["height"], data["grid"]["width"])
    assert np.abs(final - _reference_final(data, data["weights"], on)).max() < 1e-3
    # at the model's settings, the private list is the model's P spots
    py = data["model_private_spots"]
    js = _node(data, tmp_path)["private_spots"]
    assert len(js) == len(py)
    for p, j in zip(py, js, strict=True):
        ax, ay = state.fine.from_lonlat(p["lon"], p["lat"])
        bx, by = state.fine.from_lonlat(j["lon"], j["lat"])
        assert math.hypot(float(ax) - float(bx), float(ay) - float(by)) <= 30


# ---- the surfaces -------------------------------------------------------------------------------------------


def test_repick_interactive_writes_the_page(area: str) -> None:
    r = api.repick(area, interactive=True)
    page = Path(r["summary"]["outputs"]["explore"])
    assert page.name == explore.FILE_NAME and page.parent == Path(r["summary"]["outputs"]["kmz"]).parent
    assert page.stat().st_size < 10_000_000 and explore.DATA_MARK not in page.read_text()
    assert "explore" not in api.repick(area)["summary"]["outputs"]


def test_cli_interactive_flags(area: str, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    for cmd in ("analyze", "hotspots", "repick"):
        assert "--interactive" in runner.invoke(app, [cmd, "--help"]).output
    res = runner.invoke(app, ["repick", area, "--interactive", "--no-open"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["summary"]["outputs"]["explore"].endswith(explore.FILE_NAME)
    assert "weights page: file://" in res.stderr


def test_cli_opens_the_page_at_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    from cougarmap import cli

    opened: list[str] = []

    def fake_open(path: str) -> dict[str, Any]:
        opened.append(path)
        return {}

    monkeypatch.setattr(api, "open_file", fake_open)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    cli._map(dict(kmz="/x/hotspots.kmz", explore=["/x/b1/explore.html", "/x/b2/explore.html"]), True)
    assert opened == ["/x/hotspots.kmz", "/x/b1/explore.html"]
    cli._explore(dict(summary=dict(outputs=dict(kmz="/x/a.kmz"))), True)
    assert len(opened) == 2


def test_analyze_and_hotspots_interactive(tmp_path: Path) -> None:
    """A direct find_hotspots run (a small circle) passes interactive through analyze_area and lists the page."""
    with synthetic.offline():
        r = api.find_hotspots(
            f"{synthetic.LAT},{synthetic.LON}",
            radius_km=0.5,
            area_name="explore test",
            month=10,
            log=lambda *_: None,
            fast=True,
            interactive=True,
        )
    page = Path(r["summary"]["outputs"]["explore"])
    assert r["explore"] == [str(page)] and page.exists() and page.parent.name == "explore-test"
    shutil.rmtree(page.parent)


def test_scouted_hotspots_list_each_blocks_page(analyzed: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [dict(score=80.0 - i, land=f"Forest {i}", why=[], bbox=synthetic.bbox(), center={}) for i in (1, 2)]
    monkeypatch.setattr(
        api, "scout_region", lambda *a, **k: dict(location="Testville, WA", month=10, wind={}, blocks=blocks)
    )
    monkeypatch.setattr(context, "prefetch", lambda *a, **k: None)
    seen: list[dict[str, Any]] = []

    def fake_analyze(**k: Any) -> dict[str, Any]:
        seen.append(k)
        i = len(seen)
        summary = dict(analyzed["summary"], outputs=dict(analyzed["summary"]["outputs"], explore=f"/b{i}/explore.html"))
        return dict(summary=summary, candidates=analyzed["candidates"], private_candidates=[])

    monkeypatch.setattr(api, "analyze_area", fake_analyze)
    r = api.find_hotspots("Testville", radius_km=20, blocks=2, log=lambda *_: None, interactive=True)
    assert all(k["interactive"] for k in seen)
    assert r["explore"] == ["/b1/explore.html", "/b2/explore.html"]
    assert [b["explore"] for b in r["blocks"]] == r["explore"]
