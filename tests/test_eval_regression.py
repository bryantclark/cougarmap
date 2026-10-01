"""The pick-level evaluation (cougarmap.evaluate): its ranking on synthetic scores and on the analyzed synthetic
area, and its command line."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import synthetic
from cougarmap import evaluate as ev

RES = 5.0


def _peaks(n: int = 400) -> np.ndarray:
    r, c = np.mgrid[0:n, 0:n]
    s = np.zeros((n, n), "float32")
    for i, (a, b) in enumerate([(50, 50), (50, 300), (200, 200), (350, 80), (330, 330)]):
        s += (100 - 15 * i) * np.exp(-((r - a) ** 2 + (c - b) ** 2) / (2 * 8.0**2))
    return s


def test_rank_of_the_best_and_a_flat_spot() -> None:
    spots = ev.Spots.from_score(_peaks(), RES)
    assert len(spots.order) == 5
    best, beaten = spots.rank(50, 50)
    assert best == pytest.approx(1 / 5) and beaten == 0
    flat, _ = spots.rank(120, 330)  # nowhere near a peak
    assert flat == 1.0
    third, beaten = spots.rank(200, 200)
    assert third == pytest.approx(3 / 5) and beaten == 1  # the 2nd peak is 900 m away, the 1st 1.06 km


def test_top_k_distance_and_vs_random() -> None:
    spots = ev.Spots.from_score(_peaks(), RES)
    d = spots.top_k_distance(np.array([[50, 50], [50, 310], [350, 80]]), k=2)
    assert d[0] == 0 and d[1] == pytest.approx(10 * RES) and d[2] > 1000
    assert np.isinf(spots.top_k_distance(np.zeros((1, 2), int), k=0)).all()
    assert ev.vs_random(np.array([0.1, 0.2]), np.array([0.5, 0.6, 0.1])) == pytest.approx((5 / 6 + 2 / 3) / 2)


def test_evaluate_on_the_synthetic_area(
    analyzed: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole evaluation, with the analyzed synthetic area as the only area and two camera pins: one on the
    best spot the model picked, one far from any."""
    best = analyzed["candidates"][0]
    far = synthetic.lonlat(-560, 560)
    pins = "".join(
        f"<Placemark><name>{name}</name><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
        for name, lon, lat in (("Cam01", best["lon"], best["lat"]), ("Cam02", *far))
    )
    kml = tmp_path / "cams.kml"
    kml.write_text(f'<kml xmlns="http://www.opengis.net/kml/2.2"><Document>{pins}</Document></kml>')
    area = analyzed["dir"]
    out = ev.evaluate(areas={"syn": area.name}, states_dir=area.parent, kml=kml, n_random=50)
    o, a = out["overall"], out["areas"]["syn"]
    assert a["n_cams"] == 2 and out["cams"]["Cam01"]["rank_frac"] < out["cams"]["Cam02"]["rank_frac"]
    assert out["cams"]["Cam01"]["rank_frac"] <= 0.5 and 0 <= o["vs_random"] <= 1
    assert out["fixed_k"]["cams_within"]["150"] >= 0.5 and out["fixed_k"]["check_area_within"] == {}
    assert "OVERALL  median rank" in capsys.readouterr().out
    ev.report(out, by_cam=True)
    assert "Cam01:" in capsys.readouterr().out
    raw = ev.evaluate(ev.raw_score, areas={"syn": area.name}, states_dir=area.parent, kml=kml, verbose=False)
    assert raw["areas"]["syn"]["n_spots"] > 0


def test_evaluate_held_out_area_and_command_line(
    analyzed: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    best = analyzed["candidates"][0]
    kml = tmp_path / "cams.kml"
    kml.write_text(
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Cam01</name>'
        f"<Point><coordinates>{best['lon']},{best['lat']},0</coordinates></Point></Placemark></Document></kml>"
    )
    area = analyzed["dir"]
    out = ev.evaluate(areas={"syn": area.name}, states_dir=area.parent, kml=kml, n_random=20, check_area="syn")
    assert out["fixed_k"]["check_area"] == "syn" and out["fixed_k"]["check_area_within"]["150"] == 1.0
    capsys.readouterr()
    ev.main(["--kml", str(kml), "--area", f"syn={area.name}", "--states", str(area.parent), "--check", "syn"])
    assert "| syn: <=50 m" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ev.main(["--kml", str(kml)])  # no areas
    assert "no human picks" in capsys.readouterr().err


def test_eval_config_from_the_private_folder(tmp_path: Path) -> None:
    assert ev.EvalConfig.load(tmp_path / "missing.toml") == ev.EvalConfig()
    cfg = tmp_path / "eval.toml"
    cfg.write_text('kml = "picks.kml"\nstates = "states"\ncheck = "b"\n[areas]\na = "area-a"\nb = "area-b"\n')
    c = ev.EvalConfig.load(cfg)
    assert c.kml == tmp_path / "picks.kml" and c.states == tmp_path / "states" and c.check == "b"
    assert c.areas == {"a": "area-a", "b": "area-b"}
