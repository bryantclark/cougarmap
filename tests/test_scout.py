"""The regional scout on the synthetic landscape: ranked, non-adjacent blocks with reasons."""

from __future__ import annotations

import pytest

import synthetic
from cougarmap import scout


@pytest.fixture
def offline_scout(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic.install(monkeypatch)
    monkeypatch.setattr(scout, "_osm_roads", synthetic.fetch_osm)
    monkeypatch.setattr(scout, "_inat_cougars", lambda lb: [(synthetic.LAT, synthetic.LON)] * 3)


def test_scout_ranks_blocks(offline_scout: None) -> None:
    r = scout.scout(synthetic.LAT, synthetic.LON, radius_km=4, month=10, block_km=1.0, top=4, log=lambda *_: None)
    assert r["cougar_observations"] == 3 and r["wind"]["prevailing_from"] == "W" and r["month"] == 10
    blocks = r["blocks"]
    assert 1 <= len(blocks) <= 4 and [b["rank"] for b in blocks] == list(range(1, len(blocks) + 1))
    scores = [b["score"] for b in blocks]
    assert scores == sorted(scores, reverse=True) and scores[0] > 1
    for b in blocks:
        assert b["land"] == "Test National Forest" and b["block_km"] == 1.0 and b["public_fraction"] > 0
        assert set(b["parts"]) == {"terrain", "edges", "wind", "water", "prior", "development"}
        west, south, east, north = b["bbox"]
        assert west < b["center"]["lon"] < east and south < b["center"]["lat"] < north
    centers = [(b["center"]["lat"], b["center"]["lon"]) for b in blocks]
    assert len(set(centers)) == len(centers)


def test_scout_private_land_allowed_scores_more_ground(offline_scout: None) -> None:
    public = scout.scout(synthetic.LAT, synthetic.LON, 4, 10, True, block_km=1.0, top=20, log=lambda *_: None)
    anyland = scout.scout(synthetic.LAT, synthetic.LON, 4, 10, False, block_km=1.0, top=20, log=lambda *_: None)
    assert len(anyland["blocks"]) >= len(public["blocks"])
    assert any(b["land"] == "mostly private" or b["public_fraction"] < 0.5 for b in anyland["blocks"])
