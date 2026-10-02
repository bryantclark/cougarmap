"""The saved state: the chunked parallel file format (older formats still readable), the typed layer schema,
saving and loading a ModelState, migrating states saved by older versions, cheap option updates, and the in-memory
cache the MCP server's follow-up calls use."""

from __future__ import annotations

import dataclasses
import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import toys
from cougarmap import api, state, statefile
from cougarmap.analyze import site_penalty
from cougarmap.config import PROJECT_ROOT, Options
from cougarmap.grid import Grid
from cougarmap.state import SAVED, STATE_VERSION, Store, load_state, migrate, save_state, save_state_opts


def _tree() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    big = rng.normal(size=(700, 900)).astype("float32")  # spans several chunks
    return dict(
        slim=dict(opts=Options(month=10), grid=Grid(32611, 1.0, 2.0, 3.0, 900, 700), notes=["x"]),
        arrays=dict(
            big=big,
            half=big.astype(np.float16),
            mask=big > 0,
            lab=rng.integers(-1, 9, size=(400, 500)).astype("int8"),
            idx=rng.integers(0, 10**9, size=(300, 300)),
            small=np.arange(5, dtype="uint8"),
            empty=np.zeros((0, 3), "float32"),
        ),
        meta=dict(saddles=[dict(row=1, col=2)], names=("a", "b")),
    )


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    return bool(a == b)


def test_round_trip(tmp_path: Path) -> None:
    t = _tree()
    p = tmp_path / "state.pkl"
    statefile.save(t, p)
    assert statefile.is_v2(p)
    assert _same(statefile.load(p), t)
    assert not list(tmp_path.glob(".*tmp"))  # temp file renamed into place


def test_reads_older_formats(tmp_path: Path) -> None:
    t = _tree()
    with gzip.open(tmp_path / "old.pkl", "wb", compresslevel=3) as f:
        pickle.dump(t, f)
    (tmp_path / "raw.pkl").write_bytes(pickle.dumps(t))
    for name in ("old.pkl", "raw.pkl"):
        assert not statefile.is_v2(tmp_path / name)
        assert _same(statefile.load(tmp_path / name), t)


def test_update_rewrites_only_the_index(tmp_path: Path) -> None:
    p = tmp_path / "state.pkl"
    statefile.save(_tree(), p)
    before = p.read_bytes()
    statefile.update(p, lambda tree: dict(tree, slim=dict(tree["slim"], opts=Options(month=3))))
    after = p.read_bytes()
    with p.open("rb") as f:
        _, index_at = statefile._read_index(f)
    assert after[:index_at] == before[:index_at]  # array chunks untouched
    back = statefile.load(p)
    assert back["slim"]["opts"].month == 3
    assert _same(back["arrays"], _tree()["arrays"])


def test_a_failed_update_leaves_the_file_intact(tmp_path: Path) -> None:
    p = tmp_path / "state.pkl"
    statefile.save(_tree(), p)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        statefile.update(p, lambda tree: dict(tree, bad=lambda: 0))  # cannot be pickled
    assert _same(statefile.load(p), _tree())
    assert not list(tmp_path.glob(".*tmp"))


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't replace an open file (see the next test)")
def test_update_swaps_in_a_new_file(tmp_path: Path) -> None:
    """A reader that opened the file before an update (another process loading it during a re-pick) still reads
    a whole, valid file: the update never rewrites the file in place."""
    p = tmp_path / "state.pkl"
    statefile.save(_tree(), p)
    with p.open("rb") as f:
        statefile.update(p, lambda tree: dict(tree, slim=dict(tree["slim"], opts=Options(month=3))))
        old, _ = statefile._read_index(f)
    assert old["slim"]["opts"].month == 10 and statefile.load(p)["slim"]["opts"].month == 3
    assert not list(tmp_path.glob(".*tmp"))


def test_a_refused_swap_waits_for_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows refuses to replace a file someone has open: the swap retries until they let go, and gives up with
    the error (leaving the old file) if they never do."""
    p = tmp_path / "state.pkl"
    statefile.save(_tree(), p)
    real, refusals = Path.replace, [2]

    def busy(self: Path, target: Any) -> Path:
        if refusals[0]:
            refusals[0] -= 1
            raise PermissionError(13, "in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", busy)
    statefile.update(p, lambda tree: dict(tree, slim=dict(tree["slim"], opts=Options(month=3))))
    assert statefile.load(p)["slim"]["opts"].month == 3 and refusals == [0]
    refusals[0] = 10**6
    monkeypatch.setattr(statefile, "SWAP_WAIT_S", 0.1)
    with pytest.raises(PermissionError):
        statefile.update(p, lambda tree: dict(tree, slim=dict(tree["slim"], opts=Options(month=4))))
    assert statefile.load(p)["slim"]["opts"].month == 3 and not list(tmp_path.glob(".*tmp"))


def test_truncated_file_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "state.pkl"
    statefile.save(_tree(), p)
    p.write_bytes(p.read_bytes()[:-3])
    with pytest.raises(ValueError, match="truncated"):
        statefile.load(p)


# ---- the layer schema, save_state / load_state ---------------------------------------------------------------


def test_schema_saves_what_the_model_explains_with() -> None:
    assert SAVED["score"] is Store.FULL and SAVED["walk_any_m"] is Store.FULL and SAVED["wind"] is Store.HALF
    assert SAVED["usable"] is Store.EXACT and SAVED["land_names"] is Store.EXACT
    assert not {"slope_mid", "acc_mid", "edge_q", "usable_private", "final_private", "road_mid"} & set(SAVED)
    # the layers states have always kept (export.KEEP and the lookup lists before the schema existed)
    keep = """wind edges pinch water score final n_on usable public land_id lake walk_m walk_s walk_any_m walk_any_s
    walk_any_pred_mid road_dist houses flowx flown edge_downwind edge_meadow edge_ridge edge_valley edge_water
    edge_route pinch_saddle pinch_cliffbase pinch_clifftop pinch_bank pinch_funnel pinch_fence water_kind
    water_scarcity slope meadow meadow_label cliff drainx_mid drainn_mid conv_mid cos_mid drain_mid windward_mid
    landform_mid flowx_mid flown_mid windx_mid windn_mid walk_pred_mid travel travel_pos_mid context edge_density
    water_density paved_dist closed_track_mid land_names land_access water_labels saddle_points meadow_ha
    rec_dist travel_gate_mid travel_thermal_mid edge_q_drain edge_q_wind season winter_mid winter_range_mid
    trail_kind worn_lines worn_unmapped pinch_water pinch_water_kind travel_approach travel_approach_to"""
    assert set(SAVED) == set(keep.split())


def test_save_and_load_state(tmp_path: Path) -> None:
    st = toys.state()
    rng = np.random.default_rng(1)
    st.layers["score"] = rng.random(st.fine.shape).astype("float32") * 100
    st.layers["wind"] = rng.random(st.fine.shape).astype("float32")
    st.layers["usable_private"] = np.ones(st.fine.shape, bool)  # run-only: not saved
    st.chm = rng.random(st.fine.shape).astype("float32") * 30
    p = tmp_path / "state.pkl"
    save_state(st, p)
    back = load_state(p)
    assert statefile.load(p)["slim"]["version"] == STATE_VERSION
    assert back.fine == st.fine and back.opts == st.opts and back.wind == st.wind and back.unmodeled == frozenset()
    assert set(back.layers) == set(SAVED)
    assert np.array_equal(back.layers["score"], st.layers["score"])  # kept at float32
    assert back.layers["wind"].dtype == np.float32
    assert np.allclose(back.layers["wind"], st.layers["wind"], atol=1e-3)  # float16 on disk
    assert back.layers["usable"].dtype == bool and back.layers["n_on"].dtype == np.uint8
    assert np.array_equal(back.chm, np.rint(st.chm))  # canopy is kept in whole metres, rounded
    assert back.layers["land_names"] == ["Test National Forest"]

    o = Options(month=10, n_candidates=30)
    assert save_state_opts(p, o)
    assert load_state(p).opts == o
    with gzip.open(tmp_path / "old.pkl", "wb") as f:
        pickle.dump(dict(slim=dict(opts=Options()), arrays={}), f)
    assert not save_state_opts(tmp_path / "old.pkl", o)


V5_LAYERS = (
    "rec_dist",
    "travel_gate_mid",
    "travel_thermal_mid",
    "edge_q_drain",
    "edge_q_wind",
    "season",
    "winter_mid",
    "winter_range_mid",
    "trail_kind",
)
V7_LAYERS = ("pinch_water", "pinch_water_kind")
V8_LAYERS = ("travel_approach", "travel_approach_to")


def _version1_tree() -> dict[str, Any]:
    """A state as model v1 saved it: no travel lines, habitat context, paved-road distance, any-route walk,
    building counts, lakes, along-valley wind or closed tracks; Options without the fields added since."""
    st = toys.state()
    st.layers["walk_m"][:, :10] = 99.0
    st.layers["drainx_mid"][:] = 0.25
    gone = [
        "travel",
        "travel_pos_mid",
        "context",
        "edge_density",
        "water_density",
        "paved_dist",
        "walk_any_m",
        "walk_any_s",
        "walk_any_pred_mid",
        "houses",
        *V5_LAYERS,
        *V7_LAYERS,
        *V8_LAYERS,
        "lake",
        "windx_mid",
        "windn_mid",
        "closed_track_mid",
        "meadow_label",
        "meadow_ha",
    ]
    layers: dict[str, Any] = {k: v for k, v in st.layers.items() if k not in gone}
    meta = {k: layers.pop(k) for k in ("land_names", "land_access", "water_labels", "saddle_points")}
    layers["n_on"] = layers["n_on"].astype(np.int64)
    opts = Options(month=10, n_candidates=7)
    del opts.__dict__["paved_penalty"], opts.weights.__dict__["habitat"]
    del opts.__dict__["houses_exponent"]  # older states have no house cost: they get it on repick
    opts.__dict__["near_road_penalty_m"] = 40.0  # a field removed since
    slim = {f.name: getattr(st, f.name) for f in dataclasses.fields(st) if f.name not in ("layers", "unmodeled")}
    return dict(slim=dict(slim, opts=opts, meta=meta), arrays=layers)


def test_migrate_fills_what_old_states_lack(tmp_path: Path) -> None:
    tree = _version1_tree()
    assert state.state_version(tree) == 1
    with gzip.open(tmp_path / "v1.pkl", "wb") as f:  # the old gzip-pickle file format
        pickle.dump(tree, f)
    st = load_state(tmp_path / "v1.pkl")
    A = st.layers
    assert set(A) == set(SAVED)
    assert st.unmodeled == {
        "travel",
        "travel_pos_mid",
        "context",
        "edge_density",
        "water_density",
        "paved_dist",
        "houses",
        *V5_LAYERS,
        *V7_LAYERS,
        *V8_LAYERS,
    }
    assert np.array_equal(A["walk_any_m"], A["walk_m"]) and A["walk_any_m"][0, 0] == 99.0
    assert np.array_equal(A["windx_mid"], A["drainx_mid"])
    assert not A["lake"].any() and not A["houses"].any() and not A["closed_track_mid"].any()
    assert A["paved_dist"].min() >= 800 and not A["travel"].any() and (A["context"] == 1).all()
    assert st.opts.n_candidates == 7 and st.opts.paved_penalty == Options().paved_penalty
    assert st.opts.weights.habitat == Options().weights.habitat and "near_road_penalty_m" not in vars(st.opts)
    assert vars(st.opts)["houses_exponent"] == pytest.approx(0.34)
    A["houses"][0, 0] = 3.0  # what a repick of the old state now applies
    assert site_penalty(A, st.opts)[0, 0] == pytest.approx(4**-0.34 * site_penalty(A, Options(houses_exponent=0))[0, 0])

    save_state(st, tmp_path / "v3.pkl")  # what a repick of an old-format state writes
    again = load_state(tmp_path / "v3.pkl")
    assert (
        again.unmodeled == st.unmodeled
        and state.state_version(statefile.load(tmp_path / "v3.pkl")) == state.STATE_VERSION
    )


def test_version4_states_migrate_to_the_v5_layers(tmp_path: Path) -> None:
    st = toys.state()
    save_state(st, tmp_path / "v5.pkl")
    tree = statefile.load(tmp_path / "v5.pkl")
    tree["slim"]["version"] = 4
    for k in V5_LAYERS:
        del tree["arrays"][k]
    old = migrate(tree)
    A = old.layers
    assert set(V5_LAYERS) <= old.unmodeled and set(A) == set(SAVED)
    assert (A["rec_dist"] == state.PAVED_DIST_CAP_M).all() and (A["travel_gate_mid"] == 1).all()
    assert not A["edge_q_drain"].any() and not A["edge_q_wind"].any() and not A["travel_thermal_mid"].any()
    assert (A["season"] == 1).all() and not A["winter_mid"].any() and not A["winter_range_mid"].any()


def test_version6_states_have_no_water_pinch(tmp_path: Path) -> None:
    """States before ponds were pinch barriers get the component as unmodeled zeros; their saved pinch stays the
    one they were scored with, so a repick ranks them exactly as before."""
    st = toys.state()
    st.layers["pinch"][:] = 0.5
    save_state(st, tmp_path / "s.pkl")
    tree = statefile.load(tmp_path / "s.pkl")
    tree["slim"]["version"] = 6
    for k in V7_LAYERS:
        del tree["arrays"][k]
    old = migrate(tree)
    A = old.layers
    assert set(V7_LAYERS) <= old.unmodeled and set(A) == set(SAVED)
    assert not A["pinch_water"].any() and not A["pinch_water_kind"].any() and (A["pinch"] == 0.5).all()


def test_version7_states_keep_their_travel(tmp_path: Path) -> None:
    """States before the destination approaches get them as unmodeled zeros; their saved travel line stays the one
    they were scored with, so a repick ranks them exactly as before and the reasons never name an approach."""
    st = toys.state()
    st.layers["travel"][:] = 0.4
    save_state(st, tmp_path / "s.pkl")
    tree = statefile.load(tmp_path / "s.pkl")
    tree["slim"]["version"] = 7
    for k in V8_LAYERS:
        del tree["arrays"][k]
    old = migrate(tree)
    A = old.layers
    assert set(V8_LAYERS) <= old.unmodeled and set(A) == set(SAVED)
    assert not A["travel_approach"].any() and not A["travel_approach_to"].any()
    assert np.allclose(A["travel"], 0.4, atol=1e-3)


def test_states_from_a_newer_version_are_refused() -> None:
    tree = _version1_tree()
    tree["slim"]["version"] = STATE_VERSION + 1
    with pytest.raises(ValueError, match="newer CougarMap"):
        migrate(tree)


def test_state_cache_reuses_until_the_file_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.pkl"
    save_state(toys.state(), p)
    loads: list[Path] = []
    real = load_state

    def counting(q: Path) -> Any:
        loads.append(q)
        return real(q)

    monkeypatch.setattr(api, "load_state", counting)
    cache = api._StateCache()
    with cache.use(p) as a:
        a.opts.n_candidates = 42  # in-memory edits persist between uses
    with cache.use(p) as b:
        assert b is a and b.opts.n_candidates == 42
    assert len(loads) == 1

    save_state_opts(p, b.opts)
    cache.saved(p)  # the in-memory state matches what was just written
    with cache.use(p) as c:
        assert c is a
    assert len(loads) == 1

    save_state(toys.state(), p)  # e.g. an analyze job in another process rewrote it
    os.utime(p, ns=(1, 1))
    with cache.use(p) as d:
        assert d is not a and d.opts.n_candidates == 15
    assert len(loads) == 2
    cache.clear()
    with cache.use(p):
        pass
    assert len(loads) == 3


def test_state_cache_drops_a_state_an_operation_failed_on(tmp_path: Path) -> None:
    """A re-pick changes the cached state's options in place before saving them: if it fails, the cache must not
    keep options the file does not have."""
    p = tmp_path / "state.pkl"
    save_state(toys.state(), p)
    cache = api._StateCache()
    with pytest.raises(RuntimeError), cache.use(p) as a:
        a.opts.n_candidates = 42
        raise RuntimeError("write failed")
    with cache.use(p) as b:
        assert b is not a and b.opts.n_candidates == 15
    cache.clear()


def test_canopy_and_half_layers_at_saved_precision(tmp_path: Path) -> None:
    st = toys.state()
    st.chm = np.full(st.fine.shape, 11.6, np.float32)
    st.layers["slope"] = np.full(st.fine.shape, 13.2501, np.float32)
    st.layers["road_dist"] = np.full(st.fine.shape, 975.3, np.float32)  # kept at full precision
    state.to_saved_precision(st)
    assert st.chm[0, 0] == 12.0 and st.layers["road_dist"][0, 0] == np.float32(975.3)
    save_state(st, tmp_path / "state.pkl")
    back = load_state(tmp_path / "state.pkl")
    assert np.array_equal(back.chm, st.chm) and np.array_equal(back.layers["slope"], st.layers["slope"])


# ---- real saved states (slow; need the validation areas) --------------------------------------------------------


def _saved_states() -> list[Path]:
    dirs = [PROJECT_ROOT / "out", *([Path(d)] if (d := os.environ.get("COUGARMAP_EVAL_DIR")) else [])]
    return sorted(p for d in dirs for p in d.glob("*/state.pkl"))


@pytest.mark.slow
@pytest.mark.parametrize("path", _saved_states(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_real_states_load(path: Path) -> None:
    """Every saved state on this machine (out/ and COUGARMAP_EVAL_DIR) loads, with every layer of the schema."""
    st = load_state(path)
    assert set(st.layers) == set(SAVED)
    assert st.layers["score"].shape == st.fine.shape and st.layers["drain_mid"].shape == st.mid.shape


def test_states_before_worn_trails_say_they_have_none(tmp_path: Path) -> None:
    """Worn trails are on by default now; a version-5 state never computed them, so its options say off."""
    save_state(toys.state(), tmp_path / "s.pkl")
    tree = statefile.load(tmp_path / "s.pkl")
    assert migrate(tree).opts.worn_trails
    tree["slim"]["version"] = 5
    assert not migrate(tree).opts.worn_trails
