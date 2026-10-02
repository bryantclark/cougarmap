"""The GPS falsification harness (cougarmap.gps), reproducibly: download the open collar data, tile it, analyze the
tiles with the production pipeline, and report the used-vs-available table with the placebo and the ablations.

    uv run python scripts/gps_check.py all                 # every step below, in order
    uv run python scripts/gps_check.py fetch               # the collar data, into the cache folder (not git)
    uv run python scripts/gps_check.py tiles [--force]     # choose the tiles (out/gps/tiles.json)
    uv run python scripts/gps_check.py run                 # analyze each tile at each season's month, one at a time
    uv run python scripts/gps_check.py report              # out/gps/report.json and the tables (markdown)
    uv run python scripts/gps_check.py tables              # the tables again from out/gps/report.json

--datasets fishlake,olympic limits any step to some datasets. Tiles already analyzed are kept, so `run` resumes.
The data are other ecosystems with 2-8 h fixes: this can falsify, never tune (docs/VALIDATION.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from cougarmap import gps
from cougarmap.config import GPS
from cougarmap.evaluate import production_score

JSON = dict[str, Any]


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _rank(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _sign(s: JSON) -> str:
    return f"{s['above_half']}/{s['above_half'] + s['below_half']} (p={s['p_value']:.3f})" if s["p_value"] else "-"


def table_main(res: JSON) -> str:
    """Per dataset and overall: pooled ranks, Boyce, the null, and the across-animal sign test."""
    rows = [
        "| | strata | animals | rank 20 m | rank ~500 m | Boyce 20 m | Boyce ~500 m | map-shift null 20 m / 500 m "
        "| moving steps rank 20 m | animals above chance, 20 m | 500 m |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in [*res["datasets"].items(), ("**all**", res["overall"])]:
        p, a = d["pooled"], d["across_animals"]
        rows.append(
            f"| {name} | {p['n']} | {a['n_animals']} | {_rank(p['rank20'])} | {_rank(p['rank_ctx'])} "
            f"| {_pct(p['boyce20'])} | {_pct(p['boyce_ctx'])} | {_rank(p['null20'])} / {_rank(p['null_ctx'])} "
            f"| {_rank(p['moving_rank20'])} | {_sign(a['sign_rank20'])} | {_sign(a['sign_rank_ctx'])} |"
        )
    return "\n".join(rows)


def table_animals(res: JSON) -> str:
    rows = ["| animal | strata | rank 20 m | rank ~500 m | Boyce 20 m | null 20 m |", "|---|---|---|---|---|---|"]
    for a, v in res["animals"].items():
        if v["n"] >= GPS.min_strata:
            rows.append(
                f"| {a} | {v['n']} | {_rank(v['rank20'])} | {_rank(v['rank_ctx'])} | {_pct(v['boyce20'])} "
                f"| {_rank(v['null20'])} |"
            )
    return "\n".join(rows)


def _vs(c: JSON) -> str:
    return f"{c['better']}/{c['better'] + c['worse']}" + (
        f" (p={c['p_value']:.3f})" if c["p_value"] is not None else ""
    )


def table_compare(results: dict[str, JSON], ref: str) -> str:
    """Each ablation and placebo against the reference: the pooled ranks at 20 m / ~500 m, and in brackets the
    animals where the reference ranks higher at 20 m (a term helps where removing it loses most animals)."""
    datasets = list(results[ref]["datasets"])
    head = " | ".join(f"{d}: 20 m / 500 m (ref. better)" for d in datasets)
    rows = [
        f"| score | {head} | all: 20 m / 500 m | ref. better, all, 20 m | 500 m |",
        "|---|" + "---|" * (len(datasets) + 3),
    ]
    for name, r in results.items():
        c20 = gps.compare(results[ref], r, "rank20")
        cctx = gps.compare(results[ref], r, "rank_ctx")["overall"]
        cells = []
        for d in datasets:
            p = r["datasets"][d]["pooled"]
            vs = "" if name == ref else f" ({c20[d]['better']}/{c20[d]['better'] + c20[d]['worse']})"
            cells.append(f"{_rank(p['rank20'])} / {_rank(p['rank_ctx'])}{vs}")
        o = r["overall"]["pooled"]
        vs20, vs_ctx = ("(reference)", "") if name == ref else (_vs(c20["overall"]), _vs(cctx))
        rows.append(
            f"| {name} | {' | '.join(cells)} | {_rank(o['rank20'])} / {_rank(o['rank_ctx'])} | {vs20} | {vs_ctx} |"
        )
    return "\n".join(rows)


def tables(results: dict[str, JSON]) -> None:
    print("## Production\n")
    print(table_main(results["production"]))
    print("\n## Per animal (production)\n")
    print(table_animals(results["production"]))
    print("\n## Ablations and the DEM-shift placebo (against the recombined production ranking)\n")
    print(table_compare({k: v for k, v in results.items() if k != "production"}, "recombined"))
    for name in ("DEM shift E", "DEM shift N"):
        for key in ("rank20", "rank_ctx"):
            c = gps.compare(results["production"], results[name], key)
            print(f"\nproduction beats {name} ({key}), animals: " + ", ".join(f"{d} {_vs(v)}" for d, v in c.items()))


def report(datasets: list[str] | None) -> None:
    fns = {"production": production_score, **gps.ablations(), "DEM shift E": gps.dem_shift("E")}
    fns["DEM shift N"] = gps.dem_shift("N")
    results = gps.gps_check_many(fns, datasets=datasets)
    out = gps.GPS_DIR / "report.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}\n")
    tables(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "fetch", "tiles", "run", "report", "tables"])
    ap.add_argument("--datasets", help="comma-separated: " + ",".join(gps.DATASETS))
    ap.add_argument("--force", action="store_true", help="tiles: choose them again even if tiles.json exists")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # progress shows up in a redirected log
    datasets = a.datasets.split(",") if a.datasets else list(gps.DATASETS)
    if a.step in ("all", "fetch"):
        for d in datasets:
            gps.fetch(d)
    tiles_file = gps.GPS_DIR / "tiles.json"
    if a.step in ("all", "tiles") and (a.force or not tiles_file.exists()):
        tiles = []
        for d in gps.DATASETS:  # always every dataset, so tiles.json never depends on --datasets
            gps.fetch(d)
            tiles += gps.choose_tiles(gps.strata_for(d), gps.DATASETS[d].n_tiles)
        print(f"wrote {gps.write_tiles(tiles)}: {len(tiles)} tiles")
    if a.step in ("all", "run"):
        gps.analyze_tiles([t for t in gps.read_tiles() if t.dataset in datasets])
    if a.step in ("all", "report"):
        report(datasets if a.datasets else None)
    if a.step == "tables":
        tables(json.loads((gps.GPS_DIR / "report.json").read_text()))


if __name__ == "__main__":
    main()
