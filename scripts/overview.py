"""Whole-area layer maps with the camera pins (CamNN) from a KML, if one is given.

Usage: uv run python scripts/overview.py <state.pkl> <out.png> [layer,layer,...] [pins.kml]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cougarmap import api
from cougarmap.state import load_state

DEFAULT_LAYERS = ("score", "wind", "edges", "pinch", "water")


def main(state: str, out: str, layers: list[str], kml: Path | None = None) -> None:
    st = load_state(state)
    A: dict[str, Any] = dict(st.layers)
    g = st.fine
    cams = [p for p in api.list_areas(str(kml))["points"] if p["kind"] == "camera"] if kml else []
    f = max(1, int(30 / g.res))
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 11))
    for ax, name in zip(np.atleast_1d(axes), layers, strict=True):
        ax.imshow(np.where(st.aoi_mask, A[name], np.nan)[::f, ::f], cmap="magma")
        ax.set_title(name)
        for p in cams:
            r, c = g.rowcol(*g.from_lonlat(p["lon"], p["lat"]))
            if g.contains_rc(r, c) and st.aoi_mask[r, c]:
                ax.plot(c / f, r / f, "c^", ms=9, mec="k")
                ax.text(c / f + 4, r / f, p["name"][3:], color="c", fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=50)


if __name__ == "__main__":
    layers = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DEFAULT_LAYERS)
    main(sys.argv[1], sys.argv[2], layers, Path(sys.argv[4]) if len(sys.argv) > 4 else None)
