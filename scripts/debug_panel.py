"""Render the model layers around a point from a saved state.

Usage: uv run python scripts/debug_panel.py <state.pkl> <lat> <lon> <out.png> [half_m]
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.enums import Resampling

from cougarmap.state import load_state


def main(state: str, lat: float, lon: float, out: str, half_m: float = 400.0) -> None:
    st = load_state(state)
    A, g = st.layers, st.fine
    r, c = g.rowcol(*g.from_lonlat(lon, lat))
    r, c = int(r), int(c)
    k = int(half_m / g.res)
    r0, c0 = max(r - k, 0), max(c - k, 0)
    sl = np.s_[r0 : r + k, c0 : c + k]
    z = st.z[sl]
    gx, gy = np.gradient(z, g.res)
    hillshade = np.clip(0.5 + 0.5 * (-gx + gy) / np.hypot(1, np.hypot(gx, gy)), 0, 1)
    landform = g.resample_from(A["landform_mid"].astype("float32"), st.mid, Resampling.nearest)[sl]
    layers = [
        ("hillshade", hillshade),
        ("slope", A["slope"][sl]),
        ("canopy", st.chm[sl]),
        ("landform", landform),
        ("wind", A["wind"][sl]),
        ("edges", A["edges"][sl]),
        ("pinch", A["pinch"][sl]),
        ("water", A["water"][sl]),
        ("funnel", A["pinch_funnel"][sl]),
        ("meadow", A["meadow"][sl].astype(float)),
        ("score", A["score"][sl]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for ax, (name, v) in zip(axes.ravel(), layers, strict=False):
        cmap = {"hillshade": "gray", "landform": "tab10"}.get(name, "viridis")
        lims = (1, 10) if name == "landform" else (None, None)
        im = ax.imshow(v, cmap=cmap, vmin=lims[0], vmax=lims[1])
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.plot(c - c0, r - r0, "r+", ms=16, mew=2)
    plt.tight_layout()
    fig.savefig(out, dpi=55)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], float(args[1]), float(args[2]), args[3], float(args[4]) if len(args) > 4 else 400.0)
