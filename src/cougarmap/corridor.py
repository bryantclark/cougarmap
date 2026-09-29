"""Movement funnels via circuit theory (the idea behind Circuitscape/Omniscape).

The landscape is treated as a resistor network: easy ground conducts, cliffs/lakes resist. We push current
across the area wall-to-wall (north->south and west->east) and measure where it concentrates. Cells carrying far
more current than the local typical value are natural funnels: saddles, benches between cliffs, gaps between
water, etc.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from .arrays import Floats


def _solve(res_grid: Floats, axis: int) -> Floats:
    H, W = res_grid.shape
    N = H * W
    idx = np.arange(N).reshape(H, W)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    # 4-neighbour conductances: g = 2 / (R_i + R_j)
    for dr, dc in ((0, 1), (1, 0)):
        a = idx[: H - dr, : W - dc].ravel()
        b = idx[dr:, dc:].ravel()
        g = 2.0 / (res_grid[: H - dr, : W - dc].ravel() + res_grid[dr:, dc:].ravel())
        row_parts += [a, b]
        col_parts += [b, a]
        val_parts += [g, g]
    rows, cols, vals = np.concatenate(row_parts), np.concatenate(col_parts), np.concatenate(val_parts)
    G = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    deg = np.asarray(G.sum(axis=1)).ravel()
    L = (sparse.diags(deg) - G).tocsr()

    fixed = np.zeros(N, bool)
    v = np.zeros(N)
    if axis == 0:
        fixed[idx[0]] = True
        v[idx[0]] = 1.0
        fixed[idx[-1]] = True
    else:
        fixed[idx[:, 0]] = True
        v[idx[:, 0]] = 1.0
        fixed[idx[:, -1]] = True
    free = ~fixed
    A = L[free][:, free].tocsc()
    rhs = -L[free][:, fixed] @ v[fixed]
    v[free] = spsolve(A, rhs)

    # current through each cell = half the sum of |current| on its edges
    cur = np.zeros(N)
    for dr, dc in ((0, 1), (1, 0)):
        a = idx[: H - dr, : W - dc].ravel()
        b = idx[dr:, dc:].ravel()
        g = 2.0 / (res_grid[: H - dr, : W - dc].ravel() + res_grid[dr:, dc:].ravel())
        i = np.abs(g * (v[a] - v[b]))
        np.add.at(cur, a, 0.5 * i)
        np.add.at(cur, b, 0.5 * i)
    cur = cur.reshape(H, W)
    return cur / max(cur.mean(), 1e-12)


def current_density(resistance: Floats) -> Floats:
    """Average of N-S and W-E wall-to-wall current, each normalized to mean 1."""
    r = np.clip(resistance.astype(float), 1.0, 1e4)
    with ThreadPoolExecutor(2) as ex:  # independent sparse solves; SuperLU releases the GIL
        ns, we = ex.submit(_solve, r, 0), ex.submit(_solve, r, 1)
        return 0.5 * (ns.result() + we.result())


def funnel_score(resistance: Floats, res: float, local_m: float = 600.0) -> Floats:
    """0-1: how strongly movement is squeezed through each cell relative to its surroundings."""
    cur = current_density(resistance)
    size = max(3, round(local_m / res) | 1)
    local = ndimage.median_filter(cur, size=size, mode="nearest")
    ratio = cur / np.maximum(local, 1e-9)
    out: Floats = np.clip(np.log2(np.maximum(ratio, 1e-9)) / 2.5, 0, 1).astype("float32")
    return out
