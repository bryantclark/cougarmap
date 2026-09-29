"""Snow depth: NSIDC SNODAS (G02158, masked daily, contiguous US at 30 arc-seconds, ~1 km), public HTTPS, no login.

A month's typical depth is the median over the last few winters of one day each (Winter.snodas_day of the month),
in centimetres. Each day's archive is ~23 MB; only its snow-depth member (~8 MB gzipped) is kept in the disk cache,
so every area and tile shares one download per date.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import math
import tarfile
import warnings

import numpy as np
from affine import Affine

from ..arrays import Floats
from ..config import WINTER, Winter
from ..net import cached, get
from .dem import LonLatBounds

URL = "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{y}/{m:02d}_{mon}/SNODAS_{y}{m:02d}{d:02d}.tar"
# the masked grid since October 2013 (NSIDC G02158 user guide): rows x columns, cell size, upper-left corner
NROWS, NCOLS = 3351, 6935
RES_DEG = 1 / 120
ULX, ULY = -124.733749999999, 52.874583333332
DEPTH = "ssmv11036tS"  # the snow-depth product code in the member names (mm, int16 big-endian, -9999 = no data)
PAD_DEG = 0.05  # the crop reaches this far past the area, so warping to the area's grid has neighbours

__all__ = ["crop", "dates", "decode", "depth_member", "fetch_snow_depth"]


def dates(month: int, cfg: Winter = WINTER, today: dt.date | None = None) -> list[dt.date]:
    """The cfg.snodas_day of `month` in the last cfg.snodas_years years for which that day is past."""
    today = today or dt.date.today()
    y = today.year if dt.date(today.year, month, cfg.snodas_day) < today else today.year - 1
    return [dt.date(y - i, month, cfg.snodas_day) for i in range(cfg.snodas_years)]


def depth_member(tar_bytes: bytes) -> bytes:
    """The gzipped snow-depth grid out of one day's SNODAS archive."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if DEPTH in m.name and m.name.endswith(".dat.gz"):
                f = tf.extractfile(m)
                if f is not None:
                    return f.read()
    raise RuntimeError("no snow-depth grid in the SNODAS archive")


def decode(gz: bytes) -> np.ndarray:
    """The whole grid (int16 mm, -9999 = no data) from the gzipped member."""
    return np.frombuffer(gzip.decompress(gz), ">i2").reshape(NROWS, NCOLS)


def crop(grid: np.ndarray, lb: LonLatBounds, pad_deg: float = PAD_DEG) -> tuple[Floats, Affine]:
    """The part of the grid over lb (padded, clipped at the grid's edge) in cm, NaN where there is no data, and its
    lon/lat transform. Empty when lb is outside the grid."""
    w, s, e, n = lb
    c0 = max(0, math.floor((w - pad_deg - ULX) / RES_DEG))
    c1 = min(NCOLS, math.ceil((e + pad_deg - ULX) / RES_DEG))
    r0 = max(0, math.floor((ULY - n - pad_deg) / RES_DEG))
    r1 = min(NROWS, math.ceil((ULY - s + pad_deg) / RES_DEG))
    a = grid[r0 : max(r0, r1), c0 : max(c0, c1)].astype("float32")
    a[a < 0] = np.nan
    return a / 10, Affine(RES_DEG, 0, ULX + c0 * RES_DEG, 0, -RES_DEG, ULY - r0 * RES_DEG)


def _day(d: dt.date) -> bytes:
    url = URL.format(y=d.year, m=d.month, mon=d.strftime("%b"), d=d.day)
    gz: bytes = cached("snodas", d.isoformat(), lambda: depth_member(get(url, timeout=600).content))
    return gz


def fetch_snow_depth(lb: LonLatBounds, month: int, cfg: Winter = WINTER) -> tuple[Floats, Affine]:
    """The median snow depth (cm, NaN = no data) over lb in `month`, and its lon/lat transform. Days missing from
    the archive are skipped; with none at all, or outside the grid, it raises."""
    crops = []
    for d in dates(month, cfg):
        try:
            crops.append(crop(decode(_day(d)), lb))
        except RuntimeError:  # a gap in the archive, or the download failed: the other years still count
            continue
    if not crops:
        raise RuntimeError(f"no SNODAS snow depth for month {month}")
    a = np.stack([c for c, _ in crops])
    if a.size == 0:
        raise RuntimeError("outside the SNODAS grid (contiguous US only)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells (ocean, outside the mask) stay NaN
        med: Floats = np.nanmedian(a, axis=0).astype("float32")
    return med, crops[0][1]
