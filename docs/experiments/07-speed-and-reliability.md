# 07. Speed and reliability (2026-09-29 to 10-01)

## Downloads

- **Parallel lidar reads** plus GDAL HTTP tuning (merged consecutive ranges, HTTP/2 multiplexing, a VSI cache):
  elevation for a 307 km2 area went from a serial crawl to under a minute.
- **PAD-US geometry simplified on the server** (a maximum allowable offset of 0.0003 degrees): the regional query
  went from 18.5 s and 12.4 MB to 8.2 s and 3.1 MB with the same features. That fixed a scout that stalled.
- **NHD resilience:** a tile that fails is split into quadrants (up to two levels). If it still fails, the run
  continues with a note instead of crashing.
- **Stalled raster reads** fail after a minute and fall back like an unreadable tile, instead of hanging the run.

## The profile

On a 10-core laptop, a cached rerun took 30 s for a small area and 95 s for a large one (peaking at 8-9 GB), and a
first run in a new region took ~4 min, mostly downloads made one after another. The hot spots:

| Where | Time |
|---|---|
| single-threaded gzip of a 2.3 GB state pickle (13% of it a cached meshgrid) | 22-24 s |
| 240 scipy distance transforms | ~13 s |
| single-threaded GDAL warps | 8 s |
| PNG `optimize=True`, for files only ~6% smaller | 7 s |
| per-point geometry reprojection | 6 s |
| sorting in labeled min/max | 4 s |

## The changes

A chunked, threaded zlib state file with its index at the end; an exact parallel numba Felzenszwalb-Huttenlocher
distance transform; threaded warps, filters and PNG encoding; bincount label sums; sort-free flow accumulation;
features projected once; parallel downloads.

**Gate:** every layer, candidate list, explain and repick output stayed **bit-identical**, checked against saved
fingerprints, and each kernel has a test against the scipy/numpy code it replaced.

| | Before | After |
|---|---|---|
| Small area, cached | 30 s | 6 s |
| Large area, cached | 89 s | 21 s |
| Cached regional hotspot search | 37 s | 8 s |
| First run in a new region | ~4 min | ~1 min |
| `explain_point` (state in memory) | 5 s | 0.2 s |
| `repick` | 39 s | 3.4 s |

A reviewer re-timed the numbers independently.

## Platform drift

The first CI run on Linux failed four distance-transform tests that pass on macOS. The test's own cross-check used
`np.hypot`, which under glibc can differ from `sqrt(a*a + b*b)` in the last bit. The kernel and SciPy both use the
`sqrt` form, so the model was right and the test was wrong. The fix was to make the test use the same arithmetic,
because the speedups are gated on bit-exact equality.
