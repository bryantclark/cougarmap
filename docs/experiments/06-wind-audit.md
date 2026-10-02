# 06. Wind audit and model v3.1 (2026-10-01 to 10-02)

Wind is the factor our method weights most, and the hardest to check. This round audited it against weather
observations, reanalysis and a physical wind model. The shipping rule was the same as for v3: no worse overall on
the pin-free human picks, no area worse by more than ~3 points, and GPS can only veto.

## Shipped: cold air drains off hillsides too (v3.1)

Drainage strength came from the single-path (D8) upslope area with a ~3,000 m2 floor. So open hillsides and the
feet of short steep walls had no drainage at all: a synthetic 25-degree valley side scored 0.00-0.04, and the same
slope scored 0.00 / 0.09 / 0.41 at 10 / 20 / 30 m cells. It is now the multiple-flow-direction catchment per metre
of contour width (`terrain.specific_catchment`, Quinn/Freeman with p = 1.1; `factors.drain_strength` runs from 0
at 50 m2/m to 1 at 300,000 m2/m). On an open slope that is the slope length above the cell, at any cell size. It
adds 0.2-1.5 s per area.

| | v3 | **v3.1** |
|---|---|---|
| Median rank (random) | 45.6% (89%) | **43.5%** (89%) |
| vsR | 0.812 | 0.813 |
| Sites in top 25% / top 10% | 16 / 4% | 20 / 4% |
| Worst area | 45.6% | 44.1% |
| Fixed-K within 50 / 100 / 150 m | 16 / 44 / 60% | 12 / 40 / 52% |
| GPS pooled rank, 20 m | 0.514 | 0.515 (33 of 52 animals better, p = 0.02) |

Per site: 11 better, 9 worse, 5 within half a point. Most of the losers are ridge sites, because hillsides now
compete with them. A bootstrap over the picks puts the median change at -5.3 to +2.2 points (90%), and the cost
is fixed-K (60% -> 52% within 150 m). It shipped because it fixes a real defect, the picks don't object, and it is
the only wind change the GPS data lean toward.

## Shipped, no score change: honest wind labels

Our "850 hPa" input is HRRR for 99.9% of hours (Open-Meteo's best match over the US). Reports now name it and
give its consistency and most common direction next to the mean. A reason that rests on an unsteady wind (R under
0.35) says so. That matters: one test area's October "NW" wind is the mean of a two-way wind (SSW/SW or NE at
dawn) that rarely blows NW. The ground-level reference is now HRRR's 10 m wind instead of ERA5 at 100 m, which
matched valley stations worse.

## The prevailing wind against weather stations

High-pressure dawn and dusk hours, September-March 2022-2025, using our own high-pressure definition. "Within 45
degrees" happens 25% of the time by chance.

- **As a free-air wind it is good.** HRRR at 850 hPa is within 45 degrees of the nearest radiosonde 87% of the time
  on high-pressure days (ECMWF 94%, GFS 84%).
- **As the air at the ground at twilight it is not.** It is light (median 3.5-4 m/s), unsteady (R 0.11-0.54) and
  shifts from year to year (one year's monthly direction strays a median 63-92 degrees from the four-year one).
  It matches the observed wind at valley stations 8-22% of the time, and at ridge stations 7-37%: chance level.
- **HRRR's own 10 m wind does much better at valley stations** (46-88%). On ridges nothing tracks well. In one
  test area the ground wind at high-pressure dawn and dusk is a steady ENE in every month, against our NW/W. That
  looks like terrain-driven drainage.

## The code audit

- Hillsides got no drainage (fixed above).
- The "valley axis" was a fall line, not an axis.
- The hourly high-pressure filter compares each hour with the month's median pressure, and pressure dips every
  evening, so it keeps 10-11% of dusk hours against 37-42% of dawn hours.
- Wind varies least of the big factors (effective spread 0.035-0.043, against 0.07-0.09 for edges).

## Tried, measured, not shipped

| Variant (pin-free) | Median | vsR | Notes |
|---|---|---|---|
| v3.1 | 43.5% | 0.813 | GPS 0.515 |
| HRRR 10 m as the prevailing direction | 44.9% | 0.797 | one area 6.1 points worse; 10 sites better, 14 worse; GPS leans against |
| HRRR 10 m, daily high-pressure filter | 44.6% | 0.798 | one area 4.5 points worse |
| 850 hPa, daily high-pressure filter | 46.2% | 0.812 | a correctness fix, but costs 2.7 points (top-25% share 20% -> 12%) |
| Windward band reaching 150 m below the crest | 41.8% | | GPS veto at ~500 m (17 better, 28 worse) |
| Windward foot of a wall (from WindNinja) | 43.6% | | GPS vetoed it on v3 (4 better, 27 worse, p < 0.001) |
| Direction terms dropped when R < 0.35 (on v3) | 43.4% | | one area 3.9 points worse |
| A curvature (Hessian) valley axis (on v3) | 46.0% | | neutral |
| Convergence scored against the wind rose (on v3) | 44.7% | | neutral |
| Wind weight doubled (on v3) | 48.1% | 0.804 | |
| Wind removed (on v3) | 49.5% | 0.802 | |
| Wind reversed, a placebo (on v3) | 48.2% | 0.793 | so the direction carries some signal, not much |

- **The ground-level wind** is arguably the faithful reading of "the dominant wind on a high-pressure day", and it
  is what the stations see. But it costs one area 6 points and the GPS data lean against it. Users who trust a
  nearby station can pass `wind_from_deg`. An area-specific direction would be tuning.
- **WindNinja** (US Forest Service, built locally, mass-conserving solver, offline): our drainage direction matches
  its slope flow almost everywhere (mean cosine 0.96-0.97, ~98% within 45 degrees). Its strength grows with
  steepness and ours with catchment (rho ~0.4). None of nine WindNinja-based wind factors beat ours, and GPS was
  flat for all of them. Not integrated: no gain, and a 2 GB solver.

## Open questions

Is "the prevailing wind" the upper flow or the ground-level twilight flow? They disagree in at least one test
area. How far below a ridge does "the windward side" reach? A cheap anemometer or smoke puffs at dusk at a few
camera sites would settle the first. Limits: only October was evaluated, there are no wind readings at any
camera, and WindNinja ran without its momentum solver or cold-air pooling.

## Cliff base, again (in progress)

One human-picked site sits ~50 m from the toe of a ~135 m escarpment, yet its cliff-base score was ~0.1: the 60 m
base reach and the slope smoothing put the "base" on the face. The planned fix measures from the actual foot of the
escarpment, with the reach scaled to its height. Not yet evaluated.
