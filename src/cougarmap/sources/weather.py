"""Prevailing wind on high-pressure days, from Open-Meteo's historical forecasts (free, no key, 2022+). Over the
US Open-Meteo's best_match is the HRRR model (99.9% of hours at 850 hPa). Two levels:
- "upper": HRRR at 850 hPa (~1.5 km up, above the ridgetops). This is the wind the model scores with (our
  method's "dominant wind on a high-pressure day"). Under high pressure at dawn and dusk it is light and often
  unsteady (consistency R 0.2-0.5 in the inland Northwest), so every report gives R and the most common direction
  next to it.
- "10m": HRRR at 10 m, the ground-level wind, reported for reference. At valley weather stations it matches the
  observed twilight wind far better than 850 hPa does (it mostly shows cold air drainage), but at 3 km it cannot
  resolve slopes, and scoring with it did worse on the human picks (docs/experiments/06-wind-audit.md).

"High pressure" here = sea-level pressure at or above that month's median, not falling over the previous
6 hours, and no precipitation in the previous 24 hours. Periods: dawn = sunrise +/- 1.5 h, dusk = sunset +/- 1.5 h,
day = daylight in between, night = the rest. Pressure dips every evening, so far fewer dusk hours pass than dawn
hours; the dawn+dusk direction leans on dawn (a daily-mean filter was tried and did worse,
docs/experiments/06-wind-audit.md).
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NotRequired, TypedDict, cast

import numpy as np

from ..arrays import Floats, Ints, Mask
from ..net import cached, get_json

HISTFC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


class Direction(TypedDict):
    from_deg: float  # where the wind comes FROM (degrees; NaN with no hours)
    from_compass: str | None  # None with no hours
    consistency: float  # resultant length 0-1: how often it holds that direction


class PeriodWind(Direction):
    speed_ms: float  # mean 10 m wind speed
    hours: int
    high_pressure_share: float
    rose: list[float]  # share of hours from each of the 16 SECTORS


class MonthWind(TypedDict):
    dawn: PeriodWind
    dusk: PeriodWind
    day: PeriodWind
    night: PeriodWind
    all_days: Direction


class GroundWind(TypedDict):
    """The ground-level (HRRR 10 m) high-pressure dawn+dusk wind, for reference."""

    from_compass: str | None
    from_deg: float
    consistency: float
    most_common_from: str | None
    dawn_from_compass: str | None
    dusk_from_compass: str | None
    source: str


class Wind(TypedDict):
    """The prevailing wind an analysis uses (see prevailing). States saved before most_common_from and ground
    have surface_dawn_from_compass / surface_dawn_consistency (ERA5 100 m) instead: read those keys with .get."""

    from_deg: float
    from_compass: str
    consistency: float  # resultant length R, 0-1 (1 = always that direction)
    most_common_from: NotRequired[str | None]  # the sector most dawn/dusk hours blow from (None if set by user)
    source: str
    all_days_from_compass: str | None
    all_days_consistency: float
    speed_ms: float
    day_from_deg: float
    day_from_compass: str | None
    day_consistency: float
    dawn: PeriodWind
    dusk: PeriodWind
    day: PeriodWind
    ground: NotRequired[GroundWind]


def compass(deg: float) -> str:
    return SECTORS[int(((deg % 360) + 11.25) // 22.5) % 16]


# level -> (direction, speed) variables
LEVELS = {
    "upper": ("wind_direction_850hPa", "wind_speed_850hPa"),
    "10m": ("wind_direction_10m", "wind_speed_10m"),
}
UPPER_SOURCE = "HRRR 850 hPa (~1.5 km up), high-pressure dawn+dusk, Open-Meteo historical forecasts 2022+"
GROUND_SOURCE = "HRRR 10 m (ground level), high-pressure dawn+dusk, Open-Meteo historical forecasts 2022+"


def fetch_hourly(lat: float, lon: float, level: str) -> dict[str, Any]:
    last = dt.date(dt.date.today().year - 1, 12, 31)
    dkey, skey = LEVELS[level]
    start = dt.date(2022, 1, 1)
    hourly = ",".join(dict.fromkeys([dkey, skey, "wind_speed_10m", "pressure_msl", "precipitation"]))
    key = (level, round(lat, 2), round(lon, 2), str(start), str(last))

    def build() -> dict[str, Any]:
        j: dict[str, Any] = get_json(
            HISTFC,
            params=dict(
                latitude=round(lat, 2),
                longitude=round(lon, 2),
                start_date=str(start),
                end_date=str(last),
                hourly=hourly,
                daily="sunrise,sunset",
                timezone="auto",
                wind_speed_unit="ms",
            ),
            timeout=240,
        )
        return j

    out: dict[str, Any] = cached("openmeteo", key, build)
    return out


def _circ(dirs_from_deg: Floats, weights: Floats | None = None) -> tuple[float, float]:
    """Mean 'from' direction (deg) and resultant length R (0 = all over the place, 1 = always the same)."""
    if len(dirs_from_deg) == 0:
        return float("nan"), 0.0
    th = np.radians(dirs_from_deg)
    w = np.ones_like(th) if weights is None else weights
    s, c = np.sum(w * np.sin(th)), np.sum(w * np.cos(th))
    return float(np.degrees(np.arctan2(s, c)) % 360), float(np.hypot(s, c) / np.sum(w))


def high_pressure(p: Floats, rain24: Floats, months: Ints) -> Mask:
    """Hours under high pressure: sea-level pressure at or above the month's median, not falling over 6 h, and dry
    for 24 h."""
    med = np.array([np.nanmedian(p[months == m]) if (months == m).any() else np.nan for m in range(1, 13)])
    tend6 = np.concatenate([np.zeros(6), p[6:] - p[:-6]])
    out: Mask = (p >= med[months - 1]) & (tend6 >= -0.5) & (rain24 < 0.2)
    return out


def climatology(lat: float, lon: float, level: str = "upper") -> dict[int, MonthWind]:
    """Per month (1-12) and period (dawn/dusk/day/night): prevailing high-pressure wind at one LEVELS level."""
    dkey, skey = LEVELS[level]

    def build() -> dict[int, MonthWind]:
        j = fetch_hourly(lat, lon, level)
        h = j["hourly"]
        t = np.array(h["time"], dtype="datetime64[m]")
        wd = np.array(h[dkey], dtype=float)
        ws = np.array(h[skey], dtype=float)
        ws10 = np.array(h["wind_speed_10m"], dtype=float)
        p = np.array(h["pressure_msl"], dtype=float)
        pr = np.nan_to_num(np.array(h["precipitation"], dtype=float))
        months = (t.astype("datetime64[M]").astype(int) % 12) + 1
        days = t.astype("datetime64[D]")
        minute = (t - days).astype(int)

        rise = {np.datetime64(s[:10]): np.datetime64(s) for s in j["daily"]["sunrise"] if s}
        sset = {np.datetime64(s[:10]): np.datetime64(s) for s in j["daily"]["sunset"] if s}
        d_rise = np.array(
            [(rise.get(d, d + np.timedelta64(360, "m")) - d).astype("timedelta64[m]").astype(int) for d in days]
        )
        d_set = np.array(
            [(sset.get(d, d + np.timedelta64(1080, "m")) - d).astype("timedelta64[m]").astype(int) for d in days]
        )

        period = np.full(len(t), "night", dtype=object)
        day = (minute > d_rise + 90) & (minute < d_set - 90)
        period[day] = "day"
        period[np.abs(minute - d_rise) <= 90] = "dawn"
        period[np.abs(minute - d_set) <= 90] = "dusk"

        rain24 = np.convolve(pr, np.ones(24), mode="full")[: len(pr)]
        hp = high_pressure(p, rain24, months) & ~np.isnan(wd)

        def per_period(m: int, per: str) -> PeriodWind:
            sel = hp & (months == m) & (period == per)
            allsel = (months == m) & (period == per) & ~np.isnan(wd)
            d, R = _circ(wd[sel], ws[sel])
            hist = np.bincount(((wd[sel] % 360 + 11.25) // 22.5).astype(int) % 16, minlength=16)
            return PeriodWind(
                from_deg=round(d, 1),
                from_compass=compass(d) if not np.isnan(d) else None,
                consistency=round(R, 2),
                speed_ms=round(float(np.nanmean(ws10[sel])) if sel.any() else 0.0, 1),
                hours=int(sel.sum()),
                high_pressure_share=round(float(sel.sum() / max(allsel.sum(), 1)), 2),
                rose=[round(float(x), 3) for x in hist / max(int(hist.sum()), 1)],
            )

        def month(m: int) -> MonthWind:
            sel = (months == m) & ~np.isnan(wd)
            d, R = _circ(wd[sel], ws[sel])
            return MonthWind(
                dawn=per_period(m, "dawn"),
                dusk=per_period(m, "dusk"),
                day=per_period(m, "day"),
                night=per_period(m, "night"),
                all_days=Direction(from_deg=round(d, 1), from_compass=compass(d), consistency=round(R, 2)),
            )

        return {m: month(m) for m in range(1, 13)}

    out: dict[int, MonthWind] = cached("windclim", (level, round(lat, 2), round(lon, 2), "v3"), build)
    return out


def most_common(rose: list[float]) -> str | None:
    """The sector the most hours blow from (None with no hours)."""
    return SECTORS[int(np.argmax(rose))] if sum(rose) > 0 else None


def dawn_dusk(dawn: PeriodWind, dusk: PeriodWind) -> tuple[float, float, list[float]]:
    """The dawn and dusk winds combined: mean direction (weighted by hours x consistency), consistency R (the
    periods' own R, hour-weighted, times how well the two agree) and the hour-weighted rose."""
    ds = np.array([dawn["from_deg"], dusk["from_deg"]])
    ws = np.array([dawn["hours"] * dawn["consistency"], dusk["hours"] * dusk["consistency"]]) + 1e-6
    d, R = _circ(ds, ws)
    R = (
        float(np.average([dawn["consistency"], dusk["consistency"]], weights=[dawn["hours"] + 1, dusk["hours"] + 1]))
        * R
    )
    n = max(dawn["hours"] + dusk["hours"], 1)
    rose = [
        round((a * dawn["hours"] + b * dusk["hours"]) / n, 3) for a, b in zip(dawn["rose"], dusk["rose"], strict=True)
    ]
    return d, R, rose


def prevailing(lat: float, lon: float, month: int, override_from_deg: float | None = None) -> Wind:
    """The wind that matters for hunting: high-pressure dawn+dusk 850 hPa flow, combined. Also returns the
    ground-level (10 m) wind for reference."""
    with ThreadPoolExecutor(2) as ex:  # two independent multi-MB downloads on a first run
        f_up, f_10 = ex.submit(climatology, lat, lon, "upper"), ex.submit(climatology, lat, lon, "10m")
        clim, g = f_up.result(), f_10.result()[month]
    m = clim[month]
    dawn, dusk, day = m["dawn"], m["dusk"], m["day"]
    d, R, rose = dawn_dusk(dawn, dusk)
    gd, gR, grose = dawn_dusk(g["dawn"], g["dusk"])
    source, common = UPPER_SOURCE, most_common(rose)
    if override_from_deg is not None:
        d, R, source, common = float(override_from_deg) % 360, 0.7, "set by user", None
    return Wind(
        from_deg=d,
        from_compass=compass(d),
        consistency=round(R, 2),
        most_common_from=common,
        source=source,
        all_days_from_compass=m["all_days"]["from_compass"],
        all_days_consistency=m["all_days"]["consistency"],
        speed_ms=round((dawn["speed_ms"] + dusk["speed_ms"]) / 2, 1),
        day_from_deg=day["from_deg"],
        day_from_compass=day["from_compass"],
        day_consistency=day["consistency"],
        dawn=dawn,
        dusk=dusk,
        day=day,
        ground=GroundWind(
            from_compass=compass(gd) if not np.isnan(gd) else None,
            from_deg=round(gd, 1),
            consistency=round(gR, 2),
            most_common_from=most_common(grose),
            dawn_from_compass=g["dawn"]["from_compass"],
            dusk_from_compass=g["dusk"]["from_compass"],
            source=GROUND_SOURCE,
        ),
    )


UNSTEADY_R = 0.35  # below this consistency the prevailing direction holds in only about a third of the hours


def describe(w: Wind) -> dict[str, Any]:
    """The prevailing wind as reports give it: direction, its consistency R and most common direction, the source,
    and the ground-level (10 m) wind for reference (states saved before it have the ERA5 valley dawn wind)."""
    out: dict[str, Any] = dict(
        prevailing_from=w["from_compass"],
        from_deg=round(w["from_deg"]),
        consistency=w["consistency"],
        most_common_from=w.get("most_common_from"),
        source=w.get("source"),
    )
    g = w.get("ground")
    if g is not None:
        out["ground_level"] = dict(
            from_compass=g["from_compass"],
            from_deg=round(g["from_deg"]) if not np.isnan(g["from_deg"]) else None,
            consistency=g["consistency"],
            most_common_from=g["most_common_from"],
            dawn_from=g["dawn_from_compass"],
            dusk_from=g["dusk_from_compass"],
            source=g["source"],
        )
    else:
        old = cast("dict[str, Any]", w).get("surface_dawn_from_compass")
        if old is not None:
            out["valley_dawn_from"] = old  # ERA5 100 m dawn wind (older states)
    return out


def label(w: Wind) -> str:
    """'NW (HRRR 850 hPa; consistency R 0.26, most common NE)': the direction with how far to trust it."""
    if w.get("source") == "set by user":
        return f"{w['from_compass']} (set by user)"
    common = w.get("most_common_from")
    level = "HRRR 850 hPa" if "850" in str(w.get("source", "")) else str(w.get("source"))
    tail = f", most common {common}" if common and common != w["from_compass"] else ""
    return f"{w['from_compass']} ({level}; consistency R {w['consistency']:.2f}{tail})"
