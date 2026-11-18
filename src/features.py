"""Features known BEFORE booking. Used identically in training and in the app.

Never use: actual/updated time, final status, delay, realized flight counts of the
day, or anything recorded after the booking decision.

Historical aggregates (the hourly activity profile) are passed in by the caller,
computed from training data only, so evaluation stays leakage-safe.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache

import holidays
import pandas as pd

CATEGORICAL = ["airline_code", "airport_code", "country", "weekday"]
NUMERIC = ["hour", "is_weekend", "is_holiday", "near_holiday", "month", "scheduled_activity"]

# Feature sets compared in training (ablation). Each is kept only if it helps on test.
FEATURE_SETS = {
    "base": (["airline_code", "airport_code", "weekday"], ["hour"]),
    "base+activity": (["airline_code", "airport_code", "weekday"], ["hour", "scheduled_activity"]),
    "base+calendar": (["airline_code", "airport_code", "weekday"], ["hour", "is_weekend", "is_holiday", "near_holiday"]),
    "full": (["airline_code", "airport_code", "country", "weekday"],
             ["hour", "is_weekend", "is_holiday", "near_holiday", "month", "scheduled_activity"]),
}


@lru_cache(maxsize=8)
def _israel_holidays(years: tuple[int, ...]):
    return holidays.country_holidays("IL", years=list(years))


def holiday_flags(dates: pd.Series) -> pd.DataFrame:
    """is_holiday: the date is an Israeli public holiday.
    near_holiday: within 2 days before or after one (travel peaks around holidays)."""
    d = pd.to_datetime(dates).dt.date
    years = tuple(sorted({x.year for x in d} | {x.year + 1 for x in d} | {x.year - 1 for x in d}))
    hol = _israel_holidays(years)
    is_hol = d.map(lambda x: x in hol)
    near = d.map(lambda x: any((x + timedelta(days=k)) in hol for k in (-2, -1, 1, 2)))
    return pd.DataFrame({"is_holiday": is_hol.astype(int), "near_holiday": (near & ~is_hol).astype(int)},
                        index=dates.index)


def build_features(flights: pd.DataFrame, profile: pd.Series) -> pd.DataFrame:
    """Feature frame for flights (needs: airline_code, airport_code, country, scheduled).

    profile: average scheduled departures per hour (0-23) from historical/training data.
    """
    sched = pd.to_datetime(flights["scheduled"])
    out = pd.DataFrame(index=flights.index)
    out["airline_code"] = flights["airline_code"].astype(str)
    out["airport_code"] = flights["airport_code"].astype(str)
    out["country"] = flights["country"].fillna("UNKNOWN").astype(str).str.upper()
    out["weekday"] = sched.dt.day_name()
    out["hour"] = sched.dt.hour
    out["month"] = sched.dt.month
    out["is_weekend"] = sched.dt.day_of_week.isin([4, 5]).astype(int)  # Friday, Saturday in Israel
    out = out.join(holiday_flags(sched))
    out["scheduled_activity"] = out["hour"].map(profile).astype(float)
    return out[CATEGORICAL + NUMERIC]


def single_flight(airline_code: str, airport_code: str, country: str | None,
                  departure: datetime) -> pd.DataFrame:
    """One-row input frame for the app, in the same shape the pipelines were trained on."""
    return pd.DataFrame([{"airline_code": airline_code, "airport_code": airport_code,
                          "country": country, "scheduled": pd.Timestamp(departure)}])


def hourly_count_features(days: list[date]) -> pd.DataFrame:
    """Calendar features for the hourly airport-activity model: one row per (day, hour)."""
    rows = [{"service_date": d, "hour": h} for d in days for h in range(24)]
    df = pd.DataFrame(rows)
    stamps = pd.to_datetime(df["service_date"])
    df["weekday_num"] = stamps.dt.day_of_week
    df["is_weekend"] = df["weekday_num"].isin([4, 5]).astype(int)
    df["month"] = stamps.dt.month
    df = df.join(holiday_flags(stamps))
    return df
