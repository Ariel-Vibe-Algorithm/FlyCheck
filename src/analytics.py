"""Historical analytics used by the app: hourly activity, congestion index, delay rates.

Everything here is computed from historical records. Nothing here is a model
prediction; the ML delay model arrives in M3 (src/train.py, src/predict.py).
"""
from __future__ import annotations

from datetime import time

import pandas as pd

# A scheduled day counts as "complete" if it has at least this share of the busiest
# day's departures. The archive's board window cuts off the first/last days of each
# snapshot (e.g. 2 or 14 flights); real quiet days such as Saturdays keep ~40%.
COMPLETE_DAY_SHARE = 0.30
# Passengers arrive before departure, so the congestion window looks forward
# from the airport-arrival hour: departures scheduled 1-4 hours later.
WINDOW_HOURS = (1, 2, 3)
LEVEL_CUTOFFS = (33, 66)  # score < 33 low, < 66 medium, else high


# ------------------------------------------------------------------ schedule profile
def international(flights: pd.DataFrame) -> pd.DataFrame:
    return flights[~flights["is_domestic"]]


def complete_schedule_days(flights: pd.DataFrame, direction: str = "D") -> list:
    """Service dates whose schedule looks complete (see COMPLETE_DAY_SHARE)."""
    rows = international(flights)
    counts = rows[rows["direction"] == direction].groupby("service_date").size()
    if counts.empty:
        return []
    return sorted(counts[counts >= COMPLETE_DAY_SHARE * counts.max()].index)


def hourly_profile(flights: pd.DataFrame, direction: str = "D") -> pd.Series:
    """Average number of scheduled international flights per hour of day (0-23),
    over complete days. Includes flights that were later cancelled, because they
    were scheduled and passengers planned around them."""
    days = complete_schedule_days(flights, "D")
    rows = international(flights)
    rows = rows[(rows["direction"] == direction) & rows["service_date"].isin(days)]
    per_day = rows.groupby(["service_date", "hour"]).size().unstack(fill_value=0)
    per_day = per_day.reindex(index=days, columns=range(24), fill_value=0)
    return per_day.mean().rename("flights_per_hour")


def window_counts(profile: pd.Series) -> pd.Series:
    """For each airport-arrival hour, departures scheduled in the forward window."""
    return pd.Series(
        {h: sum(profile[(h + k) % 24] for k in WINDOW_HOURS) for h in range(24)},
        name="window_departures",
    )


def congestion_at(departure: time, lead_hours: float, profile: pd.Series) -> dict:
    """Relative departure-hall activity when the traveler arrives.

    score 0-100 = share of the day's arrival hours with a smaller forward-window
    count. It is a relative activity index, not a queue time or % of capacity.
    """
    dep_min = departure.hour * 60 + departure.minute
    arr_min = int(round(dep_min - lead_hours * 60)) % 1440
    arr_hour = arr_min // 60
    windows = window_counts(profile)
    value = float(windows[arr_hour])
    score = round(100 * (windows < value).sum() / (len(windows) - 1))
    level = "low" if score < LEVEL_CUTOFFS[0] else "medium" if score < LEVEL_CUTOFFS[1] else "high"
    return {
        "arrival_time": time(arr_min // 60, arr_min % 60),
        "arrival_hour": arr_hour,
        "window_hours": [(arr_hour + k) % 24 for k in WINDOW_HOURS],
        "window_departures": value,
        "score": int(score),
        "level": level,
        "crosses_midnight": arr_min > dep_min,
    }


# ------------------------------------------------------------------- delay statistics
def labelled(flights: pd.DataFrame) -> pd.DataFrame:
    """Completed international departures with a trusted delay label."""
    return flights[flights["in_model_scope"]]


def rate(rows: pd.DataFrame) -> dict:
    n = int(len(rows))
    return {
        "n": n,
        "share_delayed_30": float(rows["delayed_30"].astype(float).mean()) if n else None,
        "median_delay": float(rows["delay_minutes"].median()) if n else None,
    }


def hour_band(hour: int, width: int = 3) -> tuple[int, int]:
    start = (hour // width) * width
    return start, start + width


def delay_reference(scope: pd.DataFrame, airline: str | None = None,
                    airport: str | None = None, hour: int | None = None) -> dict:
    """Historical delay rates for the pieces of a flight the traveler chose."""
    out = {}
    if airline:
        out["airline"] = rate(scope[scope["airline_code"] == airline])
    if airport:
        out["destination"] = rate(scope[scope["airport_code"] == airport])
    if hour is not None:
        lo, hi = hour_band(hour)
        out["hour_band"] = {**rate(scope[(scope["hour"] >= lo) & (scope["hour"] < hi)]),
                            "band": (lo, hi)}
    return out


def group_delay_stats(scope: pd.DataFrame, by: str | list[str], min_n: int = 1) -> pd.DataFrame:
    """Delay rate, median delay and sample size per group, smallest samples dropped."""
    grouped = scope.groupby(by, observed=True).agg(
        n=("delayed_30", "size"),
        share_delayed_30=("delayed_30", lambda s: s.astype(float).mean()),
        median_delay=("delay_minutes", "median"),
    ).reset_index()
    return grouped[grouped["n"] >= min_n].sort_values("n", ascending=False).reset_index(drop=True)
