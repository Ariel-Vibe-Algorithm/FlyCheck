"""Tests for src.analytics and the context-data helpers."""
from datetime import date, time

import pandas as pd
import pytest

from src import analytics
from src.data_processing import chain_price_index, process_tlvflights


def flat_profile(peak_hours=()):
    """1 departure every hour, 10 in the given peak hours."""
    return pd.Series({h: (10.0 if h in peak_hours else 1.0) for h in range(24)})


def test_congestion_window_crosses_midnight():
    # depart 01:00, arrive 3h earlier = 22:00 the previous evening; window 23, 00, 01
    c = analytics.congestion_at(time(1, 0), 3, flat_profile())
    assert c["arrival_time"] == time(22, 0)
    assert c["window_hours"] == [23, 0, 1]
    assert c["crosses_midnight"] is True


def test_congestion_score_is_highest_when_window_covers_peak():
    profile = flat_profile(peak_hours=(6, 7, 8))
    busy = analytics.congestion_at(time(8, 0), 3, profile)    # arrive 05:00 -> 06,07,08
    quiet = analytics.congestion_at(time(18, 0), 3, profile)  # arrive 15:00 -> 16,17,18
    assert busy["score"] == 100 and busy["level"] == "high"
    assert quiet["level"] == "low"
    assert busy["window_departures"] == 30


def test_half_hour_lead_time():
    c = analytics.congestion_at(time(10, 15), 3.5, flat_profile())
    assert c["arrival_time"] == time(6, 45)
    assert c["arrival_hour"] == 6


def make_flights(day_counts):
    """Synthetic processed flights: day_counts = {date: number of departures at 10:00}."""
    rows = []
    for d, n in day_counts.items():
        for i in range(n):
            rows.append(dict(direction="D", service_date=d, hour=10, is_domestic=False,
                             in_model_scope=True, airline_code="LY", airport_code="ATH",
                             delayed_30=i % 2, delay_minutes=float(i)))
    return pd.DataFrame(rows)


def test_partial_snapshot_days_are_not_complete():
    flights = make_flights({date(2026, 7, 1): 2, date(2026, 7, 2): 200, date(2026, 7, 4): 90})
    assert analytics.complete_schedule_days(flights) == [date(2026, 7, 2), date(2026, 7, 4)]
    profile = analytics.hourly_profile(flights)
    assert profile[10] == pytest.approx(145.0)  # mean of 200 and 90
    assert profile[3] == 0


def test_delay_reference_and_hour_band():
    flights = make_flights({date(2026, 7, 2): 10})
    ref = analytics.delay_reference(analytics.labelled(flights), airline="LY", hour=11)
    assert ref["airline"]["n"] == 10
    assert ref["airline"]["share_delayed_30"] == pytest.approx(0.5)
    assert ref["hour_band"]["band"] == (9, 12)
    assert analytics.delay_reference(flights, airline="XX")["airline"]["n"] == 0


def test_tlvflights_rates():
    raw = pd.DataFrame([dict(airline_iata="LY", airline_name_en="El Al", total_flights=110,
                             on_time_0_15min=60, delayed_16_30min=20, delayed_31_60min=10,
                             delayed_61_120min=6, delayed_over_120min=4, cancelled=10,
                             avg_delay_minutes=30.0)])
    row = process_tlvflights(raw).iloc[0]
    assert row["late_30_rate"] == pytest.approx(0.20)   # 20 of 100 completed
    assert row["on_time_rate"] == pytest.approx(0.60)
    assert row["cancel_rate"] == pytest.approx(10 / 110)


def test_price_index_chaining_across_rebase():
    recs = [
        {"year": 2024, "month": 12, "percent": 0.0, "currBase": {"baseDesc": "Average 2022", "value": 150.0}},
        {"year": 2025, "month": 1, "percent": 10.0, "currBase": {"baseDesc": "Average 2024", "value": 110.0}},
    ]
    out = chain_price_index(recs)
    assert out["index_value"].tolist() == [100.0, 110.0]  # old base value ignored, chained by %
    assert out["base"].iloc[0] == "Average 2024"
