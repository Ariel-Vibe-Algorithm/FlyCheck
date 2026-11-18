"""Tests for features, sources, baselines, the chronological split and the training gate."""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src import features, train
from src.estimators import GroupRateClassifier
from src.sources import _to_local_naive, load_rankair


# ------------------------------------------------------------------- features
def test_features_use_no_outcome_columns():
    flights = pd.DataFrame([{"airline_code": "LY", "airport_code": "ATH", "country": "GREECE",
                             "scheduled": pd.Timestamp("2026-07-02 10:00"),
                             "updated": pd.Timestamp("2026-07-02 11:00"), "status": "DEPARTED",
                             "delay_minutes": 60.0, "delayed_30": 1}])
    X = features.build_features(flights, pd.Series(5.0, index=range(24)))
    forbidden = {"updated", "status", "delay_minutes", "delayed_30", "outcome"}
    assert forbidden.isdisjoint(X.columns)
    assert list(X.columns) == features.CATEGORICAL + features.NUMERIC


def test_holiday_flags_known_dates():
    d = pd.Series(pd.to_datetime(["2026-04-02", "2026-03-31", "2026-07-15"]))  # Passover 1st day 2026 = Apr 2
    flags = features.holiday_flags(d)
    assert flags["is_holiday"].tolist() == [1, 0, 0]
    assert flags["near_holiday"].tolist() == [0, 1, 0]


def test_weekend_is_friday_saturday():
    flights = pd.DataFrame([{"airline_code": "LY", "airport_code": "ATH", "country": "GREECE",
                             "scheduled": pd.Timestamp(f"2026-07-0{d} 10:00")} for d in (2, 3, 4, 5)])
    X = features.build_features(flights, pd.Series(1.0, index=range(24)))
    assert X["is_weekend"].tolist() == [0, 1, 1, 0]  # Thu, Fri, Sat, Sun


# ------------------------------------------------------------------- baseline
def test_group_rate_smoothing_and_unknown_groups():
    X = pd.DataFrame({"airline_code": ["A"] * 10 + ["B"] * 2})
    y = [1] * 10 + [0] * 2
    clf = GroupRateClassifier("airline_code", smoothing=2).fit(X, y)
    p = clf.predict_proba(pd.DataFrame({"airline_code": ["A", "B", "ZZ"]}))[:, 1]
    overall = 10 / 12
    assert p[0] == pytest.approx((10 + 2 * overall) / 12)
    assert p[1] == pytest.approx((0 + 2 * overall) / 4)   # shrunk, not 0%
    assert p[2] == pytest.approx(overall)                  # unknown airline -> overall rate


# ---------------------------------------------------------------------- split
def test_chronological_split_keeps_days_whole_and_test_latest():
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"service_date": np.repeat(days, 10), "x": range(100)})
    tr, te = train.chronological_split(df, test_share=0.2)
    assert set(tr["service_date"]).isdisjoint(set(te["service_date"]))
    assert max(tr["service_date"]) < min(te["service_date"])
    assert len(te) == 20


# ---------------------------------------------------------------------- sources
def test_time_modes_for_timezone_aware_exports():
    s = pd.Series(["2026-07-02 10:00:00+00:00", "2026-07-02 13:00:00"])
    conv = _to_local_naive(s, "convert")
    wall = _to_local_naive(s, "wall")
    assert conv[0] == pd.Timestamp("2026-07-02 13:00")  # 10:00 UTC = 13:00 Israel summer time
    assert wall[0] == pd.Timestamp("2026-07-02 10:00")  # clock reading kept
    assert conv[1] == wall[1] == pd.Timestamp("2026-07-02 13:00")  # naive = already local


def test_rankair_adapter_accepts_database_column_names(tmp_path):
    path = tmp_path / "export.csv"
    pd.DataFrame([{
        "airline_code": "LY", "flight_number": "315", "direction": "D", "location_iata": "LHR",
        "location_city_en": "LONDON", "country_en": "UNITED KINGDOM", "airline_name": "EL AL",
        "scheduled_time": "2026-07-02 10:00:00", "actual_time": "2026-07-02 10:45:00",
        "status_en": "DEPARTED", "terminal": "3", "scrape_timestamp": "2026-07-02 09:00:00+00:00",
    }]).to_csv(path, index=False)
    df, info = load_rankair(path)
    row = df.iloc[0]
    assert (row["airport_code"], row["status"], row["source"]) == ("LHR", "DEPARTED", "rankair")
    assert row["updated"] - row["scheduled"] == pd.Timedelta(minutes=45)
    assert row["snapshot_time"] == pd.Timestamp("2026-07-02 12:00")


def test_rankair_adapter_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"foo": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cannot find columns"):
        load_rankair(path)


# ------------------------------------------------------------------ training gate
def synthetic_flights(n_days=20, per_day=40, cancel_every=0, airline_signal=True, seed=0,
                      hour_signal=False):
    """Synthetic processed flights where airline 'BAD' is late 80% and 'GOOD' 20%.
    hour_signal: lateness depends only on the hour (afternoon 85%, morning 15%)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = date(2026, 6, 1) + timedelta(days=d)
        for i in range(per_day):
            al = "BAD" if i % 2 else "GOOD"
            sched = datetime(day.year, day.month, day.day, 5 + i % 18, 0)
            p = (0.8 if al == "BAD" else 0.2) if airline_signal else 0.5
            if hour_signal:
                p = 0.85 if sched.hour >= 14 else 0.15
            late = int(rng.random() < p)
            cancelled = cancel_every and i % cancel_every == 0
            rows.append(dict(
                airline_code=al, airline_name=al, flight_number=str(i), direction="D",
                airport_code="ATH", city="ATHENS", country="GREECE", terminal="3", status="DEPARTED",
                scheduled=sched, updated=sched + timedelta(minutes=45 if late else 5),
                snapshot_time=sched, source="test", service_date=day, hour=sched.hour,
                weekday=sched.strftime("%A"), is_domestic=False, suspect_early=False,
                outcome="cancelled" if cancelled else "completed",
                delay_minutes=np.nan if cancelled else (45.0 if late else 5.0),
                delayed_30=pd.NA if cancelled else late,
                in_model_scope=not cancelled))
    df = pd.DataFrame(rows)
    df["delayed_30"] = df["delayed_30"].astype("Int64")
    return df


def test_training_run_reports_all_tasks_and_gates_cancellation():
    metrics, artifacts = train.run(synthetic_flights())
    assert set(metrics["tasks"]) == {"delay", "cancellation", "activity"}
    # no cancellations at all -> cancellation must stay baseline-only
    assert metrics["tasks"]["cancellation"]["enough_data_for_model"] is False
    assert artifacts["cancellation"] is None
    delay = metrics["tasks"]["delay"]
    assert delay["enough_data_for_model"] is True
    assert set(delay["split"]["test_days"]).isdisjoint(set(delay["split"]["train_days"]))


def test_model_without_signal_is_not_ready():
    metrics, _ = train.run(synthetic_flights(airline_signal=False, seed=1))
    assert metrics["tasks"]["delay"]["ready_for_app"] is False


def test_model_with_signal_beyond_airline_is_ready():
    # lateness depends on the hour, which the airline baseline cannot see
    metrics, artifacts = train.run(synthetic_flights(airline_signal=False, hour_signal=True, seed=2))
    delay = metrics["tasks"]["delay"]
    assert delay["ready_for_app"] is True, delay["status_reason"]
    assert delay["brier_gain_ci95"][0] > 0
    assert artifacts["delay"] is not None
