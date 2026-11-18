"""Tests for src.data_processing using small synthetic flight-board snapshots."""
import pandas as pd
import pytest

from src.data_processing import (
    ODATA_COLUMNS,
    build_flights,
    load_flydata_log,
)


def snap(choper, fltn, stol, ptol, status, logged, aord="D", loc="FCO", term="3"):
    """One raw board row in OData-log format."""
    row = dict.fromkeys(ODATA_COLUMNS, "null")
    row.update(CHOPER=choper, CHFLTN=fltn, CHOPERD=f"{choper} AIR", CHSTOL=stol,
               CHPTOL=ptol, CHAORD=aord, CHLOC1=loc, CHLOC1T=loc, CHLOCCT="X",
               CHTERM=term, CHRMINE=status, log_timestamp=logged, log_action="INSERT")
    return row


def test_duplicate_snapshots_collapse_to_latest():
    raw = pd.DataFrame([
        snap("LY", "1", "2026-07-02T10:00:00", "2026-07-02T10:00:00", "ON TIME", "2026-07-02 08:00:00"),
        snap("LY", "1", "2026-07-02T10:00:00", "2026-07-02T10:40:00", "DEPARTED", "2026-07-02 11:00:00"),
    ])
    flights = build_flights(raw)
    assert len(flights) == 1
    row = flights.iloc[0]
    assert row["status"] == "DEPARTED"
    assert row["delay_minutes"] == 40
    assert row["delayed_30"] == 1


def test_codeshares_counted_once_and_operator_kept():
    t = ("2026-07-02T14:55:00", "2026-07-02T15:51:00")
    raw = pd.DataFrame([
        snap("AZ", "807", *t, "DEPARTED", "2026-07-02 16:00:00"),
        snap("LH", "5233", *t, "DEPARTED", "2026-07-02 16:00:00"),
    ])
    flights = build_flights(raw)
    assert len(flights) == 1
    assert flights.iloc[0]["airline_code"] == "AZ"
    assert flights.iloc[0]["codeshares"] == "AZ 807, LH 5233"


def test_cancelled_and_unresolved_have_no_target():
    raw = pd.DataFrame([
        snap("LY", "2", "2026-07-02T10:00:00", "2026-07-02T10:00:00", "CANCELED", "2026-07-02 11:00:00"),
        snap("LY", "3", "2026-07-03T10:00:00", "2026-07-03T10:00:00", "ON TIME", "2026-07-02 11:00:00"),
    ])
    flights = build_flights(raw).set_index("flight_number")
    assert flights.loc["2", "outcome"] == "cancelled"
    assert flights.loc["3", "outcome"] == "unresolved"
    assert flights["delay_minutes"].isna().all()
    assert flights["delayed_30"].isna().all()


def test_delay_across_midnight():
    raw = pd.DataFrame([
        snap("6H", "9", "2026-07-02T23:50:00", "2026-07-03T00:35:00", "DEPARTED", "2026-07-03 01:00:00"),
    ])
    row = build_flights(raw).iloc[0]
    assert row["delay_minutes"] == 45
    assert row["delayed_30"] == 1
    assert str(row["service_date"]) == "2026-07-02"  # scheduled date, not actual


def test_early_departure_is_negative_and_not_delayed():
    raw = pd.DataFrame([
        snap("W6", "5", "2026-07-02T12:00:00", "2026-07-02T11:50:00", "DEPARTED", "2026-07-02 13:00:00"),
    ])
    row = build_flights(raw).iloc[0]
    assert row["delay_minutes"] == -10
    assert row["delayed_30"] == 0


def test_suspect_early_and_domestic_are_flagged_not_deleted():
    raw = pd.DataFrame([
        # left 2h early -> probably an unrecorded schedule change
        snap("BZ", "762", "2026-07-02T08:10:00", "2026-07-02T06:03:00", "DEPARTED", "2026-07-02 09:00:00"),
        # domestic (Eilat)
        snap("IZ", "1", "2026-07-02T09:00:00", "2026-07-02T09:10:00", "DEPARTED", "2026-07-02 10:00:00", loc="ETM"),
        # normal international departure
        snap("LY", "7", "2026-07-02T10:00:00", "2026-07-02T10:20:00", "DEPARTED", "2026-07-02 11:00:00"),
    ])
    raw.loc[1, "CHLOCCT"] = "ISRAEL"
    report = {}
    flights = build_flights(raw, report=report).set_index("flight_number")
    assert len(flights) == 3  # nothing deleted
    assert bool(flights.loc["762", "suspect_early"])
    assert bool(flights.loc["1", "is_domestic"])
    assert flights["in_model_scope"].tolist().count(True) == 1
    assert bool(flights.loc["7", "in_model_scope"])
    assert report["model_scope_departures"] == 1
    assert report["completed_suspect_early_excluded"] == 1
    assert report["completed_domestic_excluded"] == 1


def test_malformed_line_is_skipped(tmp_path):
    header = ",".join(ODATA_COLUMNS)
    good = ",".join(f'"{v}"' for v in snap("LY", "1", "2026-07-02T10:00:00",
                    "2026-07-02T10:05:00", "DEPARTED", "2026-07-02 11:00:00").values())
    truncated = '"9","LY","4","EL AL","2026-07-02T10:00:00","2026-07-02T10:00:00","D","ATH","2026-07-02 11:'
    path = tmp_path / "log.csv"
    path.write_text("\n".join([header, good, truncated, good]) + "\n", encoding="utf-8")
    rows, malformed = load_flydata_log(path)
    assert malformed >= 1
    assert len(rows) >= 1


def test_unexpected_header_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_flydata_log(path)
