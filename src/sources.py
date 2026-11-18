"""Source adapters: every flight data source -> the FlyCheck internal schema.

Drop files into data/raw/ and run `python -m src.data_processing`; every source
found is loaded, converted and merged automatically.

| Source      | Where                            | Format                                        |
|-------------|----------------------------------|-----------------------------------------------|
| odata       | data/raw/odata/flydata_log.csv   | OData log of IAA board snapshots (CC BY)      |
| govil       | data/raw/govil/board_*.json      | data.gov.il board snapshots (download_data)   |
| rankair     | data/raw/rankair/*.csv           | Rank Air export (one row per flight occurrence)|

Internal schema (see data_processing.to_internal_schema): airline_code, airline_name,
flight_number, direction, airport_code, city, country, terminal, checkin_counters,
checkin_zone, status, scheduled, updated, snapshot_time, source.
All times are naive Israel local time.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_processing import (KEY_FIELDS, ROOT, internal_text_fields, load_flydata_log,
                                 to_internal_schema)

RAW_DIR = ROOT / "data" / "raw"
LOCAL_TZ = "Asia/Jerusalem"


# --------------------------------------------------------------------------- odata
def load_odata(path: Path) -> tuple[pd.DataFrame, dict]:
    raw, malformed = load_flydata_log(path)
    return to_internal_schema(raw, "odata"), {"rows": len(raw), "malformed_lines_skipped": malformed}


# --------------------------------------------------------------------------- govil
def load_govil_snapshot(path: Path) -> tuple[pd.DataFrame, dict]:
    """A data.gov.il datastore_search JSON saved by download_data (board_<UTC stamp>.json)."""
    records = json.loads(Path(path).read_text("utf-8"))["result"]["records"]
    raw = pd.DataFrame(records).astype(str)
    df = to_internal_schema(raw, "govil")
    stamp = re.search(r"(\d{8}T\d{6}Z)", Path(path).name)
    if stamp:  # retrieval time, converted to Israel local time
        utc = pd.Timestamp(datetime.strptime(stamp.group(1), "%Y%m%dT%H%M%SZ"), tz="UTC")
        df["snapshot_time"] = utc.tz_convert(LOCAL_TZ).tz_localize(None)
    return df, {"rows": len(raw)}


# ------------------------------------------------------------------------- rankair
# Rank Air publishes both its CSV (transformed) names and its database column names
# (docs/reference/FIELD_MAPPING_TABLE.md in Daniel1Maymon/israel-flights-etl).
# Each internal field lists the accepted column names, first match wins.
RANKAIR_COLUMNS = {
    "airline_code": ["airline_code", "CHOPER"],
    "airline_name": ["airline_name", "CHOPERD"],
    "flight_number": ["flight_number", "CHFLTN"],
    "direction": ["direction", "arrival_departure_code", "CHAORD"],
    "airport_code": ["location_iata", "airport_code", "CHLOC1"],
    "city": ["location_city_en", "city_name_english", "CHLOC1T"],
    "country": ["country_en", "country_name_english", "CHLOCCT"],
    "terminal": ["terminal", "terminal_number", "CHTERM"],
    "checkin_counters": ["checkin_counters", "check_in_time", "CHCINT"],
    "checkin_zone": ["checkin_zone", "check_in_zone", "CHCKZN"],
    "status": ["status_en", "status_english", "CHRMINE"],
    "scheduled": ["scheduled_time", "scheduled_departure", "scheduled_departure_time", "CHSTOL"],
    "updated": ["actual_time", "actual_departure", "actual_departure_time", "CHPTOL"],
    "snapshot_time": ["scrape_timestamp"],
}


def _pick(raw: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in raw.columns:
            return raw[name]
    return None


def _to_local_naive(values: pd.Series, mode: str) -> pd.Series:
    """Parse timestamps into naive Israel local time.

    mode="convert": timezone-aware values are true instants -> convert to Israel time.
    mode="wall":    the stored clock reading already is Israel local time (a naive local
                    value saved into a timestamptz column) -> just drop the offset.
    Naive values are taken as Israel local time in both modes.
    """
    text = values.astype(str).where(values.notna())
    aware = text.str.contains(r"(?:[+-]\d{2}(?::?\d{2})?|Z)$", regex=True, na=False)
    out = pd.to_datetime(text.where(~aware), errors="coerce")
    if aware.any():
        if mode == "wall":
            stripped = text[aware].str.replace(r"(?:[+-]\d{2}(?::?\d{2})?|Z)$", "", regex=True)
            out[aware] = pd.to_datetime(stripped, errors="coerce")
        else:
            parsed = pd.to_datetime(text[aware], errors="coerce", utc=True)
            out[aware] = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return out


def load_rankair(path: Path, time_mode: str = "convert") -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [f for f in ("airline_code", "flight_number", "direction", "airport_code", "scheduled")
               if _pick(raw, RANKAIR_COLUMNS[f]) is None]
    if missing:
        raise ValueError(f"{Path(path).name}: cannot find columns for {missing}. "
                         f"Columns present: {list(raw.columns)}")
    df = internal_text_fields(
        {name: _pick(raw, aliases) for name, aliases in RANKAIR_COLUMNS.items()}, raw.index)
    df["scheduled"] = _to_local_naive(_pick(raw, RANKAIR_COLUMNS["scheduled"]), time_mode)
    upd = _pick(raw, RANKAIR_COLUMNS["updated"])
    df["updated"] = _to_local_naive(upd, time_mode) if upd is not None else pd.NaT
    snap = _pick(raw, RANKAIR_COLUMNS["snapshot_time"])
    # scrape_timestamp is a real instant (set by their loader), so always convert it.
    df["snapshot_time"] = _to_local_naive(snap, "convert") if snap is not None else pd.NaT
    df["source"] = "rankair"
    df = df.dropna(subset=KEY_FIELDS)
    return df, {"rows": len(raw), "time_mode": time_mode}


def choose_rankair_time_mode(rankair_path: Path, reference: pd.DataFrame) -> tuple[str, dict]:
    """Pick the time interpretation under which Rank Air flights line up with a
    reference source (e.g. the OData archive) on overlapping days."""
    keys = KEY_FIELDS
    ref = set(map(tuple, reference[keys].astype(str).values))
    scores = {}
    for mode in ("convert", "wall"):
        df, _ = load_rankair(rankair_path, mode)
        overlap = df[df["scheduled"].dt.date.isin(set(reference["scheduled"].dt.date))]
        hits = sum(tuple(r) in ref for r in overlap[keys].astype(str).values)
        scores[mode] = {"overlap_rows": int(len(overlap)), "matches": int(hits)}
    best = max(scores, key=lambda m: scores[m]["matches"])
    return best, scores


# ---------------------------------------------------------------------- discovery
def load_all(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Load every source found under data/raw/. Returns (internal rows, per-source report)."""
    frames, report = [], {}
    odata_path = raw_dir / "odata" / "flydata_log.csv"
    odata = None
    if odata_path.exists():
        odata, info = load_odata(odata_path)
        frames.append(odata)
        report["odata"] = info
    for path in sorted((raw_dir / "govil").glob("board_*.json")):
        df, info = load_govil_snapshot(path)
        frames.append(df)
        report.setdefault("govil", {"files": 0, "rows": 0})
        report["govil"]["files"] += 1
        report["govil"]["rows"] += info["rows"]
    for path in sorted((raw_dir / "rankair").glob("*.csv")):
        mode, scores = ("convert", None)
        if odata is not None:
            mode, scores = choose_rankair_time_mode(path, odata)
        df, info = load_rankair(path, mode)
        frames.append(df)
        report.setdefault("rankair", {"files": []})["files"].append(
            {"file": path.name, **info, "time_check_vs_odata": scores})
    if not frames:
        raise FileNotFoundError(f"No flight data found under {raw_dir}. Run python -m src.download_data")
    return pd.concat(frames, ignore_index=True), report
