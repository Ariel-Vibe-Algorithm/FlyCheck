"""Turn raw flight-board snapshots into one clean row per physical flight.

Pipeline (python -m src.data_processing):
    raw OData log (many snapshots per flight)
      -> load_flydata_log          robust CSV read, malformed lines counted and skipped
      -> to_internal_schema        source-specific columns -> FlyCheck internal schema
      -> latest_snapshot_per_flight one row per (airline, flight number, direction, scheduled time)
      -> dedupe_codeshares         one row per physical flight
      -> add_delay_target          delay_minutes and delayed_30 for completed departures
      -> data/processed/flights.parquet

Times are naive Israel local time (verified 2026-11-18 against the live board).

The internal schema is source-independent, so a future source (e.g. Rank Air,
EUROCONTROL) only needs its own `to_internal_schema`-style adapter.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ODATA = ROOT / "data" / "raw" / "odata" / "flydata_log.csv"
RAW_TLVFLIGHTS = ROOT / "data" / "raw" / "tlvflights" / "airline_stats.csv"
RAW_CBS = ROOT / "data" / "raw" / "cbs" / "price_index_121390.json"
PROCESSED_DIR = ROOT / "data" / "processed"

ODATA_COLUMNS = [
    "_id", "CHOPER", "CHFLTN", "CHOPERD", "CHSTOL", "CHPTOL", "CHAORD", "CHLOC1",
    "CHLOC1D", "CHLOC1TH", "CHLOC1T", "CHLOC1CH", "CHLOCCT", "CHTERM", "CHCINT",
    "CHCKZN", "CHRMINE", "CHRMINH", "log_timestamp", "log_action",
]

# Final-status classes. Only COMPLETED departures get a delay label.
COMPLETED_STATUSES = {"DEPARTED", "LANDED"}
CANCELLED_STATUSES = {"CANCELED"}

DELAY_THRESHOLD_MIN = 30
# A scheduled airliner leaving more than 30 min early almost always means the
# schedule changed and CHSTOL was not updated, so we flag it rather than trust it.
SUSPECT_EARLY_MIN = -30
DOMESTIC_COUNTRY = "ISRAEL"  # e.g. Eilat (ETM); out of scope (international only)


# --------------------------------------------------------------------------- load
def load_flydata_log(path: Path | str = RAW_ODATA) -> tuple[pd.DataFrame, int]:
    """Read the OData log. Returns (rows, number_of_malformed_rows_skipped).

    The file contains at least one truncated line, whose open quote swallows the
    next line, so we read with the csv module and keep only rows with the exact
    expected number of fields.
    """
    good, bad = [], 0
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ODATA_COLUMNS:
            raise ValueError(f"Unexpected OData header: {header}")
        for row in reader:
            if len(row) == len(ODATA_COLUMNS):
                good.append(row)
            else:
                bad += 1
    return pd.DataFrame(good, columns=ODATA_COLUMNS), bad


# ------------------------------------------------------------------------ adapter
TEXT_FIELDS = {  # internal name -> IAA board column
    "airline_code": "CHOPER", "airline_name": "CHOPERD", "flight_number": "CHFLTN",
    "direction": "CHAORD",          # "D" departure, "A" arrival
    "airport_code": "CHLOC1",       # the other (non-TLV) airport
    "city": "CHLOC1T", "country": "CHLOCCT", "terminal": "CHTERM",
    "checkin_counters": "CHCINT", "checkin_zone": "CHCKZN", "status": "CHRMINE",
}
KEY_FIELDS = ["airline_code", "flight_number", "direction", "scheduled"]


def clean_text(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Strip text and turn 'null'/'None'/'' into a real missing value."""
    if values is None:
        return pd.Series(None, index=index, dtype=object)
    s = values.astype(str).str.strip()
    return s.where(~s.isin(["", "null", "None", "nan", "NaN", "NaT"]), None)


def internal_text_fields(columns: dict[str, pd.Series | None], index: pd.Index) -> pd.DataFrame:
    """Build the text part of the internal schema from {internal_name: raw column}."""
    df = pd.DataFrame({name: clean_text(columns.get(name), index) for name in TEXT_FIELDS}, index=index)
    df["airline_name"] = df["airline_name"].str.replace(r"\s+", " ", regex=True)
    df["terminal"] = df["terminal"].str.replace(r"\.0$", "", regex=True)
    return df


def to_internal_schema(raw: pd.DataFrame, source: str = "odata") -> pd.DataFrame:
    """Map IAA flight-board columns (OData log or data.gov.il) to the internal schema."""
    df = internal_text_fields({name: raw[col] for name, col in TEXT_FIELDS.items()}, raw.index)
    df["scheduled"] = pd.to_datetime(clean_text(raw["CHSTOL"], raw.index), errors="coerce")
    df["updated"] = pd.to_datetime(clean_text(raw["CHPTOL"], raw.index), errors="coerce")
    df["snapshot_time"] = (pd.to_datetime(clean_text(raw["log_timestamp"], raw.index), errors="coerce")
                           if "log_timestamp" in raw else pd.NaT)
    df["source"] = source
    return df.dropna(subset=KEY_FIELDS)


# -------------------------------------------------------------------- dedupe steps
FLIGHT_KEY = ["airline_code", "flight_number", "direction", "scheduled"]


def latest_snapshot_per_flight(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent snapshot of each flight occurrence."""
    ordered = df.sort_values("snapshot_time", kind="stable", na_position="first")
    latest = ordered.groupby(FLIGHT_KEY, as_index=False, sort=False).tail(1)
    return latest.reset_index(drop=True)


def _flight_number_rank(numbers: pd.Series) -> pd.Series:
    """Lower = more likely the operating carrier (marketing codeshares tend to use
    long/high numbers such as LH 5233 for AZ 807). A documented heuristic."""
    numeric = pd.to_numeric(numbers.str.extract(r"(\d+)")[0], errors="coerce")
    return numeric.fillna(np.inf)


def dedupe_codeshares(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse codeshare listings of the same physical flight into one row.

    Codeshares share direction, airport, scheduled time, updated time and terminal.
    The row with the lowest flight number is kept as the operating flight; all
    marketing listings are kept in `codeshares`.
    """
    phys_key = ["direction", "airport_code", "scheduled", "updated", "terminal"]
    df = df.copy()
    df["_rank"] = _flight_number_rank(df["flight_number"])
    df["_label"] = df["airline_code"] + " " + df["flight_number"]
    labels = (df.sort_values("_rank").groupby(phys_key, dropna=False)["_label"]
                .agg(lambda s: ", ".join(s)).rename("codeshares"))
    operating = (df.sort_values("_rank", kind="stable")
                   .drop_duplicates(subset=phys_key, keep="first"))
    out = operating.merge(labels, left_on=phys_key, right_index=True, how="left")
    return out.drop(columns=["_rank", "_label"]).reset_index(drop=True)


# ------------------------------------------------------------------------- target
def add_delay_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome, delay_minutes and delayed_30.

    outcome: completed | cancelled | unresolved. Only completed flights get a delay;
    cancelled and unresolved flights are kept (reported separately) with NaN target.
    """
    df = df.copy()
    df["outcome"] = np.select(
        [df["status"].isin(COMPLETED_STATUSES), df["status"].isin(CANCELLED_STATUSES)],
        ["completed", "cancelled"],
        default="unresolved",
    )
    minutes = (df["updated"] - df["scheduled"]).dt.total_seconds() / 60
    df["delay_minutes"] = minutes.where(df["outcome"] == "completed")
    df["delayed_30"] = (df["delay_minutes"] >= DELAY_THRESHOLD_MIN).astype("Int64")
    df.loc[df["delay_minutes"].isna(), "delayed_30"] = pd.NA
    # Calendar helpers (Israel local time).
    df["service_date"] = df["scheduled"].dt.date
    df["hour"] = df["scheduled"].dt.hour
    df["weekday"] = df["scheduled"].dt.day_name()
    return df


def flag_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows instead of deleting them, so every exclusion is visible and counted.

    is_domestic    flight to/from an Israeli airport (outside the project scope)
    suspect_early  'departed' > 30 min before schedule: likely an unrecorded schedule change
    in_model_scope completed international departure with a trustworthy delay label
    """
    df = df.copy()
    df["is_domestic"] = df["country"].fillna("").str.upper().eq(DOMESTIC_COUNTRY)
    df["suspect_early"] = df["delay_minutes"].lt(SUSPECT_EARLY_MIN).fillna(False)
    df["in_model_scope"] = (
        df["direction"].eq("D")
        & df["outcome"].eq("completed")
        & ~df["is_domestic"]
        & ~df["suspect_early"]
    )
    return df


def build_flights(raw: pd.DataFrame, source: str = "odata",
                  report: dict | None = None) -> pd.DataFrame:
    """Full pipeline from raw board rows to one row per physical flight.

    If `report` is given, it is filled with row counts after each cleaning step.
    """
    def log(step: str, value: int) -> None:
        if report is not None:
            report[step] = int(value)

    log("raw_rows", len(raw))
    return build_flights_from_internal(to_internal_schema(raw, source), report)


def build_flights_from_internal(internal: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Pipeline from internal-schema rows (one or many sources) to one row per physical
    flight. Across sources, the most recent snapshot of a flight wins."""
    def log(step: str, value: int) -> None:
        if report is not None:
            report[step] = int(value)

    flights = internal
    log("rows_with_valid_keys", len(flights))
    flights = latest_snapshot_per_flight(flights)
    log("unique_flight_listings", len(flights))
    flights = dedupe_codeshares(flights)
    log("physical_flights", len(flights))
    flights = flag_quality_issues(add_delay_target(flights))

    dep = flights[flights["direction"] == "D"]
    log("departures", len(dep))
    for outcome in ("completed", "cancelled", "unresolved"):
        log(f"departures_{outcome}", (dep["outcome"] == outcome).sum())
    done = dep[dep["outcome"] == "completed"]
    log("completed_domestic_excluded", done["is_domestic"].sum())
    log("completed_suspect_early_excluded", (done["suspect_early"] & ~done["is_domestic"]).sum())
    log("model_scope_departures", flights["in_model_scope"].sum())
    return flights.sort_values("scheduled").reset_index(drop=True)


# ----------------------------------------------------------------- context data
def process_tlvflights(raw: pd.DataFrame) -> pd.DataFrame:
    """Airline-level reliability from the TLVFlights export (one row per airline).

    late_30_rate uses their 31+ minute buckets over completed (non-cancelled) flights.
    TLVFlights estimates gate time as takeoff minus 15 min, so these rates are not
    directly comparable with delays computed from the IAA board.
    """
    completed = raw["total_flights"] - raw["cancelled"]
    late = raw["delayed_31_60min"] + raw["delayed_61_120min"] + raw["delayed_over_120min"]
    out = pd.DataFrame({
        "airline_code": raw["airline_iata"],
        "airline_name": raw["airline_name_en"],
        "total_flights": raw["total_flights"],
        "on_time_rate": raw["on_time_0_15min"] / completed,
        "late_30_rate": late / completed,
        "cancel_rate": raw["cancelled"] / raw["total_flights"],
        "avg_delay_minutes": raw["avg_delay_minutes"],
    })
    return out.sort_values("total_flights", ascending=False).reset_index(drop=True)


def chain_price_index(records: list[dict]) -> pd.DataFrame:
    """CBS monthly index records -> one continuous series on the latest base.

    CBS re-bases the index every two years, so raw values from different periods
    are not comparable. We start from the latest value and walk backwards using
    each month's published percent change: value[t-1] = value[t] / (1 + pct[t]/100).
    """
    recs = sorted(records, key=lambda r: (r["year"], r["month"]))
    values = [0.0] * len(recs)
    values[-1] = float(recs[-1]["currBase"]["value"])
    for i in range(len(recs) - 1, 0, -1):
        values[i - 1] = values[i] / (1 + float(recs[i]["percent"]) / 100)
    return pd.DataFrame({
        "month": pd.to_datetime([f"{r['year']}-{r['month']:02d}-01" for r in recs]),
        "index_value": [round(v, 2) for v in values],
        "base": recs[-1]["currBase"]["baseDesc"],
    })


# --------------------------------------------------------------------------- main
def model_scope_summary(flights: pd.DataFrame) -> dict:
    """Headline numbers of the labelled set (completed international departures)."""
    scope = flights[flights["in_model_scope"]]
    return {
        "n": int(len(scope)),
        "median_delay_min": float(scope["delay_minutes"].median()),
        "share_delayed_30": round(float(scope["delayed_30"].mean()), 4),
        "first_day": str(scope["service_date"].min()),
        "last_day": str(scope["service_date"].max()),
        "per_day": {str(k): int(v) for k, v in scope.groupby("service_date").size().items()},
    }


def main() -> None:
    import json

    from src import sources  # local import: sources imports this module

    internal, source_report = sources.load_all()
    report: dict = {"sources": source_report,
                    "malformed_lines_skipped": source_report.get("odata", {}).get("malformed_lines_skipped", 0),
                    "raw_rows": sum(v.get("rows", 0) for v in source_report.values() if isinstance(v, dict))}
    flights = build_flights_from_internal(internal, report=report)
    report["flights_by_source"] = flights["source"].value_counts().to_dict()
    report["model_scope_summary"] = model_scope_summary(flights)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "flights.parquet"
    flights.to_parquet(out, index=False)
    (PROCESSED_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {out.relative_to(ROOT)} and cleaning_report.json")

    if RAW_TLVFLIGHTS.exists():
        tlv = process_tlvflights(pd.read_csv(RAW_TLVFLIGHTS, encoding="utf-8-sig"))
        tlv.to_csv(PROCESSED_DIR / "airline_reliability_tlvflights.csv", index=False)
        print(f"Saved airline_reliability_tlvflights.csv ({len(tlv)} airlines)")
    if RAW_CBS.exists():
        records = json.loads(RAW_CBS.read_text("utf-8"))["month"][0]["date"]
        index = chain_price_index(records)
        index.to_csv(PROCESSED_DIR / "price_index_travel_abroad.csv", index=False)
        print(f"Saved price_index_travel_abroad.csv ({len(index)} months)")


if __name__ == "__main__":
    main()
