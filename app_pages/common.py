"""Shared, cached data loading and small UI helpers for all pages."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import analytics
from src.data_processing import PROCESSED_DIR

FLIGHTS = PROCESSED_DIR / "flights.parquet"
REPORT = PROCESSED_DIR / "cleaning_report.json"
TLVFLIGHTS = PROCESSED_DIR / "airline_reliability_tlvflights.csv"
PRICE_INDEX = PROCESSED_DIR / "price_index_travel_abroad.csv"

LEVEL_COLORS = {"low": "green", "medium": "orange", "high": "red"}
SKY, SKY_SOFT, SIGNAL, INK = "#2F6FB0", "#A9C6E4", "#F2B705", "#0E2338"


def require_data() -> None:
    """Stop the page with clear instructions if processed data is missing."""
    missing = [p.name for p in (FLIGHTS, REPORT) if not p.exists()]
    if missing:
        st.error(
            f"Processed data not found ({', '.join(missing)}). Build it first:\n\n"
            "`python -m src.download_data` then `python -m src.data_processing`"
        )
        st.stop()


@st.cache_data
def load_flights() -> pd.DataFrame:
    return pd.read_parquet(FLIGHTS)


@st.cache_data
def load_report() -> dict:
    return json.loads(REPORT.read_text("utf-8"))


@st.cache_data
def load_tlvflights() -> pd.DataFrame | None:
    return pd.read_csv(TLVFLIGHTS) if TLVFLIGHTS.exists() else None


@st.cache_data
def load_price_index() -> pd.DataFrame | None:
    return pd.read_csv(PRICE_INDEX, parse_dates=["month"]) if PRICE_INDEX.exists() else None


@st.cache_data
def load_profile() -> pd.Series:
    return analytics.hourly_profile(load_flights())


def airline_options(scope: pd.DataFrame, tlv: pd.DataFrame | None) -> dict[str, str]:
    """code -> 'CODE · Name' for airlines seen in the archive or in TLVFlights."""
    names = scope.drop_duplicates("airline_code").set_index("airline_code")["airline_name"].str.title()
    if tlv is not None:
        names = pd.concat([names, tlv.set_index("airline_code")["airline_name"]])
    names = names[~names.index.duplicated()]
    order = scope["airline_code"].value_counts().index.tolist()
    order += [c for c in names.index if c not in order]
    return {c: f"{c} · {names[c]}" for c in order}


def destination_options(scope: pd.DataFrame) -> dict[str, str]:
    counts = scope.groupby(["airport_code", "city"]).size().sort_values(ascending=False)
    return {code: f"{code} · {str(city).title()}" for code, city in counts.index}


def fmt_rate(r: dict | None, key: str = "share_delayed_30") -> str:
    return "—" if not r or r.get(key) is None else f"{r[key]:.0%}"


TASK_TITLES = {
    "delay": ("Delay of 30+ minutes", "Will a departure leave at least 30 minutes late?"),
    "cancellation": ("Cancellation", "Will a departure be cancelled?"),
    "activity": ("Airport activity", "How many international departures are scheduled per hour?"),
}


@st.cache_data
def load_metrics() -> dict | None:
    from src.train import MODELS_DIR
    path = MODELS_DIR / "metrics.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def render_model_report() -> None:
    """Baselines vs models for every target, from models/metrics.json (src.train)."""
    metrics = load_metrics()
    if metrics is None:
        st.info("No evaluation yet. Run `python -m src.train`.")
        return
    st.write("Every target is evaluated on a **chronological split**: the latest days are held out "
             "as the test set and all historical aggregates come from the training days only. "
             "A model is used in the app only if it **clearly beats the best baseline** on the test "
             "days (95% bootstrap interval of the Brier improvement above zero, and a higher ROC-AUC).")
    st.caption(f"Trained {metrics['trained_at_utc']} UTC on {metrics['data']['flights']:,} flights "
               f"({metrics['data']['first_day']} to {metrics['data']['last_day']}).")
    for task, rep in metrics["tasks"].items():
        title, question = TASK_TITLES.get(task, (task, ""))
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.caption(question)
            ready = rep.get("ready_for_app")
            (st.success if ready else st.warning)(
                ("Model in use. " if ready else "Baseline in use. ") + rep.get("status_reason", ""))
            rows = []
            for name, res in rep["results"].items():
                rows.append({"Method": name, "Type": res["kind"], **res["test"]})
            table = pd.DataFrame(rows).drop(columns=["confusion"], errors="ignore")
            st.dataframe(table, hide_index=True, width="stretch")
            split = rep.get("split", {})
            if "train_rows" in split:
                st.caption(f"Train: {split['train_rows']} rows ({split['train_days'][0]} to "
                           f"{split['train_days'][1]}, {split['train_positive_rate']:.0%} positive). "
                           f"Test: {split['test_rows']} rows on {', '.join(split['test_days'])} "
                           f"({split['test_positive_rate']:.0%} positive).")
            elif split:
                st.caption(f"Train days: {len(split['train_days'])}. Test days: {', '.join(split['test_days'])}.")
            if rep.get("ablation"):
                abl = pd.DataFrame(rep["ablation"]).T[["brier", "roc_auc", "accuracy"]]
                st.markdown("Feature-set comparison (logistic regression, same split; lower Brier is better)")
                st.dataframe(abl, width="stretch")
    st.caption("Brier = mean squared error of the predicted probability (lower is better; always "
               "predicting 50% scores 0.25). ROC-AUC = how well flights are ranked from least to most "
               "likely (0.5 = random). MAE = mean absolute error in departures per hour.")


def data_note() -> None:
    st.caption(
        "Historical data: IAA flight board archived by The Octopus (CC BY), "
        "logged on 4 days in 2026. Small sample; read rates together with their n."
    )
