"""Inference for the app. Loads only locally trained artifacts from models/.

A model output is returned only when models/metrics.json marks it ready_for_app
(it beat the best baseline on the held-out test period). Otherwise the caller gets
the reason and should show historical rates instead. This keeps the app honest
automatically when a larger dataset is added and the models are retrained.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache

import joblib
import pandas as pd

from src import features
from src.train import MODELS_DIR


@lru_cache(maxsize=1)
def load_metrics() -> dict | None:
    path = MODELS_DIR / "metrics.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


@lru_cache(maxsize=4)
def _artifact(task: str):
    path = MODELS_DIR / f"{task}_model.joblib"
    return joblib.load(path) if path.exists() else None  # trusted: produced by src.train


@lru_cache(maxsize=1)
def load_profile() -> pd.Series | None:
    path = MODELS_DIR / "hourly_profile.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0).iloc[:, 0]


def task_status(task: str) -> dict:
    metrics = load_metrics()
    if metrics is None:
        return {"ready": False, "reason": "No trained models yet. Run python -m src.train."}
    rep = metrics["tasks"].get(task, {})
    ready = bool(rep.get("ready_for_app")) and _artifact(task) is not None
    return {"ready": ready, "reason": rep.get("status_reason", ""), "report": rep}


def predict_probability(task: str, airline_code: str, airport_code: str,
                        country: str | None, departure: datetime) -> dict:
    """Probability for 'delay' (30+ min) or 'cancellation' for one planned flight."""
    status = task_status(task)
    if not status["ready"]:
        return {"probability": None, **status}
    X = features.build_features(features.single_flight(airline_code, airport_code, country, departure),
                                load_profile())
    proba = float(_artifact(task).predict_proba(X)[0, 1])
    return {"probability": proba, **status}


def predict_hourly_departures(day: date) -> dict:
    """Expected scheduled departures per hour for a date (activity model if ready,
    otherwise the historical hourly profile)."""
    status = task_status("activity")
    if status["ready"]:
        X = features.hourly_count_features([day])
        cols = status["report"]["results"]["Random forest"]["features"]
        values = _artifact("activity").predict(X[cols])
        return {"source": "model", "values": [float(v) for v in values], **status}
    profile = load_profile()
    values = None if profile is None else [float(profile.get(h, 0.0)) for h in range(24)]
    return {"source": "historical profile", "values": values, **status}
