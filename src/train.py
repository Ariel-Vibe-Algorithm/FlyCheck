"""Train and evaluate baselines and models for FlyCheck's three targets.

Run:  python -m src.train

| Target       | Question                                   | Baselines                          | Model                     |
|--------------|--------------------------------------------|------------------------------------|---------------------------|
| delay        | Will the departure leave 30+ min late?     | majority class, overall rate, airline rate | LogisticRegression, RandomForest |
| cancellation | Will the departure be cancelled?           | overall rate, airline rate (archive), airline rate (TLVFlights) | LogisticRegression (only with enough cancellations) |
| activity     | How many departures are scheduled per hour? | hourly mean, weekday-hour mean     | RandomForestRegressor     |

Everything is evaluated on a chronological split: the latest days are the test set,
and every historical aggregate is computed from the training days only. A model is
marked `ready_for_app` only if it beats the best baseline on the test set and the
data passes minimum-size checks. After evaluation, the deployable artifact is refit
on all data. The same script scales unchanged to a larger dataset.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import analytics, features
from src.data_processing import PROCESSED_DIR, ROOT
from src.estimators import GroupRateClassifier

MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SHARE = 0.20          # latest days holding at least this share of rows form the test set
MIN_TRAIN_ROWS = 300       # below this a classifier is not trusted
MIN_TEST_ROWS = 100
MIN_POSITIVES = {"train": 50, "test": 10}   # for rare targets such as cancellation
MIN_ACTIVITY_DAYS = {"train": 7, "test": 2}


# --------------------------------------------------------------------------- split
def chronological_split(df: pd.DataFrame, date_col: str = "service_date",
                        test_share: float = TEST_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest whole days go to test until they hold >= test_share of rows.
    A day is never split, so the same flight occurrence is never in both sets."""
    per_day = df.groupby(date_col).size().sort_index(ascending=False)
    cum = per_day.cumsum() / per_day.sum()
    n_test_days = int((cum < test_share).sum()) + 1
    test_days = set(per_day.index[:min(n_test_days, len(per_day) - 1)])
    is_test = df[date_col].isin(test_days)
    return df[~is_test], df[is_test]


# ------------------------------------------------------------------------- helpers
def preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])


def classifier_metrics(y_true, proba) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    pred = (proba >= 0.5).astype(int)
    both = len(set(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if both else None,
        "brier": round(brier_score_loss(y_true, proba), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def positive_proba(est, X) -> np.ndarray:
    """P(class 1), also when training saw a single class (e.g. no cancellations)."""
    proba = est.predict_proba(X)
    classes = list(getattr(est, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def brier_gain_ci(y_true, p_baseline, p_model, n_boot: int = 1000) -> tuple[float, float]:
    """95% paired-bootstrap interval of (baseline Brier - model Brier) on the test set.
    Positive = the model is better. Resampling the same rows for both keeps it paired."""
    y = np.asarray(y_true, dtype=float)
    loss_b = (np.asarray(p_baseline) - y) ** 2
    loss_m = (np.asarray(p_model) - y) ** 2
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    gains = loss_b[idx].mean(axis=1) - loss_m[idx].mean(axis=1)
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def split_info(train: pd.DataFrame, test: pd.DataFrame, y_col: str) -> dict:
    return {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_days": [str(min(train["service_date"])), str(max(train["service_date"]))],
        "test_days": sorted(str(d) for d in set(test["service_date"])),
        "train_positive_rate": round(float(train[y_col].mean()), 4),
        "test_positive_rate": round(float(test[y_col].mean()), 4),
        "train_positives": int(train[y_col].sum()), "test_positives": int(test[y_col].sum()),
    }


def _best(results: dict, kind: str) -> tuple[str, dict]:
    pool = {k: v for k, v in results.items() if v["kind"] == kind}
    name = min(pool, key=lambda k: pool[k]["test"]["brier"])
    return name, pool[name]


# ---------------------------------------------------------------- classification
def evaluate_classifier_task(rows: pd.DataFrame, y_col: str, task: str,
                             tlv_rates: dict | None = None) -> tuple[dict, Pipeline | None]:
    """Shared flow for delay and cancellation. Returns (report, deployable pipeline)."""
    train, test = chronological_split(rows)
    info = split_info(train, test, y_col)
    report = {"task": task, "target": y_col, "split": info, "results": {}, "ablation": {}}

    profile_train = analytics.hourly_profile(train_flights_for_profile(train))
    X_tr, X_te = features.build_features(train, profile_train), features.build_features(test, profile_train)
    y_tr, y_te = train[y_col].astype(int).to_numpy(), test[y_col].astype(int).to_numpy()

    candidates = {
        "Majority class": ("baseline", DummyClassifier(strategy="most_frequent")),
        "Overall rate": ("baseline", DummyClassifier(strategy="prior")),
        "Airline rate (archive)": ("baseline", GroupRateClassifier("airline_code")),
    }
    if tlv_rates:
        candidates["Airline rate (TLVFlights)"] = ("baseline", GroupRateClassifier("airline_code", prior_rates=tlv_rates))

    enough = (len(train) >= MIN_TRAIN_ROWS and len(test) >= MIN_TEST_ROWS
              and info["train_positives"] >= MIN_POSITIVES["train"]
              and info["test_positives"] >= MIN_POSITIVES["test"]
              and (len(train) - info["train_positives"]) >= MIN_POSITIVES["train"])
    report["enough_data_for_model"] = bool(enough)
    if not enough:
        report["status_reason"] = (
            f"Not enough data for a model: {len(train)} train / {len(test)} test rows, "
            f"{info['train_positives']} / {info['test_positives']} positives "
            f"(need >= {MIN_TRAIN_ROWS} / {MIN_TEST_ROWS} rows and >= "
            f"{MIN_POSITIVES['train']} / {MIN_POSITIVES['test']} positives). Baselines only.")

    probs: dict[str, np.ndarray] = {}
    for name, (kind, est) in candidates.items():
        est.fit(X_tr, y_tr)
        probs[name] = positive_proba(est, X_te)
        report["results"][name] = {"kind": kind, "test": classifier_metrics(y_te, probs[name])}

    best_pipe = None
    if enough:
        # Ablation: logistic regression on each feature set, same split.
        for set_name, (cat, num) in features.FEATURE_SETS.items():
            pipe = Pipeline([("prep", preprocessor(cat, num)),
                             ("clf", LogisticRegression(max_iter=2000, C=0.5))])
            pipe.fit(X_tr, y_tr)
            report["ablation"][set_name] = classifier_metrics(y_te, positive_proba(pipe, X_te))
        best_set = min(report["ablation"], key=lambda s: report["ablation"][s]["brier"])
        cat, num = features.FEATURE_SETS[best_set]
        report["chosen_feature_set"] = best_set
        models = {
            "Logistic regression": LogisticRegression(max_iter=2000, C=0.5),
            "Random forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
        }
        for name, clf in models.items():
            pipe = Pipeline([("prep", preprocessor(cat, num)), ("clf", clf)])
            pipe.fit(X_tr, y_tr)
            probs[name] = positive_proba(pipe, X_te)
            report["results"][name] = {"kind": "model", "features": cat + num,
                                       "test": classifier_metrics(y_te, probs[name])}
        best_model, best_m = _best(report["results"], "model")
        best_base, best_b = _best(report["results"], "baseline")
        m, b = best_m["test"], best_b["test"]
        lo, hi = brier_gain_ci(y_te, probs[best_base], probs[best_model])
        # Ready only if the Brier improvement is clearly above zero (95% bootstrap interval)
        # AND the model ranks flights better (ROC-AUC). A tiny win can be pure chance.
        beats = lo > 0 and (m["roc_auc"] or 0) > (b["roc_auc"] or 0.5)
        report.update({
            "best_model": best_model, "best_baseline": best_base,
            "brier_gain_ci95": [round(lo, 4), round(hi, 4)],
            "beats_baseline": bool(beats), "ready_for_app": bool(beats),
            "status_reason": (
                f"{best_model} beats {best_base}: Brier {m['brier']} vs {b['brier']} (gain 95% CI "
                f"{lo:.4f} to {hi:.4f}), ROC-AUC {m['roc_auc']} vs {b['roc_auc']}." if beats else
                f"No model clearly beats the best baseline ({best_base}: Brier {b['brier']}, AUC {b['roc_auc']}). "
                f"Best model {best_model}: Brier {m['brier']}, AUC {m['roc_auc']}; Brier gain 95% CI "
                f"{lo:.4f} to {hi:.4f}. The app keeps showing historical rates."),
        })
        # Deployable artifact: same recipe, refit on all rows, profile from all rows.
        profile_all = analytics.hourly_profile(train_flights_for_profile(rows))
        X_all = features.build_features(rows, profile_all)
        best_pipe = Pipeline([("prep", preprocessor(cat, num)), ("clf", models[best_model])])
        best_pipe.fit(X_all, rows[y_col].astype(int).to_numpy())
    else:
        best_base, _ = _best(report["results"], "baseline")
        report.update({"best_model": None, "best_baseline": best_base,
                       "beats_baseline": False, "ready_for_app": False})
    return report, best_pipe


_ALL_FLIGHTS: pd.DataFrame | None = None


def train_flights_for_profile(rows: pd.DataFrame) -> pd.DataFrame:
    """All scheduled flights on the same days as `rows` (the profile needs the full
    schedule of those days, not only labelled rows)."""
    days = set(rows["service_date"])
    return _ALL_FLIGHTS[_ALL_FLIGHTS["service_date"].isin(days)]


# ------------------------------------------------------------------ activity model
def evaluate_activity_task(flights: pd.DataFrame) -> tuple[dict, RandomForestRegressor | None]:
    """Hourly count of scheduled international departures on complete days."""
    days = analytics.complete_schedule_days(flights)
    dep = analytics.international(flights)
    dep = dep[(dep["direction"] == "D") & dep["service_date"].isin(days)]
    counts = dep.groupby(["service_date", "hour"]).size()
    table = features.hourly_count_features(days)
    table["departures"] = [counts.get((d, h), 0) for d, h in zip(table["service_date"], table["hour"])]

    train, test = chronological_split(table)
    report = {"task": "activity", "target": "scheduled departures per hour", "results": {},
              "split": {"train_days": sorted(str(d) for d in set(train["service_date"])),
                        "test_days": sorted(str(d) for d in set(test["service_date"]))}}
    hour_mean = train.groupby("hour")["departures"].mean()
    wd_hour = train.groupby(["weekday_num", "hour"])["departures"].mean()
    base_hour = test["hour"].map(hour_mean)
    base_wd = [wd_hour.get((w, h), hour_mean[h]) for w, h in zip(test["weekday_num"], test["hour"])]
    report["results"]["Hourly mean"] = {"kind": "baseline", "test": {"mae": round(mean_absolute_error(test["departures"], base_hour), 3)}}
    report["results"]["Weekday-hour mean"] = {"kind": "baseline", "test": {"mae": round(mean_absolute_error(test["departures"], base_wd), 3)}}

    cols = ["hour", "weekday_num", "is_weekend", "month", "is_holiday", "near_holiday"]
    n_train_days, n_test_days = len(set(train["service_date"])), len(set(test["service_date"]))
    enough = n_train_days >= MIN_ACTIVITY_DAYS["train"] and n_test_days >= MIN_ACTIVITY_DAYS["test"]
    report["enough_data_for_model"] = bool(enough)
    model = None
    if enough:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(train[cols], train["departures"])
        report["results"]["Random forest"] = {"kind": "model", "features": cols,
                                              "test": {"mae": round(mean_absolute_error(test["departures"], rf.predict(test[cols])), 3)}}
        best_base = min((k for k, v in report["results"].items() if v["kind"] == "baseline"),
                        key=lambda k: report["results"][k]["test"]["mae"])
        beats = report["results"]["Random forest"]["test"]["mae"] < report["results"][best_base]["test"]["mae"]
        report.update({"best_model": "Random forest", "best_baseline": best_base,
                       "beats_baseline": bool(beats), "ready_for_app": bool(beats),
                       "status_reason": "Random forest beats the best baseline on test MAE." if beats else
                       f"Random forest does not beat {best_base} on test MAE; the app keeps the historical profile."})
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(table[cols], table["departures"])
    else:
        report.update({"best_model": None, "ready_for_app": False, "beats_baseline": False,
                       "status_reason": f"Not enough complete days ({n_train_days} train / {n_test_days} test; "
                                        f"need >= {MIN_ACTIVITY_DAYS['train']} / {MIN_ACTIVITY_DAYS['test']})."})
    return report, model


# --------------------------------------------------------------------------- main
def run(flights: pd.DataFrame, tlv: pd.DataFrame | None = None) -> tuple[dict, dict]:
    global _ALL_FLIGHTS
    _ALL_FLIGHTS = flights
    tlv_cancel = None if tlv is None else tlv.set_index("airline_code")["cancel_rate"].to_dict()

    delay_rows = analytics.labelled(flights)
    intl_dep = analytics.international(flights[flights["direction"] == "D"])
    cancel_rows = intl_dep[intl_dep["outcome"].isin(["completed", "cancelled"])].copy()
    cancel_rows["cancelled"] = (cancel_rows["outcome"] == "cancelled").astype(int)

    delay_report, delay_pipe = evaluate_classifier_task(delay_rows, "delayed_30", "delay")
    cancel_report, cancel_pipe = evaluate_classifier_task(cancel_rows, "cancelled", "cancellation", tlv_cancel)
    activity_report, activity_model = evaluate_activity_task(flights)
    artifacts = {"delay": delay_pipe, "cancellation": cancel_pipe, "activity": activity_model}
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "data": {"flights": int(len(flights)), "sources": flights["source"].value_counts().to_dict(),
                 "first_day": str(flights["service_date"].min()), "last_day": str(flights["service_date"].max())},
        "tasks": {"delay": delay_report, "cancellation": cancel_report, "activity": activity_report},
    }
    return metrics, artifacts


def main() -> None:
    flights = pd.read_parquet(PROCESSED_DIR / "flights.parquet")
    tlv_path = PROCESSED_DIR / "airline_reliability_tlvflights.csv"
    tlv = pd.read_csv(tlv_path) if tlv_path.exists() else None
    metrics, artifacts = run(flights, tlv)

    MODELS_DIR.mkdir(exist_ok=True)
    for name, art in artifacts.items():
        path = MODELS_DIR / f"{name}_model.joblib"
        if art is not None:
            joblib.dump(art, path)
        elif path.exists():
            path.unlink()  # never leave a stale model next to fresh metrics
    analytics.hourly_profile(flights).to_csv(MODELS_DIR / "hourly_profile.csv", header=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), "utf-8")

    for task, rep in metrics["tasks"].items():
        print(f"\n== {task} ==  ready_for_app={rep['ready_for_app']}")
        print("  ", rep.get("status_reason", ""))
        for name, res in rep["results"].items():
            print(f"   {res['kind']:8s} {name:28s} {res['test']}")
        if rep.get("ablation"):
            print("   ablation (LogReg Brier):", {k: v["brier"] for k, v in rep["ablation"].items()},
                  "-> chosen:", rep.get("chosen_feature_set"))
    print(f"\nSaved metrics and artifacts in {MODELS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
