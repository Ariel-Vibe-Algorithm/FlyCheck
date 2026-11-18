"""Small scikit-learn-compatible baselines (importable, so saved pipelines can load them)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class GroupRateClassifier(ClassifierMixin, BaseEstimator):
    """Predicts the historical positive rate of the row's group (e.g. per airline).

    Rates are shrunk toward the overall rate: (positives + k * overall) / (n + k), so a
    group seen 3 times does not get an extreme 0% or 100%. Unknown groups get the overall
    rate. `prior_rates` can supply external rates (e.g. TLVFlights per-airline
    cancellation rates) instead of learning them from y.
    """

    def __init__(self, column: str = "airline_code", smoothing: float = 10.0,
                 prior_rates: dict | None = None):
        self.column = column
        self.smoothing = smoothing
        self.prior_rates = prior_rates

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.overall_ = float(y.mean())
        if self.prior_rates is not None:
            self.rates_ = dict(self.prior_rates)
        else:
            stats = y.groupby(X[self.column].astype(str)).agg(["sum", "count"])
            k = self.smoothing
            self.rates_ = ((stats["sum"] + k * self.overall_) / (stats["count"] + k)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[self.column].astype(str).map(self.rates_).fillna(self.overall_).to_numpy(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
