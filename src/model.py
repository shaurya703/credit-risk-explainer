"""Model construction, training, evaluation, and persistence.

Each model is a single sklearn Pipeline: CreditFeatureBuilder -> (scaler) ->
classifier. Persisting the whole pipeline means the API/dashboard load one object
that does preprocessing + prediction identically to training.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import CreditFeatureBuilder


def build_model(kind: str, scale_pos_weight: float | None = None) -> Pipeline:
    """Construct an untrained pipeline for the given model kind.

    Class imbalance is handled *inside the model* (class weights / scale_pos_weight)
    rather than by resampling, which keeps predicted probabilities calibrated for
    use as a risk score (see notebook 03).
    """
    if kind == "logreg":
        return Pipeline([
            ("features", CreditFeatureBuilder()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=config.RANDOM_STATE)),
        ])
    if kind == "rf":
        return Pipeline([
            ("features", CreditFeatureBuilder()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=50,
                class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE)),
        ])
    if kind == "xgboost":
        return Pipeline([
            ("features", CreditFeatureBuilder()),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=1.0, scale_pos_weight=scale_pos_weight or 1.0,
                eval_metric="logloss", n_jobs=-1, random_state=config.RANDOM_STATE)),
        ])
    raise ValueError(f"unknown model kind: {kind}")


def pos_weight(y: pd.Series) -> float:
    """negatives / positives — the XGBoost scale_pos_weight for this training set."""
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return neg / max(pos, 1)


def find_cost_threshold(y_true, y_prob,
                        cost_fn: float = config.COST_FALSE_NEGATIVE,
                        cost_fp: float = config.COST_FALSE_POSITIVE) -> float:
    """Pick the probability threshold minimizing expected business cost.

    A false negative (approving a defaulter) is `cost_fn`; a false positive
    (rejecting a good applicant) is `cost_fp`. We scan candidate thresholds and
    return the one with the lowest total cost on the given (validation) set.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, float("inf")
    y_true = np.asarray(y_true)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def evaluate(model: Pipeline, X, y, threshold: float = 0.5) -> dict:
    """Compute the honest evaluation metrics at a given decision threshold."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
        "positives": int(int(y.sum())),
    }


def save_model(model: Pipeline, path=config.MODEL_PATH) -> None:
    config.ensure_dirs()
    joblib.dump(model, path)


def load_model(path=config.MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def save_metrics(metrics: dict, path=config.METRICS_PATH) -> None:
    config.ensure_dirs()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
