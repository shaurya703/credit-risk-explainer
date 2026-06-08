"""FastAPI service: serves credit-risk predictions with SHAP explanations.

The model artifact is loaded once at startup. Both this API and the Streamlit
dashboard rely on the same `src` package, so there is no duplicated model logic.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.explain import CreditExplainer
from src.model import load_model

from .schemas import (
    ApplicantFeatures,
    Contribution,
    HealthResponse,
    PredictionResponse,
)

# Loaded at startup, shared across requests.
state: dict = {}


def _risk_band(p: float) -> str:
    if p < 0.05:
        return "Low"
    if p < 0.15:
        return "Moderate"
    if p < 0.35:
        return "Elevated"
    return "High"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_model()
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    state["model"] = model
    state["explainer"] = CreditExplainer(model)
    state["metrics"] = metrics
    state["threshold"] = float(metrics.get("threshold", 0.5))
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Explainer API",
    description="Predict loan-default risk and explain every prediction with SHAP.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in state)


@app.get("/model-info")
def model_info() -> dict:
    """Headline metrics, global feature importance, and feature stats for the UI."""
    if "metrics" not in state:
        raise HTTPException(503, "model not loaded")
    m = state["metrics"]
    return {
        "best_model": m.get("best_model"),
        "threshold": state["threshold"],
        "cost_assumption": m.get("cost_assumption"),
        "dataset": m.get("dataset"),
        "test_metrics": m.get("test"),
        "validation_comparison": m.get("validation_comparison"),
        "global_importance": m.get("global_importance"),
        "feature_stats": m.get("feature_stats"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures) -> PredictionResponse:
    """Predict default risk for one applicant and explain the prediction."""
    if "model" not in state:
        raise HTTPException(503, "model not loaded")

    X = pd.DataFrame([applicant.to_row()])[config.RAW_FEATURES]
    prob = float(state["model"].predict_proba(X)[:, 1][0])
    threshold = state["threshold"]

    exp = state["explainer"].explain_one(X)
    contributions = [Contribution(**c) for c in exp["contributions"]]

    return PredictionResponse(
        prediction=int(prob >= threshold),
        probability=prob,
        risk_band=_risk_band(prob),
        decision="decline" if prob >= threshold else "approve",
        threshold=threshold,
        base_value=exp["base_value"],
        explanation=contributions,
    )
