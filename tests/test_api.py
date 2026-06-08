"""Tests for the FastAPI service: endpoints return valid shapes."""
import pytest
from fastapi.testclient import TestClient

from src import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="model artifact missing; run `python -m src.train` first",
)

VALID_APPLICANT = {
    "revolving_utilization": 0.3,
    "age": 45,
    "num_30_59_days_past_due": 0,
    "debt_ratio": 0.35,
    "monthly_income": 5400,
    "num_open_credit_lines": 8,
    "num_90_days_late": 0,
    "num_real_estate_loans": 1,
    "num_60_89_days_past_due": 0,
    "num_dependents": 2,
}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:  # triggers lifespan -> loads model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid_shape(client):
    r = client.post("/predict", json=VALID_APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in ("approve", "decline")
    assert body["risk_band"] in ("Low", "Moderate", "Elevated", "High")
    assert len(body["explanation"]) == 15
    top = body["explanation"][0]
    assert {"feature", "label", "value", "shap_value"} <= set(top)


def test_predict_missing_income_allowed(client):
    payload = dict(VALID_APPLICANT, monthly_income=None, num_dependents=None)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0


def test_predict_rejects_invalid_input(client):
    bad = dict(VALID_APPLICANT, age=5)  # below the ge=18 bound
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "test_metrics" in body and "global_importance" in body
