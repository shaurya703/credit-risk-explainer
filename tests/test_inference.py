"""Tests for model inference and SHAP explanations against the saved artifact."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain import CreditExplainer
from src.features import OUTPUT_FEATURES
from src.model import load_model

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="model artifact missing; run `python -m src.train` first",
)


def _row(**overrides) -> pd.DataFrame:
    base = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5400.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])[config.RAW_FEATURES]


def test_probability_in_range():
    model = load_model()
    prob = model.predict_proba(_row())[:, 1][0]
    assert 0.0 <= prob <= 1.0


def test_missing_income_is_handled():
    model = load_model()
    prob = model.predict_proba(_row(MonthlyIncome=np.nan, NumberOfDependents=np.nan))[:, 1][0]
    assert 0.0 <= prob <= 1.0


def test_riskier_applicant_scores_higher():
    model = load_model()
    safe = model.predict_proba(_row(RevolvingUtilizationOfUnsecuredLines=0.1))[:, 1][0]
    risky = model.predict_proba(
        _row(RevolvingUtilizationOfUnsecuredLines=0.99, NumberOfTimes90DaysLate=3,
             **{"NumberOfTime30-59DaysPastDueNotWorse": 3}))[:, 1][0]
    assert risky > safe


def test_explanation_shape():
    model = load_model()
    exp = CreditExplainer(model).explain_one(_row())
    assert isinstance(exp["base_value"], float)
    assert len(exp["contributions"]) == len(OUTPUT_FEATURES)
    for c in exp["contributions"]:
        assert {"feature", "label", "value", "shap_value"} <= set(c)
