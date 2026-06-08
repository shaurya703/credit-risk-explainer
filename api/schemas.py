"""Pydantic request/response schemas for the prediction API.

Field names are friendly snake_case; they are mapped to the dataset's raw column
names (some of which contain hyphens) inside the handler.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    """Loan-applicant features. Income/dependents may be null (handled by the model)."""

    revolving_utilization: float = Field(..., ge=0, examples=[0.5],
                                         description="Balance on credit cards/lines / total limit")
    age: int = Field(..., ge=18, le=120, examples=[45])
    num_30_59_days_past_due: int = Field(..., ge=0, examples=[0],
                                         description="Times 30-59 days past due")
    debt_ratio: float = Field(..., ge=0, examples=[0.35],
                              description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(None, ge=0, examples=[5400])
    num_open_credit_lines: int = Field(..., ge=0, examples=[8])
    num_90_days_late: int = Field(..., ge=0, examples=[0])
    num_real_estate_loans: int = Field(..., ge=0, examples=[1])
    num_60_89_days_past_due: int = Field(..., ge=0, examples=[0])
    num_dependents: float | None = Field(None, ge=0, examples=[2])

    def to_row(self) -> dict:
        """Map to the dataset's raw column names (the order src expects)."""
        import numpy as np
        return {
            "RevolvingUtilizationOfUnsecuredLines": self.revolving_utilization,
            "age": self.age,
            "NumberOfTime30-59DaysPastDueNotWorse": self.num_30_59_days_past_due,
            "DebtRatio": self.debt_ratio,
            "MonthlyIncome": np.nan if self.monthly_income is None else self.monthly_income,
            "NumberOfOpenCreditLinesAndLoans": self.num_open_credit_lines,
            "NumberOfTimes90DaysLate": self.num_90_days_late,
            "NumberRealEstateLoansOrLines": self.num_real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": self.num_60_89_days_past_due,
            "NumberOfDependents": np.nan if self.num_dependents is None else self.num_dependents,
        }


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = predicted default at the decision threshold")
    probability: float = Field(..., description="Calibrated probability of default")
    risk_band: str
    decision: str = Field(..., description="approve / decline at the configured threshold")
    threshold: float
    base_value: float = Field(..., description="SHAP base value (avg model output, log-odds)")
    explanation: list[Contribution]


class HealthResponse(BaseModel):
    # 'model_loaded' would otherwise trip Pydantic's protected 'model_' namespace.
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
