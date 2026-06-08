"""Project-wide configuration: paths, feature definitions, and constants.

Importing these instead of hard-coding strings keeps the notebooks, API, and
dashboard consistent and runnable from any working directory.
"""
from __future__ import annotations

from pathlib import Path

# src/config.py -> src/ -> repo root
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_CSV = RAW / "cs-training.csv"
MODEL_PATH = MODELS / "model.joblib"
METRICS_PATH = MODELS / "metrics.json"

# Reproducible OpenML mirror of the Kaggle "Give Me Some Credit" competition data.
OPENML_DATA_ID = 46929
OPENML_ARFF_URL = "https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff"

RANDOM_STATE = 42

# Target column (mapped to 0/1 in load_raw).
TARGET = "default"

# The 10 raw predictor columns, in a stable order.
RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Friendlier labels for the UI / explanations.
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization ratio",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days late",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines & loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans/lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days late",
    "NumberOfDependents": "Number of dependents",
    # engineered
    "MonthlyIncome_missing": "Income was missing",
    "NumberOfDependents_missing": "Dependents was missing",
    "TotalPastDue": "Total past-due events",
    "IncomePerDependent": "Income per dependent",
    "MonthlyIncome_log": "Monthly income (log)",
}

# Business cost assumption used to pick the decision threshold.
# Approving a future defaulter (false negative) is assumed 5x costlier than
# rejecting a good applicant (false positive). Stated explicitly; tune as needed.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, MODELS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
