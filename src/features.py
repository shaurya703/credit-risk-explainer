"""Feature engineering as a single scikit-learn transformer.

Wrapping all preprocessing in one fitted transformer (learned on train only) and
placing it inside the model Pipeline guarantees the *exact same* transformation is
applied at training and at inference — no train/serve skew, no leakage, and no
logic duplicated between the notebooks, the API, and the dashboard.

Every decision here is documented in notebook 02_preprocessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

PASTDUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Final engineered feature order (what the model actually sees).
OUTPUT_FEATURES = [
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
    # engineered
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
    "TotalPastDue",
    "IncomePerDependent",
    "MonthlyIncome_log",
]

# Past-due counts of 96/98 are a known data-entry glitch; cap to a realistic max.
PASTDUE_CAP = 20
AGE_MIN = 18


class CreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Impute, cap outliers, and engineer features for the credit model.

    Learned-on-fit parameters (stored with trailing underscore, sklearn convention):
    median income/dependents and the 99th-percentile caps for the heavy-tailed
    ratio columns.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X[config.RAW_FEATURES]
        self.income_median_ = float(X["MonthlyIncome"].median())
        self.dependents_median_ = float(X["NumberOfDependents"].median())
        self.util_cap_ = float(X["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99))
        self.debt_cap_ = float(X["DebtRatio"].quantile(0.99))
        self.age_median_ = float(X["age"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[config.RAW_FEATURES].copy()
        out = pd.DataFrame(index=X.index)

        # 1) Missingness is informative -> flag before imputing.
        out["MonthlyIncome_missing"] = X["MonthlyIncome"].isna().astype(int)
        out["NumberOfDependents_missing"] = X["NumberOfDependents"].isna().astype(int)

        # 2) Impute with train medians.
        income = X["MonthlyIncome"].fillna(self.income_median_)
        deps = X["NumberOfDependents"].fillna(self.dependents_median_)

        # 3) Cap heavy-tailed / glitched values.
        util = X["RevolvingUtilizationOfUnsecuredLines"].clip(upper=self.util_cap_)
        debt = X["DebtRatio"].clip(upper=self.debt_cap_)
        age = X["age"].clip(lower=AGE_MIN)  # the lone age=0 row -> 18
        pastdue = {c: X[c].clip(upper=PASTDUE_CAP) for c in PASTDUE_COLS}

        # 4) Base (cleaned) features.
        out["RevolvingUtilizationOfUnsecuredLines"] = util
        out["age"] = age
        out["NumberOfTime30-59DaysPastDueNotWorse"] = pastdue["NumberOfTime30-59DaysPastDueNotWorse"]
        out["DebtRatio"] = debt
        out["MonthlyIncome"] = income
        out["NumberOfOpenCreditLinesAndLoans"] = X["NumberOfOpenCreditLinesAndLoans"]
        out["NumberOfTimes90DaysLate"] = pastdue["NumberOfTimes90DaysLate"]
        out["NumberRealEstateLoansOrLines"] = X["NumberRealEstateLoansOrLines"]
        out["NumberOfTime60-89DaysPastDueNotWorse"] = pastdue["NumberOfTime60-89DaysPastDueNotWorse"]
        out["NumberOfDependents"] = deps

        # 5) Engineered features (interpretable on purpose).
        out["TotalPastDue"] = sum(pastdue[c] for c in PASTDUE_COLS)
        out["IncomePerDependent"] = income / (deps + 1.0)
        out["MonthlyIncome_log"] = np.log1p(income)

        return out[OUTPUT_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(OUTPUT_FEATURES)
