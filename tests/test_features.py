"""Unit tests for the feature-engineering transformer (no network needed)."""
import numpy as np
import pandas as pd

from src import config
from src.features import CreditFeatureBuilder, OUTPUT_FEATURES


def _sample_df(n=50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2, n),
        "age": rng.integers(18, 90, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, n),
        "DebtRatio": rng.uniform(0, 3, n),
        "MonthlyIncome": rng.uniform(1000, 12000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 4, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n),
        "NumberOfDependents": rng.integers(0, 5, n).astype(float),
    })
    # Inject the messy cases the builder must handle.
    df.loc[0, "MonthlyIncome"] = np.nan
    df.loc[1, "NumberOfDependents"] = np.nan
    df.loc[2, "RevolvingUtilizationOfUnsecuredLines"] = 50000.0   # outlier
    df.loc[3, "NumberOfTimes90DaysLate"] = 98                     # sentinel glitch
    df.loc[4, "age"] = 0                                          # impossible age
    return df


def test_output_schema_and_no_nans():
    df = _sample_df()
    out = CreditFeatureBuilder().fit(df).transform(df)
    assert list(out.columns) == OUTPUT_FEATURES
    assert len(out) == len(df)
    assert not out.isna().any().any(), "transformer must not emit NaNs"


def test_missingness_flag_set():
    df = _sample_df()
    out = CreditFeatureBuilder().fit(df).transform(df)
    assert out.loc[0, "MonthlyIncome_missing"] == 1
    assert out.loc[1, "NumberOfDependents_missing"] == 1


def test_outliers_and_glitches_capped():
    df = _sample_df()
    fb = CreditFeatureBuilder().fit(df)
    out = fb.transform(df)
    assert out["RevolvingUtilizationOfUnsecuredLines"].max() <= fb.util_cap_ + 1e-9
    assert out["NumberOfTimes90DaysLate"].max() <= 20      # 98 capped
    assert out["age"].min() >= 18                           # age=0 -> 18


def test_engineered_features_present():
    df = _sample_df()
    out = CreditFeatureBuilder().fit(df).transform(df)
    for col in ("TotalPastDue", "IncomePerDependent", "MonthlyIncome_log"):
        assert col in out.columns
    # log income is monotonic in income on the imputed column
    assert (out["MonthlyIncome_log"] >= 0).all()
