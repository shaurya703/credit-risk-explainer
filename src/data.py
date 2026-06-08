"""Download, parse, and split the Give Me Some Credit dataset.

The data is pulled from OpenML (dataset 46929) — a stable, auth-free mirror of the
Kaggle "Give Me Some Credit" competition — so the whole project reproduces from
scratch without Kaggle credentials.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from . import config


def _parse_arff(text: str) -> pd.DataFrame:
    """Parse an ARFF document into a DataFrame, treating '?' as missing."""
    columns: list[str] = []
    data_start = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            # "@ATTRIBUTE <name> <type>" — name is the 2nd whitespace token.
            m = re.match(r"@attribute\s+(\S+)", stripped, flags=re.IGNORECASE)
            if m:
                columns.append(m.group(1))
        elif stripped.lower() == "@data":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("no @data section found in ARFF")

    data_text = "\n".join(lines[data_start:])
    df = pd.read_csv(
        io.StringIO(data_text),
        header=None,
        names=columns,
        na_values=["?"],
        skip_blank_lines=True,
    )
    return df


def download_raw(force: bool = False) -> pd.DataFrame:
    """Download the dataset from OpenML and cache it as data/raw/cs-training.csv."""
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    resp = requests.get(config.OPENML_ARFF_URL, timeout=120)
    resp.raise_for_status()
    df = _parse_arff(resp.text)
    df.to_csv(config.RAW_CSV, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """Return the raw data with the target mapped to a 0/1 column named 'default'.

    1 = serious delinquency within two years (the event we predict).
    """
    df = download_raw()
    target_col = "FinancialDistressNextTwoYears"
    if target_col in df.columns:
        df[config.TARGET] = df[target_col].map({"No": 0, "Yes": 1, "0": 0, "1": 1})
        df = df.drop(columns=[target_col])
    elif "SeriousDlqin2yrs" in df.columns:  # original Kaggle naming, just in case
        df = df.rename(columns={"SeriousDlqin2yrs": config.TARGET})
    # Keep only the known columns in a stable order.
    df = df[config.RAW_FEATURES + [config.TARGET]]
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def split_data(df: pd.DataFrame, random_state: int = config.RANDOM_STATE):
    """Stratified 60/20/20 train/val/test split. Returns (X_*, y_*) for each set."""
    X = df[config.RAW_FEATURES]
    y = df[config.TARGET]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
