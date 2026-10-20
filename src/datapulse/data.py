"""Loading and validation helpers for the IBM Telco churn sample."""

from pathlib import Path

import pandas as pd

EXPECTED_TARGET = "Churn"


def load_telco(path: str | Path) -> pd.DataFrame:
    """Load the CSV and normalize common source formatting issues."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    if EXPECTED_TARGET not in frame:
        raise ValueError(f"Expected target column {EXPECTED_TARGET!r}; found {list(frame.columns)}")
    return frame


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize types, missingness, and distinct values by column."""
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "missing_count": frame.isna().sum(),
        "missing_pct": frame.isna().mean().mul(100).round(2),
        "unique_count": frame.nunique(dropna=True),
    }).sort_values(["missing_count", "unique_count"], ascending=[False, True])

