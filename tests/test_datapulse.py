import pandas as pd
import pytest

from datapulse.data import load_telco, quality_report
from datapulse.metrics import threshold_for_recall


def test_load_converts_total_charges_and_reports_missing(tmp_path):
    path = tmp_path / "telco.csv"
    pd.DataFrame({"Churn": ["Yes", "No"], "TotalCharges": [" ", "42.5"]}).to_csv(path, index=False)
    frame = load_telco(path)
    assert pd.isna(frame.loc[0, "TotalCharges"])
    assert frame.loc[1, "TotalCharges"] == 42.5
    assert quality_report(frame).loc["TotalCharges", "missing_count"] == 1


def test_load_requires_churn_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"other": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Churn"):
        load_telco(path)


def test_threshold_for_recall_returns_numeric_threshold():
    threshold = threshold_for_recall([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)
    assert 0 <= threshold <= 1
