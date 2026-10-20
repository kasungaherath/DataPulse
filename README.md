# DataPulse: Customer Churn Intelligence

An end-to-end, reproducible data science notebook project that turns the IBM Telco customer sample into a practical churn-risk and retention analysis. The portfolio question is not just “Can churn be predicted?” but “How should a retention team use limited outreach capacity, and what evidence supports that choice?”

## Project highlights

- Audits quality, missingness, duplicates, class balance, and the dataset's limitations.
- Explores churn patterns with Pandas, NumPy, Seaborn, and statistical uncertainty.
- Tests selected group differences with Welch's t-test and chi-square, reporting effect sizes and multiple-testing caveats.
- Compares interpretable logistic regression with tree-based models in leakage-safe scikit-learn pipelines.
- Reports ROC-AUC, PR-AUC, recall, precision, calibration, confusion matrices, and a capacity-based top-risk lift view.
- Selects an operating threshold against a stated recall goal; frames outreach as a hypothesis requiring real-world randomized testing.
- Includes reusable `src/` utilities and focused tests. TensorFlow is intentionally optional; for this tabular dataset, a neural network adds complexity without a clear portfolio benefit.

## Quick start

Python 3.12 or newer is recommended. If Windows reports `No such file or directory` for a deeply nested Jupyter file, place the project in a shorter folder such as `C:\DataPulse` (Windows may have long path support disabled).

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --timeout 120 --retries 10 -r requirements.txt
python -m pip install --timeout 120 --retries 10 -r requirements-notebooks.txt
```

Install `requirements.txt` first for the analysis libraries. Install `requirements-notebooks.txt` only when you want JupyterLab; it adds the larger notebook server dependencies. The notebooks import the reusable package from `src/`, so editable installation is not required.

Download `Telco-Customer-Churn.csv` from the [official IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) and put it in `data/raw/`. The first notebook searches for either the IBM filename or `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

```powershell
jupyter lab
```

Run notebooks in order: `01_data_quality_eda.ipynb`, `02_statistical_analysis.ipynb`, `03_model_training_evaluation.ipynb`.

## Repository map

```text
DataPulse/
├── data/raw/                 # Local source CSV (not committed)
├── data/processed/           # Generated tables (not committed)
├── notebooks/                # Ordered analysis workflow
├── reports/figures/          # Saved plots (not committed)
├── src/datapulse/            # Reusable loading, modeling, metrics
├── tests/                    # Unit tests
├── models/                   # Optional exported model artifacts
├── pyproject.toml
└── requirements.txt
```

## Reproducibility and interpretation

The notebooks fix random seeds, split the data stratifiably, and fit all imputers, encoders, and scalers inside training pipelines. The test split is held out from model selection. Threshold selection is made on a separate validation split. Report test results only after finalizing a model and threshold. Features are observational associations, not proof of causes; the sample is a dated, fictionalized benchmark and should not be presented as representative of a current operator. No actual customer records or deployment decisions should rely on this model.

The analysis intentionally includes a model card and a results template. Replace every `TBD` with values from a clean notebook run before citing metrics on a CV. Do not invent performance claims.

## CV-ready summary (complete after running)

**DataPulse — Customer Churn Intelligence (Python, Pandas, NumPy, scikit-learn)** — Audited and analyzed [N] customer records; compared [models] using stratified holdout validation and leakage-safe preprocessing; achieved [PR-AUC / ROC-AUC] on the untouched test set. At a [recall] operating point, identified [precision/top-decile lift] to inform capacity-aware retention experiments. Reproducible notebooks, statistical analysis, tests, and documented model limitations.

## Suggested 1–2 week roadmap

1. Days 1–2: run data audit and EDA; record hypotheses and source provenance.
2. Days 3–4: complete statistical analysis and distinguish practical effects from significance.
3. Days 5–7: compare baseline models, select an operating threshold, assess calibration and error trade-offs.
4. Days 8–10: add a concise results narrative, polish charts and README, run tests from a clean environment, and publish the repository.

