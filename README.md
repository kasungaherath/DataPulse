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

The analysis includes a model card and reports the results from the saved notebook run below. These are benchmark results from one stratified holdout split, not a production estimate. Features are observational associations, not proof of causes; the dataset is a historical sample whose representativeness for any current operator is unknown. The model must not be used for actual customer decisions without fresh validation, fairness review, and a controlled intervention study.

## Key findings

- **Data quality:** 7,043 records and 21 columns; no duplicate rows or customer IDs. Eleven `TotalCharges` entries (0.16%) were blank and converted to missing values for pipeline imputation.
- **Observed patterns:** 26.5% of customers churned. Churned customers had 17.98 months' average tenure versus 37.57 for retained customers, a difference of −19.59 months (bootstrap 95% CI: −20.70 to −18.48; Cohen's d: −0.85).
- **Categorical associations:** contract type had the largest selected association with churn (Cramér's V = 0.41), followed by internet service (0.32) and payment method (0.30). These are unadjusted observational relationships, not causal effects.
- **Model comparison:** logistic regression led five-fold development cross-validation (mean PR-AUC 0.669, ROC-AUC 0.846), ahead of random forest (0.656, 0.841) and histogram gradient boosting (0.646, 0.828).
- **Held-out test:** the validation-selected logistic regression operating threshold was 0.614. Test PR-AUC was 0.634 and ROC-AUC 0.841; at that threshold, precision was 0.552, recall 0.693, and F1 0.614. The top 10% by predicted risk had 2.89× the test-set churn prevalence.
- **Decision caveat:** the threshold was chosen to target at least 70% recall on validation data; test recall was lower at 69.3%. The top-decile lift describes ranking only. It does not estimate how many customers an offer would retain.

## CV-ready summary

**DataPulse — Customer Churn Intelligence (Python, Pandas, NumPy, scikit-learn)** — Analyzed 7,043 IBM Telco sample records, tested churn associations, and compared three leakage-safe classifiers. Logistic regression achieved 0.634 PR-AUC and 0.841 ROC-AUC on a stratified held-out test split; its top-risk decile had 2.89× test-set churn prevalence. Documented threshold trade-offs, data limitations, and an experiment-first retention use case.

## Suggested 1–2 week roadmap

1. Days 1–2: run data audit and EDA; record hypotheses and source provenance.
2. Days 3–4: complete statistical analysis and distinguish practical effects from significance.
3. Days 5–7: compare baseline models, select an operating threshold, assess calibration and error trade-offs.
4. Days 8–10: polish charts and notebook narratives, rerun tests from a clean environment, and publish the repository.

