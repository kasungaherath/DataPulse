# DataPulse | Customer Churn Analysis

An end-to-end data science project exploring customer churn and evaluating how risk rankings could support retention planning. Built with Python, Pandas, NumPy, SciPy, and scikit-learn using the public IBM Telco Customer Churn sample.

## Results

- Analyzed **7,043 customers**; found 11 blank `TotalCharges` values (0.16%) and no duplicate records or customer IDs.
- Churn prevalence was **26.5%**. Churned customers had shorter average tenure than retained customers (**18.0 vs. 37.6 months**; difference: −19.6 months; bootstrap 95% CI: −20.7 to −18.5).
- Contract type showed the largest selected categorical association with churn (**Cramér's V = 0.41**). This is an unadjusted association, not evidence of causation.
- Compared logistic regression, random forest, and histogram gradient boosting in five-fold stratified cross-validation. Logistic regression led with **0.669 mean PR-AUC** and **0.846 mean ROC-AUC**.
- On the held-out test set, logistic regression achieved **0.634 PR-AUC**, **0.841 ROC-AUC**, **0.552 precision**, and **0.693 recall** at the validation-selected threshold. The highest-risk 10% had **2.89×** the test-set churn prevalence.

Metrics describe performance on this benchmark split and may not generalize to another customer population.

## Method

The notebooks progress from data quality and exploratory analysis to statistical comparisons and model evaluation. Preprocessing is fit inside scikit-learn pipelines; customer IDs are excluded; data is stratified into fit, validation, and test partitions. The threshold is selected on validation data, and final performance is reported on the held-out test set.

### Retention experiment proposal

Use the model to identify an eligible high-risk group, then randomly assign customers to an offer group or usual-service control group. Compare churn after a pre-defined follow-up period and weigh any incremental retention against offer costs. Randomization is needed to tell whether the offer helps; a risk score alone cannot answer that question.

## Run locally

Python 3.12 or newer is recommended. Download the [IBM Telco Customer Churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) into `data/raw/` as `Telco-Customer-Churn.csv`. The raw data is excluded from Git.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --timeout 120 --retries 10 -r requirements-notebooks.txt
jupyter lab
```

Run these notebooks in order:

1. `notebooks/01_data_quality_eda.ipynb`
2. `notebooks/02_statistical_analysis.ipynb`
3. `notebooks/03_model_training_evaluation.ipynb`

Run the unit tests with `python -m pytest --basetemp=.tmp\pytest`.

## Limitations

This historical sample is not evidence about a current telecom operator. Statistical comparisons are observational and do not establish causes. The model and risk ranking do not show that a retention offer will prevent churn; that requires a controlled experiment. Validate performance, fairness, calibration, and business value before considering any real-world use.
