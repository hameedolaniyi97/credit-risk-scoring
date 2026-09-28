# Credit Risk Scoring — Loan Officer Dashboard

An end-to-end machine learning project that scores loan applicants on
probability of default and explains every decision using SHAP.

**Live demo:** https://credit-risk-scoring-databoy.streamlit.app

## What it does

A loan officer enters an applicant's details and gets:
- An APPROVE / DECLINE decision
- The applicant's probability of default
- A SHAP waterfall chart showing which factors drove the decision

## Project structure

credit_risk_scoring/
├── app/            # Streamlit dashboard
├── config/         # Project constants
├── data/raw/       # UCI German Credit dataset
├── models/         # Saved model, threshold, train/test split
├── reports/        # Cost curve and SHAP figures
├── src/
│   ├── data/           # Data loading and target remapping
│   ├── features/       # Preprocessing pipeline
│   ├── models/         # Training and model comparison scripts
│   ├── evaluation/     # Cost-sensitive threshold selection
│   └── explainability/ # SHAP analysis
└── tests/          # pytest suite

## How to run it

 bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

python src/models/train_final.py
python src/evaluation/find_threshold.py
streamlit run app/streamlit_app.py

Run the tests with `pytest tests/ -v`.

## Approach

**Data:** UCI Statlog German Credit dataset (1,000 applicants, 20 features,
30% default rate). The target is remapped so 1 = default.

**Preprocessing:** a single scikit-learn ColumnTransformer (median
imputation and scaling for numeric features, most-frequent imputation and
one-hot encoding for categorical ones), bundled with the model in one
Pipeline so training and prediction always use identical transformations.

**Model selection:** logistic regression and XGBoost were compared with
stratified 5-fold cross-validation.

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.786 | 0.018 |
| XGBoost | 0.783 | 0.028 |

Performance was statistically tied. XGBoost was selected deliberately for
its native compatibility with SHAP's TreeExplainer (fast and exact
explanations), since explainability is a core feature of this project.
Experiments are tracked with MLflow.

**Cost-sensitive threshold:** a missed defaulter is assumed to cost 5x more
than wrongly declining a good customer (a business assumption, not
something the model derives). Sweeping thresholds against that cost
gives an optimal cutoff of **0.28** rather than the default 0.5.

At 0.28 on the held-out test set:
- 77% of actual defaulters are caught (vs 40% at the default 0.5 threshold on the baseline model)
- 44 good customers are wrongly declined, 14 defaulters are missed

The cost curve is nearly flat between roughly 0.20 and 0.35, so the exact
threshold can be tuned within that range for business reasons (approval
volume, customer experience) without materially increasing cost.

A first unconstrained search collapsed to a threshold of 0.12 and declined
75% of applicants, which no lender could operate. The search was
therefore restricted to a realistic operating range (threshold of 0.20 or
higher).

## Limitations

- **Small dataset.** 1,000 rows limits how much a flexible model can add
  over a linear one and makes results sensitive to the train/test split.
- **Assumed cost ratio.** The 5:1 ratio is illustrative. A real deployment
  would set it from actual loan economics.
- **Combined status/sex field.** The source data encodes personal status
  and sex as one field with no code for single or widowed women, so the
  app's marital status options depend on the sex selected.
- **Sensitive attributes.** The model uses sex and foreign-worker status as
  inputs, as in the original dataset. Production credit models are
  typically restricted from using protected attributes and a real system
  would need a fairness review.
- **Not for real lending decisions.** This is a portfolio project built on a
  public historical dataset.

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, MLflow, pytest