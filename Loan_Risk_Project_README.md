# 🏦 Bank Loan Prediction & Risk Analysis

Predicting the probability that a loan applicant will experience serious delinquency (90+ days past due) within the next 2 years, and translating that into a risk-tiered decision system — the same core problem every bank/NBFC credit risk team solves.

**[Live Demo](#) — deploy via Streamlit Cloud and add your link here**

---

## Problem Statement

Lenders need to decide: *should we approve this loan, and at what risk?* Approving too many risky loans drives up Non-Performing Assets (NPAs) — a major concern for banks and NBFCs. Rejecting too conservatively means losing good customers to competitors. This project builds a data-driven risk scoring system to support that decision.

## Dataset

**[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** — a 2011 Kaggle competition dataset, still widely used as a benchmark for credit risk modeling.

- 150,000 real anonymized loan applicant records
- Target: `SeriousDlqin2yrs` (1 = experienced 90+ day delinquency within 2 years)
- Features: revolving credit utilization, age, past-due history, debt ratio, monthly income, open credit lines, real estate loans, dependents

## Tech Stack

| Layer | Tools |
|---|---|
| Data storage / querying | PostgreSQL, SQL |
| Data cleaning & analysis | Python, pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Deployment | Streamlit |
| Dashboard | Power BI / Tableau |

## Project Structure

```
loan-risk-project/
├── data/
│   ├── raw/                  # original cs-training.csv
│   └── processed/            # loan_risk_scored.csv (dashboard-ready output)
├── notebooks/
│   ├── pipeline.py           # full EDA → cleaning → modeling pipeline
│   └── figures/              # generated EDA & evaluation charts
├── sql/
│   └── eda_queries.sql       # reference SQL queries for the analysis
├── model/
│   ├── loan_risk_model.pkl   # trained Gradient Boosting model
│   ├── scaler.pkl
│   └── feature_names.json
├── app/
│   └── streamlit_app.py      # live interactive risk predictor
├── requirements.txt
└── results_summary.json      # full run output (metrics, feature importances)
```

## Key EDA Findings

- **Overall default rate: 6.68%** (imbalanced — 1 in 15 applicants default)
- **Past-due history is the strongest risk signal**: applicants with any prior late payment default at **22.3%**, vs **2.7%** for those with a clean history — an ~8x difference
- **High revolving credit utilization (>80%) roughly doubles-to-quintuples risk**: 21.1% default rate vs 3.8% for normal utilization
- **Default risk decreases with age**: 11.2% for under-25s down to 2.4% for 65+, a pattern consistent with real-world credit behavior (younger borrowers = thinner credit history = higher risk)

*(See `sql/eda_queries.sql` for the exact queries and `notebooks/figures/` for charts.)*

## Modeling Results

Three models were trained and compared on a held-out 20% test set (stratified split):

| Model | ROC-AUC | Precision (default class) | Recall (default class) |
|---|---|---|---|
| Logistic Regression | 0.801 | 0.182 | 0.670 |
| Random Forest | 0.866 | 0.250 | 0.721 |
| **Gradient Boosting (selected)** | **0.870** | 0.568 | 0.202 |

**Why ROC-AUC, not accuracy:** the dataset is heavily imbalanced (93% non-default). A model that predicts "no default" for everyone would score ~93% accuracy while being useless. ROC-AUC measures the model's ability to *rank* risky applicants above safe ones regardless of the imbalance, which is what actually matters for a risk-scoring system.

**Model selection note:** Gradient Boosting was selected for best overall ROC-AUC, but Random Forest offers a better precision/recall trade-off for catching more true defaults (higher recall) at the cost of more false positives — worth reconsidering depending on whether the business prioritizes catching defaulters or minimizing false rejections. This trade-off is configurable via the risk-tier thresholds.

### Top Predictive Features (Random Forest importance)
1. `TotalPastDue` (engineered: sum of all past-due counts)
2. `RevolvingUtilizationOfUnsecuredLines`
3. `NumberOfTimes90DaysLate`
4. `NumberOfTime30-59DaysPastDueNotWorse`
5. `NumberOfTime60-89DaysPastDueNotWorse`

Past payment behavior dominates over income or demographics — consistent with how real-world credit bureaus (like CIBIL) weight repayment history heavily in credit scores.

## Risk Tiering

Applicants are scored into three tiers based on predicted default probability:

| Tier | Probability Range | Suggested Action |
|---|---|---|
| Low Risk | < 10% | Approve |
| Medium Risk | 10–30% | Approve with adjusted terms (higher rate / lower limit) |
| High Risk | > 30% | Manual review / decline |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (EDA + cleaning + training + saving artifacts)
python notebooks/pipeline.py

# 3. Launch the live predictor
streamlit run app/streamlit_app.py
```

## Limitations & Next Steps

- Dataset is US-based historical data (2011); production use would need re-training on current, region-specific data
- Precision on the default class is moderate (0.57) — in production, this threshold would be tuned against the actual business cost of false positives vs false negatives
- Next steps: hyperparameter tuning (GridSearch/Optuna), SHAP values for per-applicant explainability (important for regulatory transparency), cost-sensitive learning to directly optimize for business loss rather than ROC-AUC

---

*Educational project built to demonstrate an end-to-end credit risk analytics workflow: SQL → Python → Machine Learning → Deployment → Dashboard.*
