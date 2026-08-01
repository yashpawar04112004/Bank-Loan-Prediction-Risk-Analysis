"""
Bank Loan Prediction & Risk Analysis
Dataset: Give Me Some Credit (Kaggle, 2011 competition)
End-to-end pipeline: load -> EDA -> clean -> feature engineer -> model -> evaluate -> save artifacts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "cs-training.csv"
PROCESSED = BASE / "data" / "processed"
FIGS = BASE / "notebooks" / "figures"
MODEL_DIR = BASE / "model"
for d in [PROCESSED, FIGS, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
RESULTS = {}

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_csv(RAW)
df = df.drop(columns=[df.columns[0]])  # drop unnamed index column
RESULTS['raw_shape'] = list(df.shape)
RESULTS['target_balance'] = df['SeriousDlqin2yrs'].value_counts(normalize=True).round(4).to_dict()

print("Raw shape:", df.shape)
print("Target balance:\n", df['SeriousDlqin2yrs'].value_counts(normalize=True))
print("Missing values:\n", df.isnull().sum())

RESULTS['missing_before'] = df.isnull().sum().to_dict()

# ---------------------------------------------------------------
# 2. EDA PLOTS
# ---------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x='SeriousDlqin2yrs', data=df)
plt.title('Target Distribution: Serious Delinquency in 2 Years')
plt.xlabel('0 = No Default   1 = Default')
plt.tight_layout()
plt.savefig(FIGS / "target_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=50, kde=True)
plt.title('Applicant Age Distribution')
plt.tight_layout()
plt.savefig(FIGS / "age_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(8,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(FIGS / "correlation_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. CLEANING
# ---------------------------------------------------------------
# Age: a handful of rows have age=0, which is invalid -> drop
df = df[df['age'] > 0]

# MonthlyIncome: ~20% missing -> impute with median (robust to outliers)
df['MonthlyIncome_missing_flag'] = df['MonthlyIncome'].isnull().astype(int)
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

# NumberOfDependents: small % missing -> impute with mode (0 is most common)
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])

# Cap extreme outliers (96/98 are known data-entry error codes in this dataset)
for col in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate']:
    df[col] = df[col].clip(upper=df[col].quantile(0.999))

# RevolvingUtilizationOfUnsecuredLines: should logically be <= ~2 (some >1 due to fees); cap extreme outliers
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.999))
df['DebtRatio'] = df['DebtRatio'].clip(upper=df['DebtRatio'].quantile(0.999))

RESULTS['clean_shape'] = list(df.shape)
RESULTS['missing_after'] = df.isnull().sum().to_dict()

# ---------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------
df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] +
                       df['NumberOfTime60-89DaysPastDueNotWorse'] +
                       df['NumberOfTimes90DaysLate'])

df['DebtToIncomeProxy'] = df['DebtRatio'] * df['MonthlyIncome']
df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
df['CreditLinesPerAge'] = df['NumberOfOpenCreditLinesAndLoans'] / df['age']

age_bins = [0, 25, 35, 45, 55, 65, 120]
age_labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)

# ---------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
y = df['SeriousDlqin2yrs']
X = df.drop(columns=['SeriousDlqin2yrs'])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 6. MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42,
                                             class_weight='balanced', n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
}

model_results = {}
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, proba)

    model_results[name] = {
        "roc_auc": round(auc, 4),
        "precision_default": round(report['1']['precision'], 4),
        "recall_default": round(report['1']['recall'], 4),
        "f1_default": round(report['1']['f1-score'], 4),
    }
    roc_data[name] = (fpr.tolist(), tpr.tolist())
    print(f"{name}: ROC-AUC={auc:.4f}")

RESULTS['model_results'] = model_results

# Pick best model by ROC-AUC
best_name = max(model_results, key=lambda k: model_results[k]['roc_auc'])
best_model = models[best_name]
RESULTS['best_model'] = best_name

# ---------------------------------------------------------------
# 7. ROC CURVE PLOT (all models)
# ---------------------------------------------------------------
plt.figure(figsize=(6,5))
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={model_results[name]['roc_auc']:.3f})")
plt.plot([0,1],[0,1],'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "roc_curve_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 8. FEATURE IMPORTANCE (best tree-based model, fallback to RF)
# ---------------------------------------------------------------
importance_model = models["Random Forest"]
importances = pd.Series(importance_model.feature_importances_, index=feature_names).sort_values(ascending=False)
RESULTS['top_10_features'] = importances.head(10).round(4).to_dict()

plt.figure(figsize=(7,5))
importances.head(10).sort_values().plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(FIGS / "feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 9. RISK SCORING ON FULL DATASET (for dashboard)
# ---------------------------------------------------------------
full_proba = importance_model.predict_proba(X)[:, 1]
df_scored = df.copy()
df_scored['Default_Probability'] = full_proba

def risk_tier(p):
    if p < 0.10:
        return 'Low Risk'
    elif p < 0.30:
        return 'Medium Risk'
    else:
        return 'High Risk'

df_scored['Risk_Tier'] = df_scored['Default_Probability'].apply(risk_tier)
RESULTS['risk_tier_distribution'] = df_scored['Risk_Tier'].value_counts(normalize=True).round(4).to_dict()

# Save dashboard-ready CSV (trimmed columns, for Power BI / Tableau)
dashboard_cols = ['age', 'MonthlyIncome', 'NumberOfDependents', 'DebtRatio',
                   'NumberOfOpenCreditLinesAndLoans', 'TotalPastDue',
                   'Default_Probability', 'Risk_Tier', 'SeriousDlqin2yrs']
df_scored[dashboard_cols].to_csv(PROCESSED / "loan_risk_scored.csv", index=False)

plt.figure(figsize=(5,4))
sns.countplot(x='Risk_Tier', data=df_scored, order=['Low Risk','Medium Risk','High Risk'])
plt.title('Applicant Risk Tier Distribution')
plt.tight_layout()
plt.savefig(FIGS / "risk_tier_distribution.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 10. SAVE MODEL ARTIFACTS
# ---------------------------------------------------------------
joblib.dump(best_model if best_name != "Logistic Regression" else models["Random Forest"],
            MODEL_DIR / "loan_risk_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_names, MODEL_DIR / "feature_names.json.pkl")

with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(BASE / "results_summary.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("\n=== DONE ===")
print(json.dumps(RESULTS['model_results'], indent=2))
print("Best model:", best_name)
print("Top features:", list(RESULTS['top_10_features'].keys())[:5])
