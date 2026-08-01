"""
Bank Loan Prediction & Risk Analysis - Live Demo
Run locally:  streamlit run app/streamlit_app.py
Deploy free:  https://streamlit.io/cloud  (connect your GitHub repo)
"""

import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "model" / "loan_risk_model.pkl"
FEATURES_PATH = BASE / "model" / "feature_names.json"

st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return model, feature_names

model, feature_names = load_artifacts()

st.title("🏦 Bank Loan Default Risk Predictor")
st.caption("Trained on 149,999 real loan records (Give Me Some Credit dataset) · Gradient Boosting · ROC-AUC 0.87")

st.markdown("Enter applicant details to estimate the probability of serious delinquency (90+ days past due) within the next 2 years.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income (₹ or $)", min_value=0, value=50000, step=1000)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=0)
    open_credit_lines = st.number_input("Open Credit Lines / Loans", min_value=0, max_value=60, value=6)
    real_estate_loans = st.number_input("Real Estate Loans / Lines", min_value=0, max_value=30, value=1)

with col2:
    revolving_util = st.slider("Revolving Credit Utilization (0 = none, 1 = fully maxed)", 0.0, 1.5, 0.3, 0.01)
    debt_ratio = st.slider("Debt Ratio (monthly debt payments / income)", 0.0, 2.0, 0.3, 0.01)
    late_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

if st.button("Predict Default Risk", type="primary"):
    total_past_due = late_30_59 + late_60_89 + late_90
    debt_to_income_proxy = debt_ratio * monthly_income
    income_per_dependent = monthly_income / (dependents + 1)
    credit_lines_per_age = open_credit_lines / age if age > 0 else 0

    # Age group one-hot (must match training bins: <25, 25-34, 35-44, 45-54, 55-64, 65+)
    age_groups = ['25-34', '35-44', '45-54', '55-64', '65+']
    age_bin = pd.cut([age], bins=[0,25,35,45,55,65,120],
                      labels=['<25','25-34','35-44','45-54','55-64','65+'])[0]

    row = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents,
        "MonthlyIncome_missing_flag": 0,
        "TotalPastDue": total_past_due,
        "DebtToIncomeProxy": debt_to_income_proxy,
        "IncomePerDependent": income_per_dependent,
        "CreditLinesPerAge": credit_lines_per_age,
    }
    for g in age_groups:
        row[f"AgeGroup_{g}"] = 1 if age_bin == g else 0

    X = pd.DataFrame([row])
    X = X.reindex(columns=feature_names, fill_value=0)

    proba = model.predict_proba(X)[0][1]

    if proba < 0.10:
        tier, color = "Low Risk", "green"
    elif proba < 0.30:
        tier, color = "Medium Risk", "orange"
    else:
        tier, color = "High Risk", "red"

    st.metric("Default Probability (2 years)", f"{proba*100:.1f}%")
    st.markdown(f"### Risk Tier: :{color}[{tier}]")
    st.progress(min(proba, 1.0))

    if tier == "Low Risk":
        st.success("Recommendation: Approve — low predicted default risk.")
    elif tier == "Medium Risk":
        st.warning("Recommendation: Approve with caution — consider a higher interest rate or shorter tenure.")
    else:
        st.error("Recommendation: Manual review recommended — high predicted default risk.")

st.divider()
st.caption("Educational project. Not financial advice. Model trained on historical US lending data (Give Me Some Credit, Kaggle 2011).")
