# AI Revenue Intelligence & Churn Prevention System

An end-to-end **Data Science + SQL + Machine Learning + GenAI** decision-intelligence platform designed to predict SaaS customer churn, quantify financial revenue exposure, prioritize high-value retention accounts, and generate explainable retention playbooks.

---

## Business Problem

Traditional churn models stop at binary classification (`churn = 1` vs `0`), leaving Customer Success and Revenue teams without clear answers on **financial priority** or **next best action**.

This system transforms raw customer telemetry into an actionable workflow:
**Prediction → Explanation → Revenue Impact → Recommended Action**

---

## System Architecture

```text
CUSTOMER DATA (5,000 SaaS Accounts)
        │
        ▼
   PostgreSQL 17
        │
  ┌─────┴─────┐
  ▼           ▼
SQL Analytics SQL Feature Engineering (VIEW: customer_ml_features)
  │           │
  └─────┬─────┘
        ▼
  ML Churn Pipeline (Logistic Regression & Random Forest)
        │
        ▼
  Churn Probability & Configurable Risk Tiers (CRITICAL / HIGH / MEDIUM / LOW)
        │
        ▼
  Revenue Risk Engine (Expected Monthly & Annual Exposure in INR)
        │
        ▼
  GenAI Retention Layer (Gemini 2.5 Flash + Grounded Fallback)
        │
        ▼
  Streamlit + Plotly Executive Dashboard