import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "src"))
from ai_insights import generate_retention_insights

st.set_page_config(
    page_title="AI Revenue Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_data():
    scores_path = os.path.join(ROOT_DIR, "data", "customer_risk_scores.csv")
    importance_path = os.path.join(ROOT_DIR, "models", "feature_importance.csv")
    metrics_path = os.path.join(ROOT_DIR, "models", "model_metrics.csv")

    scores_df = pd.read_csv(scores_path)
    importance_df = pd.read_csv(importance_path)
    metrics_df = pd.read_csv(metrics_path)
    return scores_df, importance_df, metrics_df


df, importance_df, metrics_df = load_dashboard_data()

# --- HEADER ---
st.title("AI REVENUE INTELLIGENCE")
st.caption("AI-powered customer churn and revenue-risk decision system")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Portfolio Filters")
selected_plans = st.sidebar.multiselect(
    "Subscription Plan",
    options=sorted(df["plan"].unique()),
    default=sorted(df["plan"].unique()),
)
selected_risks = st.sidebar.multiselect(
    "Risk Level",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
)

filtered_df = df[
    (df["plan"].isin(selected_plans)) & (df["risk_level"].isin(selected_risks))
].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("Model Benchmarks")
st.sidebar.dataframe(metrics_df[["model", "roc_auc", "precision", "recall"]], hide_index=True)

# --- KPI CARDS ---
total_customers = len(filtered_df)
churn_rate = filtered_df["churn"].mean() * 100 if total_customers > 0 else 0.0
monthly_revenue = filtered_df["monthly_revenue"].sum()
annual_exposure = filtered_df["expected_annual_loss"].sum()
critical_customers = int((filtered_df["risk_level"] == "CRITICAL").sum())

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Customers", f"{total_customers:,}")
kpi2.metric("Historical Churn Rate", f"{churn_rate:.2f}%")
kpi3.metric("Monthly Recurring Revenue", f"₹{monthly_revenue:,.0f}")
kpi4.metric("Expected Annual Exposure", f"₹{annual_exposure:,.0f}")
kpi5.metric("Critical Risk Accounts", f"{critical_customers:,}")

st.markdown("---")

# --- CHARTS ROW 1 ---
col1, col2, col3 = st.columns(3)

with col1:
    plan_churn = (
        filtered_df.groupby("plan", as_index=False)
        .agg(churn_rate=("churn", lambda s: round(s.mean() * 100, 2)))
        .sort_values("churn_rate", ascending=False)
    )
    fig_churn_plan = px.bar(
        plan_churn,
        x="plan",
        y="churn_rate",
        title="1. Churn Rate by Subscription Plan (%)",
        text="churn_rate",
        color="plan",
    )
    fig_churn_plan.update_layout(showlegend=False, xaxis_title="Plan", yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig_churn_plan, use_container_width=True)

with col2:
    plan_rev = (
        filtered_df.groupby("plan", as_index=False)
        .agg(monthly_revenue=("monthly_revenue", "sum"))
        .sort_values("monthly_revenue", ascending=False)
    )
    fig_rev_plan = px.bar(
        plan_rev,
        x="plan",
        y="monthly_revenue",
        title="2. Monthly Revenue by Plan (₹)",
        color="plan",
    )
    fig_rev_plan.update_layout(showlegend=False, xaxis_title="Plan", yaxis_title="Monthly Revenue (₹)")
    st.plotly_chart(fig_rev_plan, use_container_width=True)

with col3:
    risk_counts = (
        filtered_df["risk_level"]
        .value_counts()
        .reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["risk_level", "count"]
    color_map = {
        "CRITICAL": "#d9534f",
        "HIGH": "#f0ad4e",
        "MEDIUM": "#5bc0de",
        "LOW": "#5cb85c",
    }
    fig_risk_dist = px.pie(
        risk_counts,
        names="risk_level",
        values="count",
        title="5. Risk Category Distribution",
        color="risk_level",
        color_discrete_map=color_map,
        hole=0.4,
    )
    st.plotly_chart(fig_risk_dist, use_container_width=True)

# --- CHARTS ROW 2 ---
col4, col5 = st.columns(2)

with col4:
    fig_prob_dist = px.histogram(
        filtered_df,
        x="churn_prob_pct",
        nbins=30,
        title="3. Churn Probability Distribution (%)",
        color="risk_level",
        color_discrete_map=color_map,
    )
    fig_prob_dist.update_layout(xaxis_title="Predicted Churn Probability (%)", yaxis_title="Customer Count")
    st.plotly_chart(fig_prob_dist, use_container_width=True)

with col5:
    top_signals = importance_df.head(10).sort_values("importance", ascending=True)
    fig_importance = px.bar(
        top_signals,
        x="importance",
        y="feature",
        orientation="h",
        title="4. Top 10 Predictive Churn Signals (Random Forest Importance)",
    )
    fig_importance.update_layout(xaxis_title="Relative Importance", yaxis_title="Predictive Signal")
    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")

# --- CUSTOMER RISK TABLE ---
st.subheader("Prioritized Revenue-at-Risk Accounts (Sorted by Expected Annual Loss)")
table_cols = [
    "customer_id",
    "plan",
    "contract_type",
    "monthly_revenue",
    "churn_prob_pct",
    "risk_level",
    "expected_annual_loss",
    "last_login_days",
    "payment_failures",
    "feature_usage",
    "nps_score",
]
display_df = filtered_df[table_cols].sort_values("expected_annual_loss", ascending=False).head(50)
st.dataframe(
    display_df.rename(
        columns={
            "customer_id": "Customer ID",
            "plan": "Plan",
            "contract_type": "Contract",
            "monthly_revenue": "Monthly Revenue (₹)",
            "churn_prob_pct": "Churn Probability (%)",
            "risk_level": "Risk Level",
            "expected_annual_loss": "Expected Annual Loss (₹)",
            "last_login_days": "Last Login (Days)",
            "payment_failures": "Payment Failures",
            "feature_usage": "Feature Usage (%)",
            "nps_score": "NPS Score",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# --- CUSTOMER DETAIL & GENAI RECOMMENDATION ---
st.subheader("Customer Risk Deep-Dive & AI Retention Action Plan")

if len(filtered_df) == 0:
    st.warning("No customers match the current sidebar filter selection.")
else:
    top_ids = filtered_df.sort_values("expected_annual_loss", ascending=False)["customer_id"].tolist()
    selected_id = st.selectbox(
        "Select a Customer ID to inspect risk signals and generate an AI retention plan:",
        options=top_ids,
        index=0,
    )

    cust = filtered_df[filtered_df["customer_id"] == selected_id].iloc[0]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Customer Profile", f"#{int(cust['customer_id'])} • {cust['plan']}", f"{cust['contract_type']} ({int(cust['tenure_months'])} mo)")
    p2.metric("Churn Probability", f"{cust['churn_prob_pct']}%", f"Risk: {cust['risk_level']}")
    p3.metric("Monthly Revenue", f"₹{cust['monthly_revenue']:,.2f}", f"NPS: {int(cust['nps_score'])}/10")
    p4.metric("Expected Annual Exposure", f"₹{cust['expected_annual_loss']:,.2f}", f"Monthly Risk: ₹{cust['expected_monthly_loss']:,.2f}")

    st.markdown("**Observed Behavioral Signals:**")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.info(f"Last Login: **{cust['last_login_days']} days**")
    s2.info(f"Payment Failures: **{int(cust['payment_failures'])}**")
    s3.info(f"Feature Usage: **{cust['feature_usage']}%**")
    s4.info(f"Support Tickets: **{int(cust['support_tickets'])}**")
    s5.info(f"NPS Score: **{int(cust['nps_score'])}/10**")

    st.markdown("### AI RETENTION RECOMMENDATION")
    with st.spinner("Generating grounded retention strategy..."):
        insights = generate_retention_insights(cust)

    st.caption(f"Analysis Engine: {insights.get('source', 'AI Engine')}")
    st.success(f"**Risk Summary:** {insights.get('risk_summary', '')}")

    left_ai, right_ai = st.columns(2)
    with left_ai:
        st.markdown("**Strongest Predictive Signals:**")
        for cause in insights.get("root_causes", []):
            st.markdown(f"- {cause}")

        st.markdown("**Recommended Retention Actions:**")
        for action in insights.get("recommended_actions", []):
            st.markdown(f"- {action}")

    with right_ai:
        st.markdown("**Suggested Customer Outreach Message:**")
        st.text_area(
            "Ready-to-send CSM message",
            value=insights.get("customer_message", ""),
            height=150,
        )
        st.info(f"**Expected Business Impact:** {insights.get('business_impact', '')}")