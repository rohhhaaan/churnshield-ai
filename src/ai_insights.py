import json
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def build_rule_based_fallback(customer: dict) -> dict:
    """Deterministic fallback grounded strictly in supplied metrics if API key is missing or unavailable."""
    signals = []
    actions = []

    if customer["last_login_days"] > 25:
        signals.append(f"Extended inactivity: {customer['last_login_days']} days since last login")
        actions.append("Schedule an executive re-engagement check-in and share a tailored workflow quick-start.")
    if customer["payment_failures"] >= 1:
        signals.append(f"Billing friction: {customer['payment_failures']} recent payment failure(s)")
        actions.append("Trigger immediate billing support outreach to update payment method and prevent involuntary churn.")
    if customer["feature_usage"] < 40:
        signals.append(f"Low product adoption: {customer['feature_usage']}% core feature usage")
        actions.append("Offer a 30-minute Customer Success onboarding session focused on underutilized high-value features.")
    if customer["nps_score"] <= 5:
        signals.append(f"Low satisfaction score: NPS {customer['nps_score']}/10")
        actions.append("Escalate account feedback to a senior CSM to resolve open friction points.")
    if customer["support_tickets"] >= 4:
        signals.append(f"Elevated support volume: {customer['support_tickets']} support tickets logged")
        actions.append("Audit recent support tickets and assign a dedicated technical specialist to close unresolved issues.")

    if not signals:
        signals.append(f"Combined moderate behavioral drift across tenure ({customer['tenure_months']} mo) and engagement")
        actions.append("Conduct a proactive quarterly business review (QBR) to reinforce ROI before renewal.")

    return {
        "risk_summary": (
            f"Customer {customer['customer_id']} ({customer['plan']} plan) exhibits a "
            f"{customer['churn_prob_pct']}% predicted churn probability ({customer['risk_level']} risk), "
            f"representing INR {customer['expected_annual_loss']:,.2f} in expected annual revenue exposure."
        ),
        "root_causes": signals,
        "recommended_actions": actions,
        "customer_message": (
            f"Hi there — as a valued {customer['plan']} partner, we noticed your team hasn't been able to "
            f"take full advantage of your workspace recently. I'd love to set up a quick 15-minute session "
            f"to help streamline your setup and ensure everything is running smoothly for your team."
        ),
        "business_impact": (
            f"Retaining Customer {customer['customer_id']} protects INR {customer['monthly_revenue']:,.2f}/month "
            f"(INR {customer['expected_annual_loss']:,.2f} expected annual exposure)."
        ),
    }


def generate_retention_insights(customer_row: dict | pd.Series) -> dict:
    """Generate structured, grounded retention recommendations for a customer using Gemini."""
    customer = dict(customer_row)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        result = build_rule_based_fallback(customer)
        result["source"] = "Rule-Based Fallback (Add GEMINI_API_KEY in .env for live Gemini analysis)"
        return result

    prompt = f"""
You are a senior SaaS customer-retention analyst.
Analyze ONLY the supplied customer metrics below.

STRICT RULES:
- Do NOT invent or assume any customer information not explicitly provided below.
- Do NOT claim that any metric causes churn; describe them strictly as "predictive signals" or "risk indicators".
- Keep recommendations practical, specific, and business-oriented.
- Return valid JSON ONLY matching the exact schema requested.

SUPPLIED CUSTOMER DATA:
- Customer ID: {customer['customer_id']}
- Subscription Plan: {customer['plan']} ({customer['contract_type']} contract)
- Tenure: {customer['tenure_months']} months
- Monthly Revenue: INR {customer['monthly_revenue']:,.2f}
- Churn Probability: {customer['churn_prob_pct']}%
- Risk Level: {customer['risk_level']}
- Expected Annual Revenue Loss: INR {customer['expected_annual_loss']:,.2f}
- Days Since Last Login: {customer['last_login_days']}
- Payment Failures: {customer['payment_failures']}
- Support Tickets: {customer['support_tickets']}
- Feature Usage: {customer['feature_usage']}%
- NPS Score: {customer['nps_score']}/10
- Login Count (recent period): {customer['login_count']}

Return a JSON object with these exact keys:
- "risk_summary": string (2 concise sentences summarizing risk level, probability, and INR revenue exposure)
- "root_causes": list of strings (3 to 4 strongest predictive signals / risk indicators from the data)
- "recommended_actions": list of strings (3 concrete retention actions for Customer Success / Billing)
- "customer_message": string (short, empathetic, non-pushy customer outreach email draft)
- "business_impact": string (1 sentence quantifying the INR monthly/annual revenue protected)
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        parsed = json.loads(response.text)
        parsed["source"] = "Gemini 2.5 Flash"
        return parsed
    except Exception as e:
        result = build_rule_based_fallback(customer)
        result["source"] = f"Fallback (Gemini API notice: {str(e)[:80]})"
        return result


if __name__ == "__main__":
    scores_path = os.path.join("data", "customer_risk_scores.csv")
    df = pd.read_csv(scores_path)
    top_customer = df.iloc[0]

    print(f"Testing AI Insights on #1 Risk Customer: {int(top_customer['customer_id'])}")
    insights = generate_retention_insights(top_customer)
    print(json.dumps(insights, indent=2))