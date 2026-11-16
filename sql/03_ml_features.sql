DROP VIEW IF EXISTS customer_ml_features;

CREATE VIEW customer_ml_features AS
SELECT
    customer_id,
    plan,
    contract_type,
    tenure_months,
    monthly_revenue,
    ROUND(monthly_revenue * 12, 2) AS annual_revenue,
    login_count,
    sessions,
    support_tickets,
    avg_session_minutes,
    feature_usage,
    payment_failures,
    last_login_days,
    discount_percentage,
    nps_score,

    -- Engineered Behavioral Risk Flags
    CASE WHEN last_login_days > 30 THEN 1 ELSE 0 END AS inactive_customer,
    CASE WHEN payment_failures >= 2 THEN 1 ELSE 0 END AS payment_risk,
    CASE WHEN support_tickets >= 4 THEN 1 ELSE 0 END AS support_risk,
    CASE WHEN feature_usage < 35 THEN 1 ELSE 0 END AS engagement_risk,
    CASE WHEN nps_score <= 4 THEN 1 ELSE 0 END AS detractor_risk,

    -- Composite Risk Signal Count (0 to 5)
    (
        CASE WHEN last_login_days > 30 THEN 1 ELSE 0 END +
        CASE WHEN payment_failures >= 2 THEN 1 ELSE 0 END +
        CASE WHEN support_tickets >= 4 THEN 1 ELSE 0 END +
        CASE WHEN feature_usage < 35 THEN 1 ELSE 0 END +
        CASE WHEN nps_score <= 4 THEN 1 ELSE 0 END
    ) AS total_risk_signals,

    -- Engineered Engagement & Value Ratios
    ROUND(sessions::NUMERIC / GREATEST(login_count, 1), 2) AS sessions_per_login,
    ROUND((login_count * avg_session_minutes) / 60.0, 2) AS total_active_hours,
    ROUND(monthly_revenue / GREATEST(tenure_months, 1), 2) AS revenue_per_tenure_month,

    -- Target Variable
    churn
FROM customers;