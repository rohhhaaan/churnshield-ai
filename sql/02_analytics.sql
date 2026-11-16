-- 1. Overall Customer Count & Churn Rate
SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
FROM customers;

-- 2. Revenue Breakdown by Subscription Plan
SELECT
    plan,
    COUNT(*) AS customers,
    ROUND(SUM(monthly_revenue), 2) AS total_monthly_revenue,
    ROUND(SUM(monthly_revenue) * 12, 2) AS total_annual_revenue,
    ROUND(AVG(monthly_revenue), 2) AS avg_monthly_revenue
FROM customers
GROUP BY plan
ORDER BY total_monthly_revenue DESC;

-- 3. Churn Rate by Subscription Plan
SELECT
    plan,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY plan
ORDER BY churn_rate_pct DESC;

-- 4. Historical Revenue Lost / At Risk (Where churn = 1)
SELECT
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_revenue ELSE 0 END), 2) AS monthly_revenue_at_risk,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_revenue * 12 ELSE 0 END), 2) AS annual_revenue_exposure
FROM customers;

-- 5. Top 10 High-Value, High-Risk Active/At-Risk Accounts
SELECT
    customer_id,
    plan,
    monthly_revenue,
    ROUND(monthly_revenue * 12, 2) AS annual_revenue,
    last_login_days,
    payment_failures,
    feature_usage,
    nps_score,
    support_tickets,
    churn
FROM customers
WHERE monthly_revenue >= 10000
  AND (last_login_days > 25 OR payment_failures >= 2 OR feature_usage < 35 OR nps_score <= 4)
ORDER BY monthly_revenue DESC, last_login_days DESC
LIMIT 10;