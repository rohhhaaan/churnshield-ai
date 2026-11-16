DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    plan VARCHAR(20) NOT NULL,
    tenure_months INT NOT NULL,
    monthly_revenue NUMERIC(10, 2) NOT NULL,
    login_count INT NOT NULL,
    sessions INT NOT NULL,
    support_tickets INT NOT NULL,
    avg_session_minutes NUMERIC(6, 2) NOT NULL,
    feature_usage NUMERIC(6, 2) NOT NULL,
    payment_failures INT NOT NULL,
    last_login_days NUMERIC(6, 2) NOT NULL,
    discount_percentage NUMERIC(5, 2) NOT NULL,
    nps_score INT NOT NULL,
    contract_type VARCHAR(20) NOT NULL,
    churn INT NOT NULL CHECK (churn IN (0, 1))
);