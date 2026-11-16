import os
import numpy as np
import pandas as pd


def generate_customers(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic SaaS customer dataset with realistic behavioral churn drivers."""
    np.random.seed(seed)

    customer_id = np.arange(1001, 1001 + n_customers)

    # Subscription plans with realistic B2B SaaS distribution
    plans = np.random.choice(
        ["Starter", "Growth", "Business", "Enterprise"],
        size=n_customers,
        p=[0.40, 0.32, 0.18, 0.10],
    )

    # Contract type depends loosely on plan tier
    contract_type = []
    for plan in plans:
        if plan == "Starter":
            contract_type.append(np.random.choice(["Monthly", "Annual"], p=[0.80, 0.20]))
        elif plan == "Growth":
            contract_type.append(np.random.choice(["Monthly", "Annual"], p=[0.60, 0.40]))
        elif plan == "Business":
            contract_type.append(np.random.choice(["Monthly", "Annual"], p=[0.35, 0.65]))
        else:
            contract_type.append(np.random.choice(["Monthly", "Annual"], p=[0.15, 0.85]))
    contract_type = np.array(contract_type)

    # Tenure in months (annual contracts tend to stay longer)
    base_tenure = np.random.exponential(scale=14, size=n_customers).astype(int) + 1
    tenure_months = np.clip(
        base_tenure + np.where(contract_type == "Annual", np.random.randint(3, 12, size=n_customers), 0),
        1,
        60,
    )

    # Discount percentage
    discount_percentage = np.round(
        np.where(
            contract_type == "Annual",
            np.random.uniform(5, 25, size=n_customers),
            np.random.uniform(0, 10, size=n_customers),
        ),
        1,
    )

    # Base monthly revenue in INR by plan, adjusted by discount
    base_prices = {
        "Starter": 2499,
        "Growth": 6999,
        "Business": 14999,
        "Enterprise": 34999,
    }
    raw_revenue = np.array([base_prices[p] for p in plans], dtype=float)
    monthly_revenue = np.round(raw_revenue * (1 - discount_percentage / 100.0), 2)

    # Behavioral & engagement signals
    # Create an underlying latent health factor so behavioral metrics correlate naturally
    customer_health = np.random.normal(loc=0.0, scale=1.0, size=n_customers)

    last_login_days = np.round(
        np.clip(np.random.exponential(scale=10, size=n_customers) - customer_health * 6, 0, 90),
        1,
    )

    login_count = np.clip(
        (np.random.poisson(lam=18, size=n_customers) + customer_health * 5 - last_login_days * 0.25).astype(int),
        0,
        60,
    )

    sessions = np.clip(
        (login_count * np.random.uniform(1.0, 1.8, size=n_customers)).astype(int),
        0,
        100,
    )

    avg_session_minutes = np.round(
        np.clip(np.random.normal(loc=22, scale=8, size=n_customers) + customer_health * 4, 2, 65),
        1,
    )

    feature_usage = np.round(
        np.clip(np.random.normal(loc=58, scale=20, size=n_customers) + customer_health * 12 - last_login_days * 0.3, 5, 100),
        1,
    )

    support_tickets = np.clip(
        np.random.poisson(lam=1.8, size=n_customers) + np.where(customer_health < -0.8, np.random.randint(1, 5, size=n_customers), 0),
        0,
        15,
    )

    payment_failures = np.clip(
        np.random.choice([0, 1, 2, 3, 4], size=n_customers, p=[0.74, 0.14, 0.07, 0.03, 0.02])
        + np.where(customer_health < -1.2, 1, 0),
        0,
        5,
    )

    nps_score = np.clip(
        np.round(np.random.normal(loc=7.2, scale=2.0, size=n_customers) + customer_health * 1.1 - support_tickets * 0.25).astype(int),
        0,
        10,
    )

    # Calculate realistic churn log-odds (combining signals + random noise so relationship is realistic, not deterministic)
    logit = (
        -0.35
        + 0.045 * last_login_days
        + 0.65 * payment_failures
        + 0.22 * support_tickets
        - 0.028 * feature_usage
        - 0.18 * nps_score
        - 0.025 * tenure_months
        - np.where(contract_type == "Annual", 0.55, 0.0)
        + np.random.normal(0, 0.45, size=n_customers)  # realistic unobserved noise
    )

    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, size=n_customers) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "plan": plans,
            "tenure_months": tenure_months,
            "monthly_revenue": monthly_revenue,
            "login_count": login_count,
            "sessions": sessions,
            "support_tickets": support_tickets,
            "avg_session_minutes": avg_session_minutes,
            "feature_usage": feature_usage,
            "payment_failures": payment_failures,
            "last_login_days": last_login_days,
            "discount_percentage": discount_percentage,
            "nps_score": nps_score,
            "contract_type": contract_type,
            "churn": churn,
        }
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_customers(n_customers=5000, seed=42)
    output_path = os.path.join("data", "customers.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Overall churn rate: {df['churn'].mean() * 100:.2f}%")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(2))