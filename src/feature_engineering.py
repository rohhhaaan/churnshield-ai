import os
from urllib.parse import quote_plus
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_revenue",
    "login_count",
    "sessions",
    "support_tickets",
    "avg_session_minutes",
    "feature_usage",
    "payment_failures",
    "last_login_days",
    "discount_percentage",
    "nps_score",
    "inactive_customer",
    "payment_risk",
    "support_risk",
    "engagement_risk",
    "detractor_risk",
    "total_risk_signals",
    "sessions_per_login",
    "total_active_hours",
    "revenue_per_tenure_month",
]

CATEGORICAL_FEATURES = ["plan", "contract_type"]


def get_db_engine():
    """Create a SQLAlchemy engine using credentials from .env."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "ai_revenue_intelligence")
    user = os.getenv("DB_USER", "postgres")
    password = quote_plus(os.getenv("DB_PASSWORD", ""))

    connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(connection_url)


def load_features_from_db() -> pd.DataFrame:
    """Query the engineered SQL view customer_ml_features from PostgreSQL."""
    engine = get_db_engine()
    query = "SELECT * FROM customer_ml_features ORDER BY customer_id;"
    df = pd.read_sql(query, engine)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create a Scikit-learn ColumnTransformer for numeric scaling and categorical encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


if __name__ == "__main__":
    df = load_features_from_db()
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from PostgreSQL view 'customer_ml_features'.")
    print("Columns:", list(df.columns))