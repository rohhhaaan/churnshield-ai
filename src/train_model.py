import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow running directly via `python src/train_model.py`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_features_from_db,
)

# Configurable Risk Thresholds
RISK_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH": 0.60,
    "MEDIUM": 0.40,
}


def assign_risk_level(prob: float, thresholds: dict = RISK_THRESHOLDS) -> str:
    """Map churn probability into configurable business risk tiers."""
    if prob >= thresholds["CRITICAL"]:
        return "CRITICAL"
    elif prob >= thresholds["HIGH"]:
        return "HIGH"
    elif prob >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate classification metrics with focus on Precision, Recall, F1, and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def extract_feature_importance(rf_pipeline: Pipeline) -> pd.DataFrame:
    """Extract Random Forest predictive signal importances with clean feature names."""
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    classifier = rf_pipeline.named_steps["classifier"]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names

    importance_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def train_and_score():
    df = load_features_from_db()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = []
    fitted_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    results_df = pd.DataFrame(results)
    print("\n=== MODEL EVALUATION (20% Stratified Test Set) ===")
    print(results_df.to_string(index=False))

    # Select best model by ROC-AUC (calibrated probability ranking quality)
    best_row = results_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = fitted_models[best_model_name]
    print(f"\nSelected Best Model: {best_model_name} (ROC-AUC = {best_row['roc_auc']})")

    # Extract Random Forest predictive feature importance
    importance_df = extract_feature_importance(fitted_models["Random Forest"])
    print("\n=== TOP 10 PREDICTIVE CHURN SIGNALS (Random Forest Importance) ===")
    print(importance_df.head(10).to_string(index=False))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, os.path.join("models", "churn_model.pkl"))
    importance_df.to_csv(os.path.join("models", "feature_importance.csv"), index=False)
    results_df.to_csv(os.path.join("models", "model_metrics.csv"), index=False)

    # Revenue Risk Engine across all customers
    scored_df = df.copy()
    scored_df["churn_probability"] = np.round(best_model.predict_proba(X)[:, 1], 4)
    scored_df["churn_prob_pct"] = np.round(scored_df["churn_probability"] * 100, 1)
    scored_df["risk_level"] = scored_df["churn_probability"].apply(assign_risk_level)
    scored_df["expected_monthly_loss"] = np.round(
        scored_df["monthly_revenue"] * scored_df["churn_probability"], 2
    )
    scored_df["expected_annual_loss"] = np.round(
        scored_df["expected_monthly_loss"] * 12, 2
    )

    scored_df = scored_df.sort_values("expected_annual_loss", ascending=False)
    scored_output = os.path.join("data", "customer_risk_scores.csv")
    scored_df.to_csv(scored_output, index=False)

    print("\n=== REVENUE RISK ENGINE SUMMARY ===")
    print("Risk Tier Distribution:")
    print(scored_df["risk_level"].value_counts().to_string())
    print(f"\nTotal Expected Monthly Loss: INR {scored_df['expected_monthly_loss'].sum():,.2f}")
    print(f"Total Expected Annual Exposure: INR {scored_df['expected_annual_loss'].sum():,.2f}")

    print("\n=== TOP 10 CUSTOMERS BY EXPECTED ANNUAL REVENUE LOSS ===")
    display_cols = [
        "customer_id",
        "plan",
        "monthly_revenue",
        "churn_prob_pct",
        "risk_level",
        "expected_annual_loss",
        "last_login_days",
        "payment_failures",
    ]
    print(scored_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    train_and_score()