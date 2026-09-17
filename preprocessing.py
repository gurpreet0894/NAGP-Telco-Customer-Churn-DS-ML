"""
Shared preprocessing / feature-engineering code for the Telco Customer Churn project.

This module is imported by BOTH the training notebook (notebook/churn_analysis.ipynb)
and the serving API (app.py). Keeping the logic in one place guarantees that new/unseen
customer data is transformed in exactly the same way it was during training, which
avoids train/serve skew and data leakage.

The full pipeline is:
    raw dataframe -> ChurnFeatureEngineer (cleaning + new features) -> ColumnTransformer
    (one-hot encode categoricals, pass through numerics) -> DecisionTreeClassifier

Only the DecisionTreeClassifier step differs between the model experiments in the
notebook; everything upstream of it is identical, so it is safe to fit the whole
Pipeline on the training split only and reuse it unchanged at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Raw columns expected in the incoming data (everything except the identifier and target).
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "num_addon_services", "avg_monthly_charge",
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans raw Telco churn rows and derives two extra features.

    Cleaning:
        - TotalCharges arrives as a string and is blank for the 11 brand-new
          customers with tenure == 0. It is coerced to numeric and blanks are
          filled with 0 (they have not been billed yet).

    Engineered features:
        - num_addon_services: count of the 6 optional add-on services a customer
          has subscribed to (0-6). A simple proxy for how "invested"/sticky a
          customer is - more add-ons tends to mean more switching friction.
        - avg_monthly_charge: TotalCharges / tenure (falls back to MonthlyCharges
          for tenure == 0). Flags customers whose historical spend differs from
          their current MonthlyCharges, e.g. after a plan change or promo ending.
        - tenure_group: tenure bucketed into business-friendly ranges
          (0-12, 12-24, 24-48, 48-60, 60-72 months), which makes the non-linear
          "new customers churn a lot more" pattern explicit and easy to read off
          a decision tree.

    This transformer is stateless (no statistics are learned from the training
    data), so calling it identically on train, test, or brand-new API payloads
    cannot leak information across splits.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        df["num_addon_services"] = (df[ADDON_SERVICE_COLUMNS] == "Yes").sum(axis=1)

        safe_tenure = df["tenure"].replace(0, np.nan)
        df["avg_monthly_charge"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])

        bins = [-0.1, 12, 24, 48, 60, np.inf]
        labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_pipeline(classifier):
    """Wraps a (scikit-learn compatible) classifier in the full preprocessing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def get_output_feature_names(fitted_pipeline):
    """Returns the expanded feature names (after one-hot encoding) for a fitted pipeline."""
    return fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()


def validate_and_build_dataframe(payload: dict) -> pd.DataFrame:
    """Validates a single customer's JSON payload and returns a 1-row DataFrame.

    Raises ValueError with a human-readable message on any invalid/missing input,
    which the API layer turns into a 400 response.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object of customer attributes.")

    missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    row = {col: payload[col] for col in RAW_FEATURE_COLUMNS}

    try:
        row["SeniorCitizen"] = int(row["SeniorCitizen"])
        row["tenure"] = int(row["tenure"])
        row["MonthlyCharges"] = float(row["MonthlyCharges"])
        row["TotalCharges"] = float(row["TotalCharges"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SeniorCitizen and tenure must be integers; MonthlyCharges and "
            "TotalCharges must be numbers."
        ) from exc

    if row["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1.")
    if row["tenure"] < 0:
        raise ValueError("tenure cannot be negative.")
    if row["MonthlyCharges"] < 0 or row["TotalCharges"] < 0:
        raise ValueError("MonthlyCharges and TotalCharges cannot be negative.")

    return pd.DataFrame([row])
