"""Preprocessing, model construction, and evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    """Build the serializable imputation, encoding, and scaling pipeline."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))]
    )


def build_classifier(model_name: str):
    """Return a deterministic baseline classifier."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000,
            random_state=RANDOM_STATE, solver="liblinear"
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str) -> Pipeline:
    """Combine all feature processing and classification into one pipeline."""
    return Pipeline([("preprocessor", build_preprocessor()),
                     ("classifier", build_classifier(model_name))])


def parameter_grid(model_name: str, fast: bool = False) -> dict[str, list]:
    """Return compact search spaces suitable for this small dataset."""
    if model_name == "logistic_regression":
        return {"classifier__C": [0.3, 1.0] if fast else [0.1, 0.3, 1.0, 3.0]}
    if model_name == "random_forest":
        if fast:
            return {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [None, 6],
                "classifier__min_samples_leaf": [1],
            }
        return {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 5, 8],
            "classifier__min_samples_leaf": [1, 2],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_binary_classifier(model, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Compute accuracy, precision, recall, and ROC-AUC."""
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def probability_for_positive_class(model, features: pd.DataFrame) -> np.ndarray:
    """Return positive class probabilities from a fitted binary classifier."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model does not expose predict_proba.")
    return model.predict_proba(features)[:, 1]

