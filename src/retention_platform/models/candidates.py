"""Candidate model definitions: business heuristic baseline, logistic
regression, random forest, and XGBoost.

The heuristic baseline is defined here alongside the ML models
deliberately — it passes through the identical evaluation harness used
by the ML candidates, which is what makes the lift-over-heuristic claim
credible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def predict_business_heuristic(df: pd.DataFrame) -> pd.Series:
    """Flag every customer on a month-to-month contract as at-risk.

    Operates directly on an unencoded, pre-preprocessing DataFrame shaped
    like feat_churn (must contain is_month_to_month) -- this is a fixed
    business rule, not a fitted model, so it never touches the
    preprocessing pipeline. is_month_to_month is used as-is because it is
    the single strongest, whole-population binary risk signal found in
    EDA: contract type without a fixed term correlates with substantially
    higher churn.
    """
    return df["is_month_to_month"].astype(bool)


def fit_logistic_regression(
    X_train: np.ndarray, y_train: pd.Series, random_state: int
) -> LogisticRegression:
    """Fit a Logistic Regression on already-preprocessed X_train/y_train.

    random_state is supplied by the caller (matching this project's
    reproducibility seed) rather than hardcoded here, so this function
    never falls out of sync with config/config.yaml. Every other
    hyperparameter (including class_weight) is left at scikit-learn's
    default. This model is deliberately untuned; hyperparameter search is
    a separate later stage.
    """
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train: np.ndarray, y_train: pd.Series, random_state: int
) -> RandomForestClassifier:
    """Fit a Random Forest on already-preprocessed X_train/y_train.

    random_state is supplied by the caller (matching this project's
    reproducibility seed) rather than hardcoded here, so this function
    never falls out of sync with config/config.yaml. n_estimators is
    pinned explicitly to 100 rather than relying on scikit-learn's current
    default, so this baseline's behavior doesn't silently drift if a
    future scikit-learn version changes that default. Every other
    hyperparameter (including class_weight) is left at scikit-learn's
    default. This model is deliberately untuned; hyperparameter search is
    a separate later stage.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """Return the predicted probability of voluntary churn (the positive class).

    This risk score, not a hard 0/1 label, is the primary prediction
    artifact -- downstream evaluation (Precision@K, Lift@K) ranks
    customers by score. A hard label can be derived later from a
    threshold applied to this score, but none is produced here.
    """
    return model.predict_proba(X)[:, 1]
