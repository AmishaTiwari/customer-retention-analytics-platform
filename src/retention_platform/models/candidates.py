"""Candidate model definitions: business heuristic baseline, logistic
regression, random forest, and XGBoost.

The heuristic baseline is defined here alongside the ML models
deliberately — it passes through the identical evaluation harness used
by the ML candidates, which is what makes the lift-over-heuristic claim
credible.
"""

from __future__ import annotations

import pandas as pd


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
