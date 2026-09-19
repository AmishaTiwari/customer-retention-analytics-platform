"""Budgeted hyperparameter search for the candidate models.

Each tune_* function wraps a single estimator in a RandomizedSearchCV over
a fixed search space and returns the fitted search object itself, so
callers retain access to best_params_, best_score_, and cv_results_
rather than only the winning estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

_OBJECTIVE_METRIC_TO_SKLEARN_SCORING = {
    "pr_auc": "average_precision",
}


def tune_logistic_regression(
    X_train: np.ndarray, y_train: pd.Series, config: dict
) -> RandomizedSearchCV:
    """Run a randomized hyperparameter search over Logistic Regression.

    Searches C on a log-uniform scale, l1_ratio, and whether class_weight
    is balanced, using liblinear as the solver since it supports both l1
    and l2 regularization. Budget, scoring metric, and
    cross-validation are all read from config rather than hardcoded, so
    this function stays in sync with a single source of truth rather than
    carrying its own copy of those values. The objective_metric config
    value ("pr_auc") is mapped explicitly to scikit-learn's own scorer
    name ("average_precision") rather than passed through as-is, since
    scikit-learn would not recognize "pr_auc" directly; any other value
    raises rather than silently picking a default. Returns the fitted
    search object itself, not best_estimator_, so the caller can inspect
    best_params_, best_score_, and cv_results_.
    """
    objective_metric = config["tuning"]["objective_metric"]
    if objective_metric not in _OBJECTIVE_METRIC_TO_SKLEARN_SCORING:
        raise ValueError(
            f"Unsupported objective_metric {objective_metric!r}; "
            f"expected one of {sorted(_OBJECTIVE_METRIC_TO_SKLEARN_SCORING)}"
        )
    scoring = _OBJECTIVE_METRIC_TO_SKLEARN_SCORING[objective_metric]

    random_state = config["reproducibility"]["seed"]

    param_distributions = {
        "C": loguniform(1e-3, 1e2),
        "l1_ratio": [1.0, 0.0],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
    }

    cv = RepeatedStratifiedKFold(
        n_splits=config["data"]["cv_folds"],
        n_repeats=config["data"]["cv_repeats"],
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=random_state),
        param_distributions=param_distributions,
        n_iter=config["tuning"]["budget_per_model"],
        scoring=scoring,
        cv=cv,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    return search
