"""Precision@K, Lift@K, PR-AUC, Brier Score, and related metrics.

Unit-tested against hand-computed toy cases (tests/test_metrics.py),
since a subtly wrong Precision@K would poison every downstream
conclusion and the model-selection decision itself.

Not yet implemented — scaffolded in Commit 1; implemented in Commit 9
(Evaluation & error analysis).
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(y_true, y_score, k_frac: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if not (0 < k_frac <= 1):
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac!r}")

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must be the same length, "
            f"got {len(y_true)} and {len(y_score)}"
        )

    return y_true, y_score


def precision_at_k(y_true, y_score, k_frac: float) -> float:
    """Fraction of true positives among the top k_frac of ranked predictions.

    Ranks by y_score descending, takes the top
    k = max(1, round(len(y_score) * k_frac)) rows, and returns the mean of
    y_true over that slice.
    """
    y_true, y_score = _validate_inputs(y_true, y_score, k_frac)

    k = max(1, round(len(y_score) * k_frac))
    top_k_indices = np.argsort(y_score)[::-1][:k]

    return float(np.mean(y_true[top_k_indices]))


def lift_at_k(y_true, y_score, k_frac: float) -> float:
    """Ratio of precision_at_k to the overall base rate of y_true.

    A lift of 1.0 means the top k_frac performs no better than randomly
    selecting the same number of rows; values above 1.0 indicate the
    ranking concentrates true positives above the base rate.
    """
    y_true, y_score = _validate_inputs(y_true, y_score, k_frac)

    base_rate = float(np.mean(y_true))
    return precision_at_k(y_true, y_score, k_frac) / base_rate
