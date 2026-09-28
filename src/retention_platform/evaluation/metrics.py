"""Precision@K, Lift@K, PR-AUC, Brier Score, and related metrics.

Unit-tested against hand-computed toy cases (tests/test_metrics.py),
since a subtly wrong Precision@K would poison every downstream
conclusion and the model-selection decision itself.

Not yet implemented — scaffolded in Commit 1; implemented in Commit 9
(Evaluation & error analysis).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def _validate_lengths(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must be the same length, "
            f"got {len(y_true)} and {len(y_score)}"
        )

    return y_true, y_score


def _validate_k_inputs(y_true, y_score, k_frac: float) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_score = _validate_lengths(y_true, y_score)

    if not (0 < k_frac <= 1):
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac!r}")

    return y_true, y_score


def _top_k_mask(y_true, y_score, k_frac: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Rank by y_score descending and mark the top k_frac as predicted positive.

    Returns (y_true, y_pred, k) where y_pred is a boolean array aligned
    with y_true: True for rows in the top k = max(1, round(len(y_score) *
    k_frac)) by score, False otherwise. This is the project's single
    canonical "top K by predicted risk" operating point (Business
    Design Threshold Policy), used by precision_at_k and by
    confusion_matrix_at_k so ranking logic exists in exactly one place.
    """
    y_true, y_score = _validate_k_inputs(y_true, y_score, k_frac)

    k = max(1, round(len(y_score) * k_frac))
    top_k_indices = np.argsort(y_score)[::-1][:k]

    y_pred = np.zeros(len(y_score), dtype=bool)
    y_pred[top_k_indices] = True

    return y_true, y_pred, k


def precision_at_k(y_true, y_score, k_frac: float) -> float:
    """Fraction of true positives among the top k_frac of ranked predictions.

    Ranks by y_score descending, takes the top
    k = max(1, round(len(y_score) * k_frac)) rows, and returns the mean of
    y_true over that slice.
    """
    y_true, y_pred, k = _top_k_mask(y_true, y_score, k_frac)
    return float(np.mean(y_true[y_pred]))


def lift_at_k(y_true, y_score, k_frac: float) -> float:
    """Ratio of precision_at_k to the overall base rate of y_true.

    A lift of 1.0 means the top k_frac performs no better than randomly
    selecting the same number of rows; values above 1.0 indicate the
    ranking concentrates true positives above the base rate.
    """
    y_true, y_score = _validate_k_inputs(y_true, y_score, k_frac)

    base_rate = float(np.mean(y_true))
    return precision_at_k(y_true, y_score, k_frac) / base_rate


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve.

    Thin wrapper around sklearn.metrics.average_precision_score, the same
    implementation used as the tuning objective in models/tune.py, so
    this reports the identical metric that was actually optimized.
    """
    y_true, y_score = _validate_lengths(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true, y_score) -> float:
    """Mean squared error between predicted probability and true outcome.

    Thin wrapper around sklearn.metrics.brier_score_loss. Lower is
    better; 0.0 is a perfect calibration match.
    """
    y_true, y_score = _validate_lengths(y_true, y_score)
    return float(brier_score_loss(y_true, y_score))


def confusion_matrix_at_k(y_true, y_score, k_frac: float) -> dict:
    """Confusion matrix and derived Precision/Recall/F1 at the top k_frac
    by predicted risk (the project's K-based business operating point,
    not a 0.5 probability threshold).

    Diagnostic/interpretive per ADR-0008's Supporting Metrics
    classification -- does not independently drive model selection.

    Returns a dict with keys:
        tp, fp, tn, fn   -- confusion matrix counts (int)
        precision, recall, f1  -- derived rates (float)
    """
    y_true, y_pred, k = _top_k_mask(y_true, y_score, k_frac)

    tp = int(np.sum(y_true[y_pred]))
    fp = k - tp
    total_positives = int(np.sum(y_true))
    fn = total_positives - tp
    tn = len(y_true) - k - fn

    precision = tp / k
    recall = tp / total_positives if total_positives > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
