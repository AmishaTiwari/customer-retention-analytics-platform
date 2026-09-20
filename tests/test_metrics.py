"""Hand-computed toy-case tests for evaluation/metrics.py.

A subtly wrong Precision@K would poison every downstream conclusion and
the model-selection decision itself, so these tests use small,
hand-verifiable examples rather than real pipeline data.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.metrics import average_precision_score

from retention_platform.evaluation.metrics import brier_score, lift_at_k, precision_at_k, pr_auc

# y_true has 4 positives out of 10 rows (base rate 0.4). y_score ranks
# rows 1, 4, 2, 8, 0 highest to lowest (scores 9, 8, 7, 6, 5); the other
# five rows all score lower. Top 5 by score (k_frac=0.5, k=5) are indices
# 1, 4, 2, 8, 0 -- y_true at those indices is 1, 1, 1, 1, 0, so
# precision_at_k = 4/5 = 0.8 and lift_at_k = 0.8 / 0.4 = 2.0.
Y_TRUE = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
Y_SCORE = [5, 9, 7, 3, 8, 2, 4, 1, 6, 0]


def test_precision_at_k_hand_computed():
    assert precision_at_k(Y_TRUE, Y_SCORE, 0.5) == pytest.approx(0.8)


def test_lift_at_k_equals_precision_divided_by_base_rate():
    precision = precision_at_k(Y_TRUE, Y_SCORE, 0.5)
    base_rate = np.mean(Y_TRUE)

    assert lift_at_k(Y_TRUE, Y_SCORE, 0.5) == pytest.approx(precision / base_rate)
    assert lift_at_k(Y_TRUE, Y_SCORE, 0.5) == pytest.approx(2.0)


@pytest.mark.parametrize("bad_k_frac", [0, -0.1, 1.5])
def test_invalid_k_frac_raises(bad_k_frac):
    with pytest.raises(ValueError):
        precision_at_k(Y_TRUE, Y_SCORE, bad_k_frac)
    with pytest.raises(ValueError):
        lift_at_k(Y_TRUE, Y_SCORE, bad_k_frac)


def test_mismatched_length_raises():
    with pytest.raises(ValueError):
        precision_at_k(Y_TRUE, Y_SCORE[:-1], 0.5)
    with pytest.raises(ValueError):
        lift_at_k(Y_TRUE, Y_SCORE[:-1], 0.5)


def test_k_rounds_up_to_at_least_one():
    # len=3, k_frac=0.1 -> round(3 * 0.1) = 0, but k must be at least 1.
    # Top-scoring row (index 0, score 5) is a true positive, so
    # precision_at_k with k=1 is exactly 1.0.
    y_true = [1, 0, 0]
    y_score = [5, 1, 2]

    assert precision_at_k(y_true, y_score, 0.1) == pytest.approx(1.0)


def test_pr_auc_matches_sklearn_directly():
    # PR-AUC's formula isn't simple enough to hand-verify like
    # precision_at_k -- this confirms pr_auc wraps
    # average_precision_score with the arguments in the right order,
    # not that the metric's value is independently correct.
    assert pr_auc(Y_TRUE, Y_SCORE) == pytest.approx(
        average_precision_score(Y_TRUE, Y_SCORE)
    )


def test_pr_auc_mismatched_length_raises():
    with pytest.raises(ValueError):
        pr_auc(Y_TRUE, Y_SCORE[:-1])


def test_brier_score_hand_computed():
    # Brier score = mean((y_score - y_true) ** 2).
    # (0.9-1)^2=0.01, (0.1-0)^2=0.01, (0.6-1)^2=0.16, (0.4-0)^2=0.16
    # mean = (0.01 + 0.01 + 0.16 + 0.16) / 4 = 0.085
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.6, 0.4]

    assert brier_score(y_true, y_score) == pytest.approx(0.085)


def test_brier_score_mismatched_length_raises():
    with pytest.raises(ValueError):
        brier_score(Y_TRUE, Y_SCORE[:-1])
