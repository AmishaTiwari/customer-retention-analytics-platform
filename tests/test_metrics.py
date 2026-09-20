"""Hand-computed toy-case tests for evaluation/metrics.py.

A subtly wrong Precision@K would poison every downstream conclusion and
the model-selection decision itself, so these tests use small,
hand-verifiable examples rather than real pipeline data.
"""

from __future__ import annotations

import numpy as np
import pytest

from retention_platform.evaluation.metrics import lift_at_k, precision_at_k

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
