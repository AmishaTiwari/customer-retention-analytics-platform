"""Identical-harness candidate model comparison and comparison table
construction, per the Model Comparison Strategy locked in ML System
Design v1.0.

Currently implemented: running each candidate individually through the
identical evaluation harness (evaluate_candidate, evaluate_all_candidates).
Cross-model comparison, ranking, and selection are a later step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb

from retention_platform.evaluation.metrics import (
    brier_score,
    confusion_matrix_at_k,
    lift_at_k,
    precision_at_k,
    pr_auc,
    sensitivity_report,
)
from retention_platform.models.candidates import predict_business_heuristic, predict_proba
from retention_platform.pipeline.preprocess import prepare_model_inputs
from retention_platform.pipeline.split import split_feat_churn

logger = logging.getLogger("retention_platform.evaluation.compare")

# Primary K for Precision@K, Lift@K, and the confusion matrix (ADR-0010).
PRIMARY_K_FRAC = 0.10

# Sensitivity sweep reported alongside the primary K (ADR-0010).
SENSITIVITY_K_FRACS = [0.05, 0.10, 0.15, 0.20, 0.25]


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate's metric outputs from the identical evaluation harness."""

    precision_at_k: float
    lift_at_k: float
    pr_auc: float
    brier_score: float
    confusion_matrix_at_k: dict
    sensitivity_report: list[dict]


def evaluate_candidate(y_true, y_score) -> EvaluationResult:
    """Run one candidate's predictions through every metrics.py function.

    y_score is whatever score the candidate produces for ranking --
    predicted probabilities for the ML models, or the heuristic's boolean
    flag -- metrics.py's functions only require it to be rankable, not a
    calibrated probability. Precision@K, Lift@K, and the confusion matrix
    use PRIMARY_K_FRAC; sensitivity_report sweeps SENSITIVITY_K_FRACS.
    """
    return EvaluationResult(
        precision_at_k=precision_at_k(y_true, y_score, PRIMARY_K_FRAC),
        lift_at_k=lift_at_k(y_true, y_score, PRIMARY_K_FRAC),
        pr_auc=pr_auc(y_true, y_score),
        brier_score=brier_score(y_true, y_score),
        confusion_matrix_at_k=confusion_matrix_at_k(y_true, y_score, PRIMARY_K_FRAC),
        sensitivity_report=sensitivity_report(y_true, y_score, SENSITIVITY_K_FRACS),
    )


def evaluate_all_candidates(
    conn: duckdb.DuckDBPyConnection,
    fitted_lr,
    fitted_rf,
    fitted_xgb,
) -> dict[str, EvaluationResult]:
    """Evaluate the business heuristic and the three fitted ML candidates.

    The heuristic reads the unencoded, feat_churn-shaped test split
    directly (predict_business_heuristic), while the ML candidates read
    the preprocessed test matrix (predict_proba) -- each candidate is
    called through its own correct interface rather than forcing one
    interface on both. The three ML candidates must already be fitted
    (e.g. via models.tune's tune_* functions' best_estimator_) --
    tuning them is the caller's responsibility, not this function's.
    All four candidates are scored against the same y_test, so results
    are directly comparable. Returns one EvaluationResult per candidate,
    keyed by candidate name.
    """
    split = split_feat_churn(conn)
    model_inputs = prepare_model_inputs(conn)
    y_true = model_inputs.y_test

    results: dict[str, EvaluationResult] = {}

    logger.info("Evaluating candidate: business_heuristic")
    heuristic_scores = predict_business_heuristic(split.test).astype(float)
    results["business_heuristic"] = evaluate_candidate(y_true, heuristic_scores)

    logger.info("Evaluating candidate: logistic_regression")
    lr_scores = predict_proba(fitted_lr, model_inputs.X_test)
    results["logistic_regression"] = evaluate_candidate(y_true, lr_scores)

    logger.info("Evaluating candidate: random_forest")
    rf_scores = predict_proba(fitted_rf, model_inputs.X_test)
    results["random_forest"] = evaluate_candidate(y_true, rf_scores)

    logger.info("Evaluating candidate: xgboost")
    xgb_scores = predict_proba(fitted_xgb, model_inputs.X_test)
    results["xgboost"] = evaluate_candidate(y_true, xgb_scores)

    logger.info("Evaluated %d candidates.", len(results))
    return results
