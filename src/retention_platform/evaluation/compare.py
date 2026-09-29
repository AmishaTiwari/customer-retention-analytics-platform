"""Identical-harness candidate model comparison and comparison table
construction, per the Model Comparison Strategy locked in ML System
Design v1.0.

Currently implemented: running each candidate individually through the
identical evaluation harness (evaluate_candidate, evaluate_all_candidates),
and assembling those per-candidate results into a cross-model comparison
(compare_candidates, format_comparison_table). Ranking and final selection
per ADR-0008's rule are a later step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import numpy as np

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

# The candidate every other candidate's Precision@K/Lift@K delta is measured
# against (ADR-0008's "performance relative to the business heuristic
# baseline" required-context comparison).
HEURISTIC_CANDIDATE = "business_heuristic"


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


@dataclass(frozen=True)
class CandidateKMetrics:
    """One candidate's Precision@K/Lift@K at a single K, plus its delta
    against the business heuristic at that same K.

    The deltas are None for the heuristic's own row -- it is not compared
    against itself.
    """

    k_frac: float
    precision_at_k: float
    lift_at_k: float
    precision_delta_vs_heuristic: float | None
    lift_delta_vs_heuristic: float | None


@dataclass(frozen=True)
class CandidateComparison:
    """One candidate's K-independent metrics plus its per-K breakdown."""

    name: str
    pr_auc: float
    brier_score: float
    per_k: list[CandidateKMetrics]


@dataclass(frozen=True)
class ComparisonResult:
    """Cross-model assembly of every candidate's EvaluationResult, keyed
    by candidate name. Assembly only -- no ranking or selection (ADR-0008's
    selection rule is a later step)."""

    candidates: dict[str, CandidateComparison]


def compare_candidates(results: dict[str, EvaluationResult]) -> ComparisonResult:
    """Assemble per-candidate EvaluationResults into a cross-model comparison.

    For each K in SENSITIVITY_K_FRACS, pulls each candidate's Precision@K
    and Lift@K from its own sensitivity_report (never recomputed here), and
    for every non-heuristic candidate pairs that with its delta against the
    business heuristic at the same K -- the "performance relative to the
    business heuristic baseline" required context named in ADR-0008. Each
    candidate's K-independent pr_auc and brier_score are included once.

    Does not rank or select a winner among candidates; that is ADR-0008's
    selection rule, applied in a later step.
    """
    logger.info(
        "Comparing %d candidates across %d K values.",
        len(results),
        len(SENSITIVITY_K_FRACS),
    )

    heuristic_by_k = {
        entry["k_frac"]: entry
        for entry in results[HEURISTIC_CANDIDATE].sensitivity_report
    }

    candidates: dict[str, CandidateComparison] = {}
    for name, result in results.items():
        per_k = []
        for entry in result.sensitivity_report:
            k_frac = entry["k_frac"]
            if name == HEURISTIC_CANDIDATE:
                precision_delta = None
                lift_delta = None
            else:
                heuristic_entry = heuristic_by_k[k_frac]
                precision_delta = entry["precision"] - heuristic_entry["precision"]
                lift_delta = entry["lift"] - heuristic_entry["lift"]

            per_k.append(
                CandidateKMetrics(
                    k_frac=k_frac,
                    precision_at_k=entry["precision"],
                    lift_at_k=entry["lift"],
                    precision_delta_vs_heuristic=precision_delta,
                    lift_delta_vs_heuristic=lift_delta,
                )
            )

        candidates[name] = CandidateComparison(
            name=name,
            pr_auc=result.pr_auc,
            brier_score=result.brier_score,
            per_k=per_k,
        )

    logger.info("Assembled comparison for candidates: %s", sorted(candidates))
    return ComparisonResult(candidates=candidates)


def format_comparison_table(comparison: ComparisonResult) -> str:
    """Render a ComparisonResult as a readable markdown table, one row per
    candidate per K, suitable for pasting into notes or a future ADR.

    Pure formatting -- performs no computation of its own.
    """
    header = (
        "| Candidate | K | Precision@K | Lift@K | "
        "ΔPrecision vs heuristic | ΔLift vs heuristic | PR-AUC | Brier |"
    )
    separator = "|---|---|---|---|---|---|---|---|"
    rows = [header, separator]

    for name in sorted(comparison.candidates):
        candidate = comparison.candidates[name]
        for k_metrics in candidate.per_k:
            precision_delta = (
                "n/a"
                if k_metrics.precision_delta_vs_heuristic is None
                else f"{k_metrics.precision_delta_vs_heuristic:+.4f}"
            )
            lift_delta = (
                "n/a"
                if k_metrics.lift_delta_vs_heuristic is None
                else f"{k_metrics.lift_delta_vs_heuristic:+.4f}"
            )
            rows.append(
                f"| {candidate.name} | {k_metrics.k_frac:.0%} | "
                f"{k_metrics.precision_at_k:.4f} | {k_metrics.lift_at_k:.4f} | "
                f"{precision_delta} | {lift_delta} | "
                f"{candidate.pr_auc:.4f} | {candidate.brier_score:.4f} |"
            )

    return "\n".join(rows)


@dataclass(frozen=True)
class TieBreakerEvidence:
    """Raw evidence for ADR-0008's Step 3 tie-breakers (prediction
    stability, training efficiency, model complexity), pulled from one
    fitted RandomizedSearchCV object. This is evidence gathering only --
    no weighting, scoring, or ranking of candidates happens here or
    anywhere in this module; that is ADR-0008's selection rule, applied
    in a separate, later step.
    """

    # Cross-validation score spread at the winning configuration --
    # evidence for ADR-0008's "prediction stability" tie-breaker.
    prediction_stability_std: float

    # Fit time of the winning configuration -- evidence for ADR-0008's
    # "training efficiency" tie-breaker.
    mean_fit_time_best_config: float

    # Average fit time across every sampled configuration in the search,
    # giving training-efficiency context beyond just the winning config.
    mean_fit_time_across_search: float

    # The winning hyperparameters as-is, with no derived complexity score:
    # ADR-0008 names "model complexity" as a tie-breaker but does not
    # define how to measure it, so this reports the raw search.best_params_
    # rather than inventing a complexity metric.
    best_params: dict


def extract_tiebreaker_evidence(search) -> TieBreakerEvidence:
    """Pull ADR-0008 Step 3 tie-breaker evidence from one fitted search.

    search must be a fitted RandomizedSearchCV (as returned by
    models.tune's tune_* functions), so cv_results_/best_index_/
    best_params_ are already populated.
    """
    cv_results = search.cv_results_
    best_index = search.best_index_

    return TieBreakerEvidence(
        prediction_stability_std=float(cv_results["std_test_score"][best_index]),
        mean_fit_time_best_config=float(cv_results["mean_fit_time"][best_index]),
        mean_fit_time_across_search=float(np.mean(cv_results["mean_fit_time"])),
        best_params=dict(search.best_params_),
    )


def collect_tiebreaker_evidence(lr_search, rf_search, xgb_search) -> dict[str, TieBreakerEvidence]:
    """Collect ADR-0008 Step 3 tie-breaker evidence for the three tuned ML
    candidates, keyed by candidate name.

    No business_heuristic entry: the heuristic is never tuned, so it has
    no search object and this tie-breaker evidence does not apply to it.
    """
    logger.info("Collecting tie-breaker evidence for 3 ML candidates.")

    evidence = {
        "logistic_regression": extract_tiebreaker_evidence(lr_search),
        "random_forest": extract_tiebreaker_evidence(rf_search),
        "xgboost": extract_tiebreaker_evidence(xgb_search),
    }

    logger.info("Collected tie-breaker evidence for candidates: %s", sorted(evidence))
    return evidence
