"""Identical-harness candidate model comparison and comparison table
construction, per the Model Comparison Strategy locked in ML System
Design v1.0.

Currently implemented: running each candidate individually through the
identical evaluation harness (evaluate_candidate), comparing candidates on
training-side CV (calibrate_candidate, evaluate_candidates_cv) and
assembling those results into a cross-model comparison (compare_candidates,
format_comparison_table), and -- once selection is decided elsewhere --
scoring the selected model against the held-out test set
(evaluate_final_model_on_test). Ranking and final selection per ADR-0008's
rule are a later step; this module does not perform them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict

from retention_platform.config import load_config
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

# The candidate every other candidate's Precision@K/Lift@K delta is measured
# against (ADR-0008's "performance relative to the business heuristic
# baseline" required-context comparison).
HEURISTIC_CANDIDATE = "business_heuristic"

# Calibration method for calibrate_candidate's CalibratedClassifierCV.
CALIBRATION_METHOD = "sigmoid"


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate's metric outputs from the identical evaluation harness."""

    precision_at_k: float
    lift_at_k: float
    pr_auc: float
    brier_score: float
    confusion_matrix_at_k: dict
    sensitivity_report: list[dict]


def _k_settings() -> tuple[float, list[float]]:
    """Return (primary K, sensitivity sweep) from config.

    Both are fractions of the scored population. Config is read on each
    call, not at import time.
    """
    business = load_config()["business"]
    return business["k"], business["k_sensitivity"]


def evaluate_candidate(y_true, y_score) -> EvaluationResult:
    """Run one candidate's predictions through every metrics.py function.

    y_score is whatever score the candidate produces for ranking --
    predicted probabilities for the ML models, or the heuristic's boolean
    flag -- metrics.py's functions only require it to be rankable, not a
    calibrated probability. Precision@K, Lift@K, and the confusion matrix
    use the primary K from config (business.k); sensitivity_report sweeps
    the K values in config (business.k_sensitivity).
    """
    primary_k, k_sweep = _k_settings()
    return EvaluationResult(
        precision_at_k=precision_at_k(y_true, y_score, primary_k),
        lift_at_k=lift_at_k(y_true, y_score, primary_k),
        pr_auc=pr_auc(y_true, y_score),
        brier_score=brier_score(y_true, y_score),
        confusion_matrix_at_k=confusion_matrix_at_k(y_true, y_score, primary_k),
        sensitivity_report=sensitivity_report(y_true, y_score, k_sweep),
    )


_VALID_FINAL_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}


def evaluate_final_model_on_test(
    conn: duckdb.DuckDBPyConnection,
    final_model,
    final_model_name: str,
) -> dict[str, EvaluationResult]:
    """Score the selected model and the fixed business heuristic against
    the held-out test set -- the single, final use of X_test/y_test in
    this module.

    Protocol: the test set is reserved for the final evaluation of the
    selected model plus the fixed business heuristic. Candidate
    comparison and selection never use it (see evaluate_candidates_cv).
    The heuristic is a fixed, unfitted baseline that is not part of
    selection, so scoring it on test cannot bias the selection decision
    -- it is included so the held-out report shows the selected model
    next to the baseline, as ADR-0008 requires. Call this once, only
    after selection has already been decided elsewhere.

    final_model must already be fitted (a plain predict_proba-capable
    estimator, e.g. a tune_* search's best_estimator_, or a calibrated
    estimator if calibration was selected) -- fitting or calibrating it
    is the caller's responsibility, not this function's. final_model_name
    must be one of "logistic_regression", "random_forest", or "xgboost";
    "business_heuristic" is rejected since the heuristic is handled
    separately here, not passed in as final_model.

    Loads split_feat_churn(conn) and prepare_model_inputs(conn) itself,
    so test-set access is contained to this one function. Returns
    {"business_heuristic": ..., final_model_name: ...}, the same
    dict[str, EvaluationResult] shape as the other evaluation functions.
    """
    if final_model_name not in _VALID_FINAL_MODEL_NAMES:
        raise ValueError(
            f"final_model_name must be one of {sorted(_VALID_FINAL_MODEL_NAMES)}, "
            f"got {final_model_name!r}"
        )

    split = split_feat_churn(conn)
    model_inputs = prepare_model_inputs(conn)
    y_true = model_inputs.y_test

    results: dict[str, EvaluationResult] = {}

    logger.info("Evaluating on test set: business_heuristic")
    heuristic_scores = predict_business_heuristic(split.test).astype(float)
    results["business_heuristic"] = evaluate_candidate(y_true, heuristic_scores)

    logger.info("Evaluating on test set: %s", final_model_name)
    final_model_scores = predict_proba(final_model, model_inputs.X_test)
    results[final_model_name] = evaluate_candidate(y_true, final_model_scores)

    logger.info("Evaluated final model on test set: %s", sorted(results))
    return results


def _build_cv(config: dict) -> RepeatedStratifiedKFold:
    """Build the training-side cross-validation splitter, mirroring
    models/tune.py's tune_* functions exactly so calibration and
    out-of-fold scoring use the same fold structure as Commit 8's tuning.
    """
    return RepeatedStratifiedKFold(
        n_splits=config["data"]["cv_folds"],
        n_repeats=config["data"]["cv_repeats"],
        random_state=config["reproducibility"]["seed"],
    )


def calibrate_candidate(fitted_model, config: dict) -> CalibratedClassifierCV:
    """Wrap one already-tuned candidate estimator in an unfitted
    CalibratedClassifierCV, per the Validation Strategy correction: model
    comparison and calibration belong on training-side CV, not the test
    set. fitted_model is a Commit 8 tuned estimator (e.g. a tune_*
    search's best_estimator_) -- tuning it is the caller's responsibility,
    not this function's, and no re-tuning happens here. CALIBRATION_METHOD
    ("sigmoid") is formalized in ADR-0013; see that ADR for the reasoning
    behind the choice.

    Returns the CalibratedClassifierCV unfitted: fitting happens inside
    cross_val_predict in evaluate_candidates_cv, one fold at a time, so
    each fold's calibration is learned only from that fold's own training
    portion.
    """
    return CalibratedClassifierCV(
        fitted_model, method=CALIBRATION_METHOD, cv=_build_cv(config)
    )


def evaluate_candidates_cv(
    conn: duckdb.DuckDBPyConnection,
    calibrated_lr,
    calibrated_rf,
    calibrated_xgb,
    X_train,
    y_train,
    config: dict,
) -> dict[str, EvaluationResult]:
    """Scores every candidate on out-of-fold training predictions instead of
    X_test/y_test, per the Validation Strategy correction that model
    comparison belongs on training-side CV, with the test set reserved for
    a single final evaluation (evaluate_final_model_on_test). Returns
    dict[str, EvaluationResult] keyed by business_heuristic,
    logistic_regression, random_forest, and xgboost -- a drop-in input to
    compare_candidates.

    Each ML candidate is an unfitted CalibratedClassifierCV (from
    calibrate_candidate); cross_val_predict fits and calibrates it fold by
    fold and returns each row's out-of-fold predicted probability, so no
    row is ever scored by a model that was trained on it. The heuristic
    has no fitted state to cross-validate, so it is scored directly against
    y_train through the same evaluate_candidate() call, keeping all four
    candidates in one comparable result set.
    """
    split = split_feat_churn(conn)
    cv = _build_cv(config)

    results: dict[str, EvaluationResult] = {}

    logger.info("Evaluating candidate (train CV): business_heuristic")
    heuristic_scores = predict_business_heuristic(split.train).astype(float)
    results["business_heuristic"] = evaluate_candidate(y_train, heuristic_scores)

    logger.info("Evaluating candidate (train CV): logistic_regression")
    lr_oof_scores = cross_val_predict(
        calibrated_lr, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    results["logistic_regression"] = evaluate_candidate(y_train, lr_oof_scores)

    logger.info("Evaluating candidate (train CV): random_forest")
    rf_oof_scores = cross_val_predict(
        calibrated_rf, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    results["random_forest"] = evaluate_candidate(y_train, rf_oof_scores)

    logger.info("Evaluating candidate (train CV): xgboost")
    xgb_oof_scores = cross_val_predict(
        calibrated_xgb, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    results["xgboost"] = evaluate_candidate(y_train, xgb_oof_scores)

    logger.info("Evaluated %d candidates (train CV).", len(results))
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

    For each K in the config sweep (business.k_sensitivity), pulls each
    candidate's Precision@K and Lift@K from its own sensitivity_report
    (never recomputed here), and
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
        len(_k_settings()[1]),
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
