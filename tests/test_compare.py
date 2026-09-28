"""Tests for evaluation/compare.py's per-candidate evaluation harness.

Like test_candidates.py, these tests run the full pipeline (staging,
cleaning, target construction, modeling view, feature engineering)
against the real committed data/raw/ files, then call
evaluate_all_candidates -- which tunes and evaluates all three ML
candidates via models.tune, plus the business heuristic -- to assert
claims about the harness's real behavior. This is slow: it runs the full
Commit 8 search for all three ML model families once, shared across every
test in this file via a module-scoped fixture.
"""

from __future__ import annotations

import duckdb
import pytest

from retention_platform.config import load_config
from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.evaluation.compare import (
    EvaluationResult,
    evaluate_all_candidates,
    evaluate_candidate,
)
from retention_platform.features.build import run_features
from retention_platform.models.candidates import predict_business_heuristic
from retention_platform.models.tune import (
    tune_logistic_regression,
    tune_random_forest,
    tune_xgboost,
)
from retention_platform.pipeline.preprocess import prepare_model_inputs
from retention_platform.pipeline.split import split_feat_churn

EXPECTED_CANDIDATE_NAMES = {
    "business_heuristic",
    "logistic_regression",
    "random_forest",
    "xgboost",
}


@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect(":memory:")
    run_staging(connection)
    run_cleaning(connection)
    run_target(connection)
    run_modeling_view(connection)
    run_features(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def fitted_candidates(conn, config):
    model_inputs = prepare_model_inputs(conn)
    lr_search = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)
    rf_search = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)
    xgb_search = tune_xgboost(model_inputs.X_train, model_inputs.y_train, config)
    return (
        lr_search.best_estimator_,
        rf_search.best_estimator_,
        xgb_search.best_estimator_,
    )


@pytest.fixture(scope="module")
def all_results(conn, fitted_candidates):
    fitted_lr, fitted_rf, fitted_xgb = fitted_candidates
    return evaluate_all_candidates(conn, fitted_lr, fitted_rf, fitted_xgb)


def test_all_four_candidates_present(all_results):
    assert set(all_results.keys()) == EXPECTED_CANDIDATE_NAMES


@pytest.mark.parametrize("candidate_name", sorted(EXPECTED_CANDIDATE_NAMES))
def test_each_result_contains_all_six_metrics(all_results, candidate_name):
    result = all_results[candidate_name]

    assert isinstance(result, EvaluationResult)
    assert isinstance(result.precision_at_k, float)
    assert isinstance(result.lift_at_k, float)
    assert isinstance(result.pr_auc, float)
    assert isinstance(result.brier_score, float)
    assert isinstance(result.confusion_matrix_at_k, dict)
    assert isinstance(result.sensitivity_report, list)
    assert len(result.sensitivity_report) == 5


def test_results_keyed_to_correct_candidate_name(all_results):
    for name, result in all_results.items():
        assert name in EXPECTED_CANDIDATE_NAMES
        assert isinstance(result, EvaluationResult)


def test_heuristic_result_matches_direct_computation(conn, all_results):
    # Exercises the heuristic path independently: predict_business_heuristic
    # on the raw, unencoded test split, then the same evaluate_candidate
    # harness -- should match what evaluate_all_candidates produced.
    split = split_feat_churn(conn)
    model_inputs = prepare_model_inputs(conn)
    heuristic_scores = predict_business_heuristic(split.test).astype(float)
    direct = evaluate_candidate(model_inputs.y_test, heuristic_scores)

    assert all_results["business_heuristic"].precision_at_k == pytest.approx(
        direct.precision_at_k
    )
    assert all_results["business_heuristic"].pr_auc == pytest.approx(direct.pr_auc)


def test_ml_candidates_scored_independently_of_heuristic(all_results):
    # If an ML candidate were mistakenly scored using the heuristic's raw
    # boolean flag instead of its own predicted probabilities, its pr_auc
    # would exactly equal the heuristic's. Confirming they differ checks
    # that each ML candidate went through its own predict_proba path.
    heuristic_pr_auc = all_results["business_heuristic"].pr_auc
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        assert all_results[name].pr_auc != pytest.approx(heuristic_pr_auc)
