"""Tests for evaluation/compare.py's per-candidate evaluation harness.

Like test_candidates.py, these tests run the full pipeline (staging,
cleaning, target construction, modeling view, feature engineering)
against the real committed data/raw/ files, then tune all three ML
candidates via models.tune, to assert claims about the harness's real
behavior: training-side CV comparison (calibrate_candidate,
evaluate_candidates_cv, compare_candidates, format_comparison_table), the
tie-breaker evidence functions, and the final held-out evaluation
(evaluate_final_model_on_test). This is slow: it runs the full Commit 8
search for all three ML model families once, shared across every test in
this file via a module-scoped fixture.
"""

from __future__ import annotations

import ast
import inspect

import duckdb
import pytest
from sklearn.calibration import CalibratedClassifierCV

from retention_platform.config import load_config
from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.evaluation.compare import (
    CALIBRATION_METHOD,
    SENSITIVITY_K_FRACS,
    ComparisonResult,
    EvaluationResult,
    TieBreakerEvidence,
    _build_cv,
    calibrate_candidate,
    collect_tiebreaker_evidence,
    compare_candidates,
    evaluate_candidate,
    evaluate_candidates_cv,
    evaluate_final_model_on_test,
    extract_tiebreaker_evidence,
    format_comparison_table,
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
def fitted_searches(conn, config):
    # Module-scoped so the Commit 8 search runs once and is shared by every
    # test needing the fitted search objects (fitted_candidates and the
    # tie-breaker evidence tests), rather than repeating the slow tuning.
    model_inputs = prepare_model_inputs(conn)
    lr_search = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)
    rf_search = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)
    xgb_search = tune_xgboost(model_inputs.X_train, model_inputs.y_train, config)
    return (lr_search, rf_search, xgb_search)


@pytest.fixture(scope="module")
def fitted_candidates(fitted_searches):
    lr_search, rf_search, xgb_search = fitted_searches
    return (
        lr_search.best_estimator_,
        rf_search.best_estimator_,
        xgb_search.best_estimator_,
    )


@pytest.fixture(scope="module")
def final_model_result(conn, fitted_candidates):
    # fitted_candidates[0] is the plain tuned LR -- these tests only check
    # evaluate_final_model_on_test's mechanics, so any valid final_model
    # name/estimator pair will do.
    final_model = fitted_candidates[0]
    return evaluate_final_model_on_test(conn, final_model, "logistic_regression")


def test_final_model_result_contains_expected_keys(final_model_result):
    assert set(final_model_result.keys()) == {"business_heuristic", "logistic_regression"}


@pytest.mark.parametrize("candidate_name", ["business_heuristic", "logistic_regression"])
def test_final_model_result_contains_all_six_metrics(final_model_result, candidate_name):
    result = final_model_result[candidate_name]

    assert isinstance(result, EvaluationResult)
    assert isinstance(result.precision_at_k, float)
    assert isinstance(result.lift_at_k, float)
    assert isinstance(result.pr_auc, float)
    assert isinstance(result.brier_score, float)
    assert isinstance(result.confusion_matrix_at_k, dict)
    assert isinstance(result.sensitivity_report, list)
    assert len(result.sensitivity_report) == 5


def test_final_model_heuristic_result_matches_direct_computation(conn, final_model_result):
    # Exercises the heuristic path independently: predict_business_heuristic
    # on the raw, unencoded test split, then the same evaluate_candidate
    # harness -- should match what evaluate_final_model_on_test produced.
    split = split_feat_churn(conn)
    model_inputs = prepare_model_inputs(conn)
    heuristic_scores = predict_business_heuristic(split.test).astype(float)
    direct = evaluate_candidate(model_inputs.y_test, heuristic_scores)

    assert final_model_result["business_heuristic"].precision_at_k == pytest.approx(
        direct.precision_at_k
    )
    assert final_model_result["business_heuristic"].pr_auc == pytest.approx(
        direct.pr_auc
    )


def test_final_model_pr_auc_differs_from_heuristic(final_model_result):
    # If the final model were mistakenly scored using the heuristic's raw
    # boolean flag instead of its own predicted probabilities, its pr_auc
    # would exactly equal the heuristic's. Confirming they differ checks
    # that the final model went through its own predict_proba path.
    assert final_model_result["logistic_regression"].pr_auc != pytest.approx(
        final_model_result["business_heuristic"].pr_auc
    )


def test_evaluate_final_model_on_test_rejects_unknown_name(conn, fitted_candidates):
    final_model = fitted_candidates[0]
    with pytest.raises(ValueError):
        evaluate_final_model_on_test(conn, final_model, "unknown_model")


def test_evaluate_final_model_on_test_rejects_business_heuristic(conn, fitted_candidates):
    final_model = fitted_candidates[0]
    with pytest.raises(ValueError):
        evaluate_final_model_on_test(conn, final_model, "business_heuristic")


@pytest.fixture(scope="module")
def comparison(cv_results):
    return compare_candidates(cv_results)


@pytest.mark.parametrize(
    "candidate_name", sorted(EXPECTED_CANDIDATE_NAMES - {"business_heuristic"})
)
def test_ml_candidate_deltas_match_independent_computation(
    comparison, cv_results, candidate_name
):
    heuristic_by_k = {
        entry["k_frac"]: entry
        for entry in cv_results["business_heuristic"].sensitivity_report
    }
    candidate_by_k = {
        entry["k_frac"]: entry
        for entry in cv_results[candidate_name].sensitivity_report
    }

    for k_metrics in comparison.candidates[candidate_name].per_k:
        heuristic_entry = heuristic_by_k[k_metrics.k_frac]
        candidate_entry = candidate_by_k[k_metrics.k_frac]

        expected_precision_delta = (
            candidate_entry["precision"] - heuristic_entry["precision"]
        )
        expected_lift_delta = candidate_entry["lift"] - heuristic_entry["lift"]

        assert k_metrics.precision_delta_vs_heuristic == pytest.approx(
            expected_precision_delta
        )
        assert k_metrics.lift_delta_vs_heuristic == pytest.approx(expected_lift_delta)


def test_heuristic_deltas_are_none(comparison):
    for k_metrics in comparison.candidates["business_heuristic"].per_k:
        assert k_metrics.precision_delta_vs_heuristic is None
        assert k_metrics.lift_delta_vs_heuristic is None


def test_format_comparison_table_contains_all_candidate_names(comparison):
    table = format_comparison_table(comparison)

    assert isinstance(table, str)
    assert table
    for name in EXPECTED_CANDIDATE_NAMES:
        assert name in table


@pytest.fixture(scope="module")
def tiebreaker_evidence(fitted_searches):
    lr_search, rf_search, xgb_search = fitted_searches
    return collect_tiebreaker_evidence(lr_search, rf_search, xgb_search)


def test_collect_tiebreaker_evidence_excludes_heuristic(tiebreaker_evidence):
    assert set(tiebreaker_evidence.keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


@pytest.mark.parametrize(
    "candidate_name,search_index",
    [("logistic_regression", 0), ("random_forest", 1), ("xgboost", 2)],
)
def test_best_params_matches_search_object_directly(
    tiebreaker_evidence, fitted_searches, candidate_name, search_index
):
    search = fitted_searches[search_index]
    evidence = tiebreaker_evidence[candidate_name]

    assert isinstance(evidence, TieBreakerEvidence)
    assert evidence.best_params == search.best_params_


@pytest.mark.parametrize(
    "candidate_name,search_index",
    [("logistic_regression", 0), ("random_forest", 1), ("xgboost", 2)],
)
def test_tiebreaker_numeric_fields_match_search_object_directly(
    tiebreaker_evidence, fitted_searches, candidate_name, search_index
):
    search = fitted_searches[search_index]
    evidence = tiebreaker_evidence[candidate_name]
    cv_results = search.cv_results_
    best_index = search.best_index_

    assert evidence.prediction_stability_std == pytest.approx(
        cv_results["std_test_score"][best_index]
    )
    assert evidence.mean_fit_time_best_config == pytest.approx(
        cv_results["mean_fit_time"][best_index]
    )
    assert evidence.mean_fit_time_across_search == pytest.approx(
        cv_results["mean_fit_time"].mean()
    )


@pytest.fixture(scope="module")
def calibrated_candidates(fitted_candidates, config):
    return tuple(calibrate_candidate(model, config) for model in fitted_candidates)


@pytest.mark.parametrize(
    "candidate_index", [0, 1, 2], ids=["logistic_regression", "random_forest", "xgboost"]
)
def test_calibrate_candidate_returns_unfitted_wrapper(
    fitted_candidates, calibrated_candidates, candidate_index
):
    fitted_model = fitted_candidates[candidate_index]
    calibrated = calibrated_candidates[candidate_index]

    assert isinstance(calibrated, CalibratedClassifierCV)
    assert not hasattr(calibrated, "calibrated_classifiers_")
    assert calibrated.estimator is fitted_model
    assert calibrated.method == CALIBRATION_METHOD


@pytest.fixture(scope="module")
def cv_results(conn, calibrated_candidates, config):
    # Reruns 3x 5-fold calibrated CV, so shared module-wide rather than
    # re-run per test.
    model_inputs = prepare_model_inputs(conn)
    calibrated_lr, calibrated_rf, calibrated_xgb = calibrated_candidates
    return evaluate_candidates_cv(
        conn,
        calibrated_lr,
        calibrated_rf,
        calibrated_xgb,
        model_inputs.X_train,
        model_inputs.y_train,
        config,
    )


def test_cv_results_contains_all_four_candidates(cv_results):
    assert set(cv_results.keys()) == EXPECTED_CANDIDATE_NAMES


@pytest.mark.parametrize("candidate_name", sorted(EXPECTED_CANDIDATE_NAMES))
def test_cv_result_contains_all_six_metrics(cv_results, candidate_name):
    result = cv_results[candidate_name]

    assert isinstance(result, EvaluationResult)
    assert isinstance(result.precision_at_k, float)
    assert isinstance(result.lift_at_k, float)
    assert isinstance(result.pr_auc, float)
    assert isinstance(result.brier_score, float)
    assert isinstance(result.confusion_matrix_at_k, dict)
    assert isinstance(result.sensitivity_report, list)
    assert len(result.sensitivity_report) == 5


@pytest.mark.parametrize(
    "candidate_name,candidate_index",
    [("logistic_regression", 0), ("random_forest", 1), ("xgboost", 2)],
)
def test_cv_results_differ_from_test_set_results(
    conn, fitted_candidates, cv_results, candidate_name, candidate_index
):
    # evaluate_candidates_cv scores training-CV out-of-fold predictions,
    # evaluate_final_model_on_test scores test-set predictions -- these
    # are different data, so their pr_auc values should not coincide.
    # This does not assert which is better, only that the two functions
    # are genuinely scoring different things rather than duplicating
    # each other's work.
    final_model = fitted_candidates[candidate_index]
    test_set_result = evaluate_final_model_on_test(conn, final_model, candidate_name)

    assert cv_results[candidate_name].pr_auc != pytest.approx(
        test_set_result[candidate_name].pr_auc
    )


def test_compare_candidates_accepts_cv_results(cv_results):
    comparison = compare_candidates(cv_results)

    assert isinstance(comparison, ComparisonResult)
    assert set(comparison.candidates.keys()) == EXPECTED_CANDIDATE_NAMES

    for candidate in comparison.candidates.values():
        k_fracs = [k_metrics.k_frac for k_metrics in candidate.per_k]
        assert k_fracs == SENSITIVITY_K_FRACS


def _referenced_identifiers(func) -> set[str]:
    # Parsed via ast rather than a raw substring check, so a function's
    # own docstring (which may name X_test/y_test in prose, describing
    # what it deliberately does NOT use) doesn't trip a false positive --
    # only actual identifier references (variable names, attribute
    # access) count.
    source = inspect.getsource(func)
    func_node = ast.parse(source).body[0]

    names = {node.id for node in ast.walk(func_node) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(func_node) if isinstance(node, ast.Attribute)}
    return names | attrs


_FORBIDDEN_TEST_SET_IDENTIFIERS = {"X_test", "y_test", "test"}

_FUNCTIONS_THAT_MUST_NOT_TOUCH_TEST_SET = {
    "evaluate_candidates_cv": evaluate_candidates_cv,
    "calibrate_candidate": calibrate_candidate,
    "_build_cv": _build_cv,
    "evaluate_candidate": evaluate_candidate,
    "compare_candidates": compare_candidates,
    "format_comparison_table": format_comparison_table,
    "extract_tiebreaker_evidence": extract_tiebreaker_evidence,
    "collect_tiebreaker_evidence": collect_tiebreaker_evidence,
}


@pytest.mark.parametrize(
    "func", _FUNCTIONS_THAT_MUST_NOT_TOUCH_TEST_SET.values(),
    ids=_FUNCTIONS_THAT_MUST_NOT_TOUCH_TEST_SET.keys(),
)
def test_function_never_references_test_set(func):
    # Cheap, explicit tripwire: a future edit that silently reintroduces
    # X_test/y_test/.test usage into any of these functions should fail
    # here rather than only be caught by someone reading the diff. Only
    # evaluate_final_model_on_test may touch the test set in this module.
    referenced = _referenced_identifiers(func)
    assert not (referenced & _FORBIDDEN_TEST_SET_IDENTIFIERS)


def test_evaluate_final_model_on_test_does_reference_test_set():
    # Positive control: confirms the tripwire above isn't passing
    # vacuously (e.g. because _referenced_identifiers silently returns
    # nothing) by checking it correctly flags the one function that is
    # supposed to reference the test set.
    referenced = _referenced_identifiers(evaluate_final_model_on_test)
    assert referenced & _FORBIDDEN_TEST_SET_IDENTIFIERS
