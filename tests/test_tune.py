"""Data-quality invariant tests for models/tune.py's Logistic Regression and
Random Forest searches.

Like test_candidates.py, these tests run the full pipeline (staging,
cleaning, target construction, modeling view, feature engineering)
against the real committed data/raw/ files, then call prepare_model_inputs
to get real preprocessed X_train/y_train, and assert claims about each
tune_* function's actual behavior on that data and the real loaded
config.
"""

from __future__ import annotations

import copy

import duckdb
import pytest

from retention_platform.config import load_config
from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.models.tune import tune_logistic_regression, tune_random_forest
from retention_platform.pipeline.preprocess import prepare_model_inputs


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
def model_inputs(conn):
    return prepare_model_inputs(conn)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def seed():
    return load_config()["reproducibility"]["seed"]


def test_unsupported_objective_metric_raises(model_inputs, config):
    bad_config = copy.deepcopy(config)
    bad_config["tuning"]["objective_metric"] = "not_a_real_metric"

    with pytest.raises(ValueError, match="not_a_real_metric"):
        tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, bad_config)


def test_best_score_in_valid_range(model_inputs, config):
    search = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)

    assert isinstance(search.best_score_, float)
    assert 0.0 < search.best_score_ <= 1.0


def test_best_params_keys_match_search_space(model_inputs, config):
    search = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)

    # param_distributions is local to tune_logistic_regression, not exposed
    # as a module attribute, so there is no clean way to read it back
    # without introspecting function internals -- the expected key set is
    # hardcoded here instead.
    expected_keys = {"C", "l1_ratio", "solver", "class_weight"}
    assert set(search.best_params_.keys()) == expected_keys


def test_reproducible_with_same_seed(model_inputs, config):
    # Runs the full search twice, so this test is slower than the other
    # three. The real search space and budget are used both times rather
    # than a shrunk stand-in, since a smaller search wouldn't actually
    # confirm what production runs.
    search_a = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)
    search_b = tune_logistic_regression(model_inputs.X_train, model_inputs.y_train, config)

    assert search_a.best_params_ == search_b.best_params_
    assert search_a.best_score_ == search_b.best_score_


def test_random_forest_unsupported_objective_metric_raises(model_inputs, config):
    bad_config = copy.deepcopy(config)
    bad_config["tuning"]["objective_metric"] = "not_a_real_metric"

    with pytest.raises(ValueError, match="not_a_real_metric"):
        tune_random_forest(model_inputs.X_train, model_inputs.y_train, bad_config)


def test_random_forest_best_score_in_valid_range(model_inputs, config):
    search = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)

    assert isinstance(search.best_score_, float)
    assert 0.0 < search.best_score_ <= 1.0


def test_random_forest_best_params_keys_match_search_space(model_inputs, config):
    search = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)

    # param_distributions is local to tune_random_forest, not exposed
    # as a module attribute, so there is no clean way to read it back
    # without introspecting function internals -- the expected key set is
    # hardcoded here instead.
    expected_keys = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    assert set(search.best_params_.keys()) == expected_keys


def test_random_forest_reproducible_with_same_seed(model_inputs, config):
    # Runs the full search twice, so this test is slower than the other
    # three. The real search space and budget are used both times rather
    # than a shrunk stand-in, since a smaller search wouldn't actually
    # confirm what production runs.
    search_a = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)
    search_b = tune_random_forest(model_inputs.X_train, model_inputs.y_train, config)

    assert search_a.best_params_ == search_b.best_params_
    assert search_a.best_score_ == search_b.best_score_
