"""Data-quality invariant tests for models/candidates.py's Logistic Regression.

Like test_heuristic.py, these tests run the full pipeline (staging,
cleaning, target construction, modeling view, feature engineering)
against the real committed data/raw/ files, then call
prepare_model_inputs to get real preprocessed X_train/X_test/y_train/y_test
and assert claims about the fitted model's actual behavior on that data.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.models.candidates import (
    fit_logistic_regression,
    predict_proba_logistic_regression,
)
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
def fitted_model(model_inputs):
    return fit_logistic_regression(model_inputs.X_train, model_inputs.y_train)


def test_class_weight_left_at_default(fitted_model):
    assert fitted_model.class_weight is None


def test_predicted_probabilities_within_unit_interval(fitted_model, model_inputs):
    train_proba = predict_proba_logistic_regression(fitted_model, model_inputs.X_train)
    test_proba = predict_proba_logistic_regression(fitted_model, model_inputs.X_test)

    assert ((train_proba >= 0.0) & (train_proba <= 1.0)).all()
    assert ((test_proba >= 0.0) & (test_proba <= 1.0)).all()


def test_predicted_probabilities_length_matches_input(fitted_model, model_inputs):
    train_proba = predict_proba_logistic_regression(fitted_model, model_inputs.X_train)
    test_proba = predict_proba_logistic_regression(fitted_model, model_inputs.X_test)

    assert len(train_proba) == model_inputs.X_train.shape[0]
    assert len(test_proba) == model_inputs.X_test.shape[0]


def test_refitting_with_same_random_state_is_reproducible(model_inputs):
    model_a = fit_logistic_regression(model_inputs.X_train, model_inputs.y_train)
    model_b = fit_logistic_regression(model_inputs.X_train, model_inputs.y_train)

    proba_a = predict_proba_logistic_regression(model_a, model_inputs.X_test)
    proba_b = predict_proba_logistic_regression(model_b, model_inputs.X_test)

    np.testing.assert_array_equal(proba_a, proba_b)
