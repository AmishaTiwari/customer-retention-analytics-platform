"""Data-quality invariant tests for pipeline/preprocess.py.

Like test_features.py and test_split.py, these tests run the full
pipeline (staging, cleaning, target construction, modeling view, feature
engineering) against the real committed data/raw/ files, then apply
column selection to the resulting feat_churn and assert claims about the
actual output (which columns end up in X, y, and customer_id, and that no
rows are dropped).
"""

from __future__ import annotations

import duckdb
import pytest

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.pipeline.preprocess import select_model_columns


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
def feat_churn_df(conn):
    return conn.execute("SELECT * FROM feat_churn").fetchdf()


@pytest.fixture(scope="module")
def selection_result(feat_churn_df):
    return select_model_columns(feat_churn_df)


@pytest.mark.parametrize(
    "column", ["customer_id", "country", "state", "city", "zip_code", "is_voluntary_churn"]
)
def test_excluded_columns_absent_from_X(selection_result, column):
    X, _y, _customer_id = selection_result
    assert column not in X.columns


@pytest.mark.parametrize("column", ["latitude", "longitude"])
def test_geographic_numeric_columns_present_in_X(selection_result, column):
    X, _y, _customer_id = selection_result
    assert column in X.columns


def test_y_and_customer_id_returned_correctly(feat_churn_df, selection_result):
    _X, y, customer_id = selection_result
    assert y.equals(feat_churn_df["is_voluntary_churn"])
    assert customer_id.equals(feat_churn_df["customer_id"])
    assert len(y) == len(feat_churn_df)
    assert len(customer_id) == len(feat_churn_df)


def test_row_count_unchanged(feat_churn_df, selection_result):
    X, _y, _customer_id = selection_result
    assert len(X) == len(feat_churn_df)
