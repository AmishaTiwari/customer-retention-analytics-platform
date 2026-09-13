"""Data-quality invariant tests for pipeline/preprocess.py.

Like test_features.py and test_split.py, these tests run the full
pipeline (staging, cleaning, target construction, modeling view, feature
engineering) against the real committed data/raw/ files, then apply
column selection to the resulting feat_churn and assert claims about the
actual output (which columns end up in X, y, and customer_id, and that no
rows are dropped). The pipeline-fitting tests further split into
train/test and assert claims about the fitted preprocessing transformer
(no nulls after transform, sentinel encoding, train/test shape
consistency).
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
from retention_platform.pipeline.preprocess import (
    INTERNET_TYPE_SENTINEL,
    OFFER_SENTINEL,
    build_preprocessing_pipeline,
    select_model_columns,
)
from retention_platform.pipeline.split import split_feat_churn


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


@pytest.fixture(scope="module")
def train_test_columns(conn):
    split = split_feat_churn(conn)
    X_train, _y_train, _id_train = select_model_columns(split.train)
    X_test, _y_test, _id_test = select_model_columns(split.test)
    return X_train, X_test


@pytest.fixture(scope="module")
def fitted_pipeline_output(train_test_columns):
    X_train, X_test = train_test_columns
    pipeline = build_preprocessing_pipeline()
    Xt_train = pipeline.fit_transform(X_train)
    Xt_test = pipeline.transform(X_test)
    return Xt_train, Xt_test


def test_fit_transform_succeeds_on_train_and_test(fitted_pipeline_output):
    Xt_train, Xt_test = fitted_pipeline_output
    assert Xt_train.shape[0] > 0
    assert Xt_test.shape[0] > 0


def test_transformed_output_has_no_nulls(fitted_pipeline_output):
    Xt_train, Xt_test = fitted_pipeline_output
    assert not np.isnan(Xt_train).any()
    assert not np.isnan(Xt_test).any()


@pytest.mark.parametrize(
    ("column", "sentinel"),
    [
        ("internet_type", INTERNET_TYPE_SENTINEL),
        ("offer", OFFER_SENTINEL),
    ],
)
def test_structural_null_maps_to_sentinel_category(train_test_columns, column, sentinel):
    X_train, X_test = train_test_columns
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)
    Xt_test = pipeline.transform(X_test)

    feature_names = pipeline.get_feature_names_out()
    sentinel_column = f"{column}__{column}_{sentinel}"
    sentinel_index = list(feature_names).index(sentinel_column)

    null_mask = X_test[column].isna().to_numpy()
    assert (Xt_test[null_mask, sentinel_index] == 1.0).all()
    assert (Xt_test[~null_mask, sentinel_index] == 0.0).all()


def test_transform_on_test_matches_train_column_count(fitted_pipeline_output):
    Xt_train, Xt_test = fitted_pipeline_output
    assert Xt_test.shape[1] == Xt_train.shape[1]
