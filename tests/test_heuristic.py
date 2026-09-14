"""Data-quality invariant tests for models/candidates.py's business heuristic.

Like test_features.py and test_split.py, these tests run the full
pipeline (staging, cleaning, target construction, modeling view, feature
engineering) against the real committed data/raw/ files and assert claims
about the heuristic's actual output on that data (exact correspondence
with is_month_to_month, shape, and known flagged-count/precision figures).
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
from retention_platform.models.candidates import predict_business_heuristic


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
def predictions(feat_churn_df):
    return predict_business_heuristic(feat_churn_df)


def test_predictions_exactly_match_is_month_to_month(feat_churn_df, predictions):
    assert (predictions == feat_churn_df["is_month_to_month"]).all()


def test_predictions_length_and_index_match_input(feat_churn_df, predictions):
    assert len(predictions) == len(feat_churn_df)
    assert predictions.index.equals(feat_churn_df.index)


def test_flagged_count_matches_known_figure(predictions):
    assert predictions.sum() == 3604


def test_precision_matches_known_figure(feat_churn_df, predictions):
    flagged = feat_churn_df.loc[predictions]
    precision = flagged["is_voluntary_churn"].mean()
    assert precision == pytest.approx(0.4575, abs=0.001)
