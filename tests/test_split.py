"""Data-quality invariant tests for pipeline/split.py.

Like test_features.py, these tests run the full pipeline (staging,
cleaning, target construction, modeling view, feature engineering) against
the real committed data/raw/ files, then split the resulting feat_churn
and assert claims about the actual split (disjointness, row-count
conservation, proportions, stratification).
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
from retention_platform.features.build import run_features
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
def split_result(conn):
    return split_feat_churn(conn)


@pytest.fixture(scope="module")
def feat_churn_count(conn):
    return conn.execute("SELECT COUNT(*) FROM feat_churn").fetchone()[0]


def test_train_and_test_are_disjoint(split_result):
    train_ids = set(split_result.train["customer_id"])
    test_ids = set(split_result.test["customer_id"])
    assert train_ids.isdisjoint(test_ids)


def test_train_and_test_row_counts_sum_to_full_dataset(split_result, feat_churn_count):
    assert len(split_result.train) + len(split_result.test) == feat_churn_count


def test_split_proportions_match_configured_test_size(split_result, feat_churn_count):
    configured_test_size = load_config()["data"]["test_size"]
    actual_test_size = len(split_result.test) / feat_churn_count
    assert actual_test_size == pytest.approx(configured_test_size, abs=0.01)


def test_stratification_preserves_churn_rate(conn, split_result):
    overall_rate = conn.execute("SELECT AVG(is_voluntary_churn::INTEGER) FROM feat_churn").fetchone()[0]
    train_rate = split_result.train["is_voluntary_churn"].mean()
    test_rate = split_result.test["is_voluntary_churn"].mean()

    assert train_rate == pytest.approx(overall_rate, abs=0.02)
    assert test_rate == pytest.approx(overall_rate, abs=0.02)
