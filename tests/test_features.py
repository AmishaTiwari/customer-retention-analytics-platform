"""Data-quality invariant tests for sql/05_features/.

Like test_cleaning.py, test_target.py, and test_leakage.py, these tests
run the full pipeline (staging, cleaning, target construction, modeling
view, feature engineering) against the real committed data/raw/ files and
assert claims about the actual dataset's derived features, not about
arbitrary logic that toy cases could exercise more cheaply.
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


def test_feat_churn_row_count_matches_mv_churn(conn):
    mv_count = conn.execute("SELECT COUNT(*) FROM mv_churn").fetchone()[0]
    feat_count = conn.execute("SELECT COUNT(*) FROM feat_churn").fetchone()[0]
    assert feat_count == 7037
    assert feat_count == mv_count


def test_num_addon_services_in_valid_range(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn WHERE num_addon_services NOT BETWEEN 0 AND 8"
    ).fetchone()[0]
    assert count == 0


def test_num_addon_services_zero_when_no_internet(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn "
        "WHERE internet_service = false AND num_addon_services != 0"
    ).fetchone()[0]
    assert count == 0


def test_is_month_to_month_matches_contract(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn "
        "WHERE (contract = 'Month-to-Month') != is_month_to_month"
    ).fetchone()[0]
    assert count == 0


def test_internet_without_security_flag_correct(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn "
        "WHERE (internet_service = true AND online_security = false) "
        "!= internet_without_security"
    ).fetchone()[0]
    assert count == 0


def test_no_addons_despite_internet_count(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn WHERE no_addons_despite_internet = true"
    ).fetchone()[0]
    assert count == 81


def test_no_addons_despite_internet_false_when_no_internet(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM feat_churn "
        "WHERE internet_service = false AND no_addons_despite_internet != false"
    ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize(
    "column", ["customer_id", "is_voluntary_churn", "tenure_in_months", "contract"]
)
def test_original_columns_preserved(conn, column):
    mv_count = conn.execute(f"SELECT COUNT({column}) FROM mv_churn").fetchone()[0]
    feat_count = conn.execute(f"SELECT COUNT({column}) FROM feat_churn").fetchone()[0]
    assert feat_count == mv_count == 7037
