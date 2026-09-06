"""Leakage-boundary governance tests for sql/04_modeling_view/.

Like test_cleaning.py and test_target.py, these tests run the full
pipeline (staging, cleaning, target construction, modeling view) against
the real committed data/raw/ files and assert claims about the actual
dataset's schema and row-level integrity, not about arbitrary logic that
toy cases could exercise more cheaply.
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

# Post-outcome/temporal leakage columns: only known after churn has already
# happened, not realistically available at scoring time.
POST_OUTCOME_LEAKAGE_COLUMNS = [
    "churn_score",
    "cltv",
    "churn_category",
    "churn_reason",
]

# Target-derived columns: would trivially expose the target itself.
TARGET_DERIVED_COLUMNS = [
    "customer_status",
    "churn_label",
    "churn_value",
]

EXCLUDED_COLUMNS = POST_OUTCOME_LEAKAGE_COLUMNS + TARGET_DERIVED_COLUMNS


@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect(":memory:")
    run_staging(connection)
    run_cleaning(connection)
    run_target(connection)
    run_modeling_view(connection)
    yield connection
    connection.close()


@pytest.mark.parametrize("column", EXCLUDED_COLUMNS)
def test_leakage_columns_absent_from_schema(conn, column):
    rows = conn.execute(
        "SELECT * FROM information_schema.columns "
        "WHERE table_name = 'mv_churn' AND column_name = ?",
        [column],
    ).fetchall()
    assert rows == []


@pytest.mark.parametrize("column", ["customer_id", "is_voluntary_churn"])
def test_identifier_and_target_columns_present(conn, column):
    rows = conn.execute(
        "SELECT * FROM information_schema.columns "
        "WHERE table_name = 'mv_churn' AND column_name = ?",
        [column],
    ).fetchall()
    assert len(rows) == 1


def test_row_count_matches_target_population(conn):
    count = conn.execute("SELECT COUNT(*) FROM mv_churn").fetchone()[0]
    assert count == 7037


def test_no_duplicate_customer_ids(conn):
    total = conn.execute("SELECT COUNT(*) FROM mv_churn").fetchone()[0]
    distinct = conn.execute("SELECT COUNT(DISTINCT customer_id) FROM mv_churn").fetchone()[0]
    assert total == distinct


def test_target_column_never_null(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM mv_churn WHERE is_voluntary_churn IS NULL"
    ).fetchone()[0]
    assert count == 0
