"""Data-quality invariant tests for sql/03_target/.

Like test_cleaning.py, these tests run staging, cleaning, and target
construction against the real committed data/raw/ files and assert claims
about the actual dataset (row counts, exclusion of Deceased customers, the
voluntary-churn mapping), not about arbitrary logic that toy cases could
exercise more cheaply.
"""

from __future__ import annotations

import duckdb
import pytest

from retention_platform.data.prepare import run_cleaning, run_staging, run_target


@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect(":memory:")
    run_staging(connection)
    run_cleaning(connection)
    run_target(connection)
    yield connection
    connection.close()


def test_deceased_customers_excluded(conn):
    cln_status_count = conn.execute("SELECT COUNT(*) FROM cln_status").fetchone()[0]
    tgt_churn_count = conn.execute("SELECT COUNT(*) FROM tgt_churn").fetchone()[0]
    assert cln_status_count - tgt_churn_count == 6

    deceased_count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn WHERE churn_reason = 'Deceased'"
    ).fetchone()[0]
    assert deceased_count == 0


def test_total_row_count_matches_expected(conn):
    count = conn.execute("SELECT COUNT(*) FROM tgt_churn").fetchone()[0]
    assert count == 7037


def test_stayed_and_joined_are_not_voluntary_churn(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn "
        "WHERE customer_status != 'Churned' AND is_voluntary_churn != false"
    ).fetchone()[0]
    assert count == 0


def test_all_remaining_churned_customers_are_voluntary(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn "
        "WHERE customer_status = 'Churned' AND is_voluntary_churn != true"
    ).fetchone()[0]
    assert count == 0


def test_moved_customers_treated_as_voluntary(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn WHERE churn_reason = 'Moved'"
    ).fetchone()[0]
    assert count == 46

    non_voluntary_count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn "
        "WHERE churn_reason = 'Moved' AND is_voluntary_churn != true"
    ).fetchone()[0]
    assert non_voluntary_count == 0


def test_dont_know_customers_treated_as_voluntary(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn WHERE churn_reason = 'Don''t know'"
    ).fetchone()[0]
    assert count == 130

    non_voluntary_count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn "
        "WHERE churn_reason = 'Don''t know' AND is_voluntary_churn != true"
    ).fetchone()[0]
    assert non_voluntary_count == 0


def test_poor_online_support_treated_as_voluntary(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn WHERE churn_reason = 'Poor expertise of online support'"
    ).fetchone()[0]
    assert count == 31

    non_voluntary_count = conn.execute(
        "SELECT COUNT(*) FROM tgt_churn "
        "WHERE churn_reason = 'Poor expertise of online support' AND is_voluntary_churn != true"
    ).fetchone()[0]
    assert non_voluntary_count == 0
