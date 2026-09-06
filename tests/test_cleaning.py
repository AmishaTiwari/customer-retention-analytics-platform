"""Data-quality invariant tests for sql/02_cleaning/.

Unlike hand-built toy-case tests elsewhere in this project, these tests run
staging and cleaning against the real committed data/raw/ files and assert
invariants about the actual dataset (value ranges, binary flags, boolean
conversion correctness). That is appropriate here because these are claims
about the real data itself, not about arbitrary logic that toy cases could
exercise more cheaply.
"""

from __future__ import annotations

import duckdb
import pytest

from retention_platform.data.prepare import run_cleaning, run_staging


@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect(":memory:")
    run_staging(connection)
    run_cleaning(connection)
    yield connection
    connection.close()


def test_no_negative_monthly_charge(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM stg_services WHERE monthly_charge < 0"
    ).fetchone()[0]
    assert count == 0


def test_no_negative_total_charges(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM stg_services WHERE total_charges < 0"
    ).fetchone()[0]
    assert count == 0


def test_no_negative_tenure(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM stg_services WHERE tenure_in_months < 0"
    ).fetchone()[0]
    assert count == 0


def test_satisfaction_score_in_valid_range(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM stg_status WHERE satisfaction_score NOT BETWEEN 1 AND 5"
    ).fetchone()[0]
    assert count == 0


def test_churn_value_is_binary(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM stg_status WHERE churn_value NOT IN (0, 1)"
    ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("cln_demographics", "under_30"),
        ("cln_services", "phone_service"),
        ("cln_status", "churn_label"),
    ],
)
def test_yes_no_columns_converted_to_boolean(conn, table, column):
    column_type = conn.execute(
        f"SELECT data_type FROM information_schema.columns "
        f"WHERE table_name = '{table}' AND column_name = '{column}'"
    ).fetchone()[0]
    assert column_type == "BOOLEAN"


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("cln_demographics", "under_30"),
        ("cln_demographics", "senior_citizen"),
        ("cln_demographics", "married"),
        ("cln_demographics", "dependents"),
        ("cln_services", "referred_a_friend"),
        ("cln_services", "phone_service"),
        ("cln_services", "multiple_lines"),
        ("cln_services", "internet_service"),
        ("cln_services", "online_security"),
        ("cln_services", "online_backup"),
        ("cln_services", "device_protection_plan"),
        ("cln_services", "premium_tech_support"),
        ("cln_services", "streaming_tv"),
        ("cln_services", "streaming_movies"),
        ("cln_services", "streaming_music"),
        ("cln_services", "unlimited_data"),
        ("cln_services", "paperless_billing"),
        ("cln_status", "churn_label"),
    ],
)
def test_no_nulls_introduced_by_boolean_conversion(conn, table, column):
    count = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
    ).fetchone()[0]
    assert count == 0
