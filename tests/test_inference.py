"""Tests for the inference package.

Section A: input validation rules, on a small hand-made schema and small
synthetic frames (fast, no database).
Section B: input validation on the real data and the real generated schema.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.inference import validate
from retention_platform.inference.validate import (
    InputValidationError,
    validate_scoring_table,
)
from retention_platform.pipeline.preprocess import prepare_model_inputs
from retention_platform.pipeline.schema import (
    SQL_TYPE_TO_KIND,
    build_input_schema,
    read_feat_churn_types,
)
from retention_platform.pipeline.split import split_feat_churn

# =============================================================================
# Section A: validation rules on a small hand-made schema
# =============================================================================

SMALL_SCHEMA = {
    "identifier_column": "customer_id",
    "columns": [
        {"name": "flag", "kind": "boolean-like", "nullable": False},
        {"name": "count", "kind": "integer-like", "nullable": False},
        {"name": "amount", "kind": "decimal", "nullable": False},
        {
            "name": "plan",
            "kind": "categorical",
            "nullable": False,
            "allowed_values": ["Basic", "Premium"],
        },
        {
            "name": "extra_service",
            "kind": "categorical",
            "nullable": True,
            "allowed_values": ["Cable", "DSL"],
            "reserved_values": ["No Extra"],
        },
    ],
}


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "flag": [True, False, True],
            "count": [1, 2, 3],
            "amount": [10.5, 20.0, 30.25],
            "plan": ["Basic", "Premium", "Basic"],
            "extra_service": ["Cable", None, "DSL"],
        }
    )


def with_column(frame: pd.DataFrame, name: str, values, dtype=None) -> pd.DataFrame:
    changed = frame.copy()
    changed[name] = pd.Series(values, dtype=dtype if dtype is not None else object)
    return changed


def with_object_column(frame: pd.DataFrame, name: str, values) -> pd.DataFrame:
    changed = frame.copy()
    changed[name] = pd.Series(list(values), dtype=object)
    return changed


def test_kind_constants_equal_the_schema_generator_kinds():
    assert validate.KIND_BOOLEAN == SQL_TYPE_TO_KIND["BOOLEAN"]
    assert validate.KIND_INTEGER == SQL_TYPE_TO_KIND["INTEGER"]
    assert validate.KIND_DECIMAL == SQL_TYPE_TO_KIND["DOUBLE"]
    assert validate.KIND_CATEGORICAL == SQL_TYPE_TO_KIND["VARCHAR"]
    assert {
        validate.KIND_BOOLEAN,
        validate.KIND_INTEGER,
        validate.KIND_DECIMAL,
        validate.KIND_CATEGORICAL,
    } == set(SQL_TYPE_TO_KIND.values())


def test_valid_small_frame_passes():
    assert validate_scoring_table(small_frame(), SMALL_SCHEMA) is None


def test_input_error_is_a_value_error():
    assert issubclass(InputValidationError, ValueError)


# Each case: (id, function building the broken frame, substrings the message must contain).
FAILURE_CASES = [
    # --- table-level columns ---
    (
        "missing column",
        lambda f: f.drop(columns=["amount"]),
        ["missing columns", "amount"],
    ),
    (
        "missing identifier column",
        lambda f: f.drop(columns=["customer_id"]),
        ["missing columns", "customer_id"],
    ),
    (
        "extra column",
        lambda f: f.assign(city=["a", "b", "c"]),
        ["extra columns", "city"],
    ),
    (
        "target column present",
        lambda f: f.assign(is_voluntary_churn=[0, 1, 0]),
        ["extra columns", "is_voluntary_churn"],
    ),
    (
        "duplicate column name",
        lambda f: pd.concat([f, f[["plan"]]], axis=1),
        ["duplicate column names", "plan"],
    ),
    # --- empty table ---
    ("empty table", lambda f: f.iloc[0:0], ["no rows"]),
    # --- customer_id ---
    (
        "duplicate customer_id",
        lambda f: with_object_column(f, "customer_id", ["C1", "C1", "C3"]),
        ["duplicate customer_id values", "customer_id", "row: 1", "customer_id 'C1'"],
    ),
    (
        "missing customer_id None",
        lambda f: with_object_column(f, "customer_id", ["C1", None, "C3"]),
        ["missing customer_id value", "row: 1"],
    ),
    (
        "missing customer_id NaN",
        lambda f: with_object_column(f, "customer_id", ["C1", np.nan, "C3"]),
        ["missing customer_id value", "row: 1"],
    ),
    (
        "missing customer_id pd.NA",
        lambda f: with_object_column(f, "customer_id", ["C1", pd.NA, "C3"]),
        ["missing customer_id value", "row: 1"],
    ),
    (
        "empty-string customer_id",
        lambda f: with_object_column(f, "customer_id", ["C1", "", "C3"]),
        ["empty or whitespace-only customer_id", "row: 1"],
    ),
    (
        "whitespace-only customer_id",
        lambda f: with_object_column(f, "customer_id", ["C1", "   ", "C3"]),
        ["empty or whitespace-only customer_id", "row: 1"],
    ),
    (
        "non-text customer_id",
        lambda f: with_object_column(f, "customer_id", [1, 2, 3]),
        ["customer_id is not text", "row: 0"],
    ),
    # --- missing values ---
    (
        "NaN in decimal",
        lambda f: with_object_column(f, "amount", [10.5, np.nan, 30.25]),
        ["NaN value", "amount", "row: 1", "customer_id 'C2'"],
    ),
    (
        "NaN in integer-like",
        lambda f: with_column(f, "count", [1.0, np.nan, 3.0], dtype="float64"),
        ["NaN value", "count", "row: 1"],
    ),
    (
        "positive infinity",
        lambda f: with_column(f, "amount", [10.5, np.inf, 30.25], dtype="float64"),
        ["infinite value", "amount", "row: 1"],
    ),
    (
        "negative infinity",
        lambda f: with_column(f, "amount", [10.5, -np.inf, 30.25], dtype="float64"),
        ["infinite value", "amount", "row: 1"],
    ),
    (
        "None in non-nullable categorical",
        lambda f: with_object_column(f, "plan", ["Basic", None, "Basic"]),
        ["None is not accepted", "not nullable", "plan", "row: 1"],
    ),
    (
        "None in non-nullable decimal",
        lambda f: with_object_column(f, "amount", [10.5, None, 30.25]),
        ["None is not accepted", "not nullable", "amount", "row: 1"],
    ),
    (
        "pd.NA in nullable column",
        lambda f: with_object_column(f, "extra_service", ["Cable", pd.NA, "DSL"]),
        ["pd.NA is not accepted", "extra_service", "row: 1"],
    ),
    (
        "pd.NA in non-nullable column",
        lambda f: with_object_column(f, "plan", ["Basic", pd.NA, "Basic"]),
        ["pd.NA is not accepted", "plan", "row: 1"],
    ),
    (
        "NaN in nullable column",
        lambda f: with_object_column(f, "extra_service", ["Cable", np.nan, "DSL"]),
        ["NaN value", "extra_service", "row: 1"],
    ),
    # --- categories ---
    (
        "sentinel as raw input",
        lambda f: with_object_column(f, "extra_service", ["Cable", "No Extra", "DSL"]),
        ["internal sentinel", "extra_service", "row: 1"],
    ),
    (
        "unseen category",
        lambda f: with_object_column(f, "plan", ["Basic", "Gold", "Basic"]),
        ["unseen category", "plan", "row: 1"],
    ),
    (
        "non-text category",
        lambda f: with_object_column(f, "plan", ["Basic", 5, "Basic"]),
        ["not text", "plan", "row: 1"],
    ),
    # --- numbers ---
    (
        "numeric string in integer-like",
        lambda f: with_object_column(f, "count", ["1", "2", "3"]),
        ["numeric strings are not accepted", "count", "row: 0", "3 row(s) offend"],
    ),
    (
        "numeric string in decimal",
        lambda f: with_object_column(f, "amount", [10.5, "20.0", 30.25]),
        ["numeric strings are not accepted", "amount", "row: 1"],
    ),
    (
        "12.5 in integer-like",
        lambda f: with_column(f, "count", [1.0, 12.5, 3.0], dtype="float64"),
        ["non-whole number", "count", "row: 1"],
    ),
    (
        "bool in integer-like",
        lambda f: with_object_column(f, "count", [1, True, 3]),
        ["boolean value where a number is required", "count", "row: 1"],
    ),
    (
        "bool in decimal",
        lambda f: with_object_column(f, "amount", [10.5, False, 30.25]),
        ["boolean value where a number is required", "amount", "row: 1"],
    ),
    # --- booleans ---
    (
        "boolean-like value 2",
        lambda f: with_object_column(f, "flag", [True, 2, False]),
        ["True, False, 0 or 1", "flag", "row: 1"],
    ),
    (
        "boolean-like string 'True'",
        lambda f: with_object_column(f, "flag", [True, "True", False]),
        ["True, False, 0 or 1", "flag", "row: 1"],
    ),
    (
        "boolean-like value 0.5",
        lambda f: with_object_column(f, "flag", [True, 0.5, False]),
        ["True, False, 0 or 1", "flag", "row: 1"],
    ),
    (
        "boolean-like float 1.0",
        lambda f: with_column(f, "flag", [1.0, 0.0, 1.0], dtype="float64"),
        ["True, False, 0 or 1", "flag", "row: 0"],
    ),
]


@pytest.mark.parametrize(
    "build,expected",
    [(case[1], case[2]) for case in FAILURE_CASES],
    ids=[case[0] for case in FAILURE_CASES],
)
def test_rule_failures_raise_with_condition_column_and_row(build, expected):
    broken = build(small_frame())

    with pytest.raises(InputValidationError) as exc_info:
        validate_scoring_table(broken, SMALL_SCHEMA)

    message = str(exc_info.value)
    for fragment in expected:
        assert fragment in message, f"{fragment!r} not in {message!r}"


def test_a_frame_that_is_not_a_dataframe_is_rejected():
    with pytest.raises(InputValidationError, match="DataFrame"):
        validate_scoring_table({"customer_id": ["C1"]}, SMALL_SCHEMA)


def test_table_level_failures_do_not_name_a_row():
    with pytest.raises(InputValidationError) as exc_info:
        validate_scoring_table(small_frame().iloc[0:0], SMALL_SCHEMA)

    assert "row" not in str(exc_info.value).replace("no rows", "")


def test_message_counts_every_offending_row():
    frame = with_object_column(small_frame(), "plan", ["Gold", "Basic", "Silver"])

    with pytest.raises(InputValidationError, match=r"row: 0.*2 row\(s\) offend"):
        validate_scoring_table(frame, SMALL_SCHEMA)


def _astype(column: str, dtype):
    return lambda f: f.astype({column: dtype})


ACCEPTED_CASES = [
    ("reordered columns", lambda f: f[list(reversed(f.columns))]),
    ("integer-like int32", _astype("count", "int32")),
    ("integer-like int64", _astype("count", "int64")),
    ("integer-like int8", _astype("count", "int8")),
    ("integer-like uint16", _astype("count", "uint16")),
    ("integer-like float64 whole", _astype("count", "float64")),
    ("integer-like float32 whole", _astype("count", "float32")),
    ("decimal float32", _astype("amount", "float32")),
    ("decimal float64", _astype("amount", "float64")),
    ("decimal int", lambda f: f.assign(amount=[10, 20, 30])),
    ("decimal int32", lambda f: f.assign(amount=pd.Series([10, 20, 30], dtype="int32"))),
    ("boolean-like bool", _astype("flag", "bool")),
    ("boolean-like int 0/1", lambda f: f.assign(flag=[1, 0, 1])),
    ("boolean-like int8 0/1", lambda f: f.assign(flag=pd.Series([1, 0, 1], dtype="int8"))),
    (
        "boolean-like object mix",
        lambda f: with_object_column(f, "flag", [True, 0, 1]),
    ),
    ("text object", _astype("plan", object)),
    ("text string dtype", _astype("plan", "string")),
    ("text category dtype", _astype("plan", "category")),
    ("customer_id string dtype", _astype("customer_id", "string")),
    (
        "nullable column string dtype without missing values",
        lambda f: f.assign(extra_service=pd.Series(["Cable", "DSL", "DSL"], dtype="string")),
    ),
    (
        "nullable column category dtype without missing values",
        lambda f: f.assign(
            extra_service=pd.Series(["Cable", "DSL", "DSL"], dtype="category")
        ),
    ),
    (
        "None in the nullable column",
        lambda f: with_object_column(f, "extra_service", [None, None, None]),
    ),
]


@pytest.mark.parametrize(
    "build", [case[1] for case in ACCEPTED_CASES], ids=[case[0] for case in ACCEPTED_CASES]
)
def test_compatible_inputs_are_accepted(build):
    frame = build(small_frame())

    assert validate_scoring_table(frame, SMALL_SCHEMA) is None


def test_frame_is_unchanged_after_a_successful_call():
    frame = small_frame()
    original = frame.copy(deep=True)

    validate_scoring_table(frame, SMALL_SCHEMA)

    pd.testing.assert_frame_equal(frame, original)


def test_frame_is_unchanged_after_a_failed_call():
    frame = with_object_column(small_frame(), "plan", ["Basic", "Gold", None])
    original = frame.copy(deep=True)

    with pytest.raises(InputValidationError):
        validate_scoring_table(frame, SMALL_SCHEMA)

    pd.testing.assert_frame_equal(frame, original)


def test_columns_are_checked_before_the_empty_table():
    frame = small_frame().iloc[0:0].drop(columns=["amount"])

    with pytest.raises(InputValidationError, match="missing columns"):
        validate_scoring_table(frame, SMALL_SCHEMA)


def test_empty_table_is_checked_before_customer_id():
    with pytest.raises(InputValidationError, match="no rows"):
        validate_scoring_table(small_frame().iloc[0:0], SMALL_SCHEMA)


def test_customer_id_is_checked_before_values():
    frame = with_object_column(small_frame(), "customer_id", ["C1", "C1", "C3"])
    frame = with_object_column(frame, "amount", [np.nan, 20.0, 30.25])

    with pytest.raises(InputValidationError, match="duplicate customer_id"):
        validate_scoring_table(frame, SMALL_SCHEMA)


def test_value_columns_are_checked_in_schema_order():
    frame = with_object_column(small_frame(), "flag", [True, 2, False])
    frame = with_object_column(frame, "amount", [np.nan, 20.0, 30.25])

    with pytest.raises(InputValidationError, match="column 'flag'"):
        validate_scoring_table(frame, SMALL_SCHEMA)


# =============================================================================
# Section B: validation on the real data and the real generated schema
# =============================================================================


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
def real_schema(conn):
    model_inputs = prepare_model_inputs(conn)
    return build_input_schema(read_feat_churn_types(conn), model_inputs.preprocessor)


@pytest.fixture(scope="module")
def scored_columns(real_schema):
    return [real_schema["identifier_column"], *[c["name"] for c in real_schema["columns"]]]


@pytest.fixture(scope="module")
def test_split(conn):
    return split_feat_churn(conn).test


def test_whole_real_table_validates(conn, real_schema, scored_columns):
    table = conn.execute("SELECT * FROM feat_churn").fetchdf()[scored_columns]

    assert any(dtype == np.int32 for dtype in table.dtypes)
    assert validate_scoring_table(table, real_schema) is None


def test_real_test_split_validates(real_schema, scored_columns, test_split):
    assert len(test_split) == 1408
    assert validate_scoring_table(test_split[scored_columns], real_schema) is None


@pytest.mark.parametrize("extra", ["is_voluntary_churn", "city"])
def test_adding_the_target_or_an_excluded_column_is_rejected(
    real_schema, scored_columns, test_split, extra
):
    frame = test_split[[*scored_columns, extra]]

    with pytest.raises(InputValidationError, match=f"extra columns.*{extra}"):
        validate_scoring_table(frame, real_schema)


def test_dropping_any_single_schema_column_is_rejected(
    real_schema, scored_columns, test_split
):
    for name in scored_columns:
        frame = test_split[scored_columns].drop(columns=[name])

        with pytest.raises(InputValidationError, match=f"missing columns.*{name}"):
            validate_scoring_table(frame, real_schema)


def test_sentinel_written_into_a_real_row_is_rejected(
    real_schema, scored_columns, test_split
):
    frame = test_split[scored_columns].copy()
    frame.iloc[0, frame.columns.get_loc("internet_type")] = "No Internet"

    with pytest.raises(InputValidationError, match=r"internal sentinel.*column 'internet_type'"):
        validate_scoring_table(frame, real_schema)
