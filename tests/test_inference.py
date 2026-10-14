"""Tests for the inference package.

Section A: input validation rules, on a small hand-made schema and small
synthetic frames (fast, no database).
Section B: input validation on the real data and the real generated schema.
Section C: scoring core (score_batch and top_k_count).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.inference import score, validate
from retention_platform.inference.score import ScoringError, score_batch, top_k_count
from retention_platform.inference.validate import (
    InputValidationError,
    validate_scoring_table,
)
from retention_platform.metrics import confusion_matrix_at_k
from retention_platform.models.persist import LoadedRun
from retention_platform.pipeline.preprocess import (
    build_preprocessing_pipeline,
    prepare_model_inputs,
)
from retention_platform.pipeline.schema import (
    SQL_TYPE_TO_KIND,
    TARGET_COLUMN,
    build_input_schema,
    read_feat_churn_types,
    schema_hash,
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


# =============================================================================
# Section C: scoring core (score_batch and top_k_count)
# =============================================================================

SCORING_COLUMN_TYPES = {
    "customer_id": "VARCHAR",
    "is_voluntary_churn": "BOOLEAN",
    "gender": "VARCHAR",
    "age": "INTEGER",
    "offer": "VARCHAR",
    "internet_type": "VARCHAR",
    "contract": "VARCHAR",
    "payment_method": "VARCHAR",
    "tenure_in_months": "INTEGER",
    "married": "BOOLEAN",
}
SCORING_INPUT_COLUMNS = [
    "gender",
    "age",
    "offer",
    "internet_type",
    "contract",
    "payment_method",
    "tenure_in_months",
    "married",
]
K_FRAC = 0.10


@pytest.fixture(scope="module")
def scoring_artifacts():
    """A small real fitted preprocessor, model and schema."""
    frame = pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 6,
            "age": [20 + i for i in range(12)],
            "offer": ["Offer A", None, "Offer B", None] * 3,
            "internet_type": ["DSL", "Cable", None, "Fiber Optic"] * 3,
            "contract": ["Month-to-Month", "One Year", "Two Year"] * 4,
            "payment_method": ["Credit Card", "Mailed Check", "Bank Withdrawal"] * 4,
            "tenure_in_months": [1 + 3 * i for i in range(12)],
            "married": [True, False] * 6,
        }
    )
    preprocessor = build_preprocessing_pipeline().fit(frame)
    model = LogisticRegression().fit(preprocessor.transform(frame), [0, 1] * 6)
    schema = build_input_schema(SCORING_COLUMN_TYPES, preprocessor)
    return SimpleNamespace(frame=frame, preprocessor=preprocessor, model=model, schema=schema)


def make_run(schema, preprocessor, model) -> LoadedRun:
    return LoadedRun(
        run_id="20261011T120000Z",
        run_dir=Path("unused"),
        manifest={},
        schema=schema,
        schema_hash=schema_hash(schema),
        preprocessor=preprocessor,
        model=model,
    )


def scoring_frame(artifacts, ids) -> pd.DataFrame:
    """A valid scoring table with the given customer ids, built from the
    rows the preprocessor was fitted on."""
    base = artifacts.frame
    picked = base.iloc[[position % len(base) for position in range(len(ids))]]
    frame = picked.reset_index(drop=True)
    frame.insert(0, "customer_id", pd.Series(list(ids), dtype=object))
    return frame


class ScriptedPreprocessor:
    """Records what it is given and returns a placeholder matrix."""

    def __init__(self):
        self.received = []

    def transform(self, inputs):
        self.received.append(inputs)
        return np.zeros((len(inputs), 1))


class ScriptedModel:
    """Returns chosen churn probabilities, in input row order."""

    def __init__(self, values):
        self.values = values
        self.calls = 0

    def predict_proba(self, matrix):
        self.calls += 1
        values = np.asarray(self.values, dtype=float)
        return np.column_stack([1 - values, values])


def scripted_run(artifacts, values):
    preprocessor = ScriptedPreprocessor()
    model = ScriptedModel(values)
    return make_run(artifacts.schema, preprocessor, model), preprocessor, model


def real_run(artifacts) -> LoadedRun:
    return make_run(artifacts.schema, artifacts.preprocessor, artifacts.model)


# top_k_count


@pytest.mark.parametrize(
    "n_rows,k_frac,expected",
    [(1408, 0.10, 141), (5629, 0.10, 563), (25, 0.10, 2), (1, 0.10, 1), (3, 0.10, 1), (10, 1.0, 10)],
)
def test_top_k_count_values(n_rows, k_frac, expected):
    assert top_k_count(n_rows, k_frac) == expected


@pytest.mark.parametrize("bad", [0, -5, 2.5, True, None, "10"], ids=repr)
def test_top_k_count_rejects_a_bad_row_count(bad):
    with pytest.raises(ValueError):
        top_k_count(bad, 0.10)


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, 1.0000001, True, None, "0.1"], ids=repr)
def test_top_k_count_rejects_a_bad_k(bad):
    with pytest.raises(ValueError):
        top_k_count(10, bad)


@pytest.mark.parametrize(
    "n_rows,k_frac", [(50, 0.10), (25, 0.10), (1408, 0.10), (7, 0.30), (1, 0.05), (40, 1.0)]
)
def test_top_k_count_agrees_with_the_count_metrics_py_uses(n_rows, k_frac):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n_rows)
    y_score = rng.random(n_rows)

    matrix = confusion_matrix_at_k(y_true, y_score, k_frac)

    assert top_k_count(n_rows, k_frac) == matrix["tp"] + matrix["fp"]


# Table shape


def test_the_table_has_the_exact_columns_order_and_dtypes(scoring_artifacts):
    ids = [f"C{i:02d}" for i in range(12)]
    table = score_batch(scoring_frame(scoring_artifacts, ids), real_run(scoring_artifacts), K_FRAC)

    assert list(table.columns) == ["customer_id", "churn_probability", "rank", "in_top_k"]
    assert table["customer_id"].dtype == object
    assert table["churn_probability"].dtype == np.float64
    assert table["rank"].dtype == np.int64
    assert table["in_top_k"].dtype == bool
    assert all(type(value) is str for value in table["customer_id"])
    assert list(table.index) == list(range(len(table)))


def test_one_row_per_customer_with_the_ids_unchanged(scoring_artifacts):
    ids = [f"C{i:02d}" for i in range(12)]
    table = score_batch(scoring_frame(scoring_artifacts, ids), real_run(scoring_artifacts), K_FRAC)

    assert len(table) == 12
    assert sorted(table["customer_id"]) == sorted(ids)


def test_rank_is_one_to_n_and_rows_are_in_ascending_rank(scoring_artifacts):
    table = score_batch(
        scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)]),
        real_run(scoring_artifacts),
        K_FRAC,
    )

    assert list(table["rank"]) == list(range(1, 13))


def test_probabilities_are_in_range_non_increasing_and_the_models_own_values(scoring_artifacts):
    ids = [f"C{i:02d}" for i in range(12)]
    frame = scoring_frame(scoring_artifacts, ids)
    run = real_run(scoring_artifacts)

    table = score_batch(frame, run, K_FRAC)

    expected = run.model.predict_proba(run.preprocessor.transform(frame[SCORING_INPUT_COLUMNS]))[:, 1]
    expected_by_id = dict(zip(ids, expected))
    probabilities = table["churn_probability"].to_numpy()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(np.diff(probabilities) <= 0)
    for customer_id, probability in zip(table["customer_id"], probabilities):
        assert probability == expected_by_id[customer_id]


# Ties and flags


def test_equal_probabilities_are_ordered_by_customer_id_as_text(scoring_artifacts):
    ids = ["C2", "c1", "C10", "B7", "C1"]
    run, _, _ = scripted_run(scoring_artifacts, [0.5] * 5)

    table = score_batch(scoring_frame(scoring_artifacts, ids), run, K_FRAC)

    assert list(table["customer_id"]) == ["B7", "C1", "C10", "C2", "c1"]


def test_a_tie_across_the_flag_boundary_still_flags_exactly_k_rows(scoring_artifacts):
    ids = [f"C{i:02d}" for i in range(10)]
    values = [0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2]
    run, _, _ = scripted_run(scoring_artifacts, values)

    table = score_batch(scoring_frame(scoring_artifacts, ids), run, 0.30)

    assert top_k_count(10, 0.30) == 3
    assert table["in_top_k"].sum() == 3
    assert list(table["in_top_k"]) == [True] * 3 + [False] * 7
    assert list(table["customer_id"][:3]) == ["C00", "C01", "C02"]


@pytest.mark.parametrize(
    "n_rows,k_frac,expected",
    [(1, 0.10, 1), (2, 0.10, 1), (10, 0.25, 2), (25, 0.10, 2), (30, 0.10, 3), (7, 1.0, 7), (50, 0.10, 5)],
)
def test_flag_counts_follow_the_rule_and_flags_are_the_first_rows(
    scoring_artifacts, n_rows, k_frac, expected
):
    ids = [f"C{i:03d}" for i in range(n_rows)]
    values = np.linspace(0.99, 0.01, n_rows)
    run, _, _ = scripted_run(scoring_artifacts, values)

    table = score_batch(scoring_frame(scoring_artifacts, ids), run, k_frac)

    assert expected == top_k_count(n_rows, k_frac)
    assert int(table["in_top_k"].sum()) == expected
    assert list(table["in_top_k"]) == [True] * expected + [False] * (n_rows - expected)


# Input handling


def test_the_callers_frame_is_unchanged(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    original = frame.copy(deep=True)

    score_batch(frame, real_run(scoring_artifacts), K_FRAC)

    pd.testing.assert_frame_equal(frame, original)


def test_shuffled_rows_index_and_columns_give_the_same_table(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    run = real_run(scoring_artifacts)
    expected = score_batch(frame, run, K_FRAC)

    shuffled_rows = frame.sample(frac=1, random_state=3)
    random_index = frame.copy()
    random_index.index = np.random.default_rng(1).permutation(1000)[: len(frame)]
    reordered_columns = frame[list(reversed(frame.columns))]

    for variant in (shuffled_rows, random_index, reordered_columns):
        pd.testing.assert_frame_equal(score_batch(variant, run, K_FRAC), expected)


def test_true_false_and_one_zero_in_a_boolean_like_column_give_the_same_output(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    as_integers = frame.assign(married=frame["married"].astype("int64"))
    run = real_run(scoring_artifacts)

    pd.testing.assert_frame_equal(
        score_batch(as_integers, run, K_FRAC), score_batch(frame, run, K_FRAC)
    )


def test_none_in_the_nullable_columns_reaches_the_preprocessor_as_none(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    run, preprocessor, _ = scripted_run(scoring_artifacts, np.linspace(0.9, 0.1, 12))

    score_batch(frame, run, K_FRAC)

    (received,) = preprocessor.received
    for column in ("offer", "internet_type"):
        values = received[column].tolist()
        assert any(value is None for value in values)
        assert not any(value is pd.NA or (isinstance(value, float) and value != value) for value in values)


def test_a_category_dtype_customer_id_sorts_by_text(scoring_artifacts):
    ids = ["C2", "c1", "C10", "B7"]
    frame = scoring_frame(scoring_artifacts, ids)
    frame["customer_id"] = pd.Categorical(ids, categories=["c1", "C2", "C10", "B7"])
    run, _, _ = scripted_run(scoring_artifacts, [0.5] * 4)

    table = score_batch(frame, run, K_FRAC)

    assert list(table["customer_id"]) == ["B7", "C10", "C2", "c1"]
    assert all(type(value) is str for value in table["customer_id"])


def _missing_column(frame):
    return frame.drop(columns=["age"])


def _with_target(frame):
    return frame.assign(**{TARGET_COLUMN: [0, 1] * (len(frame) // 2)})


def _duplicate_id(frame):
    changed = frame.copy()
    changed.loc[1, "customer_id"] = changed.loc[0, "customer_id"]
    return changed


@pytest.mark.parametrize(
    "damage", [_missing_column, _with_target, _duplicate_id], ids=["missing-column", "target-present", "duplicate-id"]
)
def test_validation_comes_first_and_nothing_is_transformed_or_predicted(scoring_artifacts, damage):
    frame = damage(scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)]))
    run, preprocessor, model = scripted_run(scoring_artifacts, np.linspace(0.9, 0.1, 12))

    with pytest.raises(InputValidationError):
        score_batch(frame, run, K_FRAC)

    assert preprocessor.received == []
    assert model.calls == 0


def test_the_preprocessor_receives_exactly_the_schema_input_columns(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    shuffled_columns = frame[list(reversed(frame.columns))]
    run, preprocessor, _ = scripted_run(scoring_artifacts, np.linspace(0.9, 0.1, 12))

    score_batch(shuffled_columns, run, K_FRAC)

    (received,) = preprocessor.received
    assert list(received.columns) == SCORING_INPUT_COLUMNS
    assert "customer_id" not in received.columns
    assert TARGET_COLUMN not in received.columns


# ScoringError


@pytest.mark.parametrize(
    "values,fragment",
    [
        ([0.5, np.nan, 0.5, 0.5], "non-finite"),
        ([0.5, np.inf, 0.5, 0.5], "non-finite"),
        ([0.5, 1.5, 0.5, 0.5], "outside [0, 1]"),
        ([0.5, -0.1, 0.5, 0.5], "outside [0, 1]"),
        ([0.5, 0.5, 0.5], "shape"),
        ([0.5] * 5, "shape"),
    ],
    ids=["nan", "inf", "above-one", "negative", "too-short", "too-long"],
)
def test_unusable_model_output_raises_scoring_error(scoring_artifacts, values, fragment):
    frame = scoring_frame(scoring_artifacts, ["C01", "C02", "C03", "C04"])
    run, _, _ = scripted_run(scoring_artifacts, values)

    with pytest.raises(ScoringError, match=re.escape(fragment)):
        score_batch(frame, run, K_FRAC)


def test_scoring_error_names_the_first_offending_customer(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, ["C01", "C02", "C03", "C04"])
    run, _, _ = scripted_run(scoring_artifacts, [0.5, 2.0, 0.5, 0.5])

    with pytest.raises(ScoringError, match=r"row 1 \(customer_id 'C02'"):
        score_batch(frame, run, K_FRAC)


def no_missing_scoring_frame(artifacts) -> pd.DataFrame:
    """A valid scoring table whose nullable columns hold no missing values,
    so that string and category dtypes are accepted for them too."""
    frame = scoring_frame(artifacts, [f"C{i:02d}" for i in range(12)])
    frame["offer"] = pd.Series(["Offer A", "Offer B"] * 6, dtype=object)
    frame["internet_type"] = pd.Series(["DSL", "Cable", "Fiber Optic"] * 4, dtype=object)
    return frame


TEXT_COLUMNS = ("gender", "contract", "payment_method")
NULLABLE_COLUMNS = ("offer", "internet_type")

DTYPE_VARIANTS = {
    "string-text-columns": lambda f: f.astype({name: "string" for name in TEXT_COLUMNS}),
    "category-text-columns": lambda f: f.astype({name: "category" for name in TEXT_COLUMNS}),
    "int32-numbers": lambda f: f.astype({"age": "int32", "tenure_in_months": "int32"}),
    "float64-whole-numbers": lambda f: f.astype({"age": "float64", "tenure_in_months": "float64"}),
    "string-nullable-columns": lambda f: f.astype({name: "string" for name in NULLABLE_COLUMNS}),
    "category-nullable-columns": lambda f: f.astype({name: "category" for name in NULLABLE_COLUMNS}),
}


@pytest.mark.parametrize("build", list(DTYPE_VARIANTS.values()), ids=list(DTYPE_VARIANTS))
def test_compatible_dtypes_score_the_same_table_as_the_object_baseline(scoring_artifacts, build):
    base = no_missing_scoring_frame(scoring_artifacts)
    run = real_run(scoring_artifacts)

    baseline = score_batch(base, run, K_FRAC)
    variant = score_batch(build(base), run, K_FRAC)

    pd.testing.assert_frame_equal(variant, baseline)


@pytest.mark.parametrize("bad", [0, 1.5, True], ids=repr)
def test_score_batch_rejects_a_bad_k_frac_on_a_valid_frame(scoring_artifacts, bad):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])

    with pytest.raises(ValueError, match="k_frac"):
        score_batch(frame, real_run(scoring_artifacts), bad)


def test_scoring_is_deterministic(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    run = real_run(scoring_artifacts)

    first = score_batch(frame, run, K_FRAC)
    second = score_batch(frame, run, K_FRAC)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


# Label isolation


def test_score_py_never_references_the_target_column():
    tree = ast.parse(inspect.getsource(score))
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not any(TARGET_COLUMN in text for text in strings)
    assert TARGET_COLUMN not in names | attributes
    assert "validate_scoring_table" in names
