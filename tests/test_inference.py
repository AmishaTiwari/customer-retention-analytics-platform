"""Tests for the inference package.

Section A: input validation rules, on a small hand-made schema and small
synthetic frames (fast, no database).
Section B: input validation on the real data and the real generated schema.
Section C: scoring core (score_batch).
Section D: outputs (the CSV, the run summary and their publication).
Section E: the scoring command (population selection, run_scoring, main and
its failure classes).
Section F: ranking: top_k_count and risk tiers.
Section G: drivers (structure check, averaged contributions, top-driver selection).
"""

from __future__ import annotations

import ast
import copy
import csv
import dataclasses
import inspect
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform import tracking
from retention_platform.config import REPO_ROOT, ConfigValidationError, load_config
from retention_platform.inference import outputs, score, validate
from retention_platform.inference.outputs import (
    OUTREACH_FILE,
    SNAPSHOT_NOTICE,
    SUMMARY_FILE,
    OutputGenerationError,
    OutputPublicationError,
    build_run_summary,
    outreach_csv_bytes,
    publish_outputs,
    summary_json_bytes,
)
from retention_platform.inference.score import (
    PostPublicationTrackingError,
    ScoringError,
    ScoringResult,
    run_scoring,
    score_batch,
    select_scoring_population,
)
from retention_platform.inference.drivers import (
    DIRECTION_LOWERS,
    DIRECTION_RAISES,
    DRIVER_COLUMNS,
    DRIVER_LABELS,
    EXPECTED_FOLD_MODELS,
    TOP_N_DRIVERS,
    DriverError,
    averaged_contributions,
    encoded_column_owners,
    select_top_drivers,
)
from retention_platform.inference.ranking import assign_risk_tiers, top_k_count
from retention_platform.inference.validate import (
    InputValidationError,
    validate_scoring_table,
)
from retention_platform.metrics import confusion_matrix_at_k
from retention_platform.models.persist import (
    HELD_OUT_RESULTS_REFERENCE,
    TRAINING_CV_ESTIMATE_TYPE,
    LoadedRun,
    ModelLoadError,
    build_manifest,
    publish_run,
)
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
from retention_platform.tracking import TrackingError, format_run_id

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
# Section C: scoring core (score_batch)
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


# =============================================================================
# Section D: outputs (the CSV, the run summary and their publication)
# =============================================================================

MODEL_RUN_ID = "20261011T120000Z"
SCORING_RUN_ID = "20261011T123500Z"
OUTPUT_METRICS = {
    "k": 0.1,
    "training_rows": 5629,
    "estimate_type": TRAINING_CV_ESTIMATE_TYPE,
    "precision_at_k": 0.9059,
    "lift_at_k": 3.4222,
    "pr_auc": 0.8004,
    "brier_score": 0.1046,
}
SUMMARY_KEYS = [
    "contract_version",
    "k",
    "customers_scored",
    "customers_flagged",
    "model_run_id",
    "scoring_run_id",
    "snapshot_notice",
    "model_health",
]
MODEL_HEALTH_KEYS = [
    "trained_at_utc",
    "package_version",
    "training_cv_metrics",
    "held_out_results_reference",
]
METRIC_KEYS = list(OUTPUT_METRICS)
TABLE_COLUMNS = ["customer_id", "churn_probability", "rank", "in_top_k"]


@pytest.fixture(scope="module")
def ranked_table(scoring_artifacts):
    frame = scoring_frame(scoring_artifacts, [f"C{i:02d}" for i in range(12)])
    return score_batch(frame, real_run(scoring_artifacts), K_FRAC)


@pytest.fixture(scope="module")
def output_manifest(scoring_artifacts):
    return build_manifest(
        run_id=MODEL_RUN_ID,
        instant=datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc),
        schema=scoring_artifacts.schema,
        hyperparameters={},
        training_cv_metrics=dict(OUTPUT_METRICS),
        fingerprints={"config_hash": "a" * 64},
    )


def make_summary(table, manifest, **overrides):
    arguments = {
        "k_frac": K_FRAC,
        "model_run_id": MODEL_RUN_ID,
        "scoring_run_id": SCORING_RUN_ID,
        "manifest": manifest,
    }
    arguments.update(overrides)
    return build_run_summary(table, **arguments)


def table_from(ids, probabilities, flagged=1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.Series(list(ids), dtype=object),
            "churn_probability": pd.Series(list(probabilities), dtype="float64"),
            "rank": pd.Series(range(1, len(ids) + 1), dtype="int64"),
            "in_top_k": pd.Series([position < flagged for position in range(len(ids))], dtype=bool),
        }
    )


AWKWARD_IDS = ["C1", "a,b", 'say "hi"', "Zoë"]
AWKWARD_PROBABILITIES = [2 / 3, 1 / 3, 0.1 + 0.2, 0.1]


def awkward_table() -> pd.DataFrame:
    return table_from(AWKWARD_IDS, AWKWARD_PROBABILITIES, flagged=2)


def parse_csv(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


# CSV


def test_csv_has_the_exact_header_one_row_per_customer_and_ascending_rank(ranked_table):
    rows = parse_csv(outreach_csv_bytes(ranked_table))

    assert rows[0] == TABLE_COLUMNS
    assert len(rows) - 1 == len(ranked_table)
    assert all(len(row) == 4 for row in rows)
    assert [int(row[2]) for row in rows[1:]] == list(range(1, len(ranked_table) + 1))
    assert [row[0] for row in rows[1:]] == list(ranked_table["customer_id"])


def test_csv_writes_in_top_k_as_true_and_false(ranked_table):
    rows = parse_csv(outreach_csv_bytes(ranked_table))

    assert {row[3] for row in rows[1:]} <= {"True", "False"}
    assert [row[3] == "True" for row in rows[1:]] == list(ranked_table["in_top_k"])


def test_csv_is_utf8_with_lf_endings_and_a_trailing_newline():
    data = outreach_csv_bytes(awkward_table())

    assert "Zoë" in data.decode("utf-8")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert data.count(b"\n") == len(AWKWARD_IDS) + 1


def test_csv_probabilities_round_trip_exactly():
    rows = parse_csv(outreach_csv_bytes(awkward_table()))

    assert [float(row[1]) for row in rows[1:]] == AWKWARD_PROBABILITIES
    assert rows[3][1] == "0.30000000000000004"


def test_csv_has_no_empty_cell(ranked_table):
    rows = parse_csv(outreach_csv_bytes(ranked_table))

    assert all(cell != "" for row in rows for cell in row)


def test_csv_ids_with_commas_and_quotes_survive_a_round_trip():
    rows = parse_csv(outreach_csv_bytes(awkward_table()))

    assert [row[0] for row in rows[1:]] == AWKWARD_IDS


def _drop_column(table):
    return table.drop(columns=["rank"])


def _extra_column(table):
    return table.assign(extra=1)


def _reordered_columns(table):
    return table[["churn_probability", "customer_id", "rank", "in_top_k"]]


def _nan_probability(table):
    table = table.copy()
    table.loc[1, "churn_probability"] = np.nan
    return table


def _none_id(table):
    table = table.copy()
    table.loc[1, "customer_id"] = None
    return table


def _empty_id(table):
    table = table.copy()
    table.loc[1, "customer_id"] = ""
    return table


def _blank_id(table):
    table = table.copy()
    table.loc[1, "customer_id"] = "   "
    return table


def _no_rows(table):
    return table.iloc[0:0]


def _probability_above_one(table):
    table = table.copy()
    table.loc[0, "churn_probability"] = 1.5
    return table


def _flag_not_boolean(table):
    return table.assign(in_top_k=table["in_top_k"].astype(int))


BAD_TABLES = {
    "missing-column": _drop_column,
    "extra-column": _extra_column,
    "reordered-columns": _reordered_columns,
    "nan-probability": _nan_probability,
    "none-id": _none_id,
    "empty-id": _empty_id,
    "whitespace-id": _blank_id,
    "no-rows": _no_rows,
    "probability-above-one": _probability_above_one,
    "flag-not-boolean": _flag_not_boolean,
}


@pytest.mark.parametrize("damage", list(BAD_TABLES.values()), ids=list(BAD_TABLES))
def test_csv_refuses_a_bad_table(ranked_table, damage):
    with pytest.raises(OutputGenerationError):
        outreach_csv_bytes(damage(ranked_table))


# Summary


def test_summary_has_the_exact_key_sets_in_order_with_plain_types(ranked_table, output_manifest):
    summary = make_summary(ranked_table, output_manifest)

    assert list(summary) == SUMMARY_KEYS
    assert list(summary["model_health"]) == MODEL_HEALTH_KEYS
    assert list(summary["model_health"]["training_cv_metrics"]) == METRIC_KEYS
    assert type(summary["contract_version"]) is str
    assert type(summary["k"]) is float
    assert type(summary["customers_scored"]) is int
    assert type(summary["customers_flagged"]) is int
    assert type(summary["model_run_id"]) is str
    assert type(summary["scoring_run_id"]) is str
    assert type(summary["snapshot_notice"]) is str
    health = summary["model_health"]
    assert type(health["trained_at_utc"]) is str
    assert type(health["package_version"]) is str
    assert type(health["held_out_results_reference"]) is str
    metrics = health["training_cv_metrics"]
    assert type(metrics["k"]) is float
    assert type(metrics["training_rows"]) is int
    assert type(metrics["estimate_type"]) is str
    for name in ("precision_at_k", "lift_at_k", "pr_auc", "brier_score"):
        assert type(metrics[name]) is float


def test_summary_counts_and_fixed_texts(ranked_table, output_manifest):
    summary = make_summary(ranked_table, output_manifest)
    contract = (REPO_ROOT / "docs" / "platform_output_contract.md").read_text(encoding="utf-8")

    assert summary["contract_version"] == "1"
    assert summary["customers_scored"] == len(ranked_table)
    assert summary["customers_flagged"] == int(ranked_table["in_top_k"].sum())
    assert summary["k"] == K_FRAC
    assert summary["model_run_id"] == MODEL_RUN_ID
    assert summary["scoring_run_id"] == SCORING_RUN_ID
    assert summary["snapshot_notice"] == SNAPSHOT_NOTICE
    assert f"`{SNAPSHOT_NOTICE}`" in contract
    assert summary["model_health"]["held_out_results_reference"] == "ADR-0014"
    assert summary["model_health"]["held_out_results_reference"] == HELD_OUT_RESULTS_REFERENCE


def test_summary_equals_the_expected_object_so_no_held_out_numbers_can_appear(
    ranked_table, output_manifest
):
    expected = {
        "contract_version": "1",
        "k": 0.1,
        "customers_scored": 12,
        "customers_flagged": 1,
        "model_run_id": MODEL_RUN_ID,
        "scoring_run_id": SCORING_RUN_ID,
        "snapshot_notice": (
            "Historical snapshot: this list scores the held-out test split for "
            "demonstration. It is not a live outreach list."
        ),
        "model_health": {
            "trained_at_utc": "2026-10-11T12:00:00Z",
            "package_version": output_manifest["package_version"],
            "training_cv_metrics": OUTPUT_METRICS,
            "held_out_results_reference": "ADR-0014",
        },
    }

    assert make_summary(ranked_table, output_manifest) == expected


def test_model_health_equals_the_manifest_and_is_a_deep_copy(ranked_table, output_manifest):
    manifest_before = copy.deepcopy(output_manifest)

    summary = make_summary(ranked_table, output_manifest)
    for key in MODEL_HEALTH_KEYS:
        assert summary["model_health"][key] == output_manifest[key]
    summary["model_health"]["training_cv_metrics"]["pr_auc"] = 0.0
    summary["model_health"]["package_version"] = "changed"

    assert output_manifest == manifest_before


def _manifest_without(key):
    def damage(manifest):
        del manifest[key]

    return damage


def _manifest_without_metric(manifest):
    del manifest["training_cv_metrics"]["pr_auc"]


def _manifest_extra_metric(manifest):
    manifest["training_cv_metrics"]["roc_auc"] = 0.9


def _manifest_wrong_estimate_type(manifest):
    manifest["training_cv_metrics"]["estimate_type"] = "Some other wording"


def _manifest_wrong_reference(manifest):
    manifest["held_out_results_reference"] = "ADR-0099"


def _manifest_bad_time(manifest):
    manifest["trained_at_utc"] = "2026-10-11 12:00:00"


def _manifest_bool_rows(manifest):
    manifest["training_cv_metrics"]["training_rows"] = True


MANIFEST_DAMAGE = {
    "missing-trained-at": _manifest_without("trained_at_utc"),
    "missing-package-version": _manifest_without("package_version"),
    "missing-metrics": _manifest_without("training_cv_metrics"),
    "missing-reference": _manifest_without("held_out_results_reference"),
    "missing-metric-key": _manifest_without_metric,
    "extra-metric-key": _manifest_extra_metric,
    "wrong-estimate-type": _manifest_wrong_estimate_type,
    "wrong-reference": _manifest_wrong_reference,
    "bad-trained-at": _manifest_bad_time,
    "bool-training-rows": _manifest_bool_rows,
}


@pytest.mark.parametrize("damage", list(MANIFEST_DAMAGE.values()), ids=list(MANIFEST_DAMAGE))
def test_summary_refuses_a_manifest_that_breaks_the_contract(ranked_table, output_manifest, damage):
    manifest = copy.deepcopy(output_manifest)
    damage(manifest)

    with pytest.raises(OutputGenerationError):
        make_summary(ranked_table, manifest)


@pytest.mark.parametrize(
    "overrides",
    [
        {"k_frac": 0},
        {"k_frac": 1.5},
        {"k_frac": True},
        {"model_run_id": "not-a-run-id"},
        {"scoring_run_id": "../20261011T123500Z"},
        {"scoring_run_id": ""},
    ],
    ids=["k-zero", "k-above-one", "k-bool", "bad-model-run-id", "bad-scoring-run-id", "empty-scoring-run-id"],
)
def test_summary_refuses_bad_arguments(ranked_table, output_manifest, overrides):
    with pytest.raises(OutputGenerationError):
        make_summary(ranked_table, output_manifest, **overrides)


def test_summary_refuses_an_empty_table(ranked_table, output_manifest):
    with pytest.raises(OutputGenerationError):
        make_summary(ranked_table.iloc[0:0], output_manifest)


def test_summary_bytes_are_json_with_a_trailing_newline(ranked_table, output_manifest):
    summary = make_summary(ranked_table, output_manifest)

    data = summary_json_bytes(summary)

    assert json.loads(data.decode("utf-8")) == summary
    assert data.endswith(b"\n")
    assert b"\n  " in data
    assert list(json.loads(data.decode("utf-8"))) == SUMMARY_KEYS


def test_summary_bytes_refuse_nan():
    with pytest.raises(OutputGenerationError):
        summary_json_bytes({"k": float("nan")})


# Publication


OLD_CSV = b"customer_id,churn_probability,rank,in_top_k\nOLD,0.5,1,True\n"
OLD_SUMMARY = b'{"previous": "summary"}\n'


@pytest.fixture
def publication(tmp_path, ranked_table, output_manifest):
    summary = make_summary(ranked_table, output_manifest)
    return SimpleNamespace(
        folder=tmp_path / "outputs",
        table=ranked_table,
        summary=summary,
        csv_bytes=outreach_csv_bytes(ranked_table),
        summary_bytes=summary_json_bytes(summary),
    )


def write_previous_pair(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / OUTREACH_FILE).write_bytes(OLD_CSV)
    (folder / SUMMARY_FILE).write_bytes(OLD_SUMMARY)


def folder_names(folder):
    return sorted(path.name for path in folder.iterdir())


def test_publishing_writes_both_files_with_the_exact_bytes(publication):
    csv_path, summary_path = publish_outputs(publication.folder, publication.table, publication.summary)

    assert csv_path == publication.folder / OUTREACH_FILE
    assert summary_path == publication.folder / SUMMARY_FILE
    assert csv_path.read_bytes() == publication.csv_bytes
    assert summary_path.read_bytes() == publication.summary_bytes


def test_publishing_replaces_the_previous_pair_and_keeps_no_copy_or_temp_file(publication):
    write_previous_pair(publication.folder)

    publish_outputs(publication.folder, publication.table, publication.summary)

    assert folder_names(publication.folder) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert (publication.folder / OUTREACH_FILE).read_bytes() == publication.csv_bytes
    assert (publication.folder / SUMMARY_FILE).read_bytes() == publication.summary_bytes


def test_a_missing_folder_is_created(tmp_path, publication):
    nested = tmp_path / "does" / "not" / "exist"

    publish_outputs(nested, publication.table, publication.summary)

    assert folder_names(nested) == sorted([OUTREACH_FILE, SUMMARY_FILE])


def test_leftover_temporary_files_are_overwritten(publication):
    publication.folder.mkdir()
    (publication.folder / (".tmp-" + OUTREACH_FILE)).write_bytes(b"junk")
    (publication.folder / (".tmp-" + SUMMARY_FILE)).write_bytes(b"junk")

    publish_outputs(publication.folder, publication.table, publication.summary)

    assert folder_names(publication.folder) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert (publication.folder / OUTREACH_FILE).read_bytes() == publication.csv_bytes


def test_the_csv_is_replaced_before_the_summary(publication, monkeypatch):
    real_replace = os.replace
    order = []

    def spy(source, target):
        order.append(Path(target).name)
        return real_replace(source, target)

    monkeypatch.setattr(outputs.os, "replace", spy)

    publish_outputs(publication.folder, publication.table, publication.summary)

    assert order == [OUTREACH_FILE, SUMMARY_FILE]


def assert_previous_pair_unchanged(folder):
    assert (folder / OUTREACH_FILE).read_bytes() == OLD_CSV
    assert (folder / SUMMARY_FILE).read_bytes() == OLD_SUMMARY
    assert folder_names(folder) == sorted([OUTREACH_FILE, SUMMARY_FILE])


@pytest.mark.parametrize("failing", [OUTREACH_FILE, SUMMARY_FILE], ids=["csv-temp", "summary-temp"])
def test_a_failed_temporary_write_changes_nothing(publication, monkeypatch, failing):
    write_previous_pair(publication.folder)
    real_write = outputs._write_file

    def failing_write(path, data):
        if path.name == ".tmp-" + failing:
            raise OSError("disk full")
        return real_write(path, data)

    monkeypatch.setattr(outputs, "_write_file", failing_write)

    with pytest.raises(OutputGenerationError, match="disk full"):
        publish_outputs(publication.folder, publication.table, publication.summary)

    assert_previous_pair_unchanged(publication.folder)


def test_a_failed_csv_replace_leaves_both_files_unchanged(publication, monkeypatch):
    write_previous_pair(publication.folder)

    def failing_replace(source, target):
        raise OSError("locked by another program")

    monkeypatch.setattr(outputs.os, "replace", failing_replace)

    with pytest.raises(OutputPublicationError) as exc_info:
        publish_outputs(publication.folder, publication.table, publication.summary)

    assert exc_info.value.csv_replaced is False
    assert exc_info.value.summary_replaced is False
    assert "locked by another program" in str(exc_info.value)
    assert_previous_pair_unchanged(publication.folder)


def fail_summary_replace(monkeypatch):
    real_replace = os.replace

    def replace(source, target):
        if Path(target).name == SUMMARY_FILE:
            raise OSError("summary locked")
        return real_replace(source, target)

    monkeypatch.setattr(outputs.os, "replace", replace)


def test_a_failed_summary_replace_after_the_csv_is_a_class_2b_failure(publication, monkeypatch):
    write_previous_pair(publication.folder)
    fail_summary_replace(monkeypatch)

    with pytest.raises(OutputPublicationError) as exc_info:
        publish_outputs(publication.folder, publication.table, publication.summary)

    error = exc_info.value
    message = str(error)
    assert error.csv_replaced is True
    assert error.summary_replaced is False
    assert (publication.folder / OUTREACH_FILE).read_bytes() == publication.csv_bytes
    assert (publication.folder / SUMMARY_FILE).read_bytes() == OLD_SUMMARY
    assert f"{OUTREACH_FILE} was replaced" in message
    assert f"{SUMMARY_FILE} was not replaced" in message
    assert "may be inconsistent" in message
    assert "not restored" in message
    assert "no copy" in message
    assert folder_names(publication.folder) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert isinstance(error.__cause__, OSError)


def test_the_same_failure_on_a_first_run_leaves_only_the_csv(publication, monkeypatch):
    fail_summary_replace(monkeypatch)

    with pytest.raises(OutputPublicationError) as exc_info:
        publish_outputs(publication.folder, publication.table, publication.summary)

    assert exc_info.value.csv_replaced is True
    assert exc_info.value.summary_replaced is False
    assert folder_names(publication.folder) == [OUTREACH_FILE]
    assert not (publication.folder / SUMMARY_FILE).exists()


def test_a_failing_cleanup_does_not_mask_the_original_error(publication, monkeypatch):
    write_previous_pair(publication.folder)

    def failing_replace(source, target):
        raise OSError("original failure")

    def failing_remove(path):
        raise OSError("cleanup failure")

    monkeypatch.setattr(outputs.os, "replace", failing_replace)
    monkeypatch.setattr(outputs.os, "remove", failing_remove)

    with pytest.raises(OutputPublicationError, match="original failure") as exc_info:
        publish_outputs(publication.folder, publication.table, publication.summary)

    assert "cleanup failure" not in str(exc_info.value)
    assert "original failure" in str(exc_info.value.__cause__)


def test_a_table_that_cannot_be_written_fails_before_any_file_is_touched(publication):
    write_previous_pair(publication.folder)

    with pytest.raises(OutputGenerationError):
        publish_outputs(publication.folder, publication.table.iloc[0:0], publication.summary)

    assert_previous_pair_unchanged(publication.folder)


# Source checks


def test_outputs_py_never_references_the_target_column_or_other_file_names():
    tree = ast.parse(inspect.getsource(outputs))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    assert TARGET_COLUMN not in names | attributes
    assert not any(TARGET_COLUMN in text for text in strings)
    file_names = {
        match
        for text in strings
        for match in re.findall(r"[\w.\-]+\.(?:csv|json|txt|log|tmp)\b", text)
    }
    assert file_names <= {OUTREACH_FILE, SUMMARY_FILE}
    assert file_names == {OUTREACH_FILE, SUMMARY_FILE}


# =============================================================================
# Section E: the scoring command
# =============================================================================

SCORING_INSTANT = datetime(2026, 10, 11, 12, 35, 0, tzinfo=timezone.utc)
COMMAND_RUN_ID = "20261011T123500Z"
FIRST_MODEL_INSTANT = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
SECOND_MODEL_INSTANT = datetime(2026, 10, 11, 12, 10, 0, tzinfo=timezone.utc)
SCORING_USAGE_ERROR_EXIT = 2


class FixedScoringDatetime(datetime):
    """Makes datetime.now() return SCORING_INSTANT so the run id is known."""

    @classmethod
    def now(cls, tz=None):
        return SCORING_INSTANT


class SharedConnection:
    """Wraps the shared test connection so main() cannot close it."""

    def __init__(self, connection):
        self._connection = connection
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        self.closed = True


@pytest.fixture(scope="module", autouse=True)
def real_folders_untouched():
    """The tests must never write into the real outputs/ or artifacts/."""
    folders = [REPO_ROOT / "outputs", REPO_ROOT / "artifacts" / "models", REPO_ROOT / "artifacts" / "runs"]
    before = {folder: sorted(path.name for path in folder.iterdir()) for folder in folders}
    yield
    after = {folder: sorted(path.name for path in folder.iterdir()) for folder in folders}
    assert after == before


@pytest.fixture(scope="module")
def command_model(conn):
    """A cheap real model: logistic regression on the real preprocessed
    training split, with the real generated schema."""
    inputs = prepare_model_inputs(conn)
    model = LogisticRegression(max_iter=1000).fit(inputs.X_train, inputs.y_train)
    return SimpleNamespace(
        preprocessor=inputs.preprocessor,
        model=model,
        schema=build_input_schema(read_feat_churn_types(conn), inputs.preprocessor),
        train_rows=len(inputs.y_train),
    )


def publish_model(models_dir, parts, instant) -> str:
    run_id = format_run_id(instant)
    manifest = build_manifest(
        run_id=run_id,
        instant=instant,
        schema=parts.schema,
        hyperparameters={},
        training_cv_metrics={**OUTPUT_METRICS, "training_rows": parts.train_rows},
        fingerprints={"config_hash": "a" * 64},
    )
    publish_run(
        models_dir, run_id, manifest, parts.preprocessor, parts.model, manifest["input_schema_hash"]
    )
    return run_id


def make_command_env(tmp_path, parts, model_instants=(FIRST_MODEL_INSTANT,)) -> SimpleNamespace:
    config = load_config()
    config["paths"] = {
        **config["paths"],
        "runs_dir": tmp_path / "runs",
        "models_dir": tmp_path / "models",
        "outputs_dir": tmp_path / "config_outputs_dir_not_used",
    }
    model_ids = [publish_model(config["paths"]["models_dir"], parts, i) for i in model_instants]
    return SimpleNamespace(
        config=config,
        model_ids=model_ids,
        models_dir=config["paths"]["models_dir"],
        outputs_dir=tmp_path / "outputs",
        run_dir=config["paths"]["runs_dir"] / "score" / COMMAND_RUN_ID,
    )


def patch_main(monkeypatch, env, conn) -> SharedConnection:
    """Point main() at the temporary folders, a fixed clock and the shared
    in-memory database."""
    shared = SharedConnection(conn)
    monkeypatch.setattr(score, "load_config", lambda: env.config)
    monkeypatch.setattr(score, "datetime", FixedScoringDatetime)
    monkeypatch.setattr(score, "DEFAULT_OUTPUTS_DIR", env.outputs_dir)
    monkeypatch.setattr(
        score, "duckdb", SimpleNamespace(connect=lambda *args, **kwargs: shared, Error=duckdb.Error)
    )
    return shared


def run_command(argv=(), code=1):
    with pytest.raises(SystemExit) as exc_info:
        score.main(list(argv))
    assert exc_info.value.code == code


def failure_env(tmp_path, monkeypatch, conn, parts, model_instants=(FIRST_MODEL_INSTANT,)):
    """An environment with a previous output pair already in place."""
    env = make_command_env(tmp_path, parts, model_instants)
    patch_main(monkeypatch, env, conn)
    write_previous_pair(env.outputs_dir)
    return env


def assert_no_metrics(env):
    assert not (env.run_dir / "metrics.json").exists()


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory, conn, command_model):
    """One complete successful run through main() on the reference-like
    population (n = 1,408, K = 0.10), with what was observed on the way."""
    tmp_path = tmp_path_factory.mktemp("reference")
    env = make_command_env(tmp_path, command_model)
    observed: dict = {}
    real_score_batch = score.score_batch
    real_write_metrics = tracking.TrackedRun.write_metrics

    def spy_score_batch(frame, loaded_run, k_frac):
        observed["population"] = frame.copy(deep=True)
        return real_score_batch(frame, loaded_run, k_frac)

    def spy_write_metrics(self, metrics):
        observed["outputs_at_metrics_write"] = sorted(p.name for p in env.outputs_dir.iterdir())
        return real_write_metrics(self, metrics)

    with pytest.MonkeyPatch.context() as patch:
        patch_main(patch, env, conn)
        patch.setattr(score, "score_batch", spy_score_batch)
        patch.setattr(tracking.TrackedRun, "write_metrics", spy_write_metrics)
        score.main([])

    csv_bytes = (env.outputs_dir / OUTREACH_FILE).read_bytes()
    summary = json.loads((env.outputs_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    return SimpleNamespace(
        env=env,
        observed=observed,
        csv_bytes=csv_bytes,
        rows=parse_csv(csv_bytes),
        summary=summary,
        schema=command_model.schema,
    )


# Outputs of the reference-like run


def test_reference_csv_columns_rows_and_ranking(reference_run):
    rows = reference_run.rows
    body = rows[1:]
    probabilities = [float(row[1]) for row in body]

    assert rows[0] == TABLE_COLUMNS
    assert len(body) == 1408
    assert len({row[0] for row in body}) == 1408
    assert [int(row[2]) for row in body] == list(range(1, 1409))
    assert all(a >= b for a, b in zip(probabilities, probabilities[1:]))
    assert all(
        body[i][0] <= body[i + 1][0] for i in range(1407) if probabilities[i] == probabilities[i + 1]
    )


def test_reference_flags_are_exactly_141_and_the_first_rows(reference_run):
    flags = [row[3] for row in reference_run.rows[1:]]

    assert set(flags) <= {"True", "False"}
    assert flags.count("True") == 141
    assert flags == ["True"] * 141 + ["False"] * (1408 - 141)


def test_reference_csv_has_no_empty_cell_and_lf_endings(reference_run):
    assert all(cell != "" for row in reference_run.rows for cell in row)
    assert b"\r" not in reference_run.csv_bytes
    assert reference_run.csv_bytes.endswith(b"\n")


def test_reference_summary_has_the_contract_keys_and_types(reference_run):
    summary = reference_run.summary

    assert list(summary) == SUMMARY_KEYS
    assert list(summary["model_health"]) == MODEL_HEALTH_KEYS
    assert list(summary["model_health"]["training_cv_metrics"]) == METRIC_KEYS
    assert isinstance(summary["customers_scored"], int) and not isinstance(summary["customers_scored"], bool)
    assert isinstance(summary["customers_flagged"], int)
    assert isinstance(summary["k"], float)
    assert summary["contract_version"] == "1"


def test_reference_summary_agrees_with_the_csv_and_the_run_folders(reference_run):
    env = reference_run.env
    summary = reference_run.summary
    manifest = json.loads((env.models_dir / env.model_ids[0] / "manifest.json").read_text(encoding="utf-8"))

    assert summary["customers_scored"] == len(reference_run.rows) - 1 == 1408
    assert summary["customers_flagged"] == [row[3] for row in reference_run.rows[1:]].count("True") == 141
    assert summary["k"] == 0.1
    assert summary["model_run_id"] == env.model_ids[0]
    assert (env.models_dir / summary["model_run_id"]).is_dir()
    assert summary["scoring_run_id"] == env.run_dir.name == COMMAND_RUN_ID
    assert env.run_dir.is_dir()
    assert summary["model_health"]["held_out_results_reference"] == "ADR-0014"
    for key in MODEL_HEALTH_KEYS:
        assert summary["model_health"][key] == manifest[key]


def test_the_frame_given_to_score_batch_has_only_the_id_and_the_schema_columns(reference_run):
    population = reference_run.observed["population"]
    expected_columns = ["customer_id", *[column["name"] for column in reference_run.schema["columns"]]]

    assert list(population.columns) == expected_columns
    assert len(population) == 1408
    assert TARGET_COLUMN not in population.columns
    for column in ("offer", "internet_type"):
        values = population[column].tolist()
        assert any(value is None for value in values)
        assert not any(value is pd.NA or (isinstance(value, float) and value != value) for value in values)


def test_the_tracking_folder_holds_only_config_log_and_operational_metrics(reference_run):
    run_dir = reference_run.env.run_dir

    assert sorted(path.name for path in run_dir.iterdir()) == ["config.json", "metrics.json", "run.log"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {
        "k": 0.1,
        "customers_scored": 1408,
        "customers_flagged": 141,
        "model_run_id": reference_run.env.model_ids[0],
    }


def test_metrics_are_written_only_after_both_outputs_are_published(reference_run):
    assert reference_run.observed["outputs_at_metrics_write"] == sorted([OUTREACH_FILE, SUMMARY_FILE])


def test_outputs_go_to_the_passed_folder_not_the_configured_one(reference_run):
    env = reference_run.env

    assert sorted(path.name for path in env.outputs_dir.iterdir()) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert not env.config["paths"]["outputs_dir"].exists()


def test_the_default_outputs_folder_is_the_repository_outputs_folder():
    default = inspect.signature(run_scoring).parameters["outputs_dir"].default

    assert default == REPO_ROOT / "outputs"


def test_run_scoring_writes_to_the_given_folder_and_returns_the_paths(
    tmp_path, conn, command_model
):
    env = make_command_env(tmp_path, command_model)

    result = run_scoring(env.config, conn, SCORING_INSTANT, outputs_dir=env.outputs_dir)

    assert isinstance(result, ScoringResult)
    assert result.run_id == COMMAND_RUN_ID
    assert result.model_run_id == env.model_ids[0]
    assert (result.customers_scored, result.customers_flagged) == (1408, 141)
    assert result.outreach_path == env.outputs_dir / OUTREACH_FILE
    assert result.summary_path == env.outputs_dir / SUMMARY_FILE
    assert not env.config["paths"]["outputs_dir"].exists()


def test_the_scoring_population_is_the_test_split_with_nullable_columns_kept_as_none(
    conn, command_model
):
    population = select_scoring_population(conn, command_model.schema)
    test_split = split_feat_churn(conn).test

    assert len(population) == 1408
    assert set(population["customer_id"]) == set(test_split["customer_id"])
    assert TARGET_COLUMN not in population.columns
    assert any(value is None for value in population["offer"].tolist())


# Model selection with --run-id


def test_an_explicit_run_id_selects_that_model_and_the_default_selects_the_newest(
    tmp_path, conn, command_model
):
    env = make_command_env(tmp_path, command_model, (FIRST_MODEL_INSTANT, SECOND_MODEL_INSTANT))
    older, newer = env.model_ids

    default_result = run_scoring(env.config, conn, SCORING_INSTANT, outputs_dir=env.outputs_dir)
    explicit_result = run_scoring(
        env.config,
        conn,
        SCORING_INSTANT.replace(minute=40),
        run_id=older,
        outputs_dir=env.outputs_dir,
    )

    assert default_result.model_run_id == newer
    assert explicit_result.model_run_id == older
    summary = json.loads((env.outputs_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["model_run_id"] == older


# Retention


def test_a_successful_run_replaces_the_previous_pair(tmp_path, monkeypatch, conn, command_model):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    score.main([])

    assert sorted(path.name for path in env.outputs_dir.iterdir()) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert (env.outputs_dir / OUTREACH_FILE).read_bytes() != OLD_CSV
    assert (env.outputs_dir / SUMMARY_FILE).read_bytes() != OLD_SUMMARY
    assert sorted(path.name for path in env.run_dir.iterdir()) == ["config.json", "metrics.json", "run.log"]


# Class 1


def test_class_1_a_damaged_population_publishes_nothing(
    tmp_path, monkeypatch, conn, command_model, caplog
):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)
    real_select = score.select_scoring_population

    def damaged(connection, schema):
        population = real_select(connection, schema)
        column = population.columns.get_loc("customer_id")
        population.iloc[1, column] = population.iloc[0, column]
        return population

    monkeypatch.setattr(score, "select_scoring_population", damaged)

    with caplog.at_level(logging.ERROR):
        run_command()

    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)
    assert "class 1" in caplog.text
    assert "duplicate customer_id" in caplog.text


# Class 2a


def run_2a(caplog):
    with caplog.at_level(logging.ERROR):
        run_command()
    assert "class 2a" in caplog.text
    assert "before anything was published" in caplog.text


def test_class_2a_no_valid_model(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model, model_instants=())

    run_2a(caplog)

    assert "make train" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)


@pytest.mark.parametrize("requested", ["20260101T000000Z", "../20261011T120000Z", ""], ids=["missing", "bad-format", "empty"])
def test_class_2a_an_explicit_run_id_that_is_missing_or_invalid_has_no_fallback(
    tmp_path, monkeypatch, conn, command_model, caplog, requested
):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    with caplog.at_level(logging.ERROR):
        run_command(["--run-id", requested])

    assert "class 2a" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)


def test_class_2a_an_existing_tracking_folder_stops_the_run_at_the_start(
    tmp_path, monkeypatch, conn, command_model, caplog
):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)
    env.run_dir.mkdir(parents=True)
    (env.run_dir / "keep.txt").write_text("keep", encoding="utf-8")

    run_2a(caplog)

    assert_previous_pair_unchanged(env.outputs_dir)
    assert sorted(path.name for path in env.run_dir.iterdir()) == ["keep.txt"]


def test_class_2a_a_scoring_error_from_a_scripted_model(
    tmp_path, monkeypatch, conn, command_model, caplog
):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)
    real_load = score.load_run

    def scripted_load(models_dir, run_id=None):
        loaded = real_load(models_dir, run_id)
        return dataclasses.replace(
            loaded, preprocessor=ScriptedPreprocessor(), model=ScriptedModel([np.nan] * 1408)
        )

    monkeypatch.setattr(score, "load_run", scripted_load)

    run_2a(caplog)

    assert "ScoringError" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)


def test_class_2a_a_temporary_file_write_failure(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    def failing_write(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(outputs, "_write_file", failing_write)

    run_2a(caplog)

    assert "OutputGenerationError" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)


def test_class_2a_an_unexpected_error_while_scoring(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    def broken_score_batch(frame, loaded_run, k_frac):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(score, "score_batch", broken_score_batch)

    run_2a(caplog)

    assert "RuntimeError" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)
    assert_no_metrics(env)


def test_class_2a_a_database_error(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    def broken_connect(*args, **kwargs):
        raise duckdb.IOException("cannot open the database")

    monkeypatch.setattr(
        score, "duckdb", SimpleNamespace(connect=broken_connect, Error=duckdb.Error)
    )

    run_2a(caplog)

    assert_previous_pair_unchanged(env.outputs_dir)
    assert not env.run_dir.exists()


def test_class_2a_a_configuration_error(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    def broken_config():
        raise ConfigValidationError("the configuration is broken")

    monkeypatch.setattr(score, "load_config", broken_config)

    run_2a(caplog)

    assert "the configuration is broken" in caplog.text
    assert_previous_pair_unchanged(env.outputs_dir)


# Class 2b


def test_class_2b_a_failed_summary_replacement(tmp_path, monkeypatch, conn, command_model, caplog):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)
    fail_summary_replace(monkeypatch)

    with caplog.at_level(logging.ERROR):
        run_command()

    assert "class 2b" in caplog.text
    assert f"{OUTREACH_FILE} was replaced" in caplog.text
    assert f"{SUMMARY_FILE} was not replaced" in caplog.text
    assert "may be inconsistent" in caplog.text
    new_csv = (env.outputs_dir / OUTREACH_FILE).read_bytes()
    assert new_csv != OLD_CSV
    assert len(parse_csv(new_csv)) == 1409
    assert (env.outputs_dir / SUMMARY_FILE).read_bytes() == OLD_SUMMARY
    assert sorted(path.name for path in env.outputs_dir.iterdir()) == sorted([OUTREACH_FILE, SUMMARY_FILE])
    assert_no_metrics(env)


# Class 3


def test_class_3_a_tracking_failure_after_publication_keeps_both_files(
    tmp_path, monkeypatch, conn, command_model, caplog
):
    env = failure_env(tmp_path, monkeypatch, conn, command_model)

    def broken_write_metrics(self, metrics):
        raise TrackingError("disk full")

    monkeypatch.setattr(tracking.TrackedRun, "write_metrics", broken_write_metrics)

    with caplog.at_level(logging.ERROR):
        run_command()

    assert "class 3" in caplog.text
    assert "outputs were generated" in caplog.text
    assert "tracking failed" in caplog.text
    assert COMMAND_RUN_ID in caplog.text
    assert "disk full" in caplog.text
    assert len(parse_csv((env.outputs_dir / OUTREACH_FILE).read_bytes())) == 1409
    assert (env.outputs_dir / SUMMARY_FILE).read_bytes() != OLD_SUMMARY
    assert json.loads((env.outputs_dir / SUMMARY_FILE).read_text(encoding="utf-8"))["scoring_run_id"] == COMMAND_RUN_ID
    assert_no_metrics(env)


def test_run_scoring_raises_the_class_3_error_with_the_scoring_run_id(
    tmp_path, conn, command_model, monkeypatch
):
    env = make_command_env(tmp_path, command_model)

    def broken_write_metrics(self, metrics):
        raise TrackingError("disk full")

    monkeypatch.setattr(tracking.TrackedRun, "write_metrics", broken_write_metrics)

    with pytest.raises(PostPublicationTrackingError) as exc_info:
        run_scoring(env.config, conn, SCORING_INSTANT, outputs_dir=env.outputs_dir)

    assert exc_info.value.scoring_run_id == COMMAND_RUN_ID
    assert isinstance(exc_info.value.__cause__, TrackingError)


# main: exit codes and messages for each error type


@pytest.mark.parametrize(
    "error,keywords",
    [
        (InputValidationError("column x is broken"), ["class 1", "column x is broken"]),
        (
            OutputPublicationError(
                "Publication failed part way: a.csv was replaced, b.json was not replaced",
                csv_replaced=True,
                summary_replaced=False,
            ),
            ["class 2b", "a.csv was replaced", "b.json was not replaced"],
        ),
        (
            PostPublicationTrackingError(
                COMMAND_RUN_ID,
                f"The outputs were generated and published, but tracking failed for scoring run {COMMAND_RUN_ID}: disk full",
            ),
            ["class 3", "outputs were generated", "tracking failed", COMMAND_RUN_ID],
        ),
        (TrackingError("cannot create the run folder"), ["class 2a", "before anything was published"]),
        (ModelLoadError("no valid model"), ["class 2a", "before anything was published"]),
        (ScoringError("bad probabilities"), ["class 2a", "before anything was published"]),
        (OutputGenerationError("cannot build"), ["class 2a", "before anything was published"]),
        (duckdb.IOException("db broken"), ["class 2a", "before anything was published"]),
        (RuntimeError("boom"), ["class 2a", "before anything was published", "RuntimeError"]),
    ],
    ids=[
        "class-1",
        "class-2b",
        "class-3",
        "tracking-start",
        "model-load",
        "scoring",
        "output-generation",
        "database",
        "unexpected",
    ],
)
def test_main_maps_each_error_to_its_class_and_exits_non_zero(
    tmp_path, monkeypatch, conn, command_model, caplog, error, keywords
):
    env = make_command_env(tmp_path, command_model)
    shared = patch_main(monkeypatch, env, conn)

    def raising_run_scoring(*args, **kwargs):
        raise error

    monkeypatch.setattr(score, "run_scoring", raising_run_scoring)

    with caplog.at_level(logging.ERROR):
        run_command()

    for keyword in keywords:
        assert keyword in caplog.text, keyword
    assert shared.closed is True


def test_main_succeeds_without_exiting_and_logs_the_result(
    tmp_path, monkeypatch, conn, command_model, caplog
):
    env = make_command_env(tmp_path, command_model)
    shared = patch_main(monkeypatch, env, conn)
    result = ScoringResult(
        run_id=COMMAND_RUN_ID,
        model_run_id=env.model_ids[0],
        customers_scored=1408,
        customers_flagged=141,
        outreach_path=env.outputs_dir / OUTREACH_FILE,
        summary_path=env.outputs_dir / SUMMARY_FILE,
    )
    monkeypatch.setattr(score, "run_scoring", lambda *args, **kwargs: result)

    with caplog.at_level(logging.INFO):
        score.main([])

    assert shared.closed is True
    assert "Scoring complete" in caplog.text
    assert COMMAND_RUN_ID in caplog.text
    assert env.model_ids[0] in caplog.text
    assert "1408" in caplog.text and "141" in caplog.text


def test_main_passes_the_run_id_option_to_run_scoring(tmp_path, monkeypatch, conn, command_model):
    env = make_command_env(tmp_path, command_model)
    patch_main(monkeypatch, env, conn)
    seen = {}

    def spy_run_scoring(config, connection, instant, *, run_id=None, outputs_dir=None):
        seen.update(run_id=run_id, outputs_dir=outputs_dir, instant=instant)
        return ScoringResult(COMMAND_RUN_ID, "x", 1, 1, Path("a"), Path("b"))

    monkeypatch.setattr(score, "run_scoring", spy_run_scoring)

    score.main(["--run-id", "20261011T120000Z"])
    assert seen == {
        "run_id": "20261011T120000Z",
        "outputs_dir": env.outputs_dir,
        "instant": SCORING_INSTANT,
    }
    seen.clear()
    score.main([])
    assert seen["run_id"] is None


@pytest.mark.parametrize("argv", [["--bogus"], ["--run-id"]], ids=["unknown-option", "missing-value"])
def test_a_usage_error_exits_with_argparses_own_code(argv):
    with pytest.raises(SystemExit) as exc_info:
        score.main(argv)

    assert exc_info.value.code == SCORING_USAGE_ERROR_EXIT


# =============================================================================
# Section F: ranking: top_k_count and risk tiers
# =============================================================================

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


# Risk tiers


def tier_counts(n_rows, k_frac, medium_edge):
    tiers = assign_risk_tiers(pd.Series(np.arange(1, n_rows + 1)), k_frac, medium_edge)
    return (
        int((tiers == "High").sum()),
        int((tiers == "Medium").sum()),
        int((tiers == "Low").sum()),
    )


def test_reference_population_tier_counts_and_boundaries():
    tiers = assign_risk_tiers(pd.Series(np.arange(1, 1409)), 0.10, 0.25)

    assert tiers.value_counts().to_dict() == {"Low": 1056, "Medium": 211, "High": 141}
    assert tiers[141 - 1] == "High"
    assert tiers[142 - 1] == "Medium"
    assert tiers[352 - 1] == "Medium"
    assert tiers[353 - 1] == "Low"


@pytest.mark.parametrize(
    "n_rows,expected",
    [(1, (1, 0, 0)), (2, (1, 0, 1)), (6, (1, 1, 4)), (25, (2, 4, 19))],
)
def test_small_populations_follow_the_rule_by_hand(n_rows, expected):
    assert tier_counts(n_rows, 0.10, 0.25) == expected


def test_an_edge_of_one_leaves_no_low_tier():
    assert tier_counts(10, 0.10, 1.0) == (1, 9, 0)


@pytest.mark.parametrize("k_frac,medium_edge", [(0.10, 0.25), (0.05, 0.30), (0.20, 0.21), (0.50, 1.0), (0.01, 0.02)])
@pytest.mark.parametrize("n_rows", [*range(1, 201), 1408, 5629, 10000])
def test_tier_grid_counts_add_up_and_tiers_are_contiguous(n_rows, k_frac, medium_edge):
    ranks = pd.Series(np.arange(1, n_rows + 1))

    tiers = assign_risk_tiers(ranks, k_frac, medium_edge)

    high = int((tiers == "High").sum())
    medium = int((tiers == "Medium").sum())
    low = int((tiers == "Low").sum())
    assert high + medium + low == n_rows
    assert high == int((ranks <= top_k_count(n_rows, k_frac)).sum())
    assert high <= high + medium == top_k_count(n_rows, medium_edge)
    assert list(tiers) == ["High"] * high + ["Medium"] * medium + ["Low"] * low


def test_tiers_do_not_depend_on_row_order_and_keep_the_index():
    ordered = assign_risk_tiers(pd.Series(np.arange(1, 41)), 0.10, 0.25)
    shuffled_values = np.random.default_rng(3).permutation(np.arange(1, 41))
    shuffled = pd.Series(shuffled_values, index=[f"row{i}" for i in range(40)])
    original = shuffled.copy(deep=True)

    tiers = assign_risk_tiers(shuffled, 0.10, 0.25)

    assert list(tiers.index) == list(shuffled.index)
    assert [tiers.iloc[i] for i in range(40)] == [ordered[value - 1] for value in shuffled_values]
    pd.testing.assert_series_equal(shuffled, original)


def test_tiers_are_object_dtype_with_only_the_three_labels():
    tiers = assign_risk_tiers(pd.Series(np.arange(1, 101)), 0.10, 0.25)

    assert tiers.dtype == object
    assert set(tiers) == {"High", "Medium", "Low"}


@pytest.mark.parametrize(
    "k_frac,medium_edge",
    [
        (0.25, 0.25),
        (0.30, 0.25),
        (0.10, 1.5),
        (0, 0.25),
        (-0.1, 0.25),
        (True, 0.25),
        (0.10, True),
        (None, 0.25),
        (0.10, None),
        ("0.1", 0.25),
        (0.10, "0.25"),
        (float("nan"), 0.25),
        (0.10, float("nan")),
        (float("inf"), 0.25),
        (0.10, float("inf")),
    ],
    ids=repr,
)
def test_an_invalid_pair_is_rejected_and_the_message_names_both_values(k_frac, medium_edge):
    with pytest.raises(ValueError) as exc_info:
        assign_risk_tiers(pd.Series([1, 2, 3]), k_frac, medium_edge)

    message = str(exc_info.value)
    assert repr(k_frac) in message
    assert repr(medium_edge) in message


@pytest.mark.parametrize(
    "ranks,problem",
    [
        (pd.Series([], dtype="int64"), "empty"),
        (pd.Series([1, 2, 2, 4]), "duplicate"),
        (pd.Series([0, 1, 2, 3]), "start at 1"),
        (pd.Series([1, 2, 4, 5]), "gap"),
        (pd.Series([1.0, 2.0, 3.0]), "integer dtype"),
        (pd.Series([True, False]), "integer dtype"),
        (pd.Series(["1", "2", "3"]), "integer dtype"),
        ([1, 2, 3], "Series"),
        (np.arange(1, 4), "Series"),
        (pd.Series([1, 2, None], dtype="Int64"), "missing"),
    ],
    ids=[
        "empty",
        "duplicate",
        "zero_based",
        "gap",
        "float",
        "bool",
        "text",
        "list",
        "numpy_array",
        "missing_values",
    ],
)
def test_invalid_ranks_are_rejected(ranks, problem):
    with pytest.raises(ValueError, match=problem):
        assign_risk_tiers(ranks, 0.10, 0.25)


# =============================================================================
# Section G: drivers
# =============================================================================


@pytest.fixture(scope="module")
def driver_data(conn):
    """The real preprocessor and schema, a small calibrated XGBoost model
    fitted on the real training rows, and the real scoring population."""
    inputs = prepare_model_inputs(conn)
    schema = build_input_schema(read_feat_churn_types(conn), inputs.preprocessor)
    model = CalibratedClassifierCV(
        xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=0),
        method="sigmoid",
        cv=5,
    ).fit(inputs.X_train, inputs.y_train.astype(int))
    names = [column["name"] for column in schema["columns"]]
    population = select_scoring_population(conn, schema)
    return SimpleNamespace(
        run=make_run(schema, inputs.preprocessor, model),
        names=names,
        inputs=population[names].reset_index(drop=True),
        ids=[str(value) for value in population[schema["identifier_column"]]],
    )


@pytest.fixture(scope="module")
def driver_contributions(driver_data):
    return averaged_contributions(driver_data.run, driver_data.inputs, driver_data.ids)


def dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def run_with_folds(run, folds):
    model = copy.copy(run.model)
    model.calibrated_classifiers_ = folds
    return dataclasses.replace(run, model=model)


def run_with_estimator_at(run, position, estimator):
    folds = list(run.model.calibrated_classifiers_)
    folds[position] = SimpleNamespace(estimator=estimator)
    return run_with_folds(run, folds)


def run_with_schema_columns(run, columns):
    return dataclasses.replace(run, schema={**run.schema, "columns": columns})


def run_without_owner_for_one_encoded_column(run):
    preprocessor = copy.deepcopy(run.preprocessor)
    original_names = list(preprocessor.get_feature_names_out())
    encoder = {name: step for name, step, _ in preprocessor.transformers_}["categorical"]
    encoder.categories_[0] = encoder.categories_[0][:-1]
    preprocessor.get_feature_names_out = lambda *args, **kwargs: original_names
    return dataclasses.replace(run, preprocessor=preprocessor)


def narrow_estimator():
    rng = np.random.default_rng(0)
    return xgb.XGBClassifier(n_estimators=2).fit(rng.random((30, 3)), [0, 1] * 15)


def run_with_transformers(run, rewrite):
    """A copy of the run whose preprocessor has its fitted transformer entries
    rewritten. The reported output names stay those of the original, so only
    the column mapping is affected."""
    preprocessor = copy.deepcopy(run.preprocessor)
    original_names = list(preprocessor.get_feature_names_out())
    preprocessor.transformers_ = rewrite(list(preprocessor.transformers_))
    preprocessor.get_feature_names_out = lambda *args, **kwargs: original_names
    return dataclasses.replace(run, preprocessor=preprocessor)


def entry_named(entries, label):
    return next(entry for entry in entries if entry[0] == label)


def with_entry_changed(label, *, transformer=None, columns=None):
    def rewrite(entries):
        return [
            (
                name,
                transformer if transformer is not None and name == label else step,
                columns if columns is not None and name == label else selected,
            )
            for name, step, selected in entries
        ]

    return rewrite


def with_encoder_changed(change):
    def rewrite(entries):
        change(entry_named(entries, "categorical")[1])
        return entries

    return rewrite


def with_two_encoders(entries):
    encoder = entry_named(entries, "categorical")[1]
    pipeline = Pipeline([("first", encoder), ("second", copy.deepcopy(encoder))])
    return with_entry_changed("categorical", transformer=pipeline)(entries)


def with_imputer_indicator(entries):
    entry_named(entries, "offer")[1].steps[0][1].add_indicator = True
    return entries


def run_without_feature_names(run):
    damaged = run_with_transformers(run, lambda entries: entries)
    del damaged.preprocessor.feature_names_in_
    return damaged


def set_attribute(name, value):
    return with_encoder_changed(lambda encoder: setattr(encoder, name, value))


def drop_last_learned_column(encoder):
    encoder.categories_ = encoder.categories_[:-1]


# Structure check


def test_the_structure_check_passes_and_maps_every_encoded_column(driver_data):
    owners = encoded_column_owners(driver_data.run)
    encoded_names = list(driver_data.run.preprocessor.get_feature_names_out())

    assert len(owners) == len(encoded_names)
    assert set(owners) == set(driver_data.names)
    assert all(owner in encoded_name for owner, encoded_name in zip(owners, encoded_names))
    assert EXPECTED_FOLD_MODELS == 5


STRUCTURE_FAILURES = {
    "not-calibrated": (
        lambda run: dataclasses.replace(run, model=LogisticRegression()),
        "not a CalibratedClassifierCV",
    ),
    "four-calibrated-classifiers": (
        lambda run: run_with_folds(run, list(run.model.calibrated_classifiers_)[:4]),
        "4 calibrated classifiers, expected 5",
    ),
    "estimator-not-xgboost": (
        lambda run: run_with_estimator_at(run, 1, LogisticRegression()),
        "does not wrap an XGBClassifier",
    ),
    "preprocessor-not-a-column-transformer": (
        lambda run: dataclasses.replace(run, preprocessor=object()),
        "not a fitted ColumnTransformer",
    ),
    "preprocessor-not-fitted": (
        lambda run: dataclasses.replace(run, preprocessor=build_preprocessing_pipeline()),
        "not a fitted ColumnTransformer",
    ),
    "estimator-width-differs": (
        lambda run: run_with_estimator_at(run, 2, narrow_estimator()),
        "takes 3 input features",
    ),
    "schema-input-owns-nothing": (
        lambda run: run_with_schema_columns(
            run, [*run.schema["columns"], {"name": "ghost", "kind": "decimal", "nullable": False}]
        ),
        "own no encoded column",
    ),
    "owner-not-in-schema": (
        lambda run: run_with_schema_columns(run, run.schema["columns"][1:]),
        "not in the schema",
    ),
    "encoded-column-without-owner": (
        run_without_owner_for_one_encoded_column,
        "has no owner",
    ),
    "model-not-fitted": (
        lambda run: dataclasses.replace(run, model=CalibratedClassifierCV(xgb.XGBClassifier())),
        "the model is not fitted",
    ),
    "xgboost-estimator-not-fitted": (
        lambda run: run_with_estimator_at(run, 3, xgb.XGBClassifier()),
        "XGBClassifier of calibrated classifier 3 is not fitted",
    ),
    "preprocessor-without-column-names": (
        run_without_feature_names,
        "with column names",
    ),
    "transformer-is-an-unknown-string": (
        lambda run: run_with_transformers(
            run, with_entry_changed("categorical", transformer="bogus")
        ),
        "'categorical' is 'bogus', which cannot be mapped",
    ),
    "column-selection-is-a-slice": (
        lambda run: run_with_transformers(
            run, with_entry_changed("categorical", columns=slice(0, 2))
        ),
        "'categorical' selects columns in an unsupported way",
    ),
    "column-selection-has-an-unsupported-element": (
        lambda run: run_with_transformers(run, with_entry_changed("categorical", columns=[2.5])),
        "'categorical' selects columns in an unsupported way",
    ),
    "column-position-out-of-range": (
        lambda run: run_with_transformers(run, with_entry_changed("categorical", columns=[999])),
        "column position 999, out of range",
    ),
    "two-one-hot-encoders": (
        lambda run: run_with_transformers(run, with_two_encoders),
        "more than one one-hot encoder",
    ),
    "encoder-drops-a-category": (
        lambda run: run_with_transformers(run, set_attribute("drop_idx_", np.array([0, 0, 0]))),
        "drops a category",
    ),
    "encoder-groups-by-min-frequency": (
        lambda run: run_with_transformers(run, set_attribute("min_frequency", 5)),
        "groups infrequent categories",
    ),
    "encoder-groups-by-max-categories": (
        lambda run: run_with_transformers(run, set_attribute("max_categories", 4)),
        "groups infrequent categories",
    ),
    "encoder-learned-fewer-columns": (
        lambda run: run_with_transformers(run, with_encoder_changed(drop_last_learned_column)),
        "learned 2 column(s) for 3 input column(s)",
    ),
    "imputer-adds-indicator-columns": (
        lambda run: run_with_transformers(run, with_imputer_indicator),
        "adds indicator columns",
    ),
    "step-of-another-kind": (
        lambda run: run_with_transformers(
            run,
            with_entry_changed(
                "categorical", transformer=Pipeline([("pass", FunctionTransformer())])
            ),
        ),
        "contains FunctionTransformer, which cannot be mapped",
    ),
}


@pytest.mark.parametrize(
    "damage,fragment", list(STRUCTURE_FAILURES.values()), ids=list(STRUCTURE_FAILURES)
)
def test_the_structure_check_names_each_broken_condition(driver_data, damage, fragment):
    with pytest.raises(DriverError) as exc_info:
        encoded_column_owners(damage(driver_data.run))

    assert fragment in str(exc_info.value)


def test_the_damaged_variants_leave_the_shared_run_unchanged(driver_data):
    for damage, _ in STRUCTURE_FAILURES.values():
        damage(driver_data.run)

    assert len(driver_data.run.model.calibrated_classifiers_) == 5
    assert len(encoded_column_owners(driver_data.run)) == len(
        driver_data.run.preprocessor.get_feature_names_out()
    )


def test_a_drop_entry_contributes_no_owner(driver_data):
    run = run_with_transformers(
        driver_data.run, lambda entries: [*entries, ("unused", "drop", ["gender"])]
    )

    assert encoded_column_owners(run) == encoded_column_owners(driver_data.run)


def test_a_passthrough_entry_maps_each_selected_column_to_itself(driver_data):
    run = run_with_transformers(
        driver_data.run,
        lambda entries: [
            (name, "passthrough" if name == "remainder" else step, selected)
            for name, step, selected in entries
        ],
    )

    assert encoded_column_owners(run) == encoded_column_owners(driver_data.run)


def test_moving_one_input_changes_only_the_encoded_columns_it_owns(driver_data):
    run = driver_data.run
    owners = np.array(encoded_column_owners(run))
    base = dense(run.preprocessor.transform(driver_data.inputs))

    for name in driver_data.names:
        moved = driver_data.inputs.copy()
        moved[name] = np.roll(driver_data.inputs[name].to_numpy(), 1)
        changed = np.flatnonzero((dense(run.preprocessor.transform(moved)) != base).any(axis=0))

        assert set(owners[changed]) <= {name}, name
        if driver_data.inputs[name].nunique(dropna=False) > 1:
            assert len(changed) > 0, name


# Averaged contributions


def test_contributions_have_the_expected_shape_index_columns_and_are_finite(
    driver_data, driver_contributions
):
    assert driver_contributions.shape == (len(driver_data.ids), len(driver_data.names))
    assert list(driver_contributions.index) == driver_data.ids
    assert list(driver_contributions.columns) == driver_data.names
    assert (driver_contributions.dtypes == np.float64).all()
    assert np.isfinite(driver_contributions.to_numpy()).all()


def test_owned_sums_plus_baseline_equal_the_margin_and_the_output_is_the_fold_mean(
    driver_data, driver_contributions
):
    run = driver_data.run
    owners = np.array(encoded_column_owners(run))
    matrix = dense(run.preprocessor.transform(driver_data.inputs))
    fold_sums = []

    for fold in run.model.calibrated_classifiers_:
        booster = fold.estimator.get_booster()
        dmatrix = xgb.DMatrix(matrix, feature_names=booster.feature_names)
        contributions = booster.predict(dmatrix, pred_contribs=True).astype("float64")
        margin = booster.predict(dmatrix, output_margin=True).astype("float64")
        sums = np.column_stack(
            [contributions[:, :-1][:, owners == name].sum(axis=1) for name in driver_data.names]
        )
        assert np.abs(sums.sum(axis=1) + contributions[:, -1] - margin).max() < 1e-3
        fold_sums.append(sums)

    np.testing.assert_allclose(
        driver_contributions.to_numpy(), np.mean(fold_sums, axis=0), rtol=0, atol=1e-6
    )


def test_a_customer_scored_alone_matches_the_same_customer_in_the_batch(
    driver_data, driver_contributions
):
    for position in (0, 7, 500, len(driver_data.ids) - 1):
        alone = averaged_contributions(
            driver_data.run,
            driver_data.inputs.iloc[[position]],
            [driver_data.ids[position]],
        )

        np.testing.assert_allclose(
            alone.to_numpy()[0], driver_contributions.iloc[position].to_numpy(), rtol=0, atol=1e-5
        )


def test_shuffled_rows_give_the_same_values_per_customer(driver_data, driver_contributions):
    order = np.random.default_rng(2).permutation(len(driver_data.ids))
    shuffled = averaged_contributions(
        driver_data.run,
        driver_data.inputs.iloc[order].reset_index(drop=True),
        [driver_data.ids[i] for i in order],
    )

    np.testing.assert_allclose(
        shuffled.loc[driver_data.ids].to_numpy(),
        driver_contributions.to_numpy(),
        rtol=0,
        atol=1e-5,
    )


def test_repeating_the_call_gives_identical_numbers_and_leaves_the_inputs_unchanged(
    driver_data, driver_contributions
):
    original = driver_data.inputs.copy(deep=True)

    again = averaged_contributions(driver_data.run, driver_data.inputs, driver_data.ids)

    pd.testing.assert_frame_equal(again, driver_contributions, check_exact=True)
    pd.testing.assert_frame_equal(driver_data.inputs, original)


CONTRIBUTION_CALLER_MISTAKES = {
    "missing-column": lambda d: (d.inputs.drop(columns=[d.names[0]]), d.ids),
    "extra-column": lambda d: (d.inputs.assign(extra=1), d.ids),
    "wrong-order": lambda d: (d.inputs[list(reversed(d.names))], d.ids),
    "too-few-ids": lambda d: (d.inputs, d.ids[:-1]),
    "too-many-ids": lambda d: (d.inputs, [*d.ids, "extra"]),
    "duplicate-ids": lambda d: (d.inputs, [d.ids[0], *d.ids[:-1]]),
}


@pytest.mark.parametrize(
    "build", list(CONTRIBUTION_CALLER_MISTAKES.values()), ids=list(CONTRIBUTION_CALLER_MISTAKES)
)
def test_contributions_reject_caller_mistakes(driver_data, build):
    inputs, ids = build(driver_data)

    with pytest.raises(ValueError):
        averaged_contributions(driver_data.run, inputs, ids)


def test_a_non_finite_contribution_from_a_booster_raises(driver_data):
    model = copy.deepcopy(driver_data.run.model)
    booster = model.calibrated_classifiers_[2].estimator.get_booster()
    real_predict = booster.predict

    def predict_with_a_nan(*args, **kwargs):
        output = real_predict(*args, **kwargs).copy()
        output[3, 0] = np.nan
        return output

    booster.predict = predict_with_a_nan
    run = dataclasses.replace(driver_data.run, model=model)

    with pytest.raises(DriverError, match="the model produced a contribution that is not finite"):
        averaged_contributions(run, driver_data.inputs.iloc[:20], driver_data.ids[:20])


def test_contributions_refuse_a_run_whose_structure_is_broken(driver_data):
    broken = run_with_folds(driver_data.run, list(driver_data.run.model.calibrated_classifiers_)[:4])

    with pytest.raises(DriverError, match="calibrated classifiers"):
        averaged_contributions(broken, driver_data.inputs, driver_data.ids)


# Top-driver selection

FEATURES = ["a", "b", "c", "d"]


def contribution_frame(rows, index=None, columns=FEATURES):
    return pd.DataFrame(rows, index=index, columns=columns, dtype="float64")


def test_the_top_three_are_by_absolute_size_and_a_large_negative_beats_a_small_positive():
    result = select_top_drivers(contribution_frame([[0.1, -0.9, 0.3, 0.2]]))

    assert result.iloc[0].tolist() == [
        "b", DIRECTION_LOWERS, "c", DIRECTION_RAISES, "d", DIRECTION_RAISES,
    ]


def test_drivers_are_strongest_first_with_the_right_direction():
    result = select_top_drivers(contribution_frame([[0.2, 0.5, -0.4, 0.1], [-0.3, -0.1, 0.6, 0.2]]))

    assert result.iloc[0].tolist() == [
        "b", DIRECTION_RAISES, "c", DIRECTION_LOWERS, "a", DIRECTION_RAISES,
    ]
    assert result.iloc[1].tolist() == [
        "c", DIRECTION_RAISES, "a", DIRECTION_LOWERS, "d", DIRECTION_RAISES,
    ]


def test_a_tie_in_absolute_size_is_ordered_by_column_and_keeps_each_direction():
    result = select_top_drivers(contribution_frame([[0.5, -0.5, 0.1, 0.2]]))

    assert result.iloc[0].tolist() == [
        "a", DIRECTION_RAISES, "b", DIRECTION_LOWERS, "d", DIRECTION_RAISES,
    ]


def test_a_tie_across_the_third_place_keeps_the_earlier_column():
    result = select_top_drivers(contribution_frame([[0.9, 0.5, -0.5, 0.5]]))

    assert result.iloc[0].tolist() == [
        "a", DIRECTION_RAISES, "b", DIRECTION_RAISES, "c", DIRECTION_LOWERS,
    ]


def test_a_zero_outside_the_top_three_is_fine():
    result = select_top_drivers(contribution_frame([[0.4, 0.3, 0.2, 0.0]]))

    assert result.iloc[0].tolist() == [
        "a", DIRECTION_RAISES, "b", DIRECTION_RAISES, "c", DIRECTION_RAISES,
    ]


@pytest.mark.parametrize(
    "row",
    [[0.4, 0.3, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0]],
    ids=["third-is-zero", "only-one-non-zero", "all-zero"],
)
def test_a_selected_zero_raises_and_names_the_customer_and_the_feature(row):
    frame = contribution_frame([[0.9, 0.8, 0.7, 0.6], row], index=["C1", "C7"])

    with pytest.raises(DriverError) as exc_info:
        select_top_drivers(frame)

    message = str(exc_info.value)
    assert "'C7'" in message
    assert "exactly zero" in message
    assert any(f"'{name}'" in message for name in FEATURES)


def test_a_selected_zero_names_the_feature_that_was_selected():
    with pytest.raises(DriverError) as exc_info:
        select_top_drivers(contribution_frame([[0.4, 0.3, 0.0, 0.0]], index=["C7"]))

    assert "'c'" in str(exc_info.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("position", [0, 3], ids=["selected-column", "other-column"])
def test_a_non_finite_value_anywhere_raises(bad, position):
    row = [0.9, 0.8, 0.7, 0.6]
    row[position] = bad

    with pytest.raises(DriverError, match="not finite"):
        select_top_drivers(contribution_frame([row]))


@pytest.mark.parametrize("width", [0, 1, 2])
def test_fewer_than_three_columns_raises_value_error(width):
    frame = contribution_frame(np.ones((2, width)), columns=FEATURES[:width])

    with pytest.raises(ValueError):
        select_top_drivers(frame)


def test_a_non_default_index_is_kept():
    index = ["C9", "C2"]
    result = select_top_drivers(contribution_frame([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], index=index))

    assert list(result.index) == index
    assert result.loc["C9", "driver_1"] == "a"
    assert result.loc["C2", "driver_1"] == "d"


def test_the_input_table_is_not_modified():
    frame = contribution_frame([[0.4, -0.3, 0.2, 0.1]], index=["C1"])
    original = frame.copy(deep=True)

    select_top_drivers(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_the_result_has_exactly_the_six_columns_object_dtype_and_three_different_features():
    result = select_top_drivers(contribution_frame([[0.4, -0.3, 0.2, 0.1], [0.1, 0.2, -0.3, 0.4]]))

    assert list(result.columns) == [
        "driver_1", "driver_1_direction", "driver_2", "driver_2_direction",
        "driver_3", "driver_3_direction",
    ]
    assert list(result.columns) == DRIVER_COLUMNS
    assert (result.dtypes == object).all()
    for _, row in result.iterrows():
        assert len({row["driver_1"], row["driver_2"], row["driver_3"]}) == TOP_N_DRIVERS
    assert (DIRECTION_RAISES, DIRECTION_LOWERS) == ("raises risk", "lowers risk")


def test_the_vectorised_selection_matches_a_plain_python_selection_with_forced_ties():
    rng = np.random.default_rng(11)
    columns = [f"f{i:02d}" for i in range(42)]
    values = np.round(rng.normal(scale=2.0, size=(200, 42)), 1)
    values[values == 0] = 0.1
    frame = pd.DataFrame(values, index=[f"C{i:03d}" for i in range(200)], columns=columns)

    result = select_top_drivers(frame)

    for row_number in range(200):
        row = values[row_number]
        ranked = sorted(range(42), key=lambda j: (-abs(row[j]), j))[:3]
        expected = []
        for j in ranked:
            expected += [columns[j], DIRECTION_RAISES if row[j] > 0 else DIRECTION_LOWERS]
        assert result.iloc[row_number].tolist() == expected


# End to end


def test_real_customers_get_three_different_schema_features_and_allowed_directions(
    driver_data, driver_contributions
):
    result = select_top_drivers(driver_contributions)

    assert len(result) == len(driver_data.ids) >= 300
    assert list(result.index) == driver_data.ids
    features = result[["driver_1", "driver_2", "driver_3"]]
    assert set(features.to_numpy().ravel()) <= set(driver_data.names)
    assert (features.nunique(axis=1) == TOP_N_DRIVERS).all()
    directions = result[["driver_1_direction", "driver_2_direction", "driver_3_direction"]]
    assert set(directions.to_numpy().ravel()) <= {DIRECTION_RAISES, DIRECTION_LOWERS}


# Label map

FORBIDDEN_LABEL_CHARACTERS = {
    ",": "a comma",
    '"': "a double quotation mark",
    "'": "a single quotation mark or apostrophe",
    "‘": "a typographic quotation mark",
    "’": "a typographic quotation mark",
    "“": "a typographic quotation mark",
    "”": "a typographic quotation mark",
    "\r": "a carriage return",
    "\n": "a line feed",
}


def label_problems(labels, feature_names):
    """Readable problems with a label map: it must hold exactly one clean,
    distinct label per feature name."""
    problems = []
    for name in feature_names:
        if name not in labels:
            problems.append(f"feature '{name}' has no label")
    for name in labels:
        if name not in feature_names:
            problems.append(f"label for '{name}' is not a schema feature")

    texts = {}
    for name, label in labels.items():
        if not isinstance(label, str):
            problems.append(f"label for '{name}' is not a string: {label!r}")
            continue
        if not label.strip():
            problems.append(f"label for '{name}' is empty or only whitespace: {label!r}")
            continue
        if label != label.strip():
            problems.append(f"label for '{name}' has leading or trailing whitespace: {label!r}")
        for character, description in FORBIDDEN_LABEL_CHARACTERS.items():
            if character in label:
                problems.append(f"label for '{name}' contains {description}: {label!r}")
        texts.setdefault(label.strip().casefold(), []).append(name)

    for names in texts.values():
        if len(names) > 1:
            problems.append(f"labels for {names} are the same ignoring case and whitespace")
    return problems


CLEAN_LABELS = {"a": "Alpha", "b": "Beta", "c": "Gamma"}
CLEAN_NAMES = ["a", "b", "c"]


def test_a_clean_map_has_no_problems():
    assert label_problems(CLEAN_LABELS, CLEAN_NAMES) == []


LABEL_PROBLEM_CASES = {
    "missing-label": ({"a": "Alpha", "b": "Beta"}, ["'c'", "no label"]),
    "extra-label": ({**CLEAN_LABELS, "z": "Zeta"}, ["'z'", "not a schema feature"]),
    "integer-label": ({**CLEAN_LABELS, "b": 5}, ["'b'", "not a string"]),
    "none-label": ({**CLEAN_LABELS, "b": None}, ["'b'", "not a string"]),
    "empty-label": ({**CLEAN_LABELS, "b": ""}, ["'b'", "empty"]),
    "whitespace-label": ({**CLEAN_LABELS, "b": "   "}, ["'b'", "empty"]),
    "leading-space": ({**CLEAN_LABELS, "b": " Beta"}, ["'b'", "leading or trailing"]),
    "trailing-space": ({**CLEAN_LABELS, "b": "Beta "}, ["'b'", "leading or trailing"]),
    "comma": ({**CLEAN_LABELS, "b": "Be,ta"}, ["'b'", "comma"]),
    "double-quote": ({**CLEAN_LABELS, "b": 'Be"ta'}, ["'b'", "double quotation"]),
    "apostrophe": ({**CLEAN_LABELS, "b": "Be'ta"}, ["'b'", "apostrophe"]),
    "typographic-single-open": ({**CLEAN_LABELS, "b": "Be‘ta"}, ["'b'", "typographic"]),
    "typographic-single-close": ({**CLEAN_LABELS, "b": "Be’ta"}, ["'b'", "typographic"]),
    "typographic-double-open": ({**CLEAN_LABELS, "b": "Be“ta"}, ["'b'", "typographic"]),
    "typographic-double-close": ({**CLEAN_LABELS, "b": "Be”ta"}, ["'b'", "typographic"]),
    "carriage-return": ({**CLEAN_LABELS, "b": "Be\rta"}, ["'b'", "carriage return"]),
    "line-feed": ({**CLEAN_LABELS, "b": "Be\nta"}, ["'b'", "line feed"]),
    "duplicate-ignoring-case": ({**CLEAN_LABELS, "b": "ALPHA"}, ["'a'", "'b'", "same ignoring case"]),
    "duplicate-ignoring-whitespace": (
        {**CLEAN_LABELS, "b": "Alpha "},
        ["'a'", "'b'", "same ignoring case"],
    ),
}


@pytest.mark.parametrize(
    "labels,fragments",
    list(LABEL_PROBLEM_CASES.values()),
    ids=list(LABEL_PROBLEM_CASES),
)
def test_the_label_check_reports_each_problem_with_the_offending_name(labels, fragments):
    problems = label_problems(labels, CLEAN_NAMES)

    assert problems
    text = "\n".join(problems)
    for fragment in fragments:
        assert fragment in text, f"{fragment!r} not in {text!r}"


def test_the_real_label_map_is_complete_and_clean(driver_data):
    assert label_problems(DRIVER_LABELS, driver_data.names) == []
    assert len(DRIVER_LABELS) == len(driver_data.names)
    assert list(DRIVER_LABELS) == driver_data.names


def test_every_selected_real_driver_has_a_label(driver_contributions):
    selected = select_top_drivers(driver_contributions)

    features = set(selected[["driver_1", "driver_2", "driver_3"]].to_numpy().ravel())

    assert features
    assert features <= set(DRIVER_LABELS)


RELATED_FEATURES = {
    "contract-and-month-to-month": ["contract", "is_month_to_month"],
    "referrals": ["number_of_referrals", "referred_a_friend"],
    "charges-and-revenue": ["total_charges", "total_revenue"],
    "security-add-ons": ["online_security", "internet_without_security", "no_addons_despite_internet"],
}


@pytest.mark.parametrize("group", list(RELATED_FEATURES.values()), ids=list(RELATED_FEATURES))
def test_related_features_have_different_labels(group):
    labels = [DRIVER_LABELS[name] for name in group]

    assert len({label.casefold() for label in labels}) == len(group)
