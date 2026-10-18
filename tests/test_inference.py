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
Section H: economics (expected value, break-even and sensitivity tables).
Section I: recommendation facts (tiers, group profile, importance and driver counts).
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
from retention_platform.inference import economics, outputs, recommendations, score, validate
from retention_platform.inference.economics import (
    BaseCase,
    ARPU_MULTIPLES,
    MONTHS_KEPT_SWEEP,
    OFFER_COST_SWEEP,
    BREAK_EVEN_ABOVE_ONE,
    SAVE_RATE_SWEEP,
    EconomicsAssumptions,
    EconomicsError,
    analyze_economics,
    flagged_group,
    render_economics_document,
)
from retention_platform.inference.outputs import (
    MODEL_RUN_ID_LABEL,
    OUTREACH_FILE,
    SCORING_RUN_ID_LABEL,
    SNAPSHOT_NOTICE,
    SUMMARY_FILE,
    OutputGenerationError,
    OutputPublicationError,
    build_run_summary,
    document_header,
    outreach_csv_bytes,
    publish_outputs,
    summary_json_bytes,
)
from retention_platform.inference.recommendations import (
    BREAK_EVEN_ABOVE_ONE_ACTION,
    BREAK_EVEN_ABOVE_ONE_FINDING,
    BREAK_EVEN_ACTION,
    BREAK_EVEN_FINDING,
    BREAK_EVEN_LIMIT_TEXT,
    FIBER_OPTIC_ACTION,
    FIBER_OPTIC_FINDING,
    HIGH_TIER_ACTION,
    HIGH_TIER_FINDING,
    LEAD_DRIVER_ACTION,
    LEAD_DRIVER_FINDING,
    LEAD_DRIVER_SHARE_MINIMUM,
    NO_INTERNET_CATEGORY,
    PROFILE_CATEGORICAL_COLUMNS,
    TOP_N_IMPORTANCE,
    Recommendation,
    RecommendationsError,
    compute_recommendation_facts,
    select_recommendations,
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
    global_importance,
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
    INTERNET_TYPE_SENTINEL,
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


# Document header


def test_the_header_is_exactly_four_lines():
    header = document_header("Economics Summary", "20261011T123500Z", "20261011T120000Z")

    assert header == (
        "# Economics Summary\n"
        "\n"
        "Scoring run ID: 20261011T123500Z\n"
        "Model run ID: 20261011T120000Z\n"
    )
    assert header.split("\n")[:4] == [
        "# Economics Summary",
        "",
        "Scoring run ID: 20261011T123500Z",
        "Model run ID: 20261011T120000Z",
    ]


def test_the_line_labels_are_the_constants_the_header_uses():
    assert SCORING_RUN_ID_LABEL == "Scoring run ID: "
    assert MODEL_RUN_ID_LABEL == "Model run ID: "
    lines = document_header("T", "20261011T123500Z", "20261011T120000Z").split("\n")
    assert lines[2] == SCORING_RUN_ID_LABEL + "20261011T123500Z"
    assert lines[3] == MODEL_RUN_ID_LABEL + "20261011T120000Z"


BAD_RUN_IDS = [
    "",
    "not-an-id",
    "20261011T123500",
    "20261011T123500Z\n",
    " 20261011T123500Z",
    "../20261011T123500Z",
    "20261301T123500Z",
    "20261011T256100Z",
    "2026-10-11T12:35:00Z",
    20261011123500,
    None,
]


@pytest.mark.parametrize("bad", BAD_RUN_IDS, ids=repr)
def test_a_bad_scoring_run_id_is_refused(bad):
    with pytest.raises(OutputGenerationError, match="scoring_run_id"):
        document_header("Title", bad, "20261011T120000Z")


@pytest.mark.parametrize("bad", BAD_RUN_IDS, ids=repr)
def test_a_bad_model_run_id_is_refused(bad):
    with pytest.raises(OutputGenerationError, match="model_run_id"):
        document_header("Title", "20261011T123500Z", bad)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "   ",
        "\t",
        5,
        None,
        ["Title"],
        "Line one\nLine two",
        "Line one\rLine two",
        "Line one\u2028Line two",
        "Line one\u2029Line two",
        "Line one\x85Line two",
        "Line one\x0bLine two",
        "Line one\x0cLine two",
        "Title\n",
    ],
    ids=repr,
)
def test_a_bad_title_is_refused(title):
    with pytest.raises(OutputGenerationError, match="title"):
        document_header(title, "20261011T123500Z", "20261011T120000Z")


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


# Global importance


@pytest.fixture(scope="module")
def driver_importance(driver_data):
    return global_importance(driver_data.run)


def make_get_score(scores):
    def get_score(fmap="", importance_type="weight"):
        assert importance_type == "total_gain"
        return dict(scores)

    return get_score


def run_with_scores(run, scores_by_fold, feature_names=None):
    """A copy of the run whose chosen fold boosters report the given
    get_score results (and, if given, carry the given feature names)."""
    model = copy.deepcopy(run.model)
    for position, fold in enumerate(model.calibrated_classifiers_):
        booster = fold.estimator.get_booster()
        if feature_names is not None:
            booster.feature_names = feature_names
        if position in scores_by_fold:
            booster.get_score = make_get_score(scores_by_fold[position])
    return dataclasses.replace(run, model=model)


def encoded_positions(run, input_name):
    return [i for i, owner in enumerate(encoded_column_owners(run)) if owner == input_name]


def keyed(positions_and_gains):
    return {f"f{position}": gain for position, gain in positions_and_gains.items()}


def test_global_importance_matches_an_independent_computation(driver_data, driver_importance):
    run = driver_data.run
    owners = np.array(encoded_column_owners(run))
    fold_shares = []

    for fold in run.model.calibrated_classifiers_:
        gains = np.zeros(len(owners))
        for key, gain in fold.estimator.get_booster().get_score(importance_type="total_gain").items():
            gains[int(key[1:])] = gain
        owned = np.array([gains[owners == name].sum() for name in driver_data.names])
        fold_shares.append(owned / gains.sum())
    expected = pd.Series(np.mean(fold_shares, axis=0), index=driver_data.names)

    assert abs(driver_importance["share"].sum() - 1) < 1e-9
    np.testing.assert_allclose(
        driver_importance["share"].to_numpy(),
        expected[driver_importance.index].to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_global_importance_table_shape_and_ordering(driver_data, driver_importance):
    table = driver_importance

    assert set(table.index) == set(driver_data.names)
    assert len(table) == len(driver_data.names)
    assert table.index.name == "feature"
    assert list(table.columns) == ["share", "rank"]
    assert table["share"].dtype == np.float64
    assert table["rank"].dtype == np.int64
    assert np.isfinite(table["share"].to_numpy()).all()
    assert (table["share"] >= 0).all()
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert (np.diff(table["share"].to_numpy()) <= 0).all()


def test_gain_on_two_columns_of_one_input_gives_that_input_every_share(driver_data):
    first, second, *_ = encoded_positions(driver_data.run, "offer")
    scores = {fold: keyed({first: 3.0 + fold, second: 7.0 - fold}) for fold in range(5)}

    table = global_importance(run_with_scores(driver_data.run, scores))

    assert table.index[0] == "offer"
    assert table.loc["offer", "share"] == 1.0
    assert table.loc["offer", "rank"] == 1
    assert (table.drop(index="offer")["share"] == 0.0).all()
    assert len(table) == len(driver_data.names)


def test_gains_that_differ_by_fold_are_averaged_by_hand(driver_data):
    contract = encoded_positions(driver_data.run, "contract")
    gender = encoded_positions(driver_data.run, "gender")
    scores = {
        0: keyed({contract[0]: 6.0, gender[0]: 2.0}),
        1: keyed({contract[0]: 1.0, gender[0]: 3.0}),
        2: keyed({contract[0]: 2.0, contract[1]: 2.0, gender[0]: 4.0}),
        3: keyed({contract[0]: 10.0}),
        4: keyed({contract[0]: 1.0, gender[0]: 1.0}),
    }

    table = global_importance(run_with_scores(driver_data.run, scores))

    assert table.loc["contract", "share"] == pytest.approx(0.6, abs=1e-12)
    assert table.loc["gender", "share"] == pytest.approx(0.4, abs=1e-12)
    assert list(table.index[:2]) == ["contract", "gender"]
    assert list(table["rank"][:2]) == [1, 2]
    others = [name for name in driver_data.names if name not in ("contract", "gender")]
    assert list(table.index[2:]) == others
    assert (table["share"][2:] == 0.0).all()


def test_an_encoded_column_missing_from_get_score_counts_as_zero(driver_data):
    contract = encoded_positions(driver_data.run, "contract")
    gender = encoded_positions(driver_data.run, "gender")
    scores = {fold: keyed({contract[0]: 2.0, gender[0]: 2.0}) for fold in range(5)}

    table = global_importance(run_with_scores(driver_data.run, scores))

    assert table.loc["contract", "share"] == 0.5
    assert table.loc["gender", "share"] == 0.5
    assert table["share"].sum() == 1.0


def test_equal_shares_are_ranked_by_schema_column_order(driver_data):
    names = driver_data.names
    earlier, later = "married", "contract"
    assert names.index(earlier) < names.index(later)
    assert encoded_positions(driver_data.run, later)[0] < encoded_positions(driver_data.run, earlier)[0]
    scores = {
        fold: keyed(
            {
                encoded_positions(driver_data.run, earlier)[0]: 5.0,
                encoded_positions(driver_data.run, later)[0]: 5.0,
            }
        )
        for fold in range(5)
    }

    table = global_importance(run_with_scores(driver_data.run, scores))

    assert list(table.index[:2]) == [earlier, later]
    assert list(table["rank"][:2]) == [1, 2]
    assert table.loc[earlier, "share"] == table.loc[later, "share"] == 0.5


def test_keys_that_are_feature_names_map_to_their_position(driver_data):
    width = len(encoded_column_owners(driver_data.run))
    names = [f"col{index}" for index in range(width)]
    contract = encoded_positions(driver_data.run, "contract")
    gender = encoded_positions(driver_data.run, "gender")
    scores = {
        fold: {names[contract[0]]: 3.0, names[gender[0]]: 1.0} for fold in range(5)
    }

    table = global_importance(run_with_scores(driver_data.run, scores, feature_names=names))

    assert table.loc["contract", "share"] == 0.75
    assert table.loc["gender", "share"] == 0.25


def test_a_structure_failure_propagates(driver_data):
    damage, fragment = STRUCTURE_FAILURES["four-calibrated-classifiers"]

    with pytest.raises(DriverError) as exc_info:
        global_importance(damage(driver_data.run))

    assert fragment in str(exc_info.value)


@pytest.mark.parametrize("key", ["f999", "weird", "f1.5", "f"], ids=repr)
def test_a_get_score_key_that_cannot_be_mapped_raises(driver_data, key):
    scores = {2: {"f0": 1.0, key: 2.0}}

    with pytest.raises(DriverError) as exc_info:
        global_importance(run_with_scores(driver_data.run, scores))

    message = str(exc_info.value)
    assert repr(key) in message
    assert "fold 2" in message


def test_a_feature_name_key_that_is_not_a_name_raises(driver_data):
    width = len(encoded_column_owners(driver_data.run))
    names = [f"col{index}" for index in range(width)]
    scores = {1: {"col0": 1.0, "unknown": 2.0}}

    with pytest.raises(DriverError, match="'unknown'"):
        global_importance(run_with_scores(driver_data.run, scores, feature_names=names))


def test_a_fold_with_no_gains_raises(driver_data):
    with pytest.raises(DriverError, match="fold 3: total gain is zero"):
        global_importance(run_with_scores(driver_data.run, {3: {}}))


def test_a_fold_with_only_zero_gains_raises(driver_data):
    with pytest.raises(DriverError, match="fold 3: total gain is zero"):
        global_importance(run_with_scores(driver_data.run, {3: {"f0": 0.0, "f1": 0.0}}))


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf], ids=["negative", "nan", "inf", "negative-inf"])
def test_a_negative_or_non_finite_gain_raises(driver_data, bad):
    scores = {2: {"f0": 1.0, "f1": bad}}

    with pytest.raises(DriverError, match="fold 2: .*negative or not finite"):
        global_importance(run_with_scores(driver_data.run, scores))


def test_the_shared_run_is_unchanged_after_the_damaged_importance_cases(
    driver_data, driver_importance
):
    for fold in driver_data.run.model.calibrated_classifiers_:
        booster = fold.estimator.get_booster()
        assert booster.feature_names is None
        assert len(booster.get_score(importance_type="total_gain")) > 1

    pd.testing.assert_frame_equal(
        global_importance(driver_data.run), driver_importance, check_exact=True
    )


# =============================================================================
# Section H: Economics
# =============================================================================

ECON_ASSUMPTIONS = EconomicsAssumptions(save_rate=0.20, months_kept=3, offer_cost=20.0)
ECON_K_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]


def economics_inputs(n_rows, seed=0, k_frac=0.10, shape=(5, 2)):
    """A synthetic ranked table (score_batch columns) and a monthly_charge
    Series indexed by customer_id, with the seeded values."""
    rng = np.random.default_rng(seed)
    ids = [f"C{i:05d}" for i in range(n_rows)]
    table = (
        pd.DataFrame({"customer_id": ids, "churn_probability": rng.beta(*shape, size=n_rows)})
        .sort_values(["churn_probability", "customer_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    table["rank"] = np.arange(1, n_rows + 1, dtype="int64")
    table["in_top_k"] = table["rank"] <= top_k_count(n_rows, k_frac)
    charges = pd.Series(rng.uniform(20, 120, n_rows).round(2), index=ids, name="monthly_charge")
    return table, charges


def hand_made_inputs():
    """Three customers: probabilities 0.9, 0.8, 0.7 and charges 70, 80, 90."""
    table = pd.DataFrame(
        {
            "customer_id": pd.Series(["A", "B", "C"], dtype=object),
            "churn_probability": [0.9, 0.8, 0.7],
            "rank": pd.Series([1, 2, 3], dtype="int64"),
            "in_top_k": [True, True, True],
        }
    )
    return table, pd.Series([70.0, 80.0, 90.0], index=["A", "B", "C"])


def assert_same_economics(first, second):
    assert first.base_case == second.base_case
    for name in ("one_at_a_time", "save_rate_by_months", "break_even_by_cost_and_months", "k_sweep"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)


# Hand-made group


def test_hand_made_group_base_case_figures():
    table, charges = hand_made_inputs()

    base = analyze_economics(table, charges, 1.0, [1.0], ECON_ASSUMPTIONS).base_case

    assert base.customers_contacted == 3
    assert base.total_offer_cost == pytest.approx(60.0)
    assert base.expected_would_be_churners == pytest.approx(2.4)
    assert base.expected_customers_saved == pytest.approx(0.48)
    assert base.gross_revenue_kept == pytest.approx(115.2)
    assert base.net_expected_value == pytest.approx(55.2)
    assert base.ev_per_contacted_customer == pytest.approx(18.4)
    assert base.break_even_save_rate == pytest.approx(60 / 576)
    assert base.arpu_proxy == pytest.approx(80.0)
    assert base.arpu_source == "derived"
    assert base.arpu_customers_used == 3


def test_hand_made_per_customer_values_sum_to_the_group_value():
    expected = [0.9 * 0.2 * 80 * 3 - 20, 0.8 * 0.2 * 80 * 3 - 20, 0.7 * 0.2 * 80 * 3 - 20]
    table, charges = hand_made_inputs()

    base = analyze_economics(table, charges, 1.0, [1.0], ECON_ASSUMPTIONS).base_case

    assert expected == pytest.approx([23.2, 18.4, 13.6])
    assert base.net_expected_value == pytest.approx(sum(expected))


def test_expected_value_is_zero_at_the_break_even_save_rate():
    table, charges = hand_made_inputs()
    base = analyze_economics(table, charges, 1.0, [1.0], ECON_ASSUMPTIONS).base_case

    at_break_even = EconomicsAssumptions(base.break_even_save_rate, 3, 20.0)
    result = analyze_economics(table, charges, 1.0, [1.0], at_break_even).base_case

    assert result.net_expected_value == pytest.approx(0.0, abs=1e-9)


def test_a_break_even_above_one_stays_above_one():
    table, charges = hand_made_inputs()
    table["churn_probability"] = [0.1, 0.1, 0.1]
    charges = pd.Series([10.0, 10.0, 10.0], index=["A", "B", "C"])
    assumptions = EconomicsAssumptions(0.20, 1, 20.0)

    results = analyze_economics(table, charges, 1.0, [1.0], assumptions)

    assert results.base_case.break_even_save_rate == pytest.approx(3 * 20 / (10 * 1 * 0.3))
    assert results.base_case.break_even_save_rate > 1
    assert (results.break_even_by_cost_and_months > 1).any().any()


# Independent recomputation


def plain_python_group_figures(table, charges, k_frac, assumptions):
    rows = sorted(zip(table["rank"], table["customer_id"], table["churn_probability"]))
    group = rows[: top_k_count(len(rows), k_frac)]
    arpu = assumptions.arpu_proxy
    if arpu is None:
        arpu = sum(charges[customer] for _, customer, _ in group) / len(group)
    net = 0.0
    sum_p = 0.0
    for _, _, probability in group:
        net += (
            probability * assumptions.save_rate * arpu * assumptions.months_kept
            - assumptions.offer_cost
        )
        sum_p += probability
    break_even = len(group) * assumptions.offer_cost / (arpu * assumptions.months_kept * sum_p)
    return {"n": len(group), "arpu": arpu, "net": net, "sum_p": sum_p, "break_even": break_even}


def test_group_figures_match_a_plain_python_per_customer_loop():
    table, charges = economics_inputs(300, seed=1)

    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    expected = plain_python_group_figures(table, charges, 0.10, ECON_ASSUMPTIONS)

    base = results.base_case
    assert base.customers_contacted == expected["n"] == 30
    assert base.arpu_proxy == pytest.approx(expected["arpu"], rel=1e-9)
    assert base.net_expected_value == pytest.approx(expected["net"], rel=1e-9)
    assert base.expected_would_be_churners == pytest.approx(expected["sum_p"], rel=1e-9)
    assert base.break_even_save_rate == pytest.approx(expected["break_even"], rel=1e-9)


def test_net_expected_value_is_linear_in_the_save_rate():
    table, charges = economics_inputs(300, seed=2)

    rows = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS).one_at_a_time
    save = rows[rows["input"] == "save_rate"].set_index("value")["net_expected_value"]

    slope = (save[0.40] - save[0.05]) / 0.35
    for rate in (0.10, 0.20, 0.30):
        assert save[rate] == pytest.approx(save[0.05] + slope * (rate - 0.05), rel=1e-9)


def test_expected_value_is_zero_at_the_break_even_on_synthetic_data():
    table, charges = economics_inputs(300, seed=3)
    base = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS).base_case
    assert 0 < base.break_even_save_rate < 1

    at_break_even = EconomicsAssumptions(base.break_even_save_rate, 3, 20.0)
    result = analyze_economics(table, charges, 0.10, ECON_K_VALUES, at_break_even).base_case

    assert result.net_expected_value == pytest.approx(0.0, abs=1e-8)


# Alignment by customer id


def test_a_shuffled_or_reversed_series_gives_identical_results():
    table, charges = economics_inputs(300, seed=4)
    reference = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    shuffled = charges.sample(frac=1, random_state=5)
    reversed_series = charges.iloc[::-1]

    assert_same_economics(
        reference, analyze_economics(table, shuffled, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    )
    assert_same_economics(
        reference, analyze_economics(table, reversed_series, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    )
    pd.testing.assert_frame_equal(
        flagged_group(table, shuffled, 0.10), flagged_group(table, charges, 0.10), check_exact=True
    )


def test_values_attached_to_the_wrong_ids_by_position_give_a_different_result():
    table, charges = economics_inputs(300, seed=4)
    misaligned = pd.Series(charges.to_numpy(), index=list(reversed(charges.index)))
    assert set(misaligned.index) == set(charges.index)

    correct = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    wrong = analyze_economics(table, misaligned, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)

    assert wrong.base_case.arpu_proxy != correct.base_case.arpu_proxy


# ARPU


def test_a_derived_arpu_is_the_group_mean_and_an_override_is_used_as_given():
    table, charges = economics_inputs(300, seed=6)
    group_ids = table["customer_id"][:30]

    derived = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS).base_case
    override = analyze_economics(
        table, charges, 0.10, ECON_K_VALUES, EconomicsAssumptions(0.20, 3, 20.0, arpu_proxy=95.5)
    ).base_case

    assert derived.arpu_source == "derived"
    assert derived.arpu_customers_used == 30
    assert derived.arpu_proxy == pytest.approx(charges[group_ids].mean(), rel=1e-12)
    assert override.arpu_source == "override"
    assert override.arpu_customers_used is None
    assert override.arpu_proxy == 95.5


@pytest.mark.parametrize("arpu_override", [None, 95.5], ids=["derived", "override"])
def test_the_multiples_apply_to_the_applicable_base_arpu(arpu_override):
    table, charges = economics_inputs(300, seed=6)
    assumptions = EconomicsAssumptions(0.20, 3, 20.0, arpu_proxy=arpu_override)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, assumptions)
    base = results.base_case
    rows = results.one_at_a_time
    multiples = rows[rows["input"] == "arpu_multiple"]

    for multiple, net in zip(multiples["value"], multiples["net_expected_value"]):
        expected = (
            base.expected_would_be_churners * 0.20 * base.arpu_proxy * multiple * 3
            - base.customers_contacted * 20.0
        )
        assert net == pytest.approx(expected, rel=1e-9)


def test_the_k_sweep_derives_the_arpu_per_k_without_an_override():
    table, charges = economics_inputs(300, seed=7)

    sweep = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS).k_sweep

    for k, arpu in zip(sweep["k"], sweep["arpu_proxy"]):
        size = top_k_count(300, k)
        assert arpu == pytest.approx(charges[table["customer_id"][:size]].mean(), rel=1e-12)
    assert sweep["arpu_proxy"].nunique() > 1


def test_the_k_sweep_shares_one_override_for_every_k():
    table, charges = economics_inputs(300, seed=7)
    assumptions = EconomicsAssumptions(0.20, 3, 20.0, arpu_proxy=88.0)

    sweep = analyze_economics(table, charges, 0.10, ECON_K_VALUES, assumptions).k_sweep

    assert (sweep["arpu_proxy"] == 88.0).all()


# Tables


def test_table_shapes_labels_and_orders():
    table, charges = economics_inputs(300, seed=8)

    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)

    rows = results.one_at_a_time
    assert list(rows.columns) == [
        "input", "value", "net_expected_value", "ev_per_contacted_customer", "break_even_save_rate",
    ]
    assert list(rows["input"]) == ["save_rate"] * 5 + ["months_kept"] * 4 + ["offer_cost"] * 4 + ["arpu_multiple"] * 3
    assert list(rows["value"]) == [*SAVE_RATE_SWEEP, *MONTHS_KEPT_SWEEP, *OFFER_COST_SWEEP, *ARPU_MULTIPLES]
    assert len(rows) == 16
    assert (SAVE_RATE_SWEEP, MONTHS_KEPT_SWEEP, OFFER_COST_SWEEP, ARPU_MULTIPLES) == (
        (0.05, 0.10, 0.20, 0.30, 0.40), (1, 3, 6, 12), (10, 20, 40, 60), (0.8, 1.0, 1.2),
    )

    by_months = results.save_rate_by_months
    assert by_months.shape == (5, 4)
    assert by_months.index.name == "save_rate" and list(by_months.index) == list(SAVE_RATE_SWEEP)
    assert by_months.columns.name == "months_kept" and list(by_months.columns) == list(MONTHS_KEPT_SWEEP)

    by_cost = results.break_even_by_cost_and_months
    assert by_cost.shape == (4, 4)
    assert by_cost.index.name == "months_kept" and list(by_cost.index) == list(MONTHS_KEPT_SWEEP)
    assert by_cost.columns.name == "offer_cost" and list(by_cost.columns) == list(OFFER_COST_SWEEP)

    sweep = results.k_sweep
    assert list(sweep.columns) == [
        "k", "customers_contacted", "arpu_proxy", "expected_would_be_churners",
        "net_expected_value", "ev_per_contacted_customer", "break_even_save_rate",
    ]
    assert list(sweep["k"]) == ECON_K_VALUES


def test_the_base_values_appear_once_in_each_one_at_a_time_list():
    table, charges = economics_inputs(300, seed=8)

    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    rows = results.one_at_a_time.set_index(["input", "value"])
    base_rows = [
        rows.loc[("save_rate", 0.20)], rows.loc[("months_kept", 3)],
        rows.loc[("offer_cost", 20)], rows.loc[("arpu_multiple", 1.0)],
    ]

    for row in base_rows:
        assert row["net_expected_value"] == pytest.approx(results.base_case.net_expected_value, rel=1e-12)
        assert row["ev_per_contacted_customer"] == pytest.approx(
            results.base_case.ev_per_contacted_customer, rel=1e-12
        )
        assert row["break_even_save_rate"] == pytest.approx(results.base_case.break_even_save_rate, rel=1e-12)


def test_a_save_rate_by_months_cell_and_a_break_even_cell_match_a_direct_recomputation():
    table, charges = economics_inputs(300, seed=8)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    base = results.base_case

    net = base.expected_would_be_churners * 0.30 * base.arpu_proxy * 6 - base.customers_contacted * 20.0
    break_even = base.customers_contacted * 40.0 / (base.arpu_proxy * 6 * base.expected_would_be_churners)

    assert results.save_rate_by_months.loc[0.30, 6] == pytest.approx(net, rel=1e-9)
    assert results.break_even_by_cost_and_months.loc[6, 40] == pytest.approx(break_even, rel=1e-9)


def test_the_k_sweep_row_for_k_frac_matches_the_base_case():
    table, charges = economics_inputs(300, seed=8)

    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    row = results.k_sweep[results.k_sweep["k"] == 0.10].iloc[0]

    assert row["customers_contacted"] == results.base_case.customers_contacted
    assert row["arpu_proxy"] == results.base_case.arpu_proxy
    assert row["expected_would_be_churners"] == results.base_case.expected_would_be_churners
    assert row["net_expected_value"] == results.base_case.net_expected_value
    assert row["ev_per_contacted_customer"] == results.base_case.ev_per_contacted_customer
    assert row["break_even_save_rate"] == results.base_case.break_even_save_rate


def test_group_sizes_for_a_reference_sized_table():
    table, charges = economics_inputs(1408, seed=9)

    sweep = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS).k_sweep

    assert list(sweep["customers_contacted"]) == [70, 141, 211, 282, 352]


# Consistency guard


def test_flagged_group_equals_the_in_top_k_rows():
    table, charges = economics_inputs(300, seed=10)

    group = flagged_group(table, charges, 0.10)

    assert list(group.columns) == ["customer_id", "churn_probability", "monthly_charge"]
    assert list(group["customer_id"]) == list(table[table["in_top_k"]]["customer_id"])
    assert list(group["monthly_charge"]) == [charges[c] for c in group["customer_id"]]
    assert (np.diff(group["churn_probability"].to_numpy()) <= 0).all()


def test_the_guard_refuses_an_in_top_k_column_that_is_not_boolean():
    table, charges = economics_inputs(300, seed=10)
    table["in_top_k"] = table["in_top_k"].astype(int)

    with pytest.raises(EconomicsError, match="in_top_k must be a boolean"):
        analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)


def test_the_guard_refuses_a_wrong_flag_count_and_names_both_counts():
    table, charges = economics_inputs(300, seed=10)
    table.loc[40, "in_top_k"] = True

    with pytest.raises(EconomicsError) as exc_info:
        analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)

    assert "31" in str(exc_info.value) and "30" in str(exc_info.value)


def test_the_guard_refuses_the_right_count_with_the_wrong_customers():
    table, charges = economics_inputs(300, seed=10)
    table.loc[0, "in_top_k"] = False
    table.loc[40, "in_top_k"] = True

    with pytest.raises(EconomicsError, match="not the first 30 rows by rank"):
        analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)


# Refusals


def with_probability(position, value):
    def mutate(table, charges):
        table = table.copy()
        table["churn_probability"] = table["churn_probability"].astype(object)
        table.at[position, "churn_probability"] = value
        return table, charges

    return mutate


def with_charge(position, value):
    def mutate(table, charges):
        charges = charges.astype(object)
        charges[table["customer_id"][position]] = value
        return table, charges

    return mutate


def with_rank(values, dtype="int64"):
    def mutate(table, charges):
        table = table.copy()
        table["rank"] = pd.Series(values(len(table)), dtype=dtype)
        return table, charges

    return mutate


def drop_series_id(table, charges):
    return table, charges.drop(table["customer_id"][5])


def add_series_id(table, charges):
    return table, pd.concat([charges, pd.Series([50.0], index=["UNKNOWN"])])


def duplicate_series_id(table, charges):
    return table, pd.concat([charges, charges.iloc[[3]]])


def duplicate_table_id(table, charges):
    table = table.copy()
    table.loc[7, "customer_id"] = table.loc[6, "customer_id"]
    return table, charges


def zero_charges(table, charges):
    return table, charges * 0


def zero_probabilities(table, charges):
    table = table.copy()
    table["churn_probability"] = 0.0
    return table, charges


def empty_table(table, charges):
    return table.iloc[0:0], charges


DATA_REFUSALS = {
    "probability-nan": (with_probability(2, np.nan), "churn_probability is missing"),
    "probability-none": (with_probability(2, None), "churn_probability is missing"),
    "probability-inf": (with_probability(2, np.inf), "churn_probability is not finite"),
    "probability-negative": (with_probability(2, -0.1), "outside [0, 1]"),
    "probability-above-one": (with_probability(2, 1.5), "outside [0, 1]"),
    "probability-text": (with_probability(2, "high"), "churn_probability is not a number"),
    "probability-bool": (with_probability(2, True), "churn_probability is not a number"),
    "charge-nan": (with_charge(4, np.nan), "monthly_charge is missing"),
    "charge-none": (with_charge(4, None), "monthly_charge is missing"),
    "charge-text": (with_charge(4, "80"), "monthly_charge is not a number"),
    "charge-bool": (with_charge(4, True), "monthly_charge is not a number"),
    "charge-inf": (with_charge(4, np.inf), "monthly_charge is not finite"),
    "charge-negative": (with_charge(4, -1.0), "monthly_charge is negative"),
    "series-missing-id": (drop_series_id, "monthly_charge has no value for customer"),
    "series-extra-id": (add_series_id, "unknown customer 'UNKNOWN'"),
    "series-duplicate-id": (duplicate_series_id, "duplicate customer_id in the monthly_charge index"),
    "table-duplicate-id": (duplicate_table_id, "duplicate customer_id in the table"),
    "rank-gap": (with_rank(lambda n: [*range(1, n - 1), n + 1, n + 2]), "complete 1.."),
    "rank-duplicate": (with_rank(lambda n: [1, *range(1, n)]), "complete 1.."),
    "rank-zero-based": (with_rank(lambda n: range(n)), "complete 1.."),
    "rank-float": (with_rank(lambda n: np.arange(1, n + 1), dtype="float64"), "rank must be a complete integer"),
    "probability-huge-int": (with_probability(2, 10**400), "churn_probability is not finite"),
    "charge-huge-int": (with_charge(4, 10**400), "monthly_charge is not finite"),
    "rank-bool": (with_rank(lambda n: [True] * n, dtype="bool"), "rank must be a complete integer"),
    "rank-missing": (
        with_rank(lambda n: [*range(1, n), pd.NA], dtype="Int64"),
        "rank must be a complete integer",
    ),
    "derived-arpu-not-positive": (zero_charges, "derived ARPU proxy is not above 0"),
    "probability-sum-zero": (zero_probabilities, "sum of the churn probabilities of the group is 0"),
    "empty-table": (empty_table, "no rows"),
}


@pytest.mark.parametrize(
    "mutate,fragment", list(DATA_REFUSALS.values()), ids=list(DATA_REFUSALS)
)
def test_unusable_data_is_refused_with_economics_error(mutate, fragment):
    table, charges = economics_inputs(300, seed=11)
    table, charges = mutate(table, charges)

    with pytest.raises(EconomicsError) as exc_info:
        analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)

    assert fragment in str(exc_info.value)


def test_flagged_group_refuses_unusable_data_too():
    table, charges = economics_inputs(300, seed=11)

    with pytest.raises(EconomicsError, match="monthly_charge has no value"):
        flagged_group(table, charges.drop(table["customer_id"][5]), 0.10)


CALLER_MISTAKES = {
    "table-not-a-dataframe": lambda t, c: dict(table=t.to_dict()),
    "missing-rank-column": lambda t, c: dict(table=t.drop(columns=["rank"])),
    "missing-in-top-k-column": lambda t, c: dict(table=t.drop(columns=["in_top_k"])),
    "series-not-a-series": lambda t, c: dict(monthly_charge=c.tolist()),
    "k-zero": lambda t, c: dict(k_frac=0),
    "k-above-one": lambda t, c: dict(k_frac=1.5),
    "k-bool": lambda t, c: dict(k_frac=True),
    "k-values-empty": lambda t, c: dict(k_values=[]),
    "k-values-bad-entry": lambda t, c: dict(k_values=[0.1, 0]),
    "assumptions-wrong-type": lambda t, c: dict(assumptions={"save_rate": 0.2}),
}


@pytest.mark.parametrize("build", list(CALLER_MISTAKES.values()), ids=list(CALLER_MISTAKES))
def test_caller_mistakes_raise_value_error(build):
    table, charges = economics_inputs(300, seed=12)
    arguments = dict(
        table=table, monthly_charge=charges, k_frac=0.10, k_values=ECON_K_VALUES, assumptions=ECON_ASSUMPTIONS
    )
    arguments.update(build(table, charges))

    with pytest.raises(ValueError):
        analyze_economics(**arguments)


@pytest.mark.parametrize("bad_k", [0, -0.1, 1.5, True, None, "0.1"], ids=repr)
def test_flagged_group_rejects_a_bad_k(bad_k):
    table, charges = economics_inputs(300, seed=12)

    with pytest.raises(ValueError):
        flagged_group(table, charges, bad_k)


BAD_ASSUMPTIONS = [
    ("save_rate", v) for v in (0, -0.1, 1.5, True, "0.2", np.nan, np.inf, None, 10**400)
] + [
    ("months_kept", v) for v in (0, -3, 2.5, 3.0, True, "3", np.nan, None, 10**400)
] + [
    ("offer_cost", v) for v in (0, -5, True, "20", np.nan, np.inf, None, 10**400)
] + [
    ("arpu_proxy", v) for v in (0, -80, True, "80", np.nan, np.inf, 10**400)
]


@pytest.mark.parametrize("name,value", BAD_ASSUMPTIONS)
def test_bad_assumptions_raise_value_error_and_never_overflow(name, value):
    arguments = {"save_rate": 0.20, "months_kept": 3, "offer_cost": 20.0, "arpu_proxy": None}
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        EconomicsAssumptions(**arguments)


def test_good_assumptions_are_accepted():
    assumptions = EconomicsAssumptions(0.2, 3, 20, 80)

    assert (assumptions.save_rate, assumptions.months_kept) == (0.2, 3)
    assert EconomicsAssumptions(1, 1, 0.5, None).arpu_proxy is None


def test_the_inputs_are_not_changed_by_a_call():
    table, charges = economics_inputs(300, seed=13)
    table_before, charges_before = table.copy(deep=True), charges.copy(deep=True)

    analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    flagged_group(table, charges, 0.10)

    pd.testing.assert_frame_equal(table, table_before)
    pd.testing.assert_series_equal(charges, charges_before)


def test_economics_py_imports_only_the_allowed_modules_and_never_the_target():
    tree = ast.parse(inspect.getsource(economics))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module if node.module.startswith("retention_platform") else node.module.split(".")[0])

    assert imported <= {
        "__future__",
        "math",
        "dataclasses",
        "numpy",
        "pandas",
        "retention_platform.inference.outputs",
        "retention_platform.inference.ranking",
    }
    assert TARGET_COLUMN not in inspect.getsource(economics)


# Economics document

DOC_SCORING_RUN_ID = "20261011T123500Z"
DOC_MODEL_RUN_ID = "20261011T120000Z"
DOC_HEADINGS = [
    "What this document is",
    "Assumptions",
    "Base case",
    "One input at a time",
    "Save rate and months kept",
    "Break-even save rate",
    "Outreach group size",
    "Limitations",
]


def render_document(results, assumptions, scored, k_frac=0.10):
    return render_economics_document(
        results, assumptions, k_frac, scored, DOC_SCORING_RUN_ID, DOC_MODEL_RUN_ID
    )


def hand_made_document(assumptions=ECON_ASSUMPTIONS):
    table, charges = hand_made_inputs()
    results = analyze_economics(table, charges, 1.0, [1.0], assumptions)
    return results, render_document(results, assumptions, 3, k_frac=1.0)


def seeded_document(assumptions=ECON_ASSUMPTIONS):
    table, charges = economics_inputs(300, seed=21)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, assumptions)
    return results, render_document(results, assumptions, 300)


def markdown_tables(text):
    """Every pipe table of the text as a list of rows of cell texts, with
    the header row first and the separator row removed."""
    tables, current = [], []
    for line in text.split("\n"):
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            tables.append([current[0], *current[2:]])
            current = []
    return tables


def money_text(value):
    return "{:,.2f}".format(value)


def break_even_text(value):
    return BREAK_EVEN_ABOVE_ONE if value > 1 else "{:.1f}%".format(value * 100)


def whole_percent_text(value):
    return "{}%".format(int(round(value * 100)))


def test_the_document_starts_with_the_fixed_header_and_ends_with_one_line_feed():
    _, text = seeded_document()
    lines = text.split("\n")

    assert lines[:4] == [
        "# Economics Summary",
        "",
        f"Scoring run ID: {DOC_SCORING_RUN_ID}",
        f"Model run ID: {DOC_MODEL_RUN_ID}",
    ]
    assert text.startswith(document_header("Economics Summary", DOC_SCORING_RUN_ID, DOC_MODEL_RUN_ID))
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert not any(line != line.rstrip() for line in lines)
    assert text.isascii()


@pytest.mark.parametrize("builder", [hand_made_document, seeded_document], ids=["hand-made", "seeded"])
def test_the_eight_headings_appear_once_each_in_order(builder):
    _, text = builder()

    headings = [line[3:] for line in text.split("\n") if line.startswith("## ")]

    assert headings == DOC_HEADINGS
    assert [line for line in text.split("\n") if line.startswith("# ")] == ["# Economics Summary"]


def test_rendering_twice_gives_identical_text_and_leaves_the_inputs_unchanged():
    table, charges = economics_inputs(300, seed=21)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    frames_before = {
        name: getattr(results, name).copy(deep=True)
        for name in ("one_at_a_time", "save_rate_by_months", "break_even_by_cost_and_months", "k_sweep")
    }
    base_before = results.base_case

    first = render_document(results, ECON_ASSUMPTIONS, 300)
    second = render_document(results, ECON_ASSUMPTIONS, 300)

    assert first == second
    assert results.base_case == base_before
    for name, frame in frames_before.items():
        pd.testing.assert_frame_equal(getattr(results, name), frame)


def test_the_required_phrases_appear():
    _, text = seeded_document()

    assert text.count("Simulated result under stated assumptions") == 5
    assert "currency units" in text
    assert "not estimated from the data" in text
    assert "No ROI is calculated" in text
    assert "not a recommendation to change K" in text
    assert "revenue proxy, not profit" in text
    assert SNAPSHOT_NOTICE in text
    assert re.search(r"top 10% of the scored customers: 30 of 300 customers", text)


def test_the_hand_made_base_case_prints_the_hand_computed_figures():
    _, text = hand_made_document()

    base = markdown_tables(text)[1]

    assert base[0] == ["Figure", "Value"]
    assert [row[1] for row in base[1:]] == [
        "3", "60.00", "2.40", "0.48", "115.20", "55.20", "18.40", "10.4%",
    ]
    assert [row[0] for row in base[1:]] == [
        "Customers contacted",
        "Total offer cost (currency units)",
        "Expected would-be churners",
        "Expected customers saved",
        "Gross revenue kept (currency units)",
        "Net expected value (currency units)",
        "Expected value per contacted customer (currency units)",
        "Break-even save rate",
    ]


def test_every_table_cell_equals_the_matching_result_formatted_independently():
    results, text = seeded_document()
    tables = markdown_tables(text)
    assert len(tables) == 6

    base = results.base_case
    assert [row[1] for row in tables[1][1:]] == [
        "{:,}".format(base.customers_contacted),
        money_text(base.total_offer_cost),
        money_text(base.expected_would_be_churners),
        money_text(base.expected_customers_saved),
        money_text(base.gross_revenue_kept),
        money_text(base.net_expected_value),
        money_text(base.ev_per_contacted_customer),
        break_even_text(base.break_even_save_rate),
    ]

    rows = tables[2][1:]
    assert len(rows) == 16
    labels = {"save_rate": "Save rate", "months_kept": "Months kept", "offer_cost": "Offer cost", "arpu_multiple": "ARPU multiple"}
    for shown, (_, result) in zip(rows, results.one_at_a_time.iterrows()):
        name, value = result["input"], result["value"]
        expected_value = {
            "save_rate": whole_percent_text(value),
            "months_kept": str(int(value)),
            "offer_cost": str(int(value)),
            "arpu_multiple": "{:.1f}x".format(value),
        }[name]
        assert shown == [
            labels[name],
            expected_value,
            money_text(result["net_expected_value"]),
            money_text(result["ev_per_contacted_customer"]),
            break_even_text(result["break_even_save_rate"]),
        ]

    by_months = tables[3]
    assert len(by_months) == 1 + 5 and all(len(row) == 5 for row in by_months)
    assert by_months[0] == ["Save rate", "1 month", "3 months", "6 months", "12 months"]
    for shown, rate in zip(by_months[1:], results.save_rate_by_months.index):
        assert shown == [whole_percent_text(rate)] + [
            money_text(results.save_rate_by_months.loc[rate, m]) for m in MONTHS_KEPT_SWEEP
        ]

    by_cost = tables[4]
    assert len(by_cost) == 1 + 4 and all(len(row) == 5 for row in by_cost)
    assert by_cost[0] == ["Months kept", "Offer cost 10", "Offer cost 20", "Offer cost 40", "Offer cost 60"]
    for shown, months in zip(by_cost[1:], results.break_even_by_cost_and_months.index):
        assert shown == [str(months)] + [
            break_even_text(results.break_even_by_cost_and_months.loc[months, c]) for c in OFFER_COST_SWEEP
        ]

    sweep = tables[5]
    assert len(sweep) == 1 + 5
    for shown, (_, result) in zip(sweep[1:], results.k_sweep.iterrows()):
        assert shown == [
            whole_percent_text(result["k"]),
            "{:,}".format(int(result["customers_contacted"])),
            money_text(result["arpu_proxy"]),
            money_text(result["expected_would_be_churners"]),
            money_text(result["net_expected_value"]),
            money_text(result["ev_per_contacted_customer"]),
            break_even_text(result["break_even_save_rate"]),
        ]


def test_the_derived_arpu_wording_appears_only_for_a_derived_arpu():
    _, derived = seeded_document()
    _, override = seeded_document(EconomicsAssumptions(0.20, 3, 20.0, arpu_proxy=95.5))

    assert "derived: mean monthly_charge of the 30 flagged customers" in derived
    assert "not an observed value" not in derived
    assert "derived: mean monthly_charge" not in override
    assert "not an observed value" in override
    assert "user-supplied what-if value" in override
    assert "95.50" in override
    assert "the same ARPU override of 95.50 currency units is used for every K" in override
    assert "derived separately from each K's flagged group" in derived


def test_a_break_even_above_one_prints_the_cannot_pay_off_text_and_a_normal_run_never_does():
    table, charges = hand_made_inputs()
    table["churn_probability"] = [0.1, 0.1, 0.1]
    charges = pd.Series([10.0, 10.0, 10.0], index=["A", "B", "C"])
    assumptions = EconomicsAssumptions(0.20, 1, 20.0)
    results = analyze_economics(table, charges, 1.0, [1.0], assumptions)

    text = render_document(results, assumptions, 3, k_frac=1.0)
    tables = markdown_tables(text)

    assert tables[1][-1] == ["Break-even save rate", BREAK_EVEN_ABOVE_ONE]
    assert any(BREAK_EVEN_ABOVE_ONE in cell for row in tables[4] for cell in row)
    assert "N/A" not in text
    assert "20.0" not in text
    _, normal = seeded_document()
    assert BREAK_EVEN_ABOVE_ONE not in normal


def test_exactly_one_hundred_percent_prints_as_a_number():
    table, charges = hand_made_inputs()
    base = analyze_economics(table, charges, 1.0, [1.0], ECON_ASSUMPTIONS)
    modified = dataclasses.replace(
        base,
        base_case=dataclasses.replace(base.base_case, break_even_save_rate=1.0),
    )

    text = render_document(modified, ECON_ASSUMPTIONS, 3, k_frac=1.0)

    assert markdown_tables(text)[1][-1] == ["Break-even save rate", "100.0%"]


@pytest.mark.parametrize("builder", [hand_made_document, seeded_document], ids=["hand-made", "seeded"])
def test_no_nan_inf_or_na_ever_appears(builder):
    _, text = builder()

    assert not re.search(r"\b(nan|inf|N/A)\b", text, flags=re.IGNORECASE)
    assert "None" not in text


def test_months_kept_and_offer_cost_print_as_whole_numbers():
    results, text = seeded_document()

    for row in markdown_tables(text)[2][1:]:
        if row[0] in ("Months kept", "Offer cost"):
            assert re.fullmatch(r"\d+", row[1]), row
    assert results.one_at_a_time["value"].dtype == np.float64


def test_a_save_rate_with_float_noise_prints_as_a_whole_percentage():
    noisy = 0.1 + 0.2
    assert noisy != 0.3
    assumptions = EconomicsAssumptions(noisy, 3, 20.0)
    results, text = seeded_document(assumptions)
    modified = results.one_at_a_time.copy()
    modified.loc[modified["input"] == "save_rate", "value"] = [0.05, 0.10, 0.20, noisy, 0.40]
    text = render_document(dataclasses.replace(results, one_at_a_time=modified), assumptions, 300)

    assert "save rate 30%" in text
    assert "30000000" not in text
    assert ["Save rate", "30%"] == markdown_tables(text)[2][4][:2]


DOCUMENT_REFUSALS = {
    "results-not-results": lambda r, a: dict(results={"base_case": 1}),
    "assumptions-not-assumptions": lambda r, a: dict(assumptions={"save_rate": 0.2}),
    "k-zero": lambda r, a: dict(k_frac=0),
    "k-above-one": lambda r, a: dict(k_frac=1.5),
    "k-bool": lambda r, a: dict(k_frac=True),
    "scored-below-contacted": lambda r, a: dict(customers_scored=29),
    "k-disagrees-with-the-results": lambda r, a: dict(k_frac=0.20),
    "scored-disagrees-with-the-results": lambda r, a: dict(customers_scored=1408),
    "scored-float": lambda r, a: dict(customers_scored=300.0),
    "scored-bool": lambda r, a: dict(customers_scored=True),
    "scored-text": lambda r, a: dict(customers_scored="300"),
    "scored-none": lambda r, a: dict(customers_scored=None),
    "override-disagrees-with-results": lambda r, a: dict(
        assumptions=EconomicsAssumptions(0.20, 3, 20.0, arpu_proxy=95.5)
    ),
}


@pytest.mark.parametrize("build", list(DOCUMENT_REFUSALS.values()), ids=list(DOCUMENT_REFUSALS))
def test_bad_document_arguments_raise_value_error(build):
    table, charges = economics_inputs(300, seed=21)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    arguments = dict(
        results=results,
        assumptions=ECON_ASSUMPTIONS,
        k_frac=0.10,
        customers_scored=300,
        scoring_run_id=DOC_SCORING_RUN_ID,
        model_run_id=DOC_MODEL_RUN_ID,
    )
    arguments.update(build(results, ECON_ASSUMPTIONS))

    with pytest.raises(ValueError):
        render_economics_document(**arguments)


@pytest.mark.parametrize("field", ["scoring_run_id", "model_run_id"])
@pytest.mark.parametrize("bad", ["", "not-an-id", "20261011T123500", "../20261011T123500Z", None], ids=repr)
def test_a_bad_run_id_is_refused(field, bad):
    table, charges = economics_inputs(300, seed=21)
    results = analyze_economics(table, charges, 0.10, ECON_K_VALUES, ECON_ASSUMPTIONS)
    arguments = dict(
        results=results,
        assumptions=ECON_ASSUMPTIONS,
        k_frac=0.10,
        customers_scored=300,
        scoring_run_id=DOC_SCORING_RUN_ID,
        model_run_id=DOC_MODEL_RUN_ID,
    )
    arguments[field] = bad

    with pytest.raises(OutputGenerationError):
        render_economics_document(**arguments)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_break_even_that_is_not_finite_in_the_base_case_is_refused(bad):
    table, charges = hand_made_inputs()
    base = analyze_economics(table, charges, 1.0, [1.0], ECON_ASSUMPTIONS)
    damaged = dataclasses.replace(
        base, base_case=dataclasses.replace(base.base_case, break_even_save_rate=bad)
    )

    with pytest.raises(ValueError, match="break-even save rate is not finite"):
        render_document(damaged, ECON_ASSUMPTIONS, 3, k_frac=1.0)


# =============================================================================
# Section I: Recommendation facts
# =============================================================================

FACT_K = 0.2
FACT_EDGE = 0.5
FACT_IDS = [f"C{i:02d}" for i in range(1, 11)]
FACT_PROBABILITIES = [0.95, 0.90, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05]
FACT_CHARGES = [100, 80, 60, 70, 50, 40, 90, 30, 20, 10]
FACT_TENURES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
FACT_CONTRACTS = [
    "Month-to-Month", "Month-to-Month", "One Year", "Two Year", "Month-to-Month",
    "One Year", "Two Year", "Two Year", "Month-to-Month", "One Year",
]
FACT_INTERNET = [
    "Fiber Optic", "Cable", None, "DSL", "Fiber Optic", None, "DSL", "Cable", None, "Fiber Optic",
]
R, L = DIRECTION_RAISES, DIRECTION_LOWERS
FACT_DRIVERS = [
    [("contract", R), ("monthly_charge", R), ("tenure_in_months", L)],
    [("contract", R), ("age", L), ("monthly_charge", R)],
    [("contract", L), ("age", R), ("tenure_in_months", R)],
    [("age", R), ("contract", L), ("tenure_in_months", L)],
    [("tenure_in_months", L), ("contract", R), ("age", L)],
    [("contract", L), ("monthly_charge", L), ("age", R)],
    [("monthly_charge", R), ("age", R), ("contract", R)],
    [("contract", R), ("tenure_in_months", R), ("age", L)],
    [("age", L), ("contract", L), ("monthly_charge", R)],
    [("monthly_charge", L), ("contract", R), ("tenure_in_months", L)],
]
FACT_OTHER_FEATURES = [
    "contract", "number_of_referrals", "monthly_charge", "age", "tenure_in_months", "gender",
    "offer", "dependents", "latitude", "longitude", "population", "married",
]


def fact_table():
    return pd.DataFrame(
        {
            "customer_id": pd.Series(FACT_IDS, dtype=object),
            "churn_probability": pd.Series(FACT_PROBABILITIES, dtype="float64"),
            "rank": pd.Series(range(1, 11), dtype="int64"),
            "in_top_k": pd.Series([True, True] + [False] * 8, dtype=bool),
        }
    )


def fact_inputs():
    return pd.DataFrame(
        {
            "monthly_charge": FACT_CHARGES,
            "tenure_in_months": FACT_TENURES,
            "contract": FACT_CONTRACTS,
            "internet_type": np.array(FACT_INTERNET, dtype=object),
        },
        index=pd.Index(FACT_IDS, name="customer_id"),
    )


def fact_drivers(rows=None):
    rows = FACT_DRIVERS if rows is None else rows
    flat = [[item for pair in row for item in pair] for row in rows]
    return pd.DataFrame(flat, index=pd.Index(FACT_IDS, name="customer_id"), columns=DRIVER_COLUMNS, dtype=object)


def fact_importance(internet_position):
    names = list(FACT_OTHER_FEATURES)
    names.insert(internet_position, "internet_type")
    weights = np.arange(len(names), 0, -1, dtype=float)
    return pd.DataFrame(
        {"share": weights / weights.sum(), "rank": np.arange(1, len(names) + 1, dtype="int64")},
        index=pd.Index(names, name="feature"),
    )


def fact_arguments(internet_position=2):
    return dict(
        table=fact_table(),
        inputs=fact_inputs(),
        drivers=fact_drivers(),
        importance=fact_importance(internet_position),
        k_frac=FACT_K,
        medium_edge=FACT_EDGE,
    )


def assert_same_facts(first, second):
    assert (first.customers_scored, first.customers_flagged) == (second.customers_scored, second.customers_flagged)
    pd.testing.assert_frame_equal(first.tiers, second.tiers, check_exact=True)
    pd.testing.assert_frame_equal(first.numeric_profile, second.numeric_profile, check_exact=True)
    assert list(first.categorical_profile) == list(second.categorical_profile)
    for column in first.categorical_profile:
        pd.testing.assert_frame_equal(
            first.categorical_profile[column], second.categorical_profile[column], check_exact=True
        )
    pd.testing.assert_frame_equal(first.importance_top, second.importance_top, check_exact=True)
    assert first.internet_type_importance == second.internet_type_importance
    pd.testing.assert_frame_equal(first.flagged_drivers, second.flagged_drivers, check_exact=True)
    pd.testing.assert_frame_equal(first.all_drivers, second.all_drivers, check_exact=True)


# Hand-made population


def test_part_zero_the_sentinel_equals_the_preprocessing_sentinel():
    assert NO_INTERNET_CATEGORY == INTERNET_TYPE_SENTINEL == "No Internet"


def test_hand_made_tier_facts():
    facts = compute_recommendation_facts(**fact_arguments())

    assert list(facts.tiers.index) == ["High", "Medium", "Low"]
    assert facts.tiers.index.name == "tier"
    assert list(facts.tiers["customers"]) == [2, 3, 5]
    assert list(facts.tiers["share"]) == pytest.approx([0.2, 0.3, 0.5])
    assert list(facts.tiers["min_probability"]) == pytest.approx([0.90, 0.60, 0.05])
    assert list(facts.tiers["max_probability"]) == pytest.approx([0.95, 0.80, 0.40])
    assert (facts.customers_scored, facts.customers_flagged) == (10, 2)


def test_hand_made_numeric_profile():
    facts = compute_recommendation_facts(**fact_arguments())
    profile = facts.numeric_profile

    assert list(profile.index) == ["monthly_charge", "tenure_in_months"]
    assert list(profile.columns) == ["group_mean", "all_mean"]
    assert profile.loc["monthly_charge", "group_mean"] == pytest.approx(90.0)
    assert profile.loc["monthly_charge", "all_mean"] == pytest.approx(55.0)
    assert profile.loc["tenure_in_months", "group_mean"] == pytest.approx(3.0)
    assert profile.loc["tenure_in_months", "all_mean"] == pytest.approx(11.0)


def test_hand_made_categorical_profiles_with_ties_ordered_by_name():
    facts = compute_recommendation_facts(**fact_arguments())

    contract = facts.categorical_profile["contract"]
    assert list(contract.index) == ["Month-to-Month", "One Year", "Two Year"]
    assert list(contract["group_count"]) == [2, 0, 0]
    assert list(contract["group_share"]) == pytest.approx([1.0, 0.0, 0.0])
    assert list(contract["all_count"]) == [4, 3, 3]
    assert list(contract["all_share"]) == pytest.approx([0.4, 0.3, 0.3])

    internet = facts.categorical_profile["internet_type"]
    assert list(internet.index) == ["Fiber Optic", "No Internet", "Cable", "DSL"]
    assert list(internet["group_count"]) == [1, 0, 1, 0]
    assert list(internet["group_share"]) == pytest.approx([0.5, 0.0, 0.5, 0.0])
    assert list(internet["all_count"]) == [3, 3, 2, 2]
    assert list(internet["all_share"]) == pytest.approx([0.3, 0.3, 0.2, 0.2])
    assert list(facts.categorical_profile) == ["contract", "internet_type"]


def test_hand_made_driver_facts_for_the_group_and_for_everyone():
    facts = compute_recommendation_facts(**fact_arguments())

    flagged = facts.flagged_drivers
    assert flagged.index.name == "feature"
    assert list(flagged.columns) == ["first_driver_count", "any_driver_count", "raises_count", "lowers_count"]
    assert list(flagged.index) == ["contract", "monthly_charge", "age", "tenure_in_months"]
    assert flagged.loc["contract"].tolist() == [2, 2, 2, 0]
    assert flagged.loc["monthly_charge"].tolist() == [0, 2, 2, 0]
    assert flagged.loc["age"].tolist() == [0, 1, 0, 1]
    assert flagged.loc["tenure_in_months"].tolist() == [0, 1, 0, 1]

    everyone = facts.all_drivers
    assert list(everyone.index) == ["contract", "age", "monthly_charge", "tenure_in_months"]
    assert everyone.loc["contract"].tolist() == [5, 10, 6, 4]
    assert everyone.loc["age"].tolist() == [2, 8, 4, 4]
    assert everyone.loc["monthly_charge"].tolist() == [2, 6, 4, 2]
    assert everyone.loc["tenure_in_months"].tolist() == [1, 6, 2, 4]
    assert (everyone.dtypes == np.int64).all()


@pytest.mark.parametrize("position,inside", [(2, True), (11, False)], ids=["inside-the-top-rows", "outside-the-top-rows"])
def test_the_internet_type_row_is_always_reported(position, inside):
    facts = compute_recommendation_facts(**fact_arguments(internet_position=position))
    importance = fact_importance(position)

    assert len(facts.importance_top) == TOP_N_IMPORTANCE == 10
    assert ("internet_type" in facts.importance_top.index) is inside
    assert facts.internet_type_importance.in_top_rows is inside
    assert facts.internet_type_importance.rank == position + 1
    assert facts.internet_type_importance.share == importance.loc["internet_type", "share"]
    pd.testing.assert_frame_equal(facts.importance_top, importance.head(10))


# Independent recomputation on seeded data


def seeded_fact_data(seed=5, n_rows=300, k_frac=0.10, medium_edge=0.25):
    rng = np.random.default_rng(seed)
    ids = [f"C{i:04d}" for i in range(n_rows)]
    table = (
        pd.DataFrame({"customer_id": ids, "churn_probability": rng.beta(2, 5, n_rows)})
        .sort_values(["churn_probability", "customer_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    table["rank"] = np.arange(1, n_rows + 1, dtype="int64")
    table["in_top_k"] = table["rank"] <= top_k_count(n_rows, k_frac)
    options = ["DSL", "Cable", "Fiber Optic", None]
    inputs = pd.DataFrame(
        {
            "monthly_charge": rng.uniform(20, 120, n_rows).round(2),
            "tenure_in_months": rng.integers(1, 73, n_rows),
            "contract": [["Month-to-Month", "One Year", "Two Year"][i] for i in rng.integers(0, 3, n_rows)],
            "internet_type": np.array([options[i] for i in rng.integers(0, 4, n_rows)], dtype=object),
        },
        index=pd.Index(ids, name="customer_id"),
    )
    pool = ["contract", "age", "monthly_charge", "tenure_in_months", "latitude", "dependents"]
    rows = []
    for _ in range(n_rows):
        chosen = [pool[i] for i in rng.permutation(len(pool))[:3]]
        rows.append([item for name in chosen for item in (name, [R, L][int(rng.integers(0, 2))])])
    drivers = pd.DataFrame(rows, index=inputs.index, columns=DRIVER_COLUMNS, dtype=object)
    names = [*pool, "internet_type", "gender", "offer", "married", "population", "number_of_dependents"]
    names = [names[i] for i in rng.permutation(len(names))]
    shares = np.sort(rng.random(len(names)))[::-1]
    importance = pd.DataFrame(
        {"share": shares / shares.sum(), "rank": np.arange(1, len(names) + 1, dtype="int64")},
        index=pd.Index(names, name="feature"),
    )
    return dict(
        table=table, inputs=inputs, drivers=drivers, importance=importance, k_frac=k_frac, medium_edge=medium_edge
    )


def test_seeded_facts_match_a_plain_python_recomputation():
    data = seeded_fact_data()
    facts = compute_recommendation_facts(**data)
    table, inputs, drivers = data["table"], data["inputs"], data["drivers"]
    n_rows = len(table)
    high, upto_medium = top_k_count(n_rows, 0.10), top_k_count(n_rows, 0.25)

    for tier, low_rank, high_rank in (("High", 1, high), ("Medium", high + 1, upto_medium), ("Low", upto_medium + 1, n_rows)):
        rows = [r for r in table.itertuples() if low_rank <= r.rank <= high_rank]
        row = facts.tiers.loc[tier]
        assert row["customers"] == len(rows)
        assert row["share"] == pytest.approx(len(rows) / n_rows, rel=1e-12)
        assert row["min_probability"] == min(r.churn_probability for r in rows)
        assert row["max_probability"] == max(r.churn_probability for r in rows)

    group_ids = [r.customer_id for r in table.itertuples() if r.rank <= high]
    for column in ("monthly_charge", "tenure_in_months"):
        group_mean = sum(float(inputs.loc[c, column]) for c in group_ids) / len(group_ids)
        all_mean = sum(float(v) for v in inputs[column]) / n_rows
        assert facts.numeric_profile.loc[column, "group_mean"] == pytest.approx(group_mean, rel=1e-12)
        assert facts.numeric_profile.loc[column, "all_mean"] == pytest.approx(all_mean, rel=1e-12)

    for column in ("contract", "internet_type"):
        label = lambda v: "No Internet" if v is None else v
        for category, row in facts.categorical_profile[column].iterrows():
            in_all = [c for c in inputs.index if label(inputs.loc[c, column]) == category]
            in_group = [c for c in group_ids if label(inputs.loc[c, column]) == category]
            assert row["all_count"] == len(in_all)
            assert row["group_count"] == len(in_group)
            assert row["all_share"] == pytest.approx(len(in_all) / n_rows, rel=1e-12)
            assert row["group_share"] == pytest.approx(len(in_group) / len(group_ids), rel=1e-12)
        assert facts.categorical_profile[column]["all_count"].sum() == n_rows
        assert facts.categorical_profile[column]["group_share"].sum() == pytest.approx(1.0)

    for scope, frame in ((group_ids, facts.flagged_drivers), (list(drivers.index), facts.all_drivers)):
        for feature, row in frame.iterrows():
            first = anywhere = raised = lowered = 0
            for customer in scope:
                values = drivers.loc[customer].tolist()
                names, directions = values[0::2], values[1::2]
                first += names[0] == feature
                if feature in names:
                    anywhere += 1
                    raised += directions[names.index(feature)] == R
                    lowered += directions[names.index(feature)] == L
            assert row.tolist() == [first, anywhere, raised, lowered]
        assert frame["first_driver_count"].sum() == len(scope)
        assert frame["any_driver_count"].sum() == 3 * len(scope)
        keys = [(-r.first_driver_count, -r.any_driver_count, name) for name, r in frame.iterrows()]
        assert keys == sorted(keys)


def test_shuffled_table_inputs_and_drivers_give_identical_facts():
    data = seeded_fact_data()
    reference = compute_recommendation_facts(**data)
    shuffled = dict(data)
    shuffled["table"] = data["table"].sample(frac=1, random_state=1).reset_index(drop=True)
    shuffled["inputs"] = data["inputs"].sample(frac=1, random_state=2)
    shuffled["drivers"] = data["drivers"].sample(frac=1, random_state=3)

    assert_same_facts(reference, compute_recommendation_facts(**shuffled))


def test_values_attached_to_the_wrong_ids_change_the_facts():
    data = seeded_fact_data()
    reference = compute_recommendation_facts(**data)
    wrong = dict(data)
    wrong["inputs"] = pd.DataFrame(data["inputs"].to_numpy(), index=data["inputs"].index[::-1], columns=data["inputs"].columns)

    changed = compute_recommendation_facts(**wrong)

    assert changed.numeric_profile.loc["monthly_charge", "group_mean"] != reference.numeric_profile.loc["monthly_charge", "group_mean"]


def test_the_fact_inputs_are_not_changed_by_a_call():
    data = seeded_fact_data()
    before = {name: value.copy(deep=True) for name, value in data.items() if isinstance(value, pd.DataFrame)}

    compute_recommendation_facts(**data)

    for name, frame in before.items():
        pd.testing.assert_frame_equal(data[name], frame)


# Empty tiers


def tier_sizes(n_rows, k_frac, medium_edge):
    tiers = assign_risk_tiers(pd.Series(np.arange(1, n_rows + 1)), k_frac, medium_edge)
    return [int((tiers == name).sum()) for name in ("High", "Medium", "Low")]


def assert_other_facts_are_complete(facts, n_rows):
    assert facts.customers_scored == n_rows
    assert facts.tiers["customers"].sum() == n_rows
    assert facts.tiers["share"].sum() == pytest.approx(1.0)
    for column in PROFILE_CATEGORICAL_COLUMNS:
        assert facts.categorical_profile[column]["all_share"].sum() == pytest.approx(1.0)
        assert facts.categorical_profile[column]["group_share"].sum() == pytest.approx(1.0)
    assert facts.flagged_drivers["any_driver_count"].sum() == 3 * facts.customers_flagged
    assert facts.all_drivers["first_driver_count"].sum() == n_rows
    assert np.isfinite(facts.numeric_profile.to_numpy()).all()


def test_an_empty_medium_tier_has_zero_customers_and_nan_probabilities():
    n_rows, k_frac, medium_edge = 2, 0.10, 0.25
    assert tier_sizes(n_rows, k_frac, medium_edge) == [1, 0, 1]
    data = seeded_fact_data(n_rows=n_rows, k_frac=k_frac, medium_edge=medium_edge)

    facts = compute_recommendation_facts(**data)

    medium = facts.tiers.loc["Medium"]
    assert medium["customers"] == 0
    assert medium["share"] == 0.0
    assert np.isnan(medium["min_probability"]) and np.isnan(medium["max_probability"])
    top, bottom = data["table"]["churn_probability"].tolist()
    assert facts.tiers.loc["High"].tolist() == [1, 0.5, top, top]
    assert facts.tiers.loc["Low"].tolist() == [1, 0.5, bottom, bottom]
    assert facts.customers_flagged == 1
    assert_other_facts_are_complete(facts, n_rows)


def test_an_empty_low_tier_has_zero_customers_and_nan_probabilities():
    n_rows, k_frac, medium_edge = 10, 0.10, 1.0
    assert tier_sizes(n_rows, k_frac, medium_edge) == [1, 9, 0]
    data = seeded_fact_data(n_rows=n_rows, k_frac=k_frac, medium_edge=medium_edge)

    facts = compute_recommendation_facts(**data)

    low = facts.tiers.loc["Low"]
    assert low["customers"] == 0
    assert low["share"] == 0.0
    assert np.isnan(low["min_probability"]) and np.isnan(low["max_probability"])
    probabilities = data["table"]["churn_probability"].tolist()
    assert facts.tiers.loc["High"].tolist() == [1, 0.1, probabilities[0], probabilities[0]]
    assert facts.tiers.loc["Medium"].tolist() == [9, 0.9, min(probabilities[1:]), max(probabilities[1:])]
    assert facts.customers_flagged == 1
    assert_other_facts_are_complete(facts, n_rows)


# Ties in the driver order


def test_driver_ties_are_ordered_by_first_count_then_any_count_then_name():
    rows = [
        [("b", R), ("a", R), ("z", L)],
        [("a", R), ("b", L), ("y", R)],
        [("z", R), ("a", R), ("b", R)],
        [("y", L), ("z", R), ("a", L)],
    ]
    ids = FACT_IDS[:4]
    table = fact_table().iloc[:4].copy()
    table["rank"] = pd.Series(range(1, 5), dtype="int64")
    table["in_top_k"] = pd.Series([True, False, False, False])
    arguments = dict(
        table=table,
        inputs=fact_inputs().loc[ids],
        drivers=fact_drivers(rows + FACT_DRIVERS[4:]).loc[ids],
        importance=fact_importance(2),
        k_frac=0.25,
        medium_edge=0.5,
    )

    facts = compute_recommendation_facts(**arguments)

    assert list(facts.all_drivers.index) == ["a", "b", "z", "y"]
    assert facts.all_drivers["first_driver_count"].tolist() == [1, 1, 1, 1]
    assert facts.all_drivers["any_driver_count"].tolist() == [4, 3, 3, 2]
    keys = [(-r.first_driver_count, -r.any_driver_count, n) for n, r in facts.all_drivers.iterrows()]
    assert keys == sorted(keys)
    assert list(facts.flagged_drivers.index) == ["b", "a", "z"]


# Refusals


def mutate_table(change):
    def mutate(table, inputs, drivers, importance):
        table = table.copy()
        change(table)
        return table, inputs, drivers, importance

    return mutate


def mutate_inputs(change):
    def mutate(table, inputs, drivers, importance):
        inputs = inputs.copy()
        inputs = change(inputs)
        return table, inputs, drivers, importance

    return mutate


def mutate_drivers(change):
    def mutate(table, inputs, drivers, importance):
        drivers = drivers.copy()
        drivers = change(drivers)
        return table, inputs, drivers, importance

    return mutate


def mutate_importance(change):
    def mutate(table, inputs, drivers, importance):
        importance = importance.copy()
        importance = change(importance)
        return table, inputs, drivers, importance

    return mutate


def set_cell(frame, label, column, value):
    frame[column] = frame[column].astype(object)
    frame.at[label, column] = value
    return frame


FACT_REFUSALS = {
    "empty-table": (lambda t, i, d, m: (t.iloc[0:0], i, d, m), "no rows"),
    "duplicate-id-in-table": (
        mutate_table(lambda t: t.__setitem__("customer_id", ["C01"] + FACT_IDS[1:-1] + ["C01"])),
        "duplicate customer id in the table",
    ),
    "duplicate-id-in-inputs": (
        mutate_inputs(lambda i: i.set_axis(["C01"] + FACT_IDS[1:-1] + ["C01"])),
        "duplicate customer id in the inputs",
    ),
    "duplicate-id-in-drivers": (
        mutate_drivers(lambda d: d.set_axis(["C01"] + FACT_IDS[1:-1] + ["C01"])),
        "duplicate customer id in the driver table",
    ),
    "inputs-missing-a-customer": (mutate_inputs(lambda i: i.drop(index="C05")), "the inputs has no row for customer"),
    "inputs-with-an-extra-customer": (
        mutate_inputs(lambda i: pd.concat([i, i.iloc[[0]].set_axis(["C99"])])),
        "unknown customer 'C99'",
    ),
    "drivers-missing-a-customer": (
        mutate_drivers(lambda d: d.drop(index="C05")),
        "the driver table has no row for customer",
    ),
    "drivers-with-an-extra-customer": (
        mutate_drivers(lambda d: pd.concat([d, d.iloc[[0]].set_axis(["C99"])])),
        "unknown customer 'C99'",
    ),
    "rank-gap": (mutate_table(lambda t: t.__setitem__("rank", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11])), "complete 1.."),
    "rank-float": (
        mutate_table(lambda t: t.__setitem__("rank", t["rank"].astype("float64"))),
        "rank must be a complete integer",
    ),
    "probability-nan": (
        mutate_table(lambda t: t.__setitem__("churn_probability", [np.nan] + FACT_PROBABILITIES[1:])),
        "churn_probability is missing",
    ),
    "probability-above-one": (
        mutate_table(lambda t: t.__setitem__("churn_probability", [1.5] + FACT_PROBABILITIES[1:])),
        "outside [0, 1]",
    ),
    "flags-not-boolean": (
        mutate_table(lambda t: t.__setitem__("in_top_k", t["in_top_k"].astype(int))),
        "in_top_k must be a boolean",
    ),
    "flag-count-wrong": (
        mutate_table(lambda t: t.__setitem__("in_top_k", [True] * 3 + [False] * 7)),
        "marks 3 rows but top_k_count for k=0.2 is 2",
    ),
    "right-count-wrong-customers": (
        mutate_table(lambda t: t.__setitem__("in_top_k", [True, False, True] + [False] * 7)),
        "not the first 2 rows by rank",
    ),
    "monthly-charge-missing": (
        mutate_inputs(lambda i: set_cell(i, "C03", "monthly_charge", np.nan)),
        "monthly_charge is missing for customer 'C03'",
    ),
    "tenure-missing": (
        mutate_inputs(lambda i: set_cell(i, "C03", "tenure_in_months", None)),
        "tenure_in_months is missing",
    ),
    "monthly-charge-text": (
        mutate_inputs(lambda i: set_cell(i, "C03", "monthly_charge", "60")),
        "monthly_charge is not a number",
    ),
    "tenure-infinite": (
        mutate_inputs(lambda i: set_cell(i, "C03", "tenure_in_months", np.inf)),
        "tenure_in_months is not finite",
    ),
    "contract-missing": (
        mutate_inputs(lambda i: set_cell(i, "C03", "contract", None)),
        "contract is missing for customer 'C03'",
    ),
    "contract-not-text": (
        mutate_inputs(lambda i: set_cell(i, "C03", "contract", 5)),
        "contract is not text",
    ),
    "internet-type-not-text": (
        mutate_inputs(lambda i: set_cell(i, "C01", "internet_type", 5)),
        "internet_type is not text",
    ),
    "internet-type-holds-the-reserved-category": (
        mutate_inputs(lambda i: set_cell(i, "C01", "internet_type", "No Internet")),
        "reserved category 'No Internet'",
    ),
    "driver-repeated-in-a-row": (
        mutate_drivers(lambda d: set_cell(d, "C02", "driver_2", "contract")),
        "does not have three different drivers",
    ),
    "driver-missing": (
        mutate_drivers(lambda d: set_cell(d, "C02", "driver_3", None)),
        "missing or not text",
    ),
    "driver-direction-invalid": (
        mutate_drivers(lambda d: set_cell(d, "C02", "driver_1_direction", "up")),
        "direction is not valid",
    ),
    "importance-without-internet-type": (
        mutate_importance(lambda m: m.drop(index="internet_type").assign(rank=range(1, len(m)))),
        "no internet_type row",
    ),
    "importance-ranks-wrong": (
        mutate_importance(lambda m: m.assign(rank=list(range(2, len(m) + 2)))),
        "ranks are not 1..n",
    ),
    "importance-negative-share": (
        mutate_importance(lambda m: m.assign(share=[-0.1] + list(m["share"][1:]))),
        "negative or not finite",
    ),
    "importance-empty": (mutate_importance(lambda m: m.iloc[0:0]), "no rows"),
    "importance-nan-share": (
        mutate_importance(lambda m: set_cell(m, m.index[0], "share", np.nan)),
        "negative or not finite",
    ),
    "importance-infinite-share": (
        mutate_importance(lambda m: set_cell(m, m.index[0], "share", np.inf)),
        "negative or not finite",
    ),
    "importance-huge-integer-share": (
        mutate_importance(lambda m: set_cell(m, m.index[0], "share", 10**400)),
        "negative or not finite",
    ),
    "importance-text-share": (
        mutate_importance(lambda m: set_cell(m, m.index[0], "share", "high")),
        "negative or not finite",
    ),
    "rank-bool": (
        mutate_table(lambda t: t.__setitem__("rank", [True] * 10)),
        "rank must be a complete integer",
    ),
    "probability-huge-integer": (
        mutate_table(lambda t: set_cell(t, 0, "churn_probability", 10**400)),
        "churn_probability is not finite",
    ),
    "monthly-charge-huge-integer": (
        mutate_inputs(lambda i: set_cell(i, "C03", "monthly_charge", 10**400)),
        "monthly_charge is not finite",
    ),
    "tenure-huge-integer": (
        mutate_inputs(lambda i: set_cell(i, "C03", "tenure_in_months", 10**400)),
        "tenure_in_months is not finite",
    ),
    "importance-repeated-feature": (
        mutate_importance(lambda m: m.set_axis(["contract", "contract"] + list(m.index[2:]))),
        "repeated feature",
    ),
}


@pytest.mark.parametrize("mutate,fragment", list(FACT_REFUSALS.values()), ids=list(FACT_REFUSALS))
def test_unusable_data_is_refused_with_recommendations_error(mutate, fragment):
    arguments = fact_arguments()
    table, inputs, drivers, importance = mutate(
        arguments["table"], arguments["inputs"], arguments["drivers"], arguments["importance"]
    )
    arguments.update(table=table, inputs=inputs, drivers=drivers, importance=importance)

    with pytest.raises(RecommendationsError) as exc_info:
        compute_recommendation_facts(**arguments)

    assert fragment in str(exc_info.value)


FACT_CALLER_MISTAKES = {
    "table-not-a-dataframe": lambda a: dict(table=a["table"].to_dict()),
    "table-missing-rank": lambda a: dict(table=a["table"].drop(columns=["rank"])),
    "inputs-not-a-dataframe": lambda a: dict(inputs=None),
    "inputs-missing-a-profile-column": lambda a: dict(inputs=a["inputs"].drop(columns=["contract"])),
    "drivers-not-a-dataframe": lambda a: dict(drivers=[]),
    "drivers-missing-a-column": lambda a: dict(drivers=a["drivers"].drop(columns=["driver_3_direction"])),
    "importance-not-a-dataframe": lambda a: dict(importance=None),
    "importance-wrong-columns": lambda a: dict(importance=a["importance"].rename(columns={"share": "value"})),
    "k-zero": lambda a: dict(k_frac=0),
    "k-bool": lambda a: dict(k_frac=True),
    "k-text": lambda a: dict(k_frac="0.2"),
    "k-equal-to-the-edge": lambda a: dict(k_frac=0.5),
    "k-above-the-edge": lambda a: dict(k_frac=0.6),
    "k-huge-integer": lambda a: dict(k_frac=10**400),
    "edge-huge-integer": lambda a: dict(medium_edge=10**400),
    "edge-above-one": lambda a: dict(medium_edge=1.5),
    "edge-none": lambda a: dict(medium_edge=None),
}


@pytest.mark.parametrize("build", list(FACT_CALLER_MISTAKES.values()), ids=list(FACT_CALLER_MISTAKES))
def test_fact_caller_mistakes_raise_value_error(build):
    arguments = fact_arguments()
    arguments.update(build(arguments))

    with pytest.raises(ValueError):
        compute_recommendation_facts(**arguments)


# Source checks


def test_recommendations_py_imports_only_allowed_modules_and_never_the_target():
    source = inspect.getsource(recommendations)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)

    package = {name for name in imported if name.startswith("retention_platform")}
    assert package <= {
        "retention_platform.inference.drivers",
        "retention_platform.inference.economics",
        "retention_platform.inference.ranking",
    }
    assert all(name.startswith("retention_platform.inference") for name in package)
    assert imported - package <= {"__future__", "collections.abc", "math", "dataclasses", "numpy", "pandas"}
    assert TARGET_COLUMN not in source


# Real population


def test_real_population_facts_are_consistent(driver_data, driver_contributions, driver_importance):
    frame = driver_data.inputs.copy()
    frame.insert(0, "customer_id", pd.Series(driver_data.ids, dtype=object))
    table = score_batch(frame, driver_data.run, 0.10)
    inputs = driver_data.inputs.set_axis(pd.Index(driver_data.ids, name="customer_id"))
    drivers = select_top_drivers(driver_contributions)

    facts = compute_recommendation_facts(table, inputs, drivers, driver_importance, 0.10, 0.25)

    assert facts.customers_scored == len(driver_data.ids) == 1408
    assert facts.tiers["customers"].sum() == 1408
    assert list(facts.tiers["customers"][:1]) == [top_k_count(1408, 0.10)] == [141]
    assert facts.tiers["share"].sum() == pytest.approx(1.0)
    assert facts.customers_flagged == 141
    for column in PROFILE_CATEGORICAL_COLUMNS:
        profile = facts.categorical_profile[column]
        assert profile["all_share"].sum() == pytest.approx(1.0)
        assert profile["group_share"].sum() == pytest.approx(1.0)
        assert profile["group_count"].sum() == 141
    missing = int(driver_data.inputs["internet_type"].isna().sum())
    assert facts.categorical_profile["internet_type"].loc["No Internet", "all_count"] == missing
    assert facts.flagged_drivers["first_driver_count"].sum() == 141
    assert facts.all_drivers["first_driver_count"].sum() == 1408
    assert facts.all_drivers["any_driver_count"].sum() == 3 * 1408
    assert (facts.all_drivers["raises_count"] + facts.all_drivers["lowers_count"] == facts.all_drivers["any_driver_count"]).all()
    assert facts.importance_top.index[0] == driver_importance.index[0]
    assert facts.internet_type_importance.rank == int(driver_importance.loc["internet_type", "rank"])


# Recommendations

# Phrases that would turn an association into a promise or a cause. "cause" is
# not listed because two fixed templates say "not ... the cause" and "does not
# show that ... causes"; a separate test allows those two and no other use.
BANNED_PHRASES = [
    "because",
    "prevent", "prevents", "prevented",
    "guarantee", "guarantees", "guaranteed",
    "ensure", "ensures",
    "proves", "proven",
    "will reduce", "will save", "will work", "will retain", "will keep",
    "will improve", "will increase", "will pay off", "will succeed",
]
ALL_TEMPLATES = {
    "HIGH_TIER_FINDING": HIGH_TIER_FINDING,
    "HIGH_TIER_ACTION": HIGH_TIER_ACTION,
    "LEAD_DRIVER_FINDING": LEAD_DRIVER_FINDING,
    "LEAD_DRIVER_ACTION": LEAD_DRIVER_ACTION,
    "BREAK_EVEN_FINDING": BREAK_EVEN_FINDING,
    "BREAK_EVEN_ACTION": BREAK_EVEN_ACTION,
    "BREAK_EVEN_ABOVE_ONE_FINDING": BREAK_EVEN_ABOVE_ONE_FINDING,
    "BREAK_EVEN_ABOVE_ONE_ACTION": BREAK_EVEN_ABOVE_ONE_ACTION,
    "FIBER_OPTIC_FINDING": FIBER_OPTIC_FINDING,
    "FIBER_OPTIC_ACTION": FIBER_OPTIC_ACTION,
}
RECOMMENDATION_ORDER = [
    "start-with-the-high-tier",
    "one-input-leads-the-drivers",
    "break-even-save-rate",
    "fiber-optic-on-the-list",
]


def hand_base_case(break_even=0.104):
    return BaseCase(
        customers_contacted=2,
        total_offer_cost=40.0,
        expected_would_be_churners=1.85,
        expected_customers_saved=0.37,
        gross_revenue_kept=100.0,
        net_expected_value=60.0,
        ev_per_contacted_customer=30.0,
        break_even_save_rate=break_even,
        arpu_proxy=90.0,
        arpu_source="derived",
        arpu_customers_used=2,
    )


def hand_facts():
    return compute_recommendation_facts(**fact_arguments())


def facts_without_high_customers(facts):
    tiers = facts.tiers.copy()
    tiers.loc["High", "customers"] = 0
    tiers.loc["High", "share"] = 0.0
    tiers.loc["High", "min_probability"] = np.nan
    tiers.loc["High", "max_probability"] = np.nan
    return dataclasses.replace(facts, tiers=tiers)


def facts_with_lead_count(facts, flagged, count):
    drivers = facts.flagged_drivers.copy()
    drivers.iloc[0, drivers.columns.get_loc("first_driver_count")] = count
    return dataclasses.replace(facts, customers_flagged=flagged, flagged_drivers=drivers)


def facts_with_fiber(facts, **changes):
    profile = {name: frame.copy() for name, frame in facts.categorical_profile.items()}
    for column, value in changes.items():
        profile["internet_type"].loc["Fiber Optic", column] = value
    return dataclasses.replace(facts, categorical_profile=profile)


def facts_without_fiber(facts):
    profile = {name: frame.copy() for name, frame in facts.categorical_profile.items()}
    profile["internet_type"] = profile["internet_type"].drop(index="Fiber Optic")
    return dataclasses.replace(facts, categorical_profile=profile)


def identifiers(recommendations):
    return [item.identifier for item in recommendations]


def evidence_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return "{:,}".format(value)
    return "{:.1f}%".format(value * 100)


def recommendation_variants():
    facts = hand_facts()
    big = facts_with_fiber(facts, all_count=1234)
    big = dataclasses.replace(big, customers_scored=12345)
    big = facts_with_lead_count(big, 1408, 1234)
    return [
        ("all-four", facts, hand_base_case(0.104)),
        ("above-one", facts, hand_base_case(1.5)),
        ("exactly-one", facts, hand_base_case(1.0)),
        ("large-counts", big, hand_base_case(0.5)),
        ("minimal", facts_without_fiber(facts_without_high_customers(facts_with_lead_count(facts, 2, 1))), hand_base_case(0.3)),
    ]


VARIANT_IDS = [variant[0] for variant in recommendation_variants()]


def test_hand_made_facts_give_the_exact_expected_sentences():
    recommendations = select_recommendations(hand_facts(), hand_base_case(0.104), DRIVER_LABELS)

    assert identifiers(recommendations) == RECOMMENDATION_ORDER
    assert [item.finding for item in recommendations] == [
        "The High tier holds 2 customers, with churn probabilities from 90.0% to 95.0%.",
        "Contract type is the first driver for 2 of the 2 flagged customers.",
        "In the simulated base case the break-even save rate is 10.4%: the offer would pay off "
        "only if at least that share of would-be churners were kept.",
        "Fiber Optic customers are 1 of the 2 flagged customers (50.0%) and 3 of the 10 scored "
        "customers (30.0%).",
    ]
    assert [item.action for item in recommendations] == [
        "Start outreach planning with this tier, in list order. The probabilities rank customers "
        "by modelled risk; they do not show that contact will change any outcome.",
        "Treat this as a pattern in the model's scores, not as the cause of any customer's risk. "
        "Investigate whether customers with this characteristic form a segment worth a closer "
        "look before planning offers around it.",
        "If a pilot is considered, measure the real save rate and compare it with this break-even "
        "before extending the offer. The save rate used here is an assumption, not an estimate "
        "from the data.",
        "This describes who is on the list. The model's overall reliance on the internet type is "
        "a separate measure, shown in the global importance section, and it does not show that "
        "Fiber Optic service causes churn. Understanding Fiber Optic churn would need a separate "
        "evaluation.",
    ]


def test_the_above_one_version_has_the_exact_sentences():
    recommendations = select_recommendations(hand_facts(), hand_base_case(1.5), DRIVER_LABELS)
    item = next(r for r in recommendations if r.identifier == "break-even-save-rate")

    assert item.finding == (
        "In the simulated base case the break-even save rate is above 100%: the offer cannot pay "
        "off at any save rate."
    )
    assert item.action == (
        "Review the offer cost and the months kept assumptions, and see the economics document "
        "for the sensitivity tables, before planning outreach."
    )


# Conditions


def test_the_high_tier_recommendation_needs_at_least_one_high_customer():
    facts = hand_facts()

    shown = select_recommendations(facts, hand_base_case(), DRIVER_LABELS)
    hidden = select_recommendations(facts_without_high_customers(facts), hand_base_case(), DRIVER_LABELS)

    assert "start-with-the-high-tier" in identifiers(shown)
    assert "start-with-the-high-tier" not in identifiers(hidden)


def test_the_lead_driver_boundary_is_strictly_more_than_half_the_flagged_group():
    facts = hand_facts()
    assert LEAD_DRIVER_SHARE_MINIMUM == 0.5

    exactly_half = facts_with_lead_count(facts, 140, 70)
    one_more = facts_with_lead_count(facts, 140, 71)
    far_below = facts_with_lead_count(facts, 140, 10)

    assert "one-input-leads-the-drivers" not in identifiers(select_recommendations(exactly_half, hand_base_case(), DRIVER_LABELS))
    assert "one-input-leads-the-drivers" in identifiers(select_recommendations(one_more, hand_base_case(), DRIVER_LABELS))
    assert "one-input-leads-the-drivers" not in identifiers(select_recommendations(far_below, hand_base_case(), DRIVER_LABELS))
    finding = next(r for r in select_recommendations(one_more, hand_base_case(), DRIVER_LABELS) if r.identifier == "one-input-leads-the-drivers").finding
    assert finding == "Contract type is the first driver for 71 of the 140 flagged customers."


@pytest.mark.parametrize(
    "break_even,above",
    [(0.104, False), (0.9999, False), (1.0, False), (1.0000001, True), (2.5, True)],
    ids=["normal", "just-below-one", "exactly-one", "just-above-one", "far-above-one"],
)
def test_the_break_even_recommendation_has_exactly_one_version(break_even, above):
    recommendations = select_recommendations(hand_facts(), hand_base_case(break_even), DRIVER_LABELS)

    matching = [r for r in recommendations if r.identifier == "break-even-save-rate"]

    assert len(matching) == 1
    assert ("is above 100%" in matching[0].finding) is above
    assert (matching[0].action == BREAK_EVEN_ABOVE_ONE_ACTION) is above
    if not above:
        assert matching[0].finding.startswith("In the simulated base case the break-even save rate is ")
        assert "above" not in matching[0].finding


def test_exactly_one_hundred_percent_is_the_ordinary_version_and_prints_a_number():
    item = next(
        r
        for r in select_recommendations(hand_facts(), hand_base_case(1.0), DRIVER_LABELS)
        if r.identifier == "break-even-save-rate"
    )

    assert "break-even save rate is 100.0%:" in item.finding


def test_the_fiber_optic_recommendation_conditions():
    facts = hand_facts()
    assert facts.categorical_profile["internet_type"].loc["Fiber Optic", "group_share"] > facts.categorical_profile["internet_type"].loc["Fiber Optic", "all_share"]
    equal = facts_with_fiber(facts, group_share=facts.categorical_profile["internet_type"].loc["Fiber Optic", "all_share"])
    lower = facts_with_fiber(facts, group_share=0.1)

    assert "fiber-optic-on-the-list" in identifiers(select_recommendations(facts, hand_base_case(), DRIVER_LABELS))
    assert "fiber-optic-on-the-list" not in identifiers(select_recommendations(equal, hand_base_case(), DRIVER_LABELS))
    assert "fiber-optic-on-the-list" not in identifiers(select_recommendations(lower, hand_base_case(), DRIVER_LABELS))
    assert "fiber-optic-on-the-list" not in identifiers(
        select_recommendations(facts_without_fiber(facts), hand_base_case(), DRIVER_LABELS)
    )


def test_the_order_is_fixed_and_a_subset_keeps_it():
    facts = hand_facts()

    full = select_recommendations(facts, hand_base_case(), DRIVER_LABELS)
    subset = select_recommendations(
        facts_without_fiber(facts_without_high_customers(facts)), hand_base_case(), DRIVER_LABELS
    )

    assert identifiers(full) == RECOMMENDATION_ORDER
    assert identifiers(subset) == ["one-input-leads-the-drivers", "break-even-save-rate"]


def test_the_break_even_recommendation_is_always_present():
    for _, facts, base_case in recommendation_variants():
        assert "break-even-save-rate" in identifiers(select_recommendations(facts, base_case, DRIVER_LABELS))


# Evidence and wording rules


@pytest.mark.parametrize("name,facts,base_case", recommendation_variants(), ids=VARIANT_IDS)
def test_every_evidence_value_appears_formatted_in_its_finding(name, facts, base_case):
    recommendations = select_recommendations(facts, base_case, DRIVER_LABELS)

    assert recommendations
    for item in recommendations:
        assert item.evidence
        for key, value in item.evidence.items():
            assert evidence_text(value) in item.finding, (item.identifier, key, value)


def test_large_counts_use_thousands_separators():
    _, facts, base_case = next(v for v in recommendation_variants() if v[0] == "large-counts")

    findings = {r.identifier: r.finding for r in select_recommendations(facts, base_case, DRIVER_LABELS)}

    assert "1,234 of the 1,408 flagged customers" in findings["one-input-leads-the-drivers"]
    assert "1,234 of the 12,345 scored customers" in findings["fiber-optic-on-the-list"]


@pytest.mark.parametrize("name,facts,base_case", recommendation_variants(), ids=VARIANT_IDS)
def test_no_banned_word_appears_in_any_finding_or_action(name, facts, base_case):
    for item in select_recommendations(facts, base_case, DRIVER_LABELS):
        for text in (item.finding, item.action):
            for phrase in BANNED_PHRASES:
                assert re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE) is None, (phrase, text)


def test_the_word_cause_appears_only_in_the_two_fixed_negations():
    uses = {
        name: len(re.findall(r"\bcauses?\b", text, flags=re.IGNORECASE))
        for name, text in ALL_TEMPLATES.items()
        if re.search(r"\bcauses?\b", text, flags=re.IGNORECASE)
    }

    assert uses == {"LEAD_DRIVER_ACTION": 1, "FIBER_OPTIC_ACTION": 1}
    assert "not as the cause of any customer's risk" in LEAD_DRIVER_ACTION
    assert "does not show that Fiber Optic service causes churn" in FIBER_OPTIC_ACTION


def test_the_templates_hold_no_run_specific_figure():
    digits = {name: re.findall(r"\d+", text) for name, text in ALL_TEMPLATES.items() if re.search(r"\d", text)}

    assert digits == {}
    assert BREAK_EVEN_LIMIT_TEXT == "100%"
    for name in ("HIGH_TIER_FINDING", "LEAD_DRIVER_FINDING", "BREAK_EVEN_FINDING", "FIBER_OPTIC_FINDING"):
        assert "{" in ALL_TEMPLATES[name]


@pytest.mark.parametrize("name,facts,base_case", recommendation_variants(), ids=VARIANT_IDS)
def test_all_text_is_ascii_and_deterministic(name, facts, base_case):
    first = select_recommendations(facts, base_case, DRIVER_LABELS)
    second = select_recommendations(facts, base_case, DRIVER_LABELS)

    assert first == second
    for item in first:
        assert (item.finding + item.action + item.identifier).isascii()
        assert "\n" not in item.finding and "\n" not in item.action
        assert item.finding == item.finding.strip() and item.action == item.action.strip()
    assert all(isinstance(item, Recommendation) for item in first)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first[0].finding = "changed"


# Refusals and caller mistakes


def test_a_missing_label_for_the_lead_feature_is_refused_and_names_the_feature():
    with pytest.raises(RecommendationsError, match="'contract'"):
        select_recommendations(hand_facts(), hand_base_case(), {})


def test_a_label_is_only_needed_when_the_lead_driver_recommendation_is_selected():
    facts = facts_with_lead_count(hand_facts(), 140, 10)

    recommendations = select_recommendations(facts, hand_base_case(), {})

    assert "one-input-leads-the-drivers" not in identifiers(recommendations)


RECOMMENDATION_CALLER_MISTAKES = {
    "facts-none": lambda f, b, m: (None, b, m),
    "facts-a-dict": lambda f, b, m: ({"tiers": 1}, b, m),
    "base-case-a-dict": lambda f, b, m: (f, {"break_even_save_rate": 0.1}, m),
    "base-case-none": lambda f, b, m: (f, None, m),
    "labels-a-list": lambda f, b, m: (f, b, ["Contract type"]),
    "labels-none": lambda f, b, m: (f, b, None),
    "label-not-text": lambda f, b, m: (f, b, {**m, "contract": 5}),
    "label-empty": lambda f, b, m: (f, b, {**m, "contract": "  "}),
    "break-even-nan": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate=float("nan")), m),
    "break-even-infinite": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate=float("inf")), m),
    "break-even-zero": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate=0.0), m),
    "break-even-negative": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate=-0.1), m),
    "break-even-bool": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate=True), m),
    "break-even-text": lambda f, b, m: (f, dataclasses.replace(b, break_even_save_rate="0.1"), m),
}


@pytest.mark.parametrize(
    "build", list(RECOMMENDATION_CALLER_MISTAKES.values()), ids=list(RECOMMENDATION_CALLER_MISTAKES)
)
def test_recommendation_caller_mistakes_raise_value_error(build):
    facts, base_case, labels = build(hand_facts(), hand_base_case(), dict(DRIVER_LABELS))

    with pytest.raises(ValueError):
        select_recommendations(facts, base_case, labels)


def test_the_facts_and_the_base_case_are_not_changed_by_a_call():
    facts, base_case = hand_facts(), hand_base_case()
    frames_before = {
        "tiers": facts.tiers.copy(deep=True),
        "flagged_drivers": facts.flagged_drivers.copy(deep=True),
        "all_drivers": facts.all_drivers.copy(deep=True),
        "internet": facts.categorical_profile["internet_type"].copy(deep=True),
    }
    labels = dict(DRIVER_LABELS)

    select_recommendations(facts, base_case, labels)

    pd.testing.assert_frame_equal(facts.tiers, frames_before["tiers"])
    pd.testing.assert_frame_equal(facts.flagged_drivers, frames_before["flagged_drivers"])
    pd.testing.assert_frame_equal(facts.all_drivers, frames_before["all_drivers"])
    pd.testing.assert_frame_equal(facts.categorical_profile["internet_type"], frames_before["internet"])
    assert base_case == hand_base_case()
    assert labels == DRIVER_LABELS
