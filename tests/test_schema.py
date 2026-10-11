"""Tests for pipeline/schema.py: the generated input schema, its check
against the fitted preprocessor, and its hash.

Like test_preprocess.py, these tests build feat_churn in an in-memory
DuckDB from the committed raw files and fit the real preprocessor on the
training split. The pinned column list is written by hand from the real
DESCRIBE output of feat_churn; it is not produced by the code under test.
"""

from __future__ import annotations

import copy
import json
import re

import duckdb
import pytest

from retention_platform.data.prepare import (
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import run_features
from retention_platform.pipeline.preprocess import (
    CATEGORICAL_COLUMNS,
    EXCLUDED_COLUMNS,
    IDENTIFIER_COLUMN,
    INTERNET_TYPE_SENTINEL,
    OFFER_SENTINEL,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    prepare_model_inputs,
    select_model_columns,
)
from retention_platform.pipeline.schema import (
    NON_INPUT_COLUMNS,
    SQL_TYPE_TO_KIND,
    build_input_schema,
    check_schema_matches_preprocessor,
    read_feat_churn_types,
    schema_hash,
)
from retention_platform.pipeline.split import split_feat_churn

# Every model-input column, in feat_churn order, with its schema kind.
PINNED_COLUMNS = [
    ("gender", "categorical"),
    ("age", "integer-like"),
    ("under_30", "boolean-like"),
    ("senior_citizen", "boolean-like"),
    ("married", "boolean-like"),
    ("dependents", "boolean-like"),
    ("number_of_dependents", "integer-like"),
    ("latitude", "decimal"),
    ("longitude", "decimal"),
    ("population", "integer-like"),
    ("referred_a_friend", "boolean-like"),
    ("number_of_referrals", "integer-like"),
    ("tenure_in_months", "integer-like"),
    ("offer", "categorical"),
    ("phone_service", "boolean-like"),
    ("avg_monthly_long_distance_charges", "decimal"),
    ("multiple_lines", "boolean-like"),
    ("internet_service", "boolean-like"),
    ("internet_type", "categorical"),
    ("avg_monthly_gb_download", "integer-like"),
    ("online_security", "boolean-like"),
    ("online_backup", "boolean-like"),
    ("device_protection_plan", "boolean-like"),
    ("premium_tech_support", "boolean-like"),
    ("streaming_tv", "boolean-like"),
    ("streaming_movies", "boolean-like"),
    ("streaming_music", "boolean-like"),
    ("unlimited_data", "boolean-like"),
    ("contract", "categorical"),
    ("paperless_billing", "boolean-like"),
    ("payment_method", "categorical"),
    ("monthly_charge", "decimal"),
    ("total_charges", "decimal"),
    ("total_refunds", "decimal"),
    ("total_extra_data_charges", "integer-like"),
    ("total_long_distance_charges", "decimal"),
    ("total_revenue", "decimal"),
    ("num_addon_services", "integer-like"),
    ("is_month_to_month", "boolean-like"),
    ("internet_without_security", "boolean-like"),
    ("no_addons_despite_internet", "boolean-like"),
    ("is_tenure_censored", "boolean-like"),
]


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
def column_types(conn):
    return read_feat_churn_types(conn)


@pytest.fixture(scope="module")
def model_inputs(conn):
    return prepare_model_inputs(conn)


@pytest.fixture(scope="module")
def train_features(conn):
    X_train, _, _ = select_model_columns(split_feat_churn(conn).train)
    return X_train


@pytest.fixture(scope="module")
def schema(column_types, model_inputs):
    return build_input_schema(column_types, model_inputs.preprocessor)


def _entry(schema_dict, name):
    return next(column for column in schema_dict["columns"] if column["name"] == name)


# --- SQL type mapping -------------------------------------------------------


def test_type_mapping_is_exactly_the_four_agreed_types():
    assert SQL_TYPE_TO_KIND == {
        "BOOLEAN": "boolean-like",
        "INTEGER": "integer-like",
        "DOUBLE": "decimal",
        "VARCHAR": "categorical",
    }


def test_every_model_input_column_has_a_mapped_sql_type(column_types):
    for name, sql_type in column_types.items():
        if name in NON_INPUT_COLUMNS:
            continue
        assert sql_type in SQL_TYPE_TO_KIND, f"{name} has unmapped type {sql_type}"


def test_unmapped_sql_type_raises_naming_column_and_type(model_inputs):
    hand_made = {IDENTIFIER_COLUMN: "VARCHAR", "age": "BIGINT"}

    with pytest.raises(ValueError, match=r"'age'.*'BIGINT'"):
        build_input_schema(hand_made, model_inputs.preprocessor)


# --- Generated schema -------------------------------------------------------


def test_schema_matches_the_pinned_column_list(schema):
    generated = [(column["name"], column["kind"]) for column in schema["columns"]]

    assert generated == PINNED_COLUMNS


def test_nullable_columns_are_exactly_internet_type_and_offer(schema):
    nullable = {column["name"] for column in schema["columns"] if column["nullable"]}

    assert nullable == {"internet_type", "offer"}


def test_sentinels_are_reserved_not_allowed(schema):
    internet_type = _entry(schema, "internet_type")
    offer = _entry(schema, "offer")

    assert INTERNET_TYPE_SENTINEL not in internet_type["allowed_values"]
    assert OFFER_SENTINEL not in offer["allowed_values"]
    assert internet_type["reserved_values"] == [INTERNET_TYPE_SENTINEL]
    assert offer["reserved_values"] == [OFFER_SENTINEL]


@pytest.mark.parametrize("column", CATEGORICAL_COLUMNS)
def test_allowed_values_equal_distinct_training_values(schema, train_features, column):
    expected = sorted(train_features[column].dropna().unique())

    assert sorted(_entry(schema, column)["allowed_values"]) == expected


def test_only_the_two_nullable_columns_have_reserved_values(schema):
    with_reserved = {
        column["name"] for column in schema["columns"] if "reserved_values" in column
    }

    assert with_reserved == {"internet_type", "offer"}


@pytest.mark.parametrize("column", [*EXCLUDED_COLUMNS, TARGET_COLUMN, IDENTIFIER_COLUMN])
def test_non_input_columns_are_not_model_inputs(schema, column):
    assert column not in {entry["name"] for entry in schema["columns"]}


def test_identifier_is_recorded(schema):
    assert schema["identifier_column"] == IDENTIFIER_COLUMN


def test_schema_is_json_serializable(schema):
    assert json.loads(json.dumps(schema)) == schema


def test_valid_schema_passes_the_check(schema, model_inputs):
    check_schema_matches_preprocessor(schema, model_inputs.preprocessor)


# --- Cross-check failures ---------------------------------------------------


def _drop_column(s):
    del s["columns"][3]


def _rename_column(s):
    s["columns"][1]["name"] = "age_in_years"


def _swap_order(s):
    s["columns"][1], s["columns"][2] = s["columns"][2], s["columns"][1]


def _change_allowed_value(s):
    _entry(s, "contract")["allowed_values"][0] = "Bogus Contract"


def _remove_reserved_value(s):
    del _entry(s, "internet_type")["reserved_values"]


CHECK_FAILURES = {
    "dropped column": _drop_column,
    "renamed column": _rename_column,
    "changed order": _swap_order,
    "changed allowed value": _change_allowed_value,
    "removed reserved value": _remove_reserved_value,
}


@pytest.mark.parametrize("mutate", CHECK_FAILURES.values(), ids=CHECK_FAILURES.keys())
def test_check_rejects_a_schema_that_disagrees_with_the_preprocessor(
    schema, model_inputs, mutate
):
    broken = copy.deepcopy(schema)
    mutate(broken)

    with pytest.raises(ValueError, match="does not match the fitted preprocessor"):
        check_schema_matches_preprocessor(broken, model_inputs.preprocessor)


def test_check_rejects_a_preprocessor_fitted_on_different_columns(schema, train_features):
    other = build_preprocessing_pipeline().fit(train_features.drop(columns=["age"]))

    with pytest.raises(ValueError, match="model-input columns differ"):
        check_schema_matches_preprocessor(schema, other)


def test_check_reports_every_problem_in_one_error(schema, model_inputs):
    broken = copy.deepcopy(schema)
    _drop_column(broken)
    _change_allowed_value(broken)

    with pytest.raises(ValueError) as exc_info:
        check_schema_matches_preprocessor(broken, model_inputs.preprocessor)

    message = str(exc_info.value)
    assert "model-input columns differ" in message
    assert "contract" in message


# --- Hash -------------------------------------------------------------------


def _reverse_key_order(value):
    if isinstance(value, dict):
        return {key: _reverse_key_order(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_key_order(item) for item in value]
    return value


def _change_kind(s):
    s["columns"][1]["kind"] = "decimal"


def _flip_nullable(s):
    s["columns"][1]["nullable"] = True


def _change_identifier(s):
    s["identifier_column"] = "account_id"


def _add_allowed_value(s):
    _entry(s, "gender")["allowed_values"].append("Other")


HASH_CHANGES = {
    **CHECK_FAILURES,
    "changed kind": _change_kind,
    "flipped nullable": _flip_nullable,
    "changed identifier": _change_identifier,
    "added allowed value": _add_allowed_value,
}


def test_hash_is_deterministic(schema):
    assert schema_hash(schema) == schema_hash(copy.deepcopy(schema))


def test_hash_does_not_depend_on_key_order(schema):
    reordered = _reverse_key_order(schema)

    assert list(reordered) != list(schema)
    assert schema_hash(reordered) == schema_hash(schema)


def test_hash_is_64_lowercase_hex_characters(schema):
    assert re.fullmatch(r"[0-9a-f]{64}", schema_hash(schema))


def test_hash_is_unchanged_by_a_json_round_trip(schema):
    round_tripped = json.loads(json.dumps(schema))

    assert schema_hash(round_tripped) == schema_hash(schema)


@pytest.mark.parametrize("mutate", HASH_CHANGES.values(), ids=HASH_CHANGES.keys())
def test_hash_changes_when_a_single_entry_changes(schema, mutate):
    changed = copy.deepcopy(schema)
    mutate(changed)

    assert schema_hash(changed) != schema_hash(schema)
