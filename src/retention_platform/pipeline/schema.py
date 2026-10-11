"""Input schema generated from feat_churn and the fitted preprocessor.

The schema lists the model-input columns in order, with the kind of value
each accepts, whether it may be null, and for categorical columns the
allowed values. It is a plain JSON-serializable dict, so it can be stored
and read back without importing this module.
"""

from __future__ import annotations

import hashlib
import json

import duckdb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from retention_platform.pipeline.preprocess import (
    CATEGORICAL_COLUMNS,
    EXCLUDED_COLUMNS,
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
)

# DuckDB column type -> schema kind. A model-input column with any other
# type is an error rather than a guess.
SQL_TYPE_TO_KIND = {
    "BOOLEAN": "boolean-like",
    "INTEGER": "integer-like",
    "DOUBLE": "decimal",
    "VARCHAR": "categorical",
}

# Columns of feat_churn that are not model inputs.
NON_INPUT_COLUMNS = {IDENTIFIER_COLUMN, TARGET_COLUMN, *EXCLUDED_COLUMNS}


def read_feat_churn_types(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return column name -> DuckDB type for feat_churn, in table order."""
    rows = conn.execute("DESCRIBE feat_churn").fetchall()
    return {row[0]: row[1] for row in rows}


def _fitted_parts(
    preprocessor: ColumnTransformer,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Read the fitted preprocessor.

    Returns (imputed, categories): imputed maps each column that has an
    imputer to its fill value; categories maps each one-hot encoded column
    to the categories its encoder learned, in the encoder's order.
    """
    imputed: dict[str, str] = {}
    categories: dict[str, list[str]] = {}

    for _, transformer, columns in preprocessor.transformers_:
        steps = (
            [step for _, step in transformer.steps]
            if isinstance(transformer, Pipeline)
            else [transformer]
        )
        for step in steps:
            if isinstance(step, SimpleImputer):
                for column in columns:
                    imputed[column] = step.fill_value
            elif isinstance(step, OneHotEncoder):
                for column, learned in zip(columns, step.categories_):
                    categories[column] = [str(value) for value in learned]

    return imputed, categories


def build_input_schema(
    column_types: dict[str, str], preprocessor: ColumnTransformer
) -> dict:
    """Build the input schema and check it against the fitted preprocessor.

    column_types is the feat_churn column name -> DuckDB type mapping. A
    column is nullable only if the fitted preprocessor imputes it. The
    value the imputer fills in is not allowed as raw input, so it is
    removed from the column's allowed values and recorded under
    reserved_values instead.
    """
    imputed, categories = _fitted_parts(preprocessor)

    columns = []
    for name, sql_type in column_types.items():
        if name in NON_INPUT_COLUMNS:
            continue

        kind = SQL_TYPE_TO_KIND.get(sql_type)
        if kind is None:
            raise ValueError(
                f"Column {name!r} has SQL type {sql_type!r}, which has no schema kind; "
                f"expected one of {sorted(SQL_TYPE_TO_KIND)}"
            )

        entry = {"name": name, "kind": kind, "nullable": name in imputed}
        if kind == "categorical":
            if name not in categories:
                raise ValueError(
                    f"Categorical column {name!r} has no encoder in the fitted preprocessor"
                )
            reserved = [imputed[name]] if name in imputed else []
            entry["allowed_values"] = [
                value for value in categories[name] if value not in reserved
            ]
            if reserved:
                entry["reserved_values"] = reserved
        columns.append(entry)

    schema = {"identifier_column": IDENTIFIER_COLUMN, "columns": columns}
    check_schema_matches_preprocessor(schema, preprocessor)
    return schema


def check_schema_matches_preprocessor(
    schema: dict, preprocessor: ColumnTransformer
) -> None:
    """Raise ValueError listing every way schema disagrees with the fitted
    preprocessor: the column names and order, the categorical columns, the
    nullable columns, and the allowed values plus reserved values against
    the encoder categories.
    """
    imputed, categories = _fitted_parts(preprocessor)
    problems: list[str] = []

    columns = schema.get("columns")
    if not isinstance(columns, list):
        raise ValueError("Input schema has no 'columns' list")

    if schema.get("identifier_column") != IDENTIFIER_COLUMN:
        problems.append(
            f"identifier_column is {schema.get('identifier_column')!r}, "
            f"expected {IDENTIFIER_COLUMN!r}"
        )

    names = [column.get("name") for column in columns]
    fitted_names = list(preprocessor.feature_names_in_)
    if names != fitted_names:
        missing = [name for name in fitted_names if name not in names]
        extra = [name for name in names if name not in fitted_names]
        detail = f"missing from schema: {missing}; not in preprocessor: {extra}"
        if not missing and not extra:
            detail = "same columns in a different order"
        problems.append(f"model-input columns differ from the preprocessor ({detail})")

    categorical = {c.get("name") for c in columns if c.get("kind") == "categorical"}
    if categorical != set(CATEGORICAL_COLUMNS):
        problems.append(
            f"categorical columns {sorted(categorical)} differ from "
            f"{sorted(CATEGORICAL_COLUMNS)}"
        )
    if categorical != set(categories):
        problems.append(
            f"categorical columns {sorted(categorical)} differ from the preprocessor's "
            f"encoded columns {sorted(categories)}"
        )

    nullable = {c.get("name") for c in columns if c.get("nullable")}
    if nullable != set(imputed):
        problems.append(
            f"nullable columns {sorted(nullable)} differ from the preprocessor's "
            f"imputed columns {sorted(imputed)}"
        )

    for column in columns:
        name = column.get("name")
        if column.get("kind") != "categorical" or name not in categories:
            continue
        allowed = column.get("allowed_values", [])
        reserved = column.get("reserved_values", [])
        expected_reserved = [imputed[name]] if name in imputed else []
        if reserved != expected_reserved:
            problems.append(
                f"{name}: reserved values {reserved} differ from {expected_reserved}"
            )
        if sorted(allowed + reserved) != sorted(categories[name]):
            problems.append(
                f"{name}: allowed values {allowed} plus reserved values {reserved} "
                f"differ from the encoder categories {categories[name]}"
            )

    if problems:
        raise ValueError(
            "Input schema does not match the fitted preprocessor:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )


def schema_hash(schema: dict) -> str:
    """Return the SHA-256 of the schema's canonical JSON as 64 lowercase hex
    characters. Canonical JSON has sorted keys and (",", ":") separators,
    encoded as UTF-8, so the hash does not depend on key order or on a
    JSON round trip.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
