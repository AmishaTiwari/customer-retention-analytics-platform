"""Input validation for the scoring table.

validate_scoring_table checks a pandas DataFrame against an input schema
given as a plain dict:

    {"identifier_column": str,
     "columns": [{"name": str, "kind": str, "nullable": bool,
                  "allowed_values": [...]?, "reserved_values": [...]?}, ...]}

It stops at the first failed condition, in this order:

1. Columns: missing, extra and duplicate column names (all listed at once).
2. Empty table: zero rows.
3. customer_id: missing, non-text, empty or whitespace-only, then duplicate
   values.
4. Per-column value rules, columns in schema order. Within one column the
   conditions are checked in this order: pd.NA, None in a non-nullable
   column, NaN, infinity, then the rules of the column's kind.

The frame is never changed and no label is read. This module imports
nothing from the package; the four kind strings below must equal the kinds
the schema generator writes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

KIND_BOOLEAN = "boolean-like"
KIND_INTEGER = "integer-like"
KIND_DECIMAL = "decimal"
KIND_CATEGORICAL = "categorical"


class InputValidationError(ValueError):
    """Raised when the scoring table breaks an input rule."""


# Value conditions, listed in the order they take priority within a column.
_PD_NA = "pd.NA is not accepted"
_NONE = "None is not accepted (the column is not nullable)"
_NAN = "NaN value"
_INFINITY = "infinite value"
_NOT_TEXT = "categorical value is not text"
_SENTINEL = (
    "reserved value used as raw input (it is an internal sentinel and is not accepted)"
)
_UNSEEN = "unseen category (not one of the allowed values)"
_BOOL_IN_NUMERIC = "boolean value where a number is required"
_NUMERIC_STRING = "text value where a number is required (numeric strings are not accepted)"
_NOT_WHOLE = "non-whole number where a whole number is required"
_NOT_A_NUMBER = "value that is not a number"
_BAD_BOOLEAN = "value other than True, False, 0 or 1"

_CONDITION_ORDER = [
    _PD_NA,
    _NONE,
    _NAN,
    _INFINITY,
    _NOT_TEXT,
    _SENTINEL,
    _UNSEEN,
    _BOOL_IN_NUMERIC,
    _NUMERIC_STRING,
    _NOT_WHOLE,
    _NOT_A_NUMBER,
    _BAD_BOOLEAN,
]


def _is_usable_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _failure(
    condition: str,
    column: str | None = None,
    positions: list[int] | None = None,
    ids: list | None = None,
) -> InputValidationError:
    """Build the error: the condition, the column, and the first offending
    row (0-based position, with its customer_id when that is usable text)
    and how many rows offend."""
    text = f"Input validation failed: {condition}"
    if column is not None:
        text += f" in column '{column}'"
    if positions:
        first = positions[0]
        text += f"; first offending row: {first}"
        if ids is not None and _is_usable_text(ids[first]):
            text += f" (customer_id '{ids[first]}')"
        text += f"; {len(positions)} row(s) offend"
    return InputValidationError(text)


def _check_columns(frame: pd.DataFrame, expected: list[str]) -> None:
    present = list(frame.columns)
    expected_set = set(expected)
    present_set = set(present)

    missing = [name for name in expected if name not in present_set]
    extra = [name for name in dict.fromkeys(present) if name not in expected_set]
    duplicates = [
        name
        for name in dict.fromkeys(present)
        if present.count(name) > 1
    ]

    parts = []
    if missing:
        parts.append(f"missing columns {missing}")
    if extra:
        parts.append(f"extra columns {extra}")
    if duplicates:
        parts.append(f"duplicate column names {duplicates}")
    if parts:
        raise _failure("; ".join(parts))


def _check_identifier(frame: pd.DataFrame, identifier: str) -> list:
    ids = frame[identifier].astype(object).tolist()

    missing = [
        pos
        for pos, value in enumerate(ids)
        if value is None
        or value is pd.NA
        or (isinstance(value, float) and value != value)
    ]
    if missing:
        raise _failure(f"missing {identifier} value", identifier, missing, ids)

    not_text = [pos for pos, value in enumerate(ids) if not isinstance(value, str)]
    if not_text:
        raise _failure(f"{identifier} is not text", identifier, not_text, ids)

    blank = [pos for pos, value in enumerate(ids) if not value.strip()]
    if blank:
        raise _failure(
            f"empty or whitespace-only {identifier} value", identifier, blank, ids
        )

    seen: set[str] = set()
    repeats = []
    for pos, value in enumerate(ids):
        if value in seen:
            repeats.append(pos)
        seen.add(value)
    if repeats:
        raise _failure(f"duplicate {identifier} values", identifier, repeats, ids)

    return ids


def _classify(
    value: object,
    kind: str,
    nullable: bool,
    allowed: set,
    reserved: set,
) -> str | None:
    """Return the condition a value breaks, or None if it is acceptable."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is pd.NA:
        return _PD_NA
    if value is None:
        return None if nullable else _NONE
    if isinstance(value, float):
        if value != value:
            return _NAN
        if math.isinf(value):
            return _INFINITY

    if kind == KIND_CATEGORICAL:
        if not isinstance(value, str):
            return _NOT_TEXT
        if value in reserved:
            return _SENTINEL
        if value not in allowed:
            return _UNSEEN
        return None

    if isinstance(value, bool):
        return None if kind == KIND_BOOLEAN else _BOOL_IN_NUMERIC
    if isinstance(value, str):
        return _BAD_BOOLEAN if kind == KIND_BOOLEAN else _NUMERIC_STRING

    if kind == KIND_BOOLEAN:
        return None if isinstance(value, int) and value in (0, 1) else _BAD_BOOLEAN
    if isinstance(value, (int, float)):
        if kind == KIND_INTEGER and isinstance(value, float) and not value.is_integer():
            return _NOT_WHOLE
        return None
    return _NOT_A_NUMBER


def _check_column(frame: pd.DataFrame, spec: dict, ids: list) -> None:
    name = spec["name"]
    kind = spec["kind"]
    nullable = bool(spec.get("nullable", False))
    allowed = set(spec.get("allowed_values", []))
    reserved = set(spec.get("reserved_values", []))

    found: dict[str, list[int]] = {}
    for pos, value in enumerate(frame[name].astype(object).tolist()):
        condition = _classify(value, kind, nullable, allowed, reserved)
        if condition is not None:
            found.setdefault(condition, []).append(pos)

    if found:
        condition = min(found, key=_CONDITION_ORDER.index)
        raise _failure(condition, name, found[condition], ids)


def validate_scoring_table(frame: pd.DataFrame, schema: dict) -> None:
    """Raise InputValidationError at the first failed condition; return
    nothing if the table passes. Never changes the frame."""
    if not isinstance(frame, pd.DataFrame):
        raise _failure(
            f"the scoring input must be a pandas DataFrame, got {type(frame).__name__}"
        )

    identifier = schema["identifier_column"]
    specs = schema["columns"]

    _check_columns(frame, [identifier, *[spec["name"] for spec in specs]])

    if len(frame) == 0:
        raise _failure("the table has no rows")

    ids = _check_identifier(frame, identifier)

    for spec in specs:
        _check_column(frame, spec, ids)
