"""Scoring core: turn a validated customer table and a loaded model run into
the ranked outreach table (customer_id, churn_probability, rank, in_top_k).

Nothing is written here and no tracking is started. The table is
validated first; a column that is not a model input is never given to the
model, and no label column is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention_platform.inference.validate import validate_scoring_table
from retention_platform.models.persist import LoadedRun


class ScoringError(Exception):
    """Raised when the model's output cannot be used to build the ranking."""


def top_k_count(n_rows: int, k_frac: float) -> int:
    """Number of customers flagged for outreach: max(1, round(n_rows * k_frac)).

    Python's round() sends an exact half to the nearest even integer.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValueError(f"n_rows must be an integer of at least 1, got {n_rows!r}")
    if (
        isinstance(k_frac, bool)
        or not isinstance(k_frac, (int, float))
        or not 0 < k_frac <= 1
    ):
        raise ValueError(f"k_frac must be a number with 0 < k_frac <= 1, got {k_frac!r}")
    return max(1, round(n_rows * k_frac))


def _model_probabilities(loaded_run: LoadedRun, inputs: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """The model's churn probability for each row, checked to be usable."""
    matrix = loaded_run.preprocessor.transform(inputs)
    raw = np.asarray(loaded_run.model.predict_proba(matrix))
    n_rows = len(inputs)

    if raw.ndim != 2 or raw.shape[0] != n_rows or raw.shape[1] < 2:
        raise ScoringError(
            f"the model returned an output of shape {raw.shape} for {n_rows} rows"
        )
    probabilities = raw[:, 1].astype(float)

    non_finite = np.flatnonzero(~np.isfinite(probabilities))
    if non_finite.size:
        row = int(non_finite[0])
        raise ScoringError(
            f"the model returned a non-finite probability for {non_finite.size} row(s); "
            f"first at row {row} (customer_id '{ids[row]}')"
        )
    outside = np.flatnonzero((probabilities < 0) | (probabilities > 1))
    if outside.size:
        row = int(outside[0])
        raise ScoringError(
            f"the model returned a probability outside [0, 1] for {outside.size} row(s); "
            f"first at row {row} (customer_id '{ids[row]}', value {probabilities[row]!r})"
        )
    return probabilities


def score_batch(frame: pd.DataFrame, loaded_run: LoadedRun, k_frac: float) -> pd.DataFrame:
    """Score a customer table and return the ranked outreach table.

    The table is validated first and InputValidationError propagates
    unchanged. The model then sees only the schema's input columns, in the
    schema's order. Rows are ordered by churn probability, highest first,
    with ties broken by customer_id ascending as text. rank runs 1 to n
    and in_top_k is true for the first top_k_count(n, k_frac) rows. The
    caller's frame is not changed.
    """
    validate_scoring_table(frame, loaded_run.schema)

    identifier = loaded_run.schema["identifier_column"]
    input_columns = [column["name"] for column in loaded_run.schema["columns"]]
    inputs = frame[input_columns].copy()
    ids = [str(value) for value in frame[identifier].astype(object).tolist()]

    probabilities = _model_probabilities(loaded_run, inputs, ids)

    table = pd.DataFrame(
        {
            "customer_id": pd.Series(ids, dtype=object),
            "churn_probability": pd.Series(probabilities, dtype="float64"),
        }
    )
    table = table.sort_values(
        by=["churn_probability", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)

    n_rows = len(table)
    table["rank"] = np.arange(1, n_rows + 1, dtype="int64")
    table["in_top_k"] = table["rank"] <= top_k_count(n_rows, k_frac)
    return table[["customer_id", "churn_probability", "rank", "in_top_k"]]
