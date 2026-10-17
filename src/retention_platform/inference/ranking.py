"""Ranking rules: the flag count for the top-K list and the rule that turns
ranks into High, Medium and Low risk tiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def _check_ranks(rank: pd.Series) -> np.ndarray:
    """Return the ranks as an int64 array after checking they are 1..n once each."""
    if not isinstance(rank, pd.Series):
        raise ValueError(f"rank must be a pandas Series, got {type(rank).__name__}")
    if len(rank) == 0:
        raise ValueError("rank is empty; it must hold each of 1..n exactly once with n >= 1")
    if not pd.api.types.is_integer_dtype(rank.dtype):
        raise ValueError(f"rank must have an integer dtype, got {rank.dtype}")
    if rank.isna().any():
        raise ValueError("rank contains missing values")

    values = rank.to_numpy(dtype="int64")
    n_rows = len(values)
    if len(np.unique(values)) != n_rows:
        raise ValueError("rank contains duplicate values")
    if values.min() != 1:
        raise ValueError(f"rank must start at 1, got a minimum of {int(values.min())}")
    if values.max() != n_rows:
        raise ValueError(
            f"rank has a gap: {n_rows} rows but the largest rank is {int(values.max())}"
        )
    return values


def _is_plain_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assign_risk_tiers(rank: pd.Series, k_frac: float, medium_edge: float) -> pd.Series:
    """Label each rank "High", "Medium" or "Low".

    rank holds each of 1..n exactly once, in any order. Both boundaries use
    the top_k_count rule: ranks up to top_k_count(n, k_frac) are High, ranks
    above that up to top_k_count(n, medium_edge) are Medium, the rest are
    Low. With very small n the Medium tier can be empty. Requires
    0 < k_frac < medium_edge <= 1. Returns a new object-dtype Series with
    the input's index; the input is not changed.
    """
    if not (
        _is_plain_number(k_frac)
        and _is_plain_number(medium_edge)
        and 0 < k_frac < medium_edge <= 1
    ):
        raise ValueError(
            f"k_frac ({k_frac!r}) and medium_edge ({medium_edge!r}) must be numbers "
            "with 0 < k_frac < medium_edge <= 1"
        )
    values = _check_ranks(rank)

    n_rows = len(values)
    high = top_k_count(n_rows, k_frac)
    upto_medium = top_k_count(n_rows, medium_edge)

    tiers = np.full(n_rows, "Low", dtype=object)
    tiers[values <= upto_medium] = "Medium"
    tiers[values <= high] = "High"
    return pd.Series(tiers, index=rank.index, dtype=object)
