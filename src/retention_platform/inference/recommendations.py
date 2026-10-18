"""The facts a recommendations document can quote, computed from the ranked
scoring table, the scoring inputs, the customer drivers and the global
importance. Numbers only: no wording and no thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from retention_platform.inference.drivers import (
    DIRECTION_LOWERS,
    DIRECTION_RAISES,
    DRIVER_COLUMNS,
    TOP_N_DRIVERS,
)
from retention_platform.inference.ranking import assign_risk_tiers, top_k_count

PROFILE_NUMERIC_COLUMNS = ("monthly_charge", "tenure_in_months")
PROFILE_CATEGORICAL_COLUMNS = ("contract", "internet_type")
TOP_N_IMPORTANCE = 10
IMPORTANCE_FEATURE = "internet_type"
NO_INTERNET_CATEGORY = "No Internet"
TIER_ORDER = ("High", "Medium", "Low")

TABLE_COLUMNS = ["customer_id", "churn_probability", "rank", "in_top_k"]
_DRIVER_FEATURE_COLUMNS = DRIVER_COLUMNS[0::2]
_DRIVER_DIRECTION_COLUMNS = DRIVER_COLUMNS[1::2]


class RecommendationsError(Exception):
    """Raised when the data cannot be used to compute the facts."""


@dataclass(frozen=True)
class InternetTypeImportance:
    """The internet_type row of the global importance, and whether it is
    already among the top rows."""

    share: float
    rank: int
    in_top_rows: bool


@dataclass(frozen=True, eq=False)
class RecommendationFacts:
    """Every fact the recommendations document quotes."""

    customers_scored: int
    customers_flagged: int
    tiers: pd.DataFrame
    numeric_profile: pd.DataFrame
    categorical_profile: dict
    importance_top: pd.DataFrame
    internet_type_importance: InternetTypeImportance
    flagged_drivers: pd.DataFrame
    all_drivers: pd.DataFrame


def _is_fraction(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return 0 < float(value) <= 1
    except OverflowError:
        return False


def _floats(values: pd.Series, what: str, ids: list) -> np.ndarray:
    """The values as float64; a missing, non-numeric or non-finite value
    raises an error that names the first offending customer."""
    result = np.empty(len(values), dtype="float64")
    for position, value in enumerate(values.tolist()):
        customer = ids[position]
        if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and value != value):
            raise RecommendationsError(f"{what} is missing for customer {customer!r}")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise RecommendationsError(f"{what} is not a number for customer {customer!r}: {value!r}")
        try:
            number = float(value)
        except OverflowError:
            raise RecommendationsError(f"{what} is not finite for customer {customer!r}") from None
        if not math.isfinite(number):
            raise RecommendationsError(f"{what} is not finite for customer {customer!r}")
        result[position] = number
    return result


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and value != value)


def _require_frame(value: object, name: str, columns: list[str]) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a DataFrame, got {type(value).__name__}")
    missing = [column for column in columns if column not in value.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _unique_ids(ids: list, what: str) -> None:
    if len(set(ids)) != len(ids):
        duplicate = next(value for value in ids if ids.count(value) > 1)
        raise RecommendationsError(f"duplicate customer id in {what}: {duplicate!r}")


def _check_same_ids(ids: list, other: list, what: str) -> None:
    known, other_known = set(ids), set(other)
    absent = [value for value in ids if value not in other_known]
    if absent:
        raise RecommendationsError(f"{what} has no row for customer {absent[0]!r}")
    extra = [value for value in other if value not in known]
    if extra:
        raise RecommendationsError(f"{what} has a row for unknown customer {extra[0]!r}")


def _check_flags(table: pd.DataFrame, ids: list, ranks: np.ndarray, k_frac: float) -> None:
    flags = table["in_top_k"]
    if flags.dtype != np.bool_:
        raise RecommendationsError(f"in_top_k must be a boolean column, got dtype {flags.dtype}")
    expected = top_k_count(len(ids), k_frac)
    flagged = int(flags.sum())
    if flagged != expected:
        raise RecommendationsError(
            f"in_top_k marks {flagged} rows but top_k_count for k={k_frac!r} is {expected}"
        )
    first_by_rank = {ids[p] for p in np.argsort(ranks, kind="stable")[:expected]}
    if {ids[p] for p in np.flatnonzero(flags.to_numpy())} != first_by_rank:
        raise RecommendationsError(f"the in_top_k rows are not the first {expected} rows by rank")


def _tier_facts(tiers: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    n_rows = len(tiers)
    rows = []
    for tier in TIER_ORDER:
        mask = (tiers == tier).to_numpy()
        count = int(mask.sum())
        rows.append(
            {
                "customers": count,
                "share": count / n_rows,
                "min_probability": float(probabilities[mask].min()) if count else float("nan"),
                "max_probability": float(probabilities[mask].max()) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(list(TIER_ORDER), name="tier"))


def _category(column: str, value: object, customer: object) -> str:
    if _is_missing(value):
        if column == IMPORTANCE_FEATURE:
            return NO_INTERNET_CATEGORY
        raise RecommendationsError(f"{column} is missing for customer {customer!r}")
    if not isinstance(value, str) or not value.strip():
        raise RecommendationsError(f"{column} is not text for customer {customer!r}: {value!r}")
    if column == IMPORTANCE_FEATURE and value == NO_INTERNET_CATEGORY:
        raise RecommendationsError(
            f"{column} holds the reserved category {NO_INTERNET_CATEGORY!r} for customer {customer!r}"
        )
    return value


def _category_table(categories: list[str], flagged: np.ndarray) -> pd.DataFrame:
    all_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for category, in_group in zip(categories, flagged):
        all_counts[category] = all_counts.get(category, 0) + 1
        if in_group:
            group_counts[category] = group_counts.get(category, 0) + 1
    n_all, n_group = len(categories), int(flagged.sum())
    names = sorted(all_counts, key=lambda name: (-all_counts[name], name))
    return pd.DataFrame(
        {
            "group_count": [group_counts.get(name, 0) for name in names],
            "group_share": [group_counts.get(name, 0) / n_group for name in names],
            "all_count": [all_counts[name] for name in names],
            "all_share": [all_counts[name] / n_all for name in names],
        },
        index=pd.Index(names, name="category"),
    )


def _driver_facts(features: np.ndarray, directions: np.ndarray) -> pd.DataFrame:
    """features and directions have one row per customer and three columns."""
    first: dict[str, int] = {}
    anywhere: dict[str, int] = {}
    raises: dict[str, int] = {}
    lowers: dict[str, int] = {}
    for feature_row, direction_row in zip(features, directions):
        first[feature_row[0]] = first.get(feature_row[0], 0) + 1
        for feature, direction in zip(feature_row, direction_row):
            anywhere[feature] = anywhere.get(feature, 0) + 1
            target = raises if direction == DIRECTION_RAISES else lowers
            target[feature] = target.get(feature, 0) + 1
    names = sorted(anywhere, key=lambda name: (-first.get(name, 0), -anywhere[name], name))
    return pd.DataFrame(
        {
            "first_driver_count": np.array([first.get(n, 0) for n in names], dtype="int64"),
            "any_driver_count": np.array([anywhere[n] for n in names], dtype="int64"),
            "raises_count": np.array([raises.get(n, 0) for n in names], dtype="int64"),
            "lowers_count": np.array([lowers.get(n, 0) for n in names], dtype="int64"),
        },
        index=pd.Index(names, name="feature"),
    )


def _checked_drivers(drivers: pd.DataFrame, ids: list) -> tuple[np.ndarray, np.ndarray]:
    """The driver features and directions in the table's row order, after
    checking that each customer has three different features and valid directions."""
    ordered = drivers.loc[ids]
    features = ordered[_DRIVER_FEATURE_COLUMNS].to_numpy(dtype=object)
    directions = ordered[_DRIVER_DIRECTION_COLUMNS].to_numpy(dtype=object)
    for position, customer in enumerate(ids):
        names = list(features[position])
        if not all(isinstance(name, str) and name for name in names):
            raise RecommendationsError(f"a driver is missing or not text for customer {customer!r}")
        if len(set(names)) != TOP_N_DRIVERS:
            raise RecommendationsError(f"customer {customer!r} does not have three different drivers")
        if not all(direction in (DIRECTION_RAISES, DIRECTION_LOWERS) for direction in directions[position]):
            raise RecommendationsError(f"a driver direction is not valid for customer {customer!r}")
    return features, directions


def _checked_importance(importance: pd.DataFrame) -> None:
    if not isinstance(importance, pd.DataFrame):
        raise ValueError(f"importance must be a DataFrame, got {type(importance).__name__}")
    if list(importance.columns) != ["share", "rank"]:
        raise ValueError(f"importance must have exactly the columns ['share', 'rank'], got {list(importance.columns)}")
    if len(importance) == 0:
        raise RecommendationsError("the global importance has no rows")
    if importance.index.has_duplicates:
        raise RecommendationsError("the global importance has a repeated feature")
    try:
        shares = importance["share"].to_numpy(dtype="float64")
    except (OverflowError, TypeError, ValueError):
        shares = None
    if shares is None or not np.isfinite(shares).all() or (shares < 0).any():
        raise RecommendationsError("the global importance has a share that is negative or not finite")
    if list(importance["rank"]) != list(range(1, len(importance) + 1)):
        raise RecommendationsError("the global importance ranks are not 1..n in table order")
    if IMPORTANCE_FEATURE not in importance.index:
        raise RecommendationsError(f"the global importance has no {IMPORTANCE_FEATURE} row")


def compute_recommendation_facts(
    table: pd.DataFrame,
    inputs: pd.DataFrame,
    drivers: pd.DataFrame,
    importance: pd.DataFrame,
    k_frac: float,
    medium_edge: float,
) -> RecommendationFacts:
    """The tier, group-profile, global-importance and driver facts.

    table is the score_batch output; inputs holds the scoring inputs and
    drivers the select_top_drivers output, both indexed by customer_id;
    importance is the global_importance frame. Rows are matched by customer
    id, never by position. The inputs are not changed.
    """
    _require_frame(table, "table", TABLE_COLUMNS)
    _require_frame(inputs, "inputs", [*PROFILE_NUMERIC_COLUMNS, *PROFILE_CATEGORICAL_COLUMNS])
    _require_frame(drivers, "drivers", DRIVER_COLUMNS)
    if not (_is_fraction(k_frac) and _is_fraction(medium_edge) and 0 < k_frac < medium_edge <= 1):
        raise ValueError(
            f"k_frac ({k_frac!r}) and medium_edge ({medium_edge!r}) must be numbers "
            "with 0 < k_frac < medium_edge <= 1"
        )
    _checked_importance(importance)

    n_rows = len(table)
    if n_rows == 0:
        raise RecommendationsError("the table has no rows")
    ids = table["customer_id"].tolist()
    _unique_ids(ids, "the table")
    _unique_ids(inputs.index.tolist(), "the inputs")
    _unique_ids(drivers.index.tolist(), "the driver table")
    _check_same_ids(ids, inputs.index.tolist(), "the inputs")
    _check_same_ids(ids, drivers.index.tolist(), "the driver table")

    rank = table["rank"]
    if rank.dtype == np.bool_ or not pd.api.types.is_integer_dtype(rank.dtype) or rank.isna().any():
        raise RecommendationsError(f"rank must be a complete integer column, got dtype {rank.dtype}")
    ranks = rank.to_numpy(dtype="int64")
    if not np.array_equal(np.sort(ranks), np.arange(1, n_rows + 1)):
        raise RecommendationsError(f"rank is not a complete 1..{n_rows} sequence")
    probabilities = _floats(table["churn_probability"], "churn_probability", ids)
    if ((probabilities < 0) | (probabilities > 1)).any():
        position = int(np.flatnonzero((probabilities < 0) | (probabilities > 1))[0])
        raise RecommendationsError(
            f"churn_probability is outside [0, 1] for customer {ids[position]!r}: {probabilities[position]!r}"
        )
    _check_flags(table, ids, ranks, k_frac)
    flagged = table["in_top_k"].to_numpy(dtype=bool)

    aligned = inputs.loc[ids]
    tiers = assign_risk_tiers(pd.Series(ranks, dtype="int64"), k_frac, medium_edge)
    tier_facts = _tier_facts(tiers, probabilities)

    numeric_rows = []
    for column in PROFILE_NUMERIC_COLUMNS:
        values = _floats(aligned[column], column, ids)
        numeric_rows.append(
            {
                "group_mean": math.fsum(values[flagged].tolist()) / int(flagged.sum()),
                "all_mean": math.fsum(values.tolist()) / n_rows,
            }
        )
    numeric_profile = pd.DataFrame(numeric_rows, index=pd.Index(list(PROFILE_NUMERIC_COLUMNS), name="column"))

    categorical_profile = {}
    for column in PROFILE_CATEGORICAL_COLUMNS:
        categories = [_category(column, value, ids[p]) for p, value in enumerate(aligned[column].tolist())]
        categorical_profile[column] = _category_table(categories, flagged)

    top_rows = importance.head(TOP_N_IMPORTANCE).copy()
    internet_row = importance.loc[IMPORTANCE_FEATURE]
    internet_importance = InternetTypeImportance(
        share=float(internet_row["share"]),
        rank=int(internet_row["rank"]),
        in_top_rows=IMPORTANCE_FEATURE in top_rows.index,
    )

    features, directions = _checked_drivers(drivers, ids)

    return RecommendationFacts(
        customers_scored=n_rows,
        customers_flagged=int(flagged.sum()),
        tiers=tier_facts,
        numeric_profile=numeric_profile,
        categorical_profile=categorical_profile,
        importance_top=top_rows,
        internet_type_importance=internet_importance,
        flagged_drivers=_driver_facts(features[flagged], directions[flagged]),
        all_drivers=_driver_facts(features, directions),
    )
