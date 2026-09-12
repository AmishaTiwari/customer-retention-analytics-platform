"""Column selection for the model feature matrix.

Splits feat_churn into the feature matrix X, the target y, and the
retained customer_id, before any encoding, scaling, or missing-value
handling is applied.
"""

from __future__ import annotations

import pandas as pd

# Columns retained separately, never part of X.
IDENTIFIER_COLUMN = "customer_id"
TARGET_COLUMN = "is_voluntary_churn"

# Columns dropped from the feature matrix entirely. This is a choice of
# which columns serve as the model's input representation, not a change
# to their disposition as valid, non-leakage features -- city and
# zip_code stay correctly classified as Feature columns, they are just
# not the representation used here, the same way lat_long was dropped
# in favor of latitude/longitude as separate numeric columns.
EXCLUDED_COLUMNS = [
    IDENTIFIER_COLUMN,
    "country",  # single distinct value in this dataset, zero information
    "state",  # single distinct value in this dataset, zero information
    "city",  # high-cardinality; geographic signal already captured by latitude/longitude
    "zip_code",  # high-cardinality; geographic signal already captured by latitude/longitude
]


def select_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split df into the feature matrix X, the target y, and customer_id.

    X excludes EXCLUDED_COLUMNS and TARGET_COLUMN; every remaining column
    is a candidate model feature.
    """
    customer_id = df[IDENTIFIER_COLUMN]
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[*EXCLUDED_COLUMNS, TARGET_COLUMN])

    return X, y, customer_id
