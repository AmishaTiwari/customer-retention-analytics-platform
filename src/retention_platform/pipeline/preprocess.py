"""Column selection and preprocessing pipeline for the model feature matrix.

Splits feat_churn into the feature matrix X, the target y, and the
retained customer_id, then builds the shared scikit-learn preprocessing
transformer (encoding + scaling) applied identically across Logistic
Regression, Random Forest, and XGBoost.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


# The five low-cardinality columns encoded as one-hot categoricals. Every
# other column returned in X by select_model_columns is treated as
# numeric and scaled.
CATEGORICAL_COLUMNS = [
    "gender",
    "offer",
    "internet_type",
    "contract",
    "payment_method",
]

# internet_type is null exactly for customers with internet_service =
# false (confirmed by matching null counts against that column) -- "No
# Internet" fills that gap with an explicit category rather than leaving
# a null for the encoder to handle.
INTERNET_TYPE_SENTINEL = "No Internet"

# offer is null for customers with no promotional offer on file. Unlike
# internet_type, this isn't cross-checked against another column here --
# it's our working interpretation of what a null means, not a confirmed
# data-definition fact. "No Offer" fills that gap with an explicit
# category rather than leaving a null for the encoder to handle.
OFFER_SENTINEL = "No Offer"


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Construct the shared preprocessing transformer, unfitted.

    Categorical columns are constant-filled (internet_type and offer
    only, with their own sentinel values) then one-hot encoded; every
    other column in X is scaled as numeric. Callers must .fit() this on
    the train split only (see pipeline.split.split_feat_churn) and use
    .transform() on test/inference data -- never .fit() or .fit_transform()
    on the full dataset or on test.
    """
    # DuckDB's fetchdf() represents SQL NULL as Python None in these
    # object-dtype columns, not NaN, so missing_values must be set
    # explicitly -- SimpleImputer's default (NaN) does not mask None on
    # object arrays.
    internet_type_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=INTERNET_TYPE_SENTINEL,
                    missing_values=None,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    offer_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=OFFER_SENTINEL,
                    missing_values=None,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    other_categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column not in ("internet_type", "offer")
    ]

    return ColumnTransformer(
        transformers=[
            ("internet_type", internet_type_pipeline, ["internet_type"]),
            ("offer", offer_pipeline, ["offer"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), other_categorical_columns),
        ],
        # Every column not named above -- i.e. every column in X that
        # isn't one of the five CATEGORICAL_COLUMNS -- is treated as
        # numeric and scaled.
        remainder=StandardScaler(),
    )
