"""Per-customer risk drivers: contributions from the five fold models,
summed from the encoded columns back to the model's input features, and the
choice of the three strongest drivers with a direction.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from retention_platform.models.persist import LoadedRun

EXPECTED_FOLD_MODELS = 5
TOP_N_DRIVERS = 3

DIRECTION_RAISES = "raises risk"
DIRECTION_LOWERS = "lowers risk"

DRIVER_COLUMNS = [
    "driver_1",
    "driver_1_direction",
    "driver_2",
    "driver_2_direction",
    "driver_3",
    "driver_3_direction",
]

# One fixed label per model input feature, in schema order. A label names
# the feature and does not state a customer's value.
DRIVER_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "under_30": "Age under 30",
    "senior_citizen": "Senior citizen status",
    "married": "Marital status",
    "dependents": "Dependents",
    "number_of_dependents": "Number of dependents",
    "latitude": "Customer location (latitude)",
    "longitude": "Customer location (longitude)",
    "population": "Population of the customer zip code area",
    "referred_a_friend": "Referred a friend",
    "number_of_referrals": "Number of referrals",
    "tenure_in_months": "Tenure in months",
    "offer": "Promotional offer",
    "phone_service": "Phone service",
    "avg_monthly_long_distance_charges": "Average monthly long-distance charges",
    "multiple_lines": "Multiple phone lines",
    "internet_service": "Internet service",
    "internet_type": "Internet connection type",
    "avg_monthly_gb_download": "Average monthly data download (GB)",
    "online_security": "Online security add-on",
    "online_backup": "Online backup add-on",
    "device_protection_plan": "Device protection plan",
    "premium_tech_support": "Premium tech support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
    "streaming_music": "Streaming music",
    "unlimited_data": "Unlimited data plan",
    "contract": "Contract type",
    "paperless_billing": "Paperless billing",
    "payment_method": "Payment method",
    "monthly_charge": "Monthly charge",
    "total_charges": "Total charges billed to date",
    "total_refunds": "Total refunds to date",
    "total_extra_data_charges": "Total extra data charges to date",
    "total_long_distance_charges": "Total long-distance charges to date",
    "total_revenue": "Total revenue from the customer to date",
    "num_addon_services": "Number of add-on services",
    "is_month_to_month": "Month-to-month contract",
    "internet_without_security": "Internet service without online security",
    "no_addons_despite_internet": "Internet service with no add-on services",
    "is_tenure_censored": "Tenure at the 72-month recording limit",
}


class DriverError(Exception):
    """Raised when the loaded model cannot be used for drivers, when a
    selected driver has a contribution of exactly zero, or when a
    contribution is not finite."""


def _is_fitted(obj: object) -> bool:
    try:
        check_is_fitted(obj)
    except NotFittedError:
        return False
    return True


def _fold_estimators(model: object) -> list[xgb.XGBClassifier]:
    """The fitted XGBClassifier of each calibrated fold, after checking the model's shape."""
    if not isinstance(model, CalibratedClassifierCV):
        raise DriverError(
            f"the model is not a CalibratedClassifierCV, got {type(model).__name__}"
        )
    if not _is_fitted(model):
        raise DriverError("the model is not fitted")

    folds = list(model.calibrated_classifiers_)
    if len(folds) != EXPECTED_FOLD_MODELS:
        raise DriverError(
            f"the model has {len(folds)} calibrated classifiers, expected {EXPECTED_FOLD_MODELS}"
        )

    estimators = []
    for position, fold in enumerate(folds):
        estimator = getattr(fold, "estimator", None)
        if not isinstance(estimator, xgb.XGBClassifier):
            raise DriverError(
                f"calibrated classifier {position} does not wrap an XGBClassifier, "
                f"got {type(estimator).__name__}"
            )
        try:
            estimator.get_booster()
        except NotFittedError as exc:
            raise DriverError(f"the XGBClassifier of calibrated classifier {position} is not fitted") from exc
        estimators.append(estimator)
    return estimators


def _column_names(columns: object, fitted_names: list[str], label: str) -> list[str]:
    """Input-column names behind one fitted transformer's column selection."""
    if isinstance(columns, (str, int, np.integer)):
        columns = [columns]
    if not isinstance(columns, (list, tuple, np.ndarray)):
        raise DriverError(f"transformer '{label}' selects columns in an unsupported way")

    names = []
    for column in columns:
        if isinstance(column, str):
            names.append(column)
        elif isinstance(column, (int, np.integer)) and not isinstance(column, bool):
            if not 0 <= column < len(fitted_names):
                raise DriverError(f"transformer '{label}' selects column position {column}, out of range")
            names.append(fitted_names[column])
        else:
            raise DriverError(f"transformer '{label}' selects columns in an unsupported way")
    return names


def _transformer_owners(label: str, transformer: object, names: list[str]) -> list[str]:
    """The owning input column of each output column of one fitted transformer."""
    steps = [step for _, step in transformer.steps] if isinstance(transformer, Pipeline) else [transformer]

    owners = names
    encoders = 0
    for step in steps:
        if isinstance(step, OneHotEncoder):
            encoders += 1
            if encoders > 1:
                raise DriverError(f"transformer '{label}' has more than one one-hot encoder")
            if step.drop_idx_ is not None:
                raise DriverError(f"the one-hot encoder in '{label}' drops a category")
            if step.min_frequency is not None or step.max_categories is not None:
                raise DriverError(f"the one-hot encoder in '{label}' groups infrequent categories")
            if len(step.categories_) != len(names):
                raise DriverError(
                    f"the one-hot encoder in '{label}' learned {len(step.categories_)} "
                    f"column(s) for {len(names)} input column(s)"
                )
            owners = [name for name, categories in zip(names, step.categories_) for _ in categories]
        elif isinstance(step, SimpleImputer):
            if step.add_indicator:
                raise DriverError(f"the imputer in '{label}' adds indicator columns")
        elif not isinstance(step, StandardScaler):
            raise DriverError(f"transformer '{label}' contains {type(step).__name__}, which cannot be mapped")
    return owners


def encoded_column_owners(loaded_run: LoadedRun) -> list[str]:
    """Check the structure the contribution arithmetic relies on, then return
    the input column that each encoded column comes from, in the
    preprocessor's output order.

    Raises DriverError naming the broken condition.
    """
    estimators = _fold_estimators(loaded_run.model)

    preprocessor = loaded_run.preprocessor
    if not isinstance(preprocessor, ColumnTransformer) or not _is_fitted(preprocessor):
        raise DriverError(
            f"the preprocessor is not a fitted ColumnTransformer, got {type(preprocessor).__name__}"
        )
    width = len(preprocessor.get_feature_names_out())
    for position, estimator in enumerate(estimators):
        inputs = estimator.get_booster().num_features()
        if inputs != width:
            raise DriverError(
                f"the preprocessor produces {width} encoded columns but the estimator of "
                f"calibrated classifier {position} takes {inputs} input features"
            )

    if not hasattr(preprocessor, "feature_names_in_"):
        raise DriverError("the preprocessor was not fitted on a table with column names")
    fitted_names = list(preprocessor.feature_names_in_)

    owners: list[str] = []
    for label, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str):
            if transformer == "drop":
                continue
            if transformer != "passthrough":
                raise DriverError(f"transformer '{label}' is '{transformer}', which cannot be mapped")
            owners.extend(_column_names(columns, fitted_names, label))
            continue
        owners.extend(
            _transformer_owners(label, transformer, _column_names(columns, fitted_names, label))
        )

    if len(owners) != width:
        raise DriverError(
            f"{width} encoded columns but {len(owners)} owning input columns; "
            "an encoded column has no owner"
        )

    schema_names = [column["name"] for column in loaded_run.schema["columns"]]
    not_in_schema = sorted(set(owners) - set(schema_names))
    if not_in_schema:
        raise DriverError(f"encoded columns are owned by inputs that are not in the schema: {not_in_schema}")
    owns_nothing = [name for name in schema_names if name not in set(owners)]
    if owns_nothing:
        raise DriverError(f"schema input columns that own no encoded column: {owns_nothing}")
    return owners


def _owner_matrix(owners: list[str], schema_names: list[str]) -> np.ndarray:
    """0/1 matrix with one row per encoded column and one column per input
    feature; a 1 marks the input that owns the encoded column."""
    matrix = np.zeros((len(owners), len(schema_names)), dtype="float64")
    position_of = {name: position for position, name in enumerate(schema_names)}
    for column, owner in enumerate(owners):
        matrix[column, position_of[owner]] = 1.0
    return matrix


def averaged_contributions(
    loaded_run: LoadedRun, inputs: pd.DataFrame, customer_ids: list[str]
) -> pd.DataFrame:
    """Per-customer contribution of each input feature to the log-odds of churn.

    inputs holds the schema's input columns in schema order; customer_ids
    holds one unique id per row. Each fold model's built-in contributions
    are summed from the encoded columns to their input feature and the five
    results are averaged. The baseline column, the calibrators, the
    probabilities and any label are not used.
    """
    schema_names = [column["name"] for column in loaded_run.schema["columns"]]
    if list(inputs.columns) != schema_names:
        raise ValueError("inputs must have exactly the schema's input columns, in the schema's order")
    if len(customer_ids) != len(inputs):
        raise ValueError(f"{len(customer_ids)} customer ids for {len(inputs)} rows")
    if len(set(customer_ids)) != len(customer_ids):
        raise ValueError("customer_ids contains duplicates")

    owner_matrix = _owner_matrix(encoded_column_owners(loaded_run), schema_names)

    matrix = loaded_run.preprocessor.transform(inputs)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)

    per_fold = []
    for estimator in _fold_estimators(loaded_run.model):
        booster = estimator.get_booster()
        contributions = booster.predict(
            xgb.DMatrix(matrix, feature_names=booster.feature_names), pred_contribs=True
        ).astype("float64")
        per_fold.append(contributions[:, :-1] @ owner_matrix)

    averaged = np.mean(per_fold, axis=0)
    if not np.isfinite(averaged).all():
        raise DriverError("the model produced a contribution that is not finite")
    return pd.DataFrame(averaged, index=list(customer_ids), columns=schema_names, dtype="float64")


def select_top_drivers(contributions: pd.DataFrame) -> pd.DataFrame:
    """The three strongest drivers of each customer, by absolute contribution.

    contributions has one row per customer and one column per input feature.
    Equal absolute values are ordered by column order. Direction is
    "raises risk" for a positive value and "lowers risk" for a negative one.
    The input is not changed.
    """
    if contributions.shape[1] < TOP_N_DRIVERS:
        raise ValueError(
            f"at least {TOP_N_DRIVERS} feature columns are required, got {contributions.shape[1]}"
        )
    values = contributions.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise DriverError(
            f"contribution for customer {contributions.index[row]!r}, "
            f"feature {contributions.columns[column]!r} is not finite"
        )

    order = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :TOP_N_DRIVERS]
    selected = np.take_along_axis(values, order, axis=1)
    zero_positions = np.argwhere(selected == 0)
    if len(zero_positions):
        zero_row, zero_rank = zero_positions[0]
        raise DriverError(
            f"customer {contributions.index[zero_row]!r}: the contribution of selected feature "
            f"{contributions.columns[order[zero_row, zero_rank]]!r} is exactly zero"
        )

    features = np.asarray(contributions.columns, dtype=object)[order]
    directions = np.where(selected > 0, DIRECTION_RAISES, DIRECTION_LOWERS)

    result = np.empty((len(values), 2 * TOP_N_DRIVERS), dtype=object)
    result[:, 0::2] = features
    result[:, 1::2] = directions
    return pd.DataFrame(result, index=contributions.index, columns=DRIVER_COLUMNS, dtype=object)


def _encoded_position(key: str, feature_names: list[str] | None, width: int) -> int | None:
    """The encoded column a get_score key refers to, or None if it is not exactly one."""
    if feature_names is None:
        match = re.fullmatch(r"f(\d+)", key)
        position = int(match.group(1)) if match else None
    else:
        matches = [index for index, name in enumerate(feature_names) if name == key]
        position = matches[0] if len(matches) == 1 else None
    if position is None or not 0 <= position < width:
        return None
    return position


def _fold_gains(booster: xgb.Booster, width: int, fold: int) -> np.ndarray:
    """Total gain of each encoded column in one fold booster; a column never
    used in a split is absent from get_score and counts as zero."""
    gains = np.zeros(width, dtype="float64")
    for key, gain in booster.get_score(importance_type="total_gain").items():
        position = _encoded_position(key, booster.feature_names, width)
        if position is None:
            raise DriverError(
                f"fold {fold}: get_score key {key!r} cannot be mapped to exactly one encoded column"
            )
        if not np.isfinite(gain) or gain < 0:
            raise DriverError(f"fold {fold}: gain for key {key!r} is negative or not finite")
        gains[position] = gain
    if gains.sum() == 0:
        raise DriverError(f"fold {fold}: total gain is zero")
    return gains


def global_importance(loaded_run: LoadedRun) -> pd.DataFrame:
    """How much the model relies on each input feature overall.

    Each fold's total gain is summed from the encoded columns to their input
    feature and divided by the fold's total gain, so a fold's shares add up
    to 1; the five folds' shares are then averaged. The result has one row
    per input feature, named "feature", with columns share and rank (1 for
    the largest share, equal shares ranked by schema column order), sorted
    by rank. The shares are a different measure from the per-customer
    contributions and are not comparable with them on one scale.
    """
    owners = encoded_column_owners(loaded_run)
    schema_names = [column["name"] for column in loaded_run.schema["columns"]]
    owner_matrix = _owner_matrix(owners, schema_names)

    fold_shares = []
    for fold, estimator in enumerate(_fold_estimators(loaded_run.model)):
        gains = _fold_gains(estimator.get_booster(), len(owners), fold)
        fold_shares.append((gains @ owner_matrix) / gains.sum())

    shares = np.mean(fold_shares, axis=0)
    order = np.argsort(-shares, kind="stable")
    return pd.DataFrame(
        {"share": shares[order], "rank": np.arange(1, len(order) + 1, dtype="int64")},
        index=pd.Index(np.asarray(schema_names, dtype=object)[order], name="feature"),
    )
