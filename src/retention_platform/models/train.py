"""Production training: tune XGBoost on the training split, check the tuned
parameters against a reference, fit the calibrated model, and publish it
with its manifest.

Run as a script: `python -m retention_platform.models.train`. Everything
happens inside one tracked run; the training run's folder holds config.json
and run.log, and metrics.json is written last, only after the model is
published. Only the training rows are used for tuning, fitting and
calibration; the test split is used only for its row count.
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from sklearn.model_selection import cross_val_predict

from retention_platform.config import ConfigValidationError, load_config
from retention_platform.logging_setup import setup_logging
from retention_platform.metrics import brier_score, lift_at_k, pr_auc, precision_at_k
from retention_platform.models import persist
from retention_platform.models.calibrate import build_cv, calibrate_candidate
from retention_platform.models.tune import tune_xgboost
from retention_platform.pipeline.preprocess import (
    build_preprocessing_pipeline,
    select_model_columns,
)
from retention_platform.pipeline.schema import build_input_schema, read_feat_churn_types
from retention_platform.pipeline.split import split_feat_churn
from retention_platform.tracking import TrackedRun, TrackingError, track_run

logger = logging.getLogger(__name__)

# The tuned XGBoost parameters the production model must have. Training
# stops if the search returns anything else; there is no override.
REFERENCE_HYPERPARAMETERS = {
    "n_estimators": 215,
    "max_depth": 8,
    "learning_rate": 0.02171140223187999,
    "subsample": 0.9442922333025374,
    "colsample_bytree": 0.6440207698110707,
    "min_child_weight": 9,
    "scale_pos_weight": 1,
}


class ReferenceMismatchError(Exception):
    """Raised when the tuned parameters differ from the reference."""


@dataclass(frozen=True)
class TrainingResult:
    """What a finished training run produced."""

    run_id: str
    model_path: Path
    metrics: dict


def check_against_reference(best_params: dict) -> None:
    """Raise ReferenceMismatchError unless best_params holds exactly the
    reference parameters with exactly equal values (no tolerance). NumPy
    scalars compare by value; a bool never equals a number."""
    problems = []

    missing = [name for name in REFERENCE_HYPERPARAMETERS if name not in best_params]
    unexpected = [name for name in best_params if name not in REFERENCE_HYPERPARAMETERS]
    if missing:
        problems.append(f"missing parameters: {missing}")
    if unexpected:
        problems.append(f"unexpected parameters: {unexpected}")

    for name, reference in REFERENCE_HYPERPARAMETERS.items():
        if name not in best_params:
            continue
        found = best_params[name]
        if isinstance(found, np.generic):
            found = found.item()
        if isinstance(found, bool) or found != reference:
            problems.append(f"{name}: found {found!r}, reference {reference!r}")

    if problems:
        raise ReferenceMismatchError(
            "The tuned hyperparameters differ from the reference: "
            + "; ".join(problems)
            + ". The production model would no longer be the model that was "
            "evaluated. There is no override."
        )


def compute_training_cv_metrics(calibrated_model, X_train, y_train, config: dict) -> dict:
    """Headline metrics from out-of-fold predictions on the training rows.

    calibrated_model must be unfitted: each outer fold fits and calibrates
    it on that fold's training portion and predicts the held-out portion.
    """
    k = config["business"]["k"]
    scores = cross_val_predict(
        calibrated_model, X_train, y_train, cv=build_cv(config), method="predict_proba"
    )[:, 1]
    return {
        "k": float(k),
        "training_rows": int(len(y_train)),
        "estimate_type": persist.TRAINING_CV_ESTIMATE_TYPE,
        "precision_at_k": float(precision_at_k(y_train, scores, k)),
        "lift_at_k": float(lift_at_k(y_train, scores, k)),
        "pr_auc": float(pr_auc(y_train, scores)),
        "brier_score": float(brier_score(y_train, scores)),
    }


def train_and_publish(
    config: dict,
    conn: duckdb.DuckDBPyConnection,
    run: TrackedRun,
    instant: datetime,
) -> TrainingResult:
    """Train, publish the model, then write the run's metrics.json.

    Must be called inside an already started tracked run. Any failure is
    logged before it leaves, so it is recorded in the run's log. If the
    model is published but metrics.json cannot be written, the model stays
    published and the error names its run id.
    """
    try:
        models_dir = config["paths"]["models_dir"]

        logger.info("Training run %s: checking the model path", run.run_id)
        persist.ensure_run_path_free(models_dir, run.run_id)

        logger.info("Loading and preprocessing the training split")
        split = split_feat_churn(conn)
        X_train_raw, y_train, _ = select_model_columns(split.train)
        preprocessor = build_preprocessing_pipeline()
        X_train = preprocessor.fit_transform(X_train_raw)
        train_rows = len(y_train)
        test_rows = len(split.test)
        logger.info("Training rows: %d (test rows, count only: %d)", train_rows, test_rows)

        logger.info("Running the hyperparameter search")
        search = tune_xgboost(X_train, y_train, config)

        logger.info("Checking the tuned hyperparameters against the reference")
        check_against_reference(search.best_params_)

        logger.info("Fitting the calibrated model on the training rows")
        calibrated = calibrate_candidate(search.best_estimator_, config)
        calibrated.fit(X_train, y_train)

        logger.info("Computing training cross-validation metrics")
        training_cv_metrics = compute_training_cv_metrics(
            calibrate_candidate(search.best_estimator_, config), X_train, y_train, config
        )

        logger.info("Building the input schema, fingerprints and manifest")
        schema = build_input_schema(read_feat_churn_types(conn), preprocessor)
        fingerprints = persist.build_fingerprints(config, train_rows, test_rows)
        manifest = persist.build_manifest(
            run_id=run.run_id,
            instant=instant,
            schema=schema,
            hyperparameters=copy.deepcopy(REFERENCE_HYPERPARAMETERS),
            training_cv_metrics=training_cv_metrics,
            fingerprints=fingerprints,
        )

        logger.info("Publishing the model run")
        model_path = persist.publish_run(
            models_dir,
            run.run_id,
            manifest,
            preprocessor,
            calibrated,
            manifest["input_schema_hash"],
        )
        logger.info("Published model run %s at %s", run.run_id, model_path)

        try:
            run.write_metrics(manifest["training_cv_metrics"])
        except TrackingError as exc:
            raise TrackingError(
                f"Model run {run.run_id} was published at {model_path}, "
                f"but tracking failed: {exc}"
            ) from exc

        return TrainingResult(
            run_id=run.run_id, model_path=model_path, metrics=manifest["training_cv_metrics"]
        )
    except Exception:
        logger.exception("Training run %s failed", run.run_id)
        raise


def main() -> None:
    setup_logging()

    conn = None
    try:
        config = load_config()
        instant = datetime.now(timezone.utc)
        with track_run(config, "train", instant) as run:
            try:
                conn = duckdb.connect(str(config["paths"]["interim_db"]), read_only=True)
            except duckdb.Error:
                logger.exception("Training run %s failed", run.run_id)
                raise
            result = train_and_publish(config, conn, run, instant)
    except ConfigValidationError as exc:
        logger.error("Configuration error:\n%s", exc)
        sys.exit(1)
    except TrackingError as exc:
        logger.error("Tracking error: %s", exc)
        sys.exit(1)
    except persist.PublicationError as exc:
        logger.error("Model publication error: %s", exc)
        sys.exit(1)
    except ReferenceMismatchError as exc:
        logger.error("Hyperparameter check failed: %s", exc)
        sys.exit(1)
    except duckdb.Error as exc:
        logger.error("Database error:\n%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error during training: %s: %s", type(exc).__name__, exc)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    logger.info(
        "Training complete: run %s, model at %s, precision@k %.4f, lift@k %.2f, "
        "pr_auc %.4f, brier %.4f",
        result.run_id,
        result.model_path,
        result.metrics["precision_at_k"],
        result.metrics["lift_at_k"],
        result.metrics["pr_auc"],
        result.metrics["brier_score"],
    )


if __name__ == "__main__":
    main()
