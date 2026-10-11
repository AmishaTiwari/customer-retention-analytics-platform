"""Calibration method, the cross-validation splitter, and the unfitted
calibration wrapper.

build_cv builds the same fold structure as the tuning searches, so
calibration and out-of-fold scoring use the same train/validation
splits. Every function takes config as a plain dict.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold

# Calibration method for calibrate_candidate's CalibratedClassifierCV.
CALIBRATION_METHOD = "sigmoid"


def build_cv(config: dict) -> RepeatedStratifiedKFold:
    """Build the training-side cross-validation splitter, mirroring
    models/tune.py's tune_* functions exactly so calibration and
    out-of-fold scoring use the same fold structure as Commit 8's tuning.
    """
    return RepeatedStratifiedKFold(
        n_splits=config["data"]["cv_folds"],
        n_repeats=config["data"]["cv_repeats"],
        random_state=config["reproducibility"]["seed"],
    )


def calibrate_candidate(fitted_model, config: dict) -> CalibratedClassifierCV:
    """Wrap one already-tuned candidate estimator in an unfitted
    CalibratedClassifierCV, per the Validation Strategy correction: model
    comparison and calibration belong on training-side CV, not the test
    set. fitted_model is a Commit 8 tuned estimator (e.g. a tune_*
    search's best_estimator_) -- tuning it is the caller's responsibility,
    not this function's, and no re-tuning happens here. CALIBRATION_METHOD
    ("sigmoid") is formalized in ADR-0013; see that ADR for the reasoning
    behind the choice.

    Returns the CalibratedClassifierCV unfitted: fitting happens inside
    cross_val_predict in evaluate_candidates_cv, one fold at a time, so
    each fold's calibration is learned only from that fold's own training
    portion.
    """
    return CalibratedClassifierCV(
        fitted_model, method=CALIBRATION_METHOD, cv=build_cv(config)
    )
