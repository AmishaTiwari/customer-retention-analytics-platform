"""Tests for models/train.py: the production training procedure.

No real data and no real search: a small synthetic feat_churn table is
built in DuckDB, tune_xgboost is replaced by a fake search that returns a
real XGBClassifier with the reference parameters, and everything else
(preprocessing, schema, calibration, persistence, tracking) is the real
code. All files are written under tmp_path.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import math
from datetime import datetime, timezone
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from retention_platform import tracking
from retention_platform.config import load_config
from retention_platform.metrics import brier_score, lift_at_k, pr_auc, precision_at_k
from retention_platform.models import persist, train
from retention_platform.models.calibrate import calibrate_candidate
from retention_platform.models.persist import (
    TRAINING_CV_ESTIMATE_TYPE,
    load_run,
)
from retention_platform.models.train import (
    REFERENCE_HYPERPARAMETERS,
    ReferenceMismatchError,
    check_against_reference,
    compute_training_cv_metrics,
)
from retention_platform.pipeline.preprocess import (
    build_preprocessing_pipeline,
    prepare_model_inputs,
    select_model_columns,
)
from retention_platform.pipeline.split import split_feat_churn
from retention_platform.tracking import TrackingError

FIXED_INSTANT = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
RUN_ID = "20261011T120000Z"


class FixedDatetime(datetime):
    """Makes datetime.now() return FIXED_INSTANT so the run id is known."""

    @classmethod
    def now(cls, tz=None):
        return FIXED_INSTANT


# --- Synthetic data and helpers ---------------------------------------------


def synthetic_frame(rows: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    tenure = rng.integers(1, 72, rows)
    monthly = rng.normal(70, 20, rows).round(2)
    contract = rng.choice(["Month-to-Month", "One Year", "Two Year"], rows, p=[0.55, 0.25, 0.2])
    internet_choices = ["DSL", "Cable", "Fiber Optic", None]
    offer_choices = ["Offer A", "Offer B", None]
    logit = -0.8 + 1.0 * (contract == "Month-to-Month") - 0.03 * tenure + 0.01 * (monthly - 70)
    churn = rng.random(rows) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame(
        {
            "customer_id": [f"C{index:05d}" for index in range(rows)],
            "is_voluntary_churn": churn.tolist(),
            "gender": rng.choice(["Female", "Male"], rows).tolist(),
            "age": age.tolist(),
            "under_30": (age < 30).tolist(),
            "monthly_charge": monthly.tolist(),
            "offer": [offer_choices[i] for i in rng.integers(0, 3, rows)],
            "internet_type": [internet_choices[i] for i in rng.integers(0, 4, rows)],
            "contract": contract.tolist(),
            "payment_method": rng.choice(
                ["Credit Card", "Mailed Check", "Bank Withdrawal"], rows
            ).tolist(),
            "tenure_in_months": tenure.tolist(),
            "country": ["United States"] * rows,
            "state": ["California"] * rows,
            "city": [f"City {index % 5}" for index in range(rows)],
            "zip_code": [f"9000{index % 7}" for index in range(rows)],
        }
    )


def build_database(target) -> duckdb.DuckDBPyConnection:
    """Create the synthetic feat_churn table with the four SQL types."""
    connection = duckdb.connect(str(target))
    connection.execute(
        """
        CREATE TABLE feat_churn (
            customer_id VARCHAR, is_voluntary_churn BOOLEAN, gender VARCHAR,
            age INTEGER, under_30 BOOLEAN, monthly_charge DOUBLE, offer VARCHAR,
            internet_type VARCHAR, contract VARCHAR, payment_method VARCHAR,
            tenure_in_months INTEGER, country VARCHAR, state VARCHAR,
            city VARCHAR, zip_code VARCHAR
        )
        """
    )
    connection.register("synthetic_frame", synthetic_frame())
    connection.execute("INSERT INTO feat_churn SELECT * FROM synthetic_frame")
    connection.unregister("synthetic_frame")
    return connection


def make_config(tmp_path) -> dict:
    config = load_config()
    config["paths"] = {
        **config["paths"],
        "models_dir": tmp_path / "models",
        "runs_dir": tmp_path / "runs",
        "interim_db": tmp_path / "synthetic.duckdb",
    }
    return config


def make_fake_tune(observations: dict, params: dict | None = None):
    """A stand-in for tune_xgboost: a fitted real XGBClassifier built from
    the given (default: reference) parameters."""
    chosen = dict(REFERENCE_HYPERPARAMETERS if params is None else params)

    def fake_tune(X_train, y_train, config):
        observations["tune_calls"] = observations.get("tune_calls", 0) + 1
        observations["X_train"] = X_train
        estimator = XGBClassifier(
            eval_metric="logloss", random_state=config["reproducibility"]["seed"], **chosen
        )
        estimator.fit(X_train, y_train)
        return SimpleNamespace(best_params_=dict(chosen), best_estimator_=estimator)

    return fake_tune


def make_env(tmp_path, monkeypatch, params: dict | None = None) -> SimpleNamespace:
    config = make_config(tmp_path)
    build_database(config["paths"]["interim_db"]).close()
    observations: dict = {}
    monkeypatch.setattr(train, "load_config", lambda: config)
    monkeypatch.setattr(train, "tune_xgboost", make_fake_tune(observations, params))
    monkeypatch.setattr(train, "datetime", FixedDatetime)
    return SimpleNamespace(
        config=config,
        observations=observations,
        models_dir=config["paths"]["models_dir"],
        run_dir=config["paths"]["runs_dir"] / "train" / RUN_ID,
    )


def tree_state(root):
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for path in sorted(root.rglob("*"))
    }


def names_in(folder):
    return sorted(path.name for path in folder.iterdir())


def to_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


@pytest.fixture(autouse=True)
def info_level_logging():
    """Run logs are written at INFO, so raise the root level for each test."""
    root = logging.getLogger()
    saved = root.level
    root.setLevel(logging.INFO)
    yield
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved)


# --- Reference constant and check -------------------------------------------


def test_reference_constant_has_exactly_the_seven_values():
    assert REFERENCE_HYPERPARAMETERS == {
        "n_estimators": 215,
        "max_depth": 8,
        "learning_rate": 0.02171140223187999,
        "subsample": 0.9442922333025374,
        "colsample_bytree": 0.6440207698110707,
        "min_child_weight": 9,
        "scale_pos_weight": 1,
    }


def test_the_reference_values_are_accepted():
    check_against_reference(dict(REFERENCE_HYPERPARAMETERS))


def test_numpy_scalars_compare_by_value():
    numpy_values = {
        "n_estimators": np.int64(215),
        "max_depth": np.int32(8),
        "learning_rate": np.float64(0.02171140223187999),
        "subsample": np.float64(0.9442922333025374),
        "colsample_bytree": np.float64(0.6440207698110707),
        "min_child_weight": np.int64(9),
        "scale_pos_weight": np.int64(1),
    }

    check_against_reference(numpy_values)


@pytest.mark.parametrize("name", list(REFERENCE_HYPERPARAMETERS))
def test_the_smallest_possible_change_to_any_parameter_is_refused(name):
    params = dict(REFERENCE_HYPERPARAMETERS)
    value = params[name]
    params[name] = math.nextafter(value, math.inf) if isinstance(value, float) else value + 1

    with pytest.raises(ReferenceMismatchError, match=name):
        check_against_reference(params)


def test_scale_pos_weight_none_is_refused():
    params = {**REFERENCE_HYPERPARAMETERS, "scale_pos_weight": None}

    with pytest.raises(ReferenceMismatchError, match="scale_pos_weight"):
        check_against_reference(params)


def test_a_bool_never_equals_a_number():
    params = {**REFERENCE_HYPERPARAMETERS, "scale_pos_weight": True}

    with pytest.raises(ReferenceMismatchError, match="scale_pos_weight"):
        check_against_reference(params)


def test_a_missing_key_is_refused():
    params = dict(REFERENCE_HYPERPARAMETERS)
    del params["max_depth"]

    with pytest.raises(ReferenceMismatchError, match="missing parameters.*max_depth"):
        check_against_reference(params)


def test_an_extra_key_is_refused():
    params = {**REFERENCE_HYPERPARAMETERS, "gamma": 0.1}

    with pytest.raises(ReferenceMismatchError, match="unexpected parameters.*gamma"):
        check_against_reference(params)


def test_the_message_names_every_difference_and_says_there_is_no_override():
    params = {**REFERENCE_HYPERPARAMETERS, "n_estimators": 216, "max_depth": 9}

    with pytest.raises(ReferenceMismatchError) as exc_info:
        check_against_reference(params)

    message = str(exc_info.value)
    assert "n_estimators: found 216, reference 215" in message
    assert "max_depth: found 9, reference 8" in message
    assert "no override" in message
    assert "no longer be the model that was evaluated" in message


# --- Headline metrics -------------------------------------------------------


def test_training_cv_metrics_match_an_independent_computation(tmp_path):
    config = make_config(tmp_path)
    connection = build_database(":memory:")
    split = split_feat_churn(connection)
    X_raw, y_train, _ = select_model_columns(split.train)
    X_train = build_preprocessing_pipeline().fit_transform(X_raw)
    seed = config["reproducibility"]["seed"]
    estimator = XGBClassifier(
        n_estimators=20, max_depth=3, eval_metric="logloss", random_state=seed
    )

    result = compute_training_cv_metrics(
        calibrate_candidate(estimator, config), X_train, y_train, config
    )

    independent = cross_val_predict(
        CalibratedClassifierCV(
            estimator,
            method="sigmoid",
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed),
        ),
        X_train,
        y_train,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed),
        method="predict_proba",
    )[:, 1]
    k = config["business"]["k"]
    assert set(result) == {
        "k",
        "training_rows",
        "estimate_type",
        "precision_at_k",
        "lift_at_k",
        "pr_auc",
        "brier_score",
    }
    assert result["estimate_type"] == TRAINING_CV_ESTIMATE_TYPE
    assert result["k"] == k
    assert result["training_rows"] == len(y_train)
    assert result["precision_at_k"] == pytest.approx(precision_at_k(y_train, independent, k))
    assert result["lift_at_k"] == pytest.approx(lift_at_k(y_train, independent, k))
    assert result["pr_auc"] == pytest.approx(pr_auc(y_train, independent))
    assert result["brier_score"] == pytest.approx(brier_score(y_train, independent))
    assert type(result["k"]) is float
    assert type(result["training_rows"]) is int
    for name in ("precision_at_k", "lift_at_k", "pr_auc", "brier_score"):
        assert type(result[name]) is float
    connection.close()


# --- A full successful run through main() -----------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One complete training run through main(), with observations."""
    tmp_path = tmp_path_factory.mktemp("trained")
    config = make_config(tmp_path)
    build_database(config["paths"]["interim_db"]).close()
    observations: dict = {"fit_rows": [], "at_metrics_write": None}

    original_fit = CalibratedClassifierCV.fit
    original_write_metrics = tracking.TrackedRun.write_metrics

    def spy_fit(self, X, y, *args, **kwargs):
        observations["fit_rows"].append(X.shape[0])
        return original_fit(self, X, y, *args, **kwargs)

    def spy_write_metrics(self, metrics):
        model_dir = config["paths"]["models_dir"] / RUN_ID
        observations["at_metrics_write"] = {
            "model_files": names_in(model_dir) if model_dir.exists() else None,
            "metrics_exists": (self.folder / "metrics.json").exists(),
        }
        return original_write_metrics(self, metrics)

    def no_transform(self, X):
        raise AssertionError("the preprocessor must not transform rows after fitting")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "load_config", lambda: config)
        patch.setattr(train, "tune_xgboost", make_fake_tune(observations))
        patch.setattr(train, "datetime", FixedDatetime)
        patch.setattr(CalibratedClassifierCV, "fit", spy_fit)
        patch.setattr(tracking.TrackedRun, "write_metrics", spy_write_metrics)
        patch.setattr(ColumnTransformer, "transform", no_transform)
        train.main()

    return SimpleNamespace(
        config=config,
        observations=observations,
        models_dir=config["paths"]["models_dir"],
        model_dir=config["paths"]["models_dir"] / RUN_ID,
        run_dir=config["paths"]["runs_dir"] / "train" / RUN_ID,
        db_path=config["paths"]["interim_db"],
    )


def split_sizes(db_path):
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        split = split_feat_churn(connection)
        return len(split.train), len(split.test)
    finally:
        connection.close()


def test_success_publishes_exactly_the_two_files(trained):
    assert names_in(trained.model_dir) == ["bundle.joblib", "manifest.json"]
    assert names_in(trained.models_dir) == [RUN_ID]


def test_success_records_one_run_id_everywhere(trained):
    manifest = json.loads((trained.model_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == trained.model_dir.name == trained.run_dir.name == RUN_ID
    run_id_instant = datetime.strptime(manifest["run_id"], "%Y%m%dT%H%M%SZ")
    trained_at = datetime.strptime(manifest["trained_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert run_id_instant == trained_at


def test_success_manifest_holds_the_reference_hyperparameters(trained):
    manifest = json.loads((trained.model_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["hyperparameters"] == REFERENCE_HYPERPARAMETERS


def test_success_manifest_metrics_equal_the_run_metrics_file(trained):
    manifest = json.loads((trained.model_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((trained.run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert manifest["training_cv_metrics"] == metrics
    assert metrics["estimate_type"] == TRAINING_CV_ESTIMATE_TYPE
    assert metrics["k"] == trained.config["business"]["k"]


def test_success_fingerprints_have_four_keys_and_the_split_sizes(trained):
    manifest = json.loads((trained.model_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows, test_rows = split_sizes(trained.db_path)

    fingerprints = manifest["fingerprints"]
    assert set(fingerprints) == {
        "config_hash",
        "sql_hash",
        "data_fingerprint",
        "hash_normalization",
    }
    assert fingerprints["data_fingerprint"]["train_rows"] == train_rows
    assert fingerprints["data_fingerprint"]["test_rows"] == test_rows
    assert manifest["training_cv_metrics"]["training_rows"] == train_rows


def test_success_run_folder_holds_config_log_and_metrics(trained):
    assert names_in(trained.run_dir) == ["config.json", "metrics.json", "run.log"]


def test_success_model_loads_and_predicts(trained):
    loaded = load_run(trained.models_dir)
    connection = duckdb.connect(str(trained.db_path), read_only=True)
    try:
        X_raw, _, _ = select_model_columns(split_feat_churn(connection).train)
    finally:
        connection.close()

    probabilities = loaded.model.predict_proba(loaded.preprocessor.transform(X_raw))

    assert loaded.run_id == RUN_ID
    assert probabilities.shape == (len(X_raw), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_metrics_json_is_written_only_after_the_model_is_published(trained):
    observed = trained.observations["at_metrics_write"]

    assert observed["model_files"] == ["bundle.joblib", "manifest.json"]
    assert observed["metrics_exists"] is False


def test_the_fits_see_only_training_rows(trained):
    train_rows, _ = split_sizes(trained.db_path)
    fit_rows = trained.observations["fit_rows"]

    assert train_rows in fit_rows
    assert max(fit_rows) == train_rows


def test_preprocessing_equals_prepare_model_inputs(trained):
    connection = duckdb.connect(str(trained.db_path), read_only=True)
    try:
        expected = prepare_model_inputs(connection).X_train
    finally:
        connection.close()

    assert np.array_equal(to_array(trained.observations["X_train"]), to_array(expected))


def test_train_py_never_references_test_split_identifiers():
    tree = ast.parse(inspect.getsource(train))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    referenced = names | attributes

    assert not (referenced & {"X_test", "y_test", "id_test"})
    assert "split_feat_churn" in referenced


# --- Failures ---------------------------------------------------------------


def run_main_expecting_exit(code: int = 1):
    with pytest.raises(SystemExit) as exc_info:
        train.main()
    assert exc_info.value.code == code


def test_hyperparameter_mismatch_writes_nothing_under_models_dir(tmp_path, monkeypatch):
    changed = {
        **REFERENCE_HYPERPARAMETERS,
        "learning_rate": math.nextafter(REFERENCE_HYPERPARAMETERS["learning_rate"], 1.0),
    }
    env = make_env(tmp_path, monkeypatch, params=changed)

    run_main_expecting_exit()

    assert not env.models_dir.exists()
    assert names_in(env.run_dir) == ["config.json", "run.log"]
    log = (env.run_dir / "run.log").read_text(encoding="utf-8")
    assert "learning_rate" in log
    assert "no override" in log


def test_existing_model_path_stops_before_any_tuning(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    existing = env.models_dir / RUN_ID
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    before = tree_state(env.models_dir)

    run_main_expecting_exit()

    assert env.observations.get("tune_calls", 0) == 0
    assert tree_state(env.models_dir) == before
    assert names_in(env.run_dir) == ["config.json", "run.log"]
    assert "already exists" in (env.run_dir / "run.log").read_text(encoding="utf-8")


def test_tracking_failure_after_publication_keeps_the_model(tmp_path, monkeypatch, caplog):
    env = make_env(tmp_path, monkeypatch)

    def broken_write_metrics(self, metrics):
        raise TrackingError("disk full")

    monkeypatch.setattr(tracking.TrackedRun, "write_metrics", broken_write_metrics)

    with caplog.at_level(logging.ERROR):
        run_main_expecting_exit()

    assert names_in(env.models_dir / RUN_ID) == ["bundle.joblib", "manifest.json"]
    assert load_run(env.models_dir).run_id == RUN_ID
    assert not (env.run_dir / "metrics.json").exists()
    message = f"Model run {RUN_ID} was published at"
    assert message in caplog.text
    assert "tracking failed: disk full" in caplog.text
    assert message in (env.run_dir / "run.log").read_text(encoding="utf-8")


def test_tracking_failure_at_the_start_changes_nothing_else(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file", encoding="utf-8")
    env.config["paths"]["runs_dir"] = blocker

    run_main_expecting_exit()

    assert env.observations.get("tune_calls", 0) == 0
    assert not env.models_dir.exists()


def test_a_failed_bundle_write_leaves_no_temporary_directory(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(persist.joblib, "dump", broken_dump)

    run_main_expecting_exit()

    assert names_in(env.models_dir) == []
    assert names_in(env.run_dir) == ["config.json", "run.log"]


def test_a_missing_database_ends_with_exit_code_one(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    env.config["paths"]["interim_db"].unlink()

    run_main_expecting_exit()

    assert env.observations.get("tune_calls", 0) == 0
    assert not env.models_dir.exists()
    assert names_in(env.run_dir) == ["config.json", "run.log"]
