"""Tests for models/persist.py: the fingerprints, the manifest, the
existence check, the publication of a model run, and run selection and
loading.

Synthetic files live under tmp_path; the real repository folders are only
read, never written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import tomllib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from retention_platform.config import REPO_ROOT, load_config
from retention_platform.data import ingest
from retention_platform.data.prepare import (
    CLEANING_SQL_DIR,
    MODELING_VIEW_SQL_DIR,
    STAGING_SQL_DIR,
    TARGET_SQL_DIR,
    run_cleaning,
    run_modeling_view,
    run_staging,
    run_target,
)
from retention_platform.features.build import FEATURES_SQL_DIR, run_features
from retention_platform.models import persist
from retention_platform.models.persist import (
    BUNDLE_FILE,
    HASH_NORMALIZATION,
    HELD_OUT_RESULTS_REFERENCE,
    MANIFEST_FILE,
    SQL_FOLDERS,
    TEMP_DIR_PREFIX,
    TRAINING_CV_ESTIMATE_TYPE,
    LoadedRun,
    ModelLoadError,
    PublicationError,
    build_fingerprints,
    build_manifest,
    config_hash,
    data_fingerprint,
    ensure_run_path_free,
    find_valid_run_ids,
    is_valid_run_id,
    library_versions,
    load_run,
    package_version,
    publish_run,
    select_run_id,
    sha256_normalized,
    sql_hash,
)
from retention_platform.pipeline.preprocess import build_preprocessing_pipeline
from retention_platform.pipeline.schema import build_input_schema, schema_hash
from retention_platform.tracking import RUN_ID_FORMAT, config_snapshot, format_run_id

HEX64 = re.compile(r"[0-9a-f]{64}")


def independent_hash(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def write_bytes(folder, name: str, data: bytes):
    path = folder / name
    path.write_bytes(data)
    return path


# --- Normalized file hash ---------------------------------------------------


def test_crlf_and_lf_versions_hash_identically(tmp_path):
    lf = write_bytes(tmp_path, "lf.txt", b"one\ntwo\nthree\n")
    crlf = write_bytes(tmp_path, "crlf.txt", b"one\r\ntwo\r\nthree\r\n")

    assert sha256_normalized(lf) == sha256_normalized(crlf)


@pytest.mark.parametrize(
    "data",
    [b"", b"plain", b"a\r\nb\r\n", b"a\nb\r\nc\n", b"x\r\n\r\ny", "café\r\n✓\r\n".encode()],
    ids=["empty", "plain", "crlf", "mixed", "blank-line", "utf8"],
)
def test_result_equals_an_independent_computation(tmp_path, data):
    path = write_bytes(tmp_path, "f.bin", data)

    assert sha256_normalized(path) == independent_hash(data)


CHUNK_SPLIT_DATA = b"line1\r\nline2\r\n\r\nend\r\r\n\rx\r\n"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, None], ids=["1", "2", "3", "5", "default"])
def test_crlf_pair_split_across_chunks_is_handled(tmp_path, chunk_size):
    path = write_bytes(tmp_path, "f.bin", CHUNK_SPLIT_DATA)
    whole_file = independent_hash(CHUNK_SPLIT_DATA)

    if chunk_size is None:
        assert sha256_normalized(path) == whole_file
    else:
        assert sha256_normalized(path, chunk_size=chunk_size) == whole_file


def test_lone_cr_is_preserved(tmp_path):
    cr = write_bytes(tmp_path, "cr.txt", b"a\rb")
    lf = write_bytes(tmp_path, "lf.txt", b"a\nb")

    assert sha256_normalized(cr) != sha256_normalized(lf)
    assert sha256_normalized(cr) == hashlib.sha256(b"a\rb").hexdigest()


def test_cr_cr_lf_behaves_like_cr_then_lf(tmp_path):
    path = write_bytes(tmp_path, "f.bin", b"\r\r\n")

    assert sha256_normalized(path) == hashlib.sha256(b"\r\n").hexdigest()
    assert sha256_normalized(path, chunk_size=1) == hashlib.sha256(b"\r\n").hexdigest()


@pytest.mark.parametrize("chunk_size", [1, 2, None], ids=["1", "2", "default"])
def test_file_ending_in_cr_keeps_it(tmp_path, chunk_size):
    path = write_bytes(tmp_path, "f.bin", b"abc\r")
    expected = hashlib.sha256(b"abc\r").hexdigest()

    if chunk_size is None:
        assert sha256_normalized(path) == expected
    else:
        assert sha256_normalized(path, chunk_size=chunk_size) == expected


def test_empty_file_works(tmp_path):
    path = write_bytes(tmp_path, "empty", b"")

    assert sha256_normalized(path) == hashlib.sha256(b"").hexdigest()


def test_file_on_disk_is_unchanged(tmp_path):
    data = b"a\r\nb\r\nc\r"
    path = write_bytes(tmp_path, "f.bin", data)

    sha256_normalized(path, chunk_size=2)

    assert path.read_bytes() == data


def test_digest_is_64_lowercase_hex(tmp_path):
    path = write_bytes(tmp_path, "f.bin", b"abc")

    assert HEX64.fullmatch(sha256_normalized(path))


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_below_one_raises(tmp_path, chunk_size):
    path = write_bytes(tmp_path, "f.bin", b"abc")

    with pytest.raises(ValueError):
        sha256_normalized(path, chunk_size=chunk_size)


# --- config_hash ------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {"raw_data": tmp_path / "raw", "runs_dir": tmp_path / "runs"},
        "reproducibility": {"seed": 42},
        "data": {"test_size": 0.2, "cv_folds": 5, "cv_repeats": 1},
        "business": {"k": 0.10, "k_sensitivity": [0.05, 0.10]},
        "tuning": {"budget_per_model": 30, "objective_metric": "pr_auc"},
        "inference": {"input_schema": None, "output_path": None},
    }


def canonical_hash(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_config_hash_equals_the_canonical_hash_of_the_snapshot(config):
    assert config_hash(config) == canonical_hash(config_snapshot(config))


def test_config_hash_ignores_paths(config, tmp_path):
    before = config_hash(config)
    config["paths"]["raw_data"] = tmp_path / "somewhere_else"
    config["paths"]["extra"] = tmp_path

    assert config_hash(config) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["reproducibility"].update(seed=43),
        lambda c: c["data"].update(cv_folds=3),
        lambda c: c["business"]["k_sensitivity"].append(0.15),
        lambda c: c["tuning"].update(objective_metric="other"),
        lambda c: c["inference"].update(output_path="x"),
    ],
    ids=["reproducibility", "data", "business", "tuning", "inference"],
)
def test_config_hash_changes_when_any_other_section_changes(config, change):
    before = config_hash(config)

    change(config)

    assert config_hash(config) != before


def test_config_hash_is_independent_of_key_order(config):
    reordered = {key: config[key] for key in reversed(list(config))}
    reordered["data"] = {key: config["data"][key] for key in reversed(list(config["data"]))}

    assert list(reordered) != list(config)
    assert config_hash(reordered) == config_hash(config)


def test_config_hash_is_64_lowercase_hex(config):
    assert HEX64.fullmatch(config_hash(config))


# --- sql_hash on a synthetic tree -------------------------------------------


def make_tree(root, files: dict[str, dict[str, bytes]]):
    """Create every SQL folder; files maps folder -> {file name: bytes}."""
    for folder in SQL_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for folder, scripts in files.items():
        for name, data in scripts.items():
            (root / folder / name).write_bytes(data)
    return root


BASE_FILES = {
    "01_staging": {"01_a.sql": b"SELECT 1;\n", "02_b.sql": b"SELECT 2;\n"},
    "02_cleaning": {"01_c.sql": b"SELECT 3;\n"},
    "05_features": {"01_d.sql": b"SELECT 4;\n"},
}


def test_gitkeep_and_other_files_are_ignored(tmp_path):
    plain = make_tree(tmp_path / "plain", BASE_FILES)
    extra = make_tree(tmp_path / "extra", BASE_FILES)
    (extra / "01_staging" / ".gitkeep").write_bytes(b"")
    (extra / "01_staging" / "notes.txt").write_bytes(b"not sql")
    (extra / "02_cleaning" / "README.md").write_bytes(b"not sql")

    assert sql_hash(extra) == sql_hash(plain)


def test_changing_a_files_content_changes_the_hash(tmp_path):
    before = sql_hash(make_tree(tmp_path / "a", BASE_FILES))
    changed = {**BASE_FILES, "02_cleaning": {"01_c.sql": b"SELECT 33;\n"}}

    assert sql_hash(make_tree(tmp_path / "b", changed)) != before


def test_renaming_a_file_changes_the_hash(tmp_path):
    before = sql_hash(make_tree(tmp_path / "a", BASE_FILES))
    renamed = {**BASE_FILES, "02_cleaning": {"01_renamed.sql": b"SELECT 3;\n"}}

    assert sql_hash(make_tree(tmp_path / "b", renamed)) != before


def test_swapping_the_folder_order_changes_the_hash(tmp_path):
    one = {"01_staging": {"01_a.sql": b"SELECT 1;\n"}, "02_cleaning": {"01_b.sql": b"SELECT 2;\n"}}
    swapped = {"01_staging": {"01_b.sql": b"SELECT 2;\n"}, "02_cleaning": {"01_a.sql": b"SELECT 1;\n"}}

    assert sql_hash(make_tree(tmp_path / "a", one)) != sql_hash(make_tree(tmp_path / "b", swapped))


def test_crlf_copy_of_the_tree_hashes_the_same(tmp_path):
    lf = make_tree(tmp_path / "lf", BASE_FILES)
    crlf_files = {
        folder: {name: data.replace(b"\n", b"\r\n") for name, data in scripts.items()}
        for folder, scripts in BASE_FILES.items()
    }
    crlf = make_tree(tmp_path / "crlf", crlf_files)

    assert sql_hash(crlf) == sql_hash(lf)


def test_script_contribution_is_name_nul_and_normalized_contents(tmp_path):
    files = {
        "01_staging": {"01_a.sql": b"A\r\n", "02_b.sql": b"B\n"},
        "03_target": {"01_t.sql": b"T"},
    }
    root = make_tree(tmp_path, files)

    expected = hashlib.sha256()
    expected.update(b"01_a.sql" + b"\x00" + b"A\n")
    expected.update(b"02_b.sql" + b"\x00" + b"B\n")
    expected.update(b"01_t.sql" + b"\x00" + b"T")

    assert sql_hash(root) == expected.hexdigest()


def test_a_missing_sql_folder_raises(tmp_path):
    root = make_tree(tmp_path, BASE_FILES)
    (root / "03_target").rmdir()

    with pytest.raises(FileNotFoundError, match="03_target"):
        sql_hash(root)


# --- sql_hash on the real repository ----------------------------------------


LOADER_DIRS = [
    STAGING_SQL_DIR,
    CLEANING_SQL_DIR,
    TARGET_SQL_DIR,
    MODELING_VIEW_SQL_DIR,
    FEATURES_SQL_DIR,
]


class _Result:
    def fetchone(self):
        return (0,)


class RecordingConnection:
    """Stands in for a DuckDB connection and records the SQL the loaders run."""

    def __init__(self):
        self.statements = []

    def execute(self, sql):
        if not sql.lstrip().upper().startswith("SELECT COUNT"):
            self.statements.append(sql)
        return _Result()


def test_folder_order_constant_equals_the_loaders_order():
    assert list(SQL_FOLDERS) == [directory.name for directory in LOADER_DIRS]
    assert {directory.parent for directory in LOADER_DIRS} == {REPO_ROOT / "sql"}


def test_loaders_execute_scripts_in_the_order_the_hash_reads_them():
    raw_dir = load_config()["paths"]["raw_data"]
    connection = RecordingConnection()

    # The order the data preparation entry point and the feature build run them.
    run_staging(connection, raw_dir=raw_dir)
    run_cleaning(connection)
    run_target(connection)
    run_modeling_view(connection)
    run_features(connection)

    expected = []
    for folder in SQL_FOLDERS:
        for script in sorted((REPO_ROOT / "sql" / folder).glob("*.sql")):
            text = script.read_text(encoding="utf-8")
            if folder == "01_staging":
                text = text.replace("{raw_dir}", raw_dir.resolve().as_posix())
            expected.append(text)

    assert connection.statements == expected


def test_real_sql_hash_equals_an_independent_computation():
    digest = hashlib.sha256()
    for directory in LOADER_DIRS:
        for script in sorted(directory.glob("*.sql")):
            digest.update(script.name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(script.read_bytes().replace(b"\r\n", b"\n"))

    assert sql_hash() == digest.hexdigest()
    assert HEX64.fullmatch(sql_hash())


# --- data_fingerprint and build_fingerprints --------------------------------


def test_data_fingerprint_on_a_synthetic_raw_folder(tmp_path):
    write_bytes(tmp_path, "b.csv", b"x,y\r\n1,2\r\n")
    write_bytes(tmp_path, "a.csv", b"p,q\n3,4\n")
    write_bytes(tmp_path, "notes.txt", b"ignored")
    (tmp_path / "folder.csv").mkdir()

    result = data_fingerprint(tmp_path, 10, 3)

    assert list(result["raw_files_sha256"]) == ["a.csv", "b.csv"]
    assert result["raw_files_sha256"]["a.csv"] == independent_hash(b"p,q\n3,4\n")
    assert result["raw_files_sha256"]["b.csv"] == independent_hash(b"x,y\n1,2\n")
    assert all(HEX64.fullmatch(value) for value in result["raw_files_sha256"].values())
    assert result["train_rows"] == 10
    assert result["test_rows"] == 3
    assert set(result) == {"raw_files_sha256", "train_rows", "test_rows"}


def test_data_fingerprint_rejects_a_missing_folder_and_no_csv_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_fingerprint(tmp_path / "missing", 1, 1)

    write_bytes(tmp_path, "notes.txt", b"ignored")
    with pytest.raises(FileNotFoundError):
        data_fingerprint(tmp_path, 1, 1)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "7", None])
def test_data_fingerprint_rejects_invalid_counts(tmp_path, bad):
    write_bytes(tmp_path, "a.csv", b"x\n")

    with pytest.raises(ValueError):
        data_fingerprint(tmp_path, bad, 1)
    with pytest.raises(ValueError):
        data_fingerprint(tmp_path, 1, bad)


def test_real_raw_folder_lists_the_files_the_ingest_stage_declares():
    raw_dir = load_config()["paths"]["raw_data"]

    result = data_fingerprint(raw_dir, 1, 1)

    assert set(result["raw_files_sha256"]) == {spec.filename for spec in ingest.RAW_FILE_SPECS}
    assert all(HEX64.fullmatch(value) for value in result["raw_files_sha256"].values())


def test_build_fingerprints_has_exactly_the_four_keys():
    config = load_config()

    result = build_fingerprints(config, 5629, 1408)

    assert set(result) == {"config_hash", "sql_hash", "data_fingerprint", "hash_normalization"}
    assert result["hash_normalization"] == "CRLF to LF before hashing"
    assert result["hash_normalization"] == HASH_NORMALIZATION
    assert result["config_hash"] == config_hash(config)
    assert result["sql_hash"] == sql_hash()
    assert result["data_fingerprint"] == data_fingerprint(config["paths"]["raw_data"], 5629, 1408)
    assert result["data_fingerprint"]["train_rows"] == 5629
    assert result["data_fingerprint"]["test_rows"] == 1408


def test_build_fingerprints_survives_a_json_round_trip():
    result = build_fingerprints(load_config(), 5629, 1408)

    assert json.loads(json.dumps(result)) == result


# --- Run id and package metadata --------------------------------------------

RUN_ID = "20261011T120000Z"
INSTANT = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def test_a_valid_run_id_is_accepted():
    assert is_valid_run_id(RUN_ID)


@pytest.mark.parametrize(
    "text",
    [
        "20261011T120000z",
        "20261011T120000",
        "202610110T120000Z",
        "2026101T120000Z",
        " 20261011T120000Z",
        "20261011T120000Z ",
        "20261011T120000Z\n",
        "../20261011T120000Z",
        "",
        "20261340T250000Z",
        "20261011t120000Z",
        None,
        20261011,
    ],
    ids=[
        "lowercase-z",
        "missing-Z",
        "extra-digit",
        "short-date",
        "leading-space",
        "trailing-space",
        "trailing-newline",
        "path-traversal",
        "empty",
        "impossible-date-time",
        "lowercase-t",
        "none",
        "integer",
    ],
)
def test_invalid_run_ids_are_rejected(text):
    assert not is_valid_run_id(text)


def test_package_version_comes_from_the_installed_metadata():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    name = pyproject["project"]["name"]

    assert package_version() == importlib.metadata.version(name)
    assert package_version()


def test_library_versions_has_the_five_libraries():
    versions = library_versions()

    assert set(versions) == {"scikit-learn", "xgboost", "numpy", "pandas", "joblib"}
    for name, version in versions.items():
        assert isinstance(version, str) and version
        assert version == importlib.metadata.version(name)


# --- Manifest ---------------------------------------------------------------

SMALL_SCHEMA = {
    "identifier_column": "customer_id",
    "columns": [
        {"name": "age", "kind": "integer-like", "nullable": False},
        {"name": "plan", "kind": "categorical", "nullable": False, "allowed_values": ["A", "B"]},
    ],
}

METRICS = {
    "k": 0.1,
    "training_rows": 5629,
    "estimate_type": TRAINING_CV_ESTIMATE_TYPE,
    "precision_at_k": 0.9,
    "lift_at_k": 3.4,
    "pr_auc": 0.8,
    "brier_score": 0.1,
}

FINGERPRINTS = {"config_hash": "a" * 64, "hash_normalization": HASH_NORMALIZATION}

MANIFEST_KEYS = {
    "run_id",
    "input_schema",
    "input_schema_hash",
    "hyperparameters",
    "library_versions",
    "trained_at_utc",
    "package_version",
    "training_cv_metrics",
    "held_out_results_reference",
    "fingerprints",
}


def make_manifest(**overrides):
    arguments = {
        "run_id": RUN_ID,
        "instant": INSTANT,
        "schema": SMALL_SCHEMA,
        "hyperparameters": {"n_estimators": 215, "learning_rate": 0.02},
        "training_cv_metrics": dict(METRICS),
        "fingerprints": FINGERPRINTS,
    }
    arguments.update(overrides)
    return build_manifest(**arguments)


def test_estimate_type_wording_is_the_one_in_the_contract():
    contract = (REPO_ROOT / "docs" / "platform_output_contract.md").read_text(encoding="utf-8")

    assert f"`{TRAINING_CV_ESTIMATE_TYPE}`" in contract


def test_manifest_has_exactly_the_ten_keys_and_the_given_values():
    manifest = make_manifest()

    assert set(manifest) == MANIFEST_KEYS
    assert manifest["run_id"] == RUN_ID
    assert manifest["input_schema_hash"] == schema_hash(SMALL_SCHEMA)
    assert manifest["input_schema"] == SMALL_SCHEMA
    assert manifest["training_cv_metrics"] == METRICS
    assert manifest["fingerprints"] == FINGERPRINTS
    assert manifest["package_version"] == package_version()
    assert manifest["library_versions"] == library_versions()
    assert manifest["held_out_results_reference"] == "ADR-0014" == HELD_OUT_RESULTS_REFERENCE


def test_trained_at_and_run_id_come_from_the_same_instant():
    plus_two = timezone(timedelta(hours=2))
    instant = datetime(2026, 10, 11, 14, 0, 0, 999999, tzinfo=plus_two)

    manifest = make_manifest(instant=instant)

    assert manifest["trained_at_utc"] == "2026-10-11T12:00:00Z"
    parsed_run_id = datetime.strptime(manifest["run_id"], RUN_ID_FORMAT)
    parsed_trained_at = datetime.strptime(manifest["trained_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    assert parsed_run_id == parsed_trained_at


def test_manifest_rejects_a_naive_instant():
    with pytest.raises(ValueError):
        make_manifest(instant=datetime(2026, 10, 11, 12, 0, 0))


@pytest.mark.parametrize("run_id", ["20261011T120001Z", "../x", "", "20261011T120000z"])
def test_manifest_rejects_a_run_id_that_is_invalid_or_does_not_match_the_instant(run_id):
    with pytest.raises(ValueError):
        make_manifest(run_id=run_id)


def test_manifest_rejects_metrics_with_a_missing_key():
    metrics = dict(METRICS)
    del metrics["pr_auc"]

    with pytest.raises(ValueError, match="pr_auc"):
        make_manifest(training_cv_metrics=metrics)


def test_manifest_rejects_metrics_with_an_extra_key():
    with pytest.raises(ValueError, match="roc_auc"):
        make_manifest(training_cv_metrics={**METRICS, "roc_auc": 0.9})


def test_manifest_rejects_a_wrong_estimate_type():
    with pytest.raises(ValueError, match="estimate_type"):
        make_manifest(training_cv_metrics={**METRICS, "estimate_type": "Some other text"})


def test_manifest_has_a_held_out_reference_but_no_held_out_results():
    manifest = make_manifest()

    assert [key for key in manifest if "held" in key] == ["held_out_results_reference"]
    assert manifest["held_out_results_reference"] == "ADR-0014"


def test_numpy_scalars_in_hyperparameters_become_plain_values():
    hyperparameters = {
        "n_estimators": np.int64(215),
        "learning_rate": np.float64(0.0217),
        "scale_pos_weight": np.int32(1),
        "nested": {"values": [np.int64(1), np.float32(0.5)]},
    }

    manifest = make_manifest(hyperparameters=hyperparameters)

    assert type(manifest["hyperparameters"]["n_estimators"]) is int
    assert type(manifest["hyperparameters"]["learning_rate"]) is float
    assert type(manifest["hyperparameters"]["scale_pos_weight"]) is int
    assert type(manifest["hyperparameters"]["nested"]["values"][0]) is int
    assert json.loads(json.dumps(manifest)) == manifest


def test_unserialisable_hyperparameters_raise_value_error():
    with pytest.raises(ValueError):
        make_manifest(hyperparameters={"estimator": object()})


def test_nan_in_the_manifest_raises_value_error():
    with pytest.raises(ValueError):
        make_manifest(training_cv_metrics={**METRICS, "pr_auc": float("nan")})


def test_manifest_does_not_share_state_with_its_inputs():
    schema = json.loads(json.dumps(SMALL_SCHEMA))

    manifest = make_manifest(schema=schema)
    manifest["input_schema"]["columns"].append({"name": "extra"})

    assert schema == SMALL_SCHEMA


# --- Existence check --------------------------------------------------------


def test_free_path_passes_and_creates_nothing(tmp_path):
    models_dir = tmp_path / "models"

    ensure_run_path_free(models_dir, RUN_ID)

    assert not models_dir.exists()


def test_existing_free_path_inside_an_existing_models_dir_passes(tmp_path):
    (tmp_path / "20260101T000000Z").mkdir()

    ensure_run_path_free(tmp_path, RUN_ID)

    assert [path.name for path in tmp_path.iterdir()] == ["20260101T000000Z"]


def _empty_folder(path):
    path.mkdir()


def _half_filled_folder(path):
    path.mkdir()
    (path / BUNDLE_FILE).write_bytes(b"partial")


def _complete_run(path):
    path.mkdir()
    (path / BUNDLE_FILE).write_bytes(b"bundle")
    (path / MANIFEST_FILE).write_text("{}", encoding="utf-8")


def _plain_file(path):
    path.write_bytes(b"a file")


@pytest.mark.parametrize(
    "create",
    [_empty_folder, _half_filled_folder, _complete_run, _plain_file],
    ids=["empty-folder", "half-filled-folder", "complete-run", "plain-file"],
)
def test_existing_path_in_any_form_is_refused_and_left_unchanged(tmp_path, create):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / RUN_ID
    create(target)
    before = sorted(str(p.relative_to(models_dir)) for p in models_dir.rglob("*"))

    with pytest.raises(PublicationError):
        ensure_run_path_free(models_dir, RUN_ID)

    assert sorted(str(p.relative_to(models_dir)) for p in models_dir.rglob("*")) == before


def test_existence_check_refuses_an_invalid_run_id(tmp_path):
    with pytest.raises(ValueError):
        ensure_run_path_free(tmp_path, "../elsewhere")


# --- Publication ------------------------------------------------------------


@pytest.fixture
def parts():
    return StandardScaler(), LogisticRegression()


def publish(models_dir, parts, **overrides):
    manifest = overrides.pop("manifest", None) or make_manifest()
    arguments = {
        "models_dir": models_dir,
        "run_id": RUN_ID,
        "manifest": manifest,
        "preprocessor": parts[0],
        "model": parts[1],
        "schema_hash": manifest["input_schema_hash"],
    }
    arguments.update(overrides)
    return publish_run(**arguments)


def names_in(folder):
    return sorted(path.name for path in folder.iterdir())


def test_publishing_creates_exactly_the_two_files(tmp_path, parts):
    models_dir = tmp_path / "models"

    final = publish(models_dir, parts)

    assert final == models_dir / RUN_ID
    assert names_in(final) == [BUNDLE_FILE, MANIFEST_FILE]
    assert names_in(models_dir) == [RUN_ID]

    manifest = json.loads((final / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["run_id"] == final.name
    bundle = joblib.load(final / BUNDLE_FILE)
    assert isinstance(bundle, dict)
    assert set(bundle) == {"preprocessor", "model", "schema_hash"}
    assert bundle["schema_hash"] == manifest["input_schema_hash"]
    assert isinstance(bundle["preprocessor"], StandardScaler)
    assert isinstance(bundle["model"], LogisticRegression)


def test_manifest_file_is_indented_utf8_json_with_a_trailing_newline(tmp_path, parts):
    manifest = make_manifest(hyperparameters={"label": "café ✓"})

    final = publish(tmp_path / "models", parts, manifest=manifest)

    raw = (final / MANIFEST_FILE).read_bytes()
    text = raw.decode("utf-8")
    assert b"\r" not in raw
    assert text.endswith("\n")
    assert "café ✓" in text
    assert "\n  " in text
    assert json.loads(text) == manifest


def test_models_dir_is_created_when_missing(tmp_path, parts):
    models_dir = tmp_path / "deep" / "models"

    publish(models_dir, parts)

    assert (models_dir / RUN_ID).is_dir()


def test_a_non_empty_folder_at_the_final_path_is_refused(tmp_path, parts):
    models_dir = tmp_path / "models"
    existing = models_dir / RUN_ID
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(PublicationError, match=r"collided.*nothing was replaced"):
        publish(models_dir, parts)

    assert names_in(existing) == ["keep.txt"]
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert names_in(models_dir) == [RUN_ID]


def test_a_race_after_the_existence_check_is_still_refused(tmp_path, parts):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ensure_run_path_free(models_dir, RUN_ID)

    existing = models_dir / RUN_ID
    existing.mkdir()
    (existing / "appeared.txt").write_text("late", encoding="utf-8")

    with pytest.raises(PublicationError, match=r"collided.*nothing was replaced"):
        publish(models_dir, parts)

    assert names_in(existing) == ["appeared.txt"]
    assert names_in(models_dir) == [RUN_ID]


def test_a_failed_write_leaves_no_temporary_directory(tmp_path, parts, monkeypatch):
    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(persist.joblib, "dump", broken_dump)
    models_dir = tmp_path / "models"

    with pytest.raises(OSError, match="disk full"):
        publish(models_dir, parts)

    assert names_in(models_dir) == []


def test_a_failed_publication_reraises_the_original_error_and_cleans_up(
    tmp_path, parts, monkeypatch
):
    failure = OSError("rename failed for another reason")

    def broken_rename(source, target):
        raise failure

    monkeypatch.setattr(persist.os, "rename", broken_rename)
    models_dir = tmp_path / "models"

    with pytest.raises(OSError) as exc_info:
        publish(models_dir, parts)

    assert exc_info.value is failure
    assert names_in(models_dir) == []


def test_a_failing_cleanup_never_hides_the_original_error(tmp_path, parts, monkeypatch):
    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    def broken_rmtree(*args, **kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(persist.joblib, "dump", broken_dump)
    monkeypatch.setattr(persist.shutil, "rmtree", broken_rmtree)
    models_dir = tmp_path / "models"

    with pytest.raises(OSError, match="disk full"):
        publish(models_dir, parts)

    assert all(name.startswith(TEMP_DIR_PREFIX) for name in names_in(models_dir))


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": "20261011T120001Z"},
        {"schema_hash": "0" * 64},
        {"run_id": "../x"},
        {"run_id": ""},
    ],
    ids=["manifest-run-id-differs", "schema-hash-differs", "invalid-run-id", "empty-run-id"],
)
def test_mismatches_raise_value_error_and_write_nothing(tmp_path, parts, overrides):
    models_dir = tmp_path / "models"

    with pytest.raises(ValueError):
        publish(models_dir, parts, **overrides)

    assert not models_dir.exists()
    assert list(tmp_path.iterdir()) == []


# --- Run selection and loading ----------------------------------------------

LOGGER_NAME = "retention_platform.models.persist"

SMALL_COLUMN_TYPES = {
    "customer_id": "VARCHAR",
    "is_voluntary_churn": "BOOLEAN",
    "gender": "VARCHAR",
    "age": "INTEGER",
    "offer": "VARCHAR",
    "internet_type": "VARCHAR",
    "contract": "VARCHAR",
    "payment_method": "VARCHAR",
    "tenure_in_months": "INTEGER",
    "married": "BOOLEAN",
}


@pytest.fixture(scope="module")
def artifacts():
    """A small fitted preprocessor and model, and the schema built from them."""
    frame = pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 6,
            "age": [20 + i for i in range(12)],
            "offer": ["Offer A", None, "Offer B", None] * 3,
            "internet_type": ["DSL", "Cable", None, "Fiber Optic"] * 3,
            "contract": ["Month-to-Month", "One Year", "Two Year"] * 4,
            "payment_method": ["Credit Card", "Mailed Check", "Bank Withdrawal"] * 4,
            "tenure_in_months": [1 + 3 * i for i in range(12)],
            "married": [True, False] * 6,
        }
    )
    preprocessor = build_preprocessing_pipeline().fit(frame)
    matrix = preprocessor.transform(frame)
    model = LogisticRegression().fit(matrix, [0, 1] * 6)
    return SimpleNamespace(
        frame=frame,
        preprocessor=preprocessor,
        model=model,
        matrix=matrix,
        schema=build_input_schema(SMALL_COLUMN_TYPES, preprocessor),
    )


def at(day: int) -> datetime:
    return datetime(2026, 10, day, 12, 0, 0, tzinfo=timezone.utc)


def publish_real_run(models_dir, artifacts, instant) -> str:
    run_id = format_run_id(instant)
    manifest = build_manifest(
        run_id=run_id,
        instant=instant,
        schema=artifacts.schema,
        hyperparameters={},
        training_cv_metrics=dict(METRICS),
        fingerprints=FINGERPRINTS,
    )
    publish_run(
        models_dir,
        run_id,
        manifest,
        artifacts.preprocessor,
        artifacts.model,
        manifest["input_schema_hash"],
    )
    return run_id


@pytest.fixture
def published(tmp_path, artifacts):
    models_dir = tmp_path / "models"
    run_id = publish_real_run(models_dir, artifacts, at(11))
    return SimpleNamespace(models_dir=models_dir, run_id=run_id, run_dir=models_dir / run_id)


def read_manifest(run_dir):
    return json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))


def write_manifest(run_dir, manifest):
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def edit_manifest(run_dir, mutate):
    manifest = read_manifest(run_dir)
    mutate(manifest)
    write_manifest(run_dir, manifest)


def edit_bundle(run_dir, mutate):
    bundle = joblib.load(run_dir / BUNDLE_FILE)
    mutate(bundle)
    joblib.dump(bundle, run_dir / BUNDLE_FILE)


def edit_schema_consistently(run_dir, mutate_schema):
    """Edit the manifest's schema and update both stored hashes to match, so
    only the checks after the hash checks can catch the edit."""
    manifest = read_manifest(run_dir)
    mutate_schema(manifest["input_schema"])
    new_hash = schema_hash(manifest["input_schema"])
    manifest["input_schema_hash"] = new_hash
    write_manifest(run_dir, manifest)
    edit_bundle(run_dir, lambda bundle: bundle.update(schema_hash=new_hash))


def tree_state(root):
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for path in sorted(root.rglob("*"))
    }


def warnings_from_persist(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]


# Selection


def test_highest_name_is_selected_even_when_modification_times_are_reversed(tmp_path, artifacts):
    models_dir = tmp_path / "models"
    ids = [publish_real_run(models_dir, artifacts, at(day)) for day in (1, 2, 3)]
    for offset, run_id in enumerate(ids):
        timestamp = 2_000_000_000 - offset * 1000
        os.utime(models_dir / run_id, (timestamp, timestamp))

    assert (models_dir / ids[-1]).stat().st_mtime < (models_dir / ids[0]).stat().st_mtime
    assert select_run_id(models_dir) == ids[-1]
    assert find_valid_run_ids(models_dir) == sorted(ids)


def _only_bundle(path):
    path.mkdir()
    (path / BUNDLE_FILE).write_bytes(b"x")


def _only_manifest(path):
    path.mkdir()
    (path / MANIFEST_FILE).write_text(json.dumps({"run_id": path.name}), encoding="utf-8")


def _manifest_with_other_run_id(path):
    path.mkdir()
    (path / BUNDLE_FILE).write_bytes(b"x")
    (path / MANIFEST_FILE).write_text(json.dumps({"run_id": "20261231T000000Z"}), encoding="utf-8")


def _unreadable_manifest(path):
    path.mkdir()
    (path / BUNDLE_FILE).write_bytes(b"x")
    (path / MANIFEST_FILE).write_text("{not json", encoding="utf-8")


SKIPPED_FOLDERS = {
    ".tmp-abc123": (_empty_folder, "temporary"),
    "20271001T000000Z": (_empty_folder, "bundle.joblib is missing"),
    "20271002T000000Z": (_only_bundle, "manifest.json is missing"),
    "20271003T000000Z": (_only_manifest, "bundle.joblib is missing"),
    "20271004T000000Z": (_manifest_with_other_run_id, "differs from the folder name"),
    "20271005T000000Z": (_unreadable_manifest, "cannot be read"),
    "not-a-run": (_empty_folder, "not a run id"),
}


def test_invalid_folders_are_skipped_with_one_warning_each(tmp_path, artifacts, caplog):
    models_dir = tmp_path / "models"
    valid_id = publish_real_run(models_dir, artifacts, at(11))
    for name, (create, _) in SKIPPED_FOLDERS.items():
        create(models_dir / name)
    (models_dir / ".gitkeep").write_bytes(b"")
    (models_dir / "notes.txt").write_text("plain file", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        found = find_valid_run_ids(models_dir)

    assert found == [valid_id]
    warned = warnings_from_persist(caplog)
    assert len(warned) == len(SKIPPED_FOLDERS)
    for name, (_, reason) in SKIPPED_FOLDERS.items():
        matching = [record.getMessage() for record in warned if name in record.getMessage()]
        assert len(matching) == 1, name
        assert reason in matching[0], matching[0]
    assert not any(".gitkeep" in record.getMessage() for record in warned)
    assert not any("notes.txt" in record.getMessage() for record in warned)


def test_a_valid_run_is_still_selected_when_higher_folders_are_invalid(tmp_path, artifacts):
    models_dir = tmp_path / "models"
    valid_id = publish_real_run(models_dir, artifacts, at(11))
    for name, (create, _) in SKIPPED_FOLDERS.items():
        create(models_dir / name)

    assert select_run_id(models_dir) == valid_id


def test_a_missing_models_dir_has_no_valid_runs(tmp_path):
    assert find_valid_run_ids(tmp_path / "missing") == []


@pytest.mark.parametrize(
    "requested",
    ["../20261011T120000Z", "", "20261011T120000z", "20261011T120000Z\n"],
    ids=["path-traversal", "empty", "lowercase-z", "trailing-newline"],
)
def test_a_badly_formatted_requested_run_id_is_refused(published, requested):
    with pytest.raises(ModelLoadError, match="run id format"):
        select_run_id(published.models_dir, requested)


def test_the_format_is_checked_before_any_path_is_built(tmp_path):
    models_dir = tmp_path / "does_not_exist"

    with pytest.raises(ModelLoadError, match="run id format"):
        select_run_id(models_dir, "../x")

    assert not models_dir.exists()


def test_a_requested_run_that_does_not_exist_is_refused(published):
    with pytest.raises(ModelLoadError, match="not found"):
        select_run_id(published.models_dir, "20260101T000000Z")


@pytest.mark.parametrize(
    "create",
    [_empty_folder, _only_bundle, _manifest_with_other_run_id],
    ids=["empty", "one-file", "mismatched-manifest"],
)
def test_an_invalid_requested_run_is_refused_without_fallback(published, create):
    invalid_id = "20271001T000000Z"
    create(published.models_dir / invalid_id)

    with pytest.raises(ModelLoadError, match=f"{invalid_id} is not valid"):
        select_run_id(published.models_dir, invalid_id)
    with pytest.raises(ModelLoadError, match=invalid_id):
        load_run(published.models_dir, invalid_id)


def test_a_requested_plain_file_is_refused(published):
    (published.models_dir / "20271001T000000Z").write_bytes(b"a file")

    with pytest.raises(ModelLoadError, match="not a folder"):
        select_run_id(published.models_dir, "20271001T000000Z")


def test_a_valid_requested_run_is_selected(tmp_path, artifacts):
    models_dir = tmp_path / "models"
    older = publish_real_run(models_dir, artifacts, at(1))
    publish_real_run(models_dir, artifacts, at(2))

    assert select_run_id(models_dir, older) == older


@pytest.mark.parametrize("prepare", ["missing", "empty", "only-invalid"])
def test_no_valid_run_tells_the_person_to_run_make_train(tmp_path, prepare):
    models_dir = tmp_path / "models"
    if prepare != "missing":
        models_dir.mkdir()
    if prepare == "only-invalid":
        _only_bundle(models_dir / "20271002T000000Z")
        _empty_folder(models_dir / ".tmp-abc")

    with pytest.raises(ModelLoadError, match="make train"):
        select_run_id(models_dir)
    with pytest.raises(ModelLoadError, match="make train"):
        load_run(models_dir)


def test_the_selected_run_id_is_logged_at_info_level(published, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        select_run_id(published.models_dir)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.INFO
    ]
    assert any(published.run_id in message for message in messages)


# Loading


def _to_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_round_trip_returns_the_published_run(published, artifacts):
    loaded = load_run(published.models_dir)

    assert isinstance(loaded, LoadedRun)
    assert loaded.run_id == published.run_id
    assert loaded.run_dir == published.run_dir
    assert loaded.manifest == read_manifest(published.run_dir)
    assert loaded.schema == artifacts.schema
    assert loaded.schema == loaded.manifest["input_schema"]
    assert loaded.schema_hash == loaded.manifest["input_schema_hash"] == schema_hash(artifacts.schema)
    assert np.allclose(
        loaded.model.predict_proba(artifacts.matrix), artifacts.model.predict_proba(artifacts.matrix)
    )
    assert np.allclose(
        _to_array(loaded.preprocessor.transform(artifacts.frame)), _to_array(artifacts.matrix)
    )


def test_loaded_run_is_frozen(published):
    loaded = load_run(published.models_dir)

    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.run_id = "other"


def test_load_run_uses_the_highest_run_or_the_requested_one(tmp_path, artifacts):
    models_dir = tmp_path / "models"
    older = publish_real_run(models_dir, artifacts, at(1))
    newer = publish_real_run(models_dir, artifacts, at(2))

    assert load_run(models_dir).run_id == newer
    assert load_run(models_dir, older).run_id == older


def _break_manifest_unreadable(run_dir):
    (run_dir / MANIFEST_FILE).write_text("{not json", encoding="utf-8")


LOAD_BREAKS = {
    "bundle-schema-hash-differs": (
        lambda d: edit_bundle(d, lambda b: b.update(schema_hash="0" * 64)),
        "differs from the manifest",
    ),
    "schema-edited-after-publication": (
        lambda d: edit_manifest(
            d, lambda m: m["input_schema"]["columns"][0]["allowed_values"].append("Other")
        ),
        "does not match its schema hash",
    ),
    "unknown-kind": (
        lambda d: edit_schema_consistently(d, lambda s: s["columns"][1].update(kind="weird")),
        "expected one of",
    ),
    "schema-disagrees-with-preprocessor": (
        lambda d: edit_schema_consistently(
            d, lambda s: s["columns"][0]["allowed_values"].append("Other")
        ),
        "does not match the preprocessor",
    ),
    "bundle-not-a-dict": (
        lambda d: joblib.dump(["not", "a", "dict"], d / BUNDLE_FILE),
        "must be a dict",
    ),
    "bundle-missing-a-key": (
        lambda d: edit_bundle(d, lambda b: b.pop("model")),
        "exactly preprocessor, model and schema_hash",
    ),
    "bundle-with-an-extra-key": (
        lambda d: edit_bundle(d, lambda b: b.update(extra=1)),
        "exactly preprocessor, model and schema_hash",
    ),
    "bundle-file-corrupt": (
        lambda d: (d / BUNDLE_FILE).write_bytes(b"this is not a joblib file"),
        "cannot be read",
    ),
    "manifest-lacks-schema-hash": (
        lambda d: edit_manifest(d, lambda m: m.pop("input_schema_hash")),
        "no input_schema_hash",
    ),
    "manifest-lacks-schema": (
        lambda d: edit_manifest(d, lambda m: m.pop("input_schema")),
        "no input_schema",
    ),
    "schema-without-a-column-list": (
        lambda d: edit_manifest(d, lambda m: m.update(input_schema={"columns": "none"})),
        "no column list",
    ),
}


@pytest.mark.parametrize(
    "damage,fragment",
    list(LOAD_BREAKS.values()),
    ids=list(LOAD_BREAKS),
)
def test_a_damaged_run_raises_model_load_error_naming_the_run(published, damage, fragment):
    damage(published.run_dir)

    with pytest.raises(ModelLoadError) as exc_info:
        load_run(published.models_dir)

    assert published.run_id in str(exc_info.value)
    assert fragment in str(exc_info.value)


def test_an_unreadable_manifest_is_refused_naming_the_run(published):
    _break_manifest_unreadable(published.run_dir)

    with pytest.raises(ModelLoadError, match=published.run_id):
        load_run(published.models_dir, published.run_id)


@pytest.mark.parametrize("damage", [value[0] for value in LOAD_BREAKS.values()], ids=list(LOAD_BREAKS))
def test_a_failed_load_changes_nothing_under_models_dir(published, damage):
    damage(published.run_dir)
    before = tree_state(published.models_dir)

    with pytest.raises(ModelLoadError):
        load_run(published.models_dir)

    assert tree_state(published.models_dir) == before


def test_changed_fingerprints_and_library_versions_still_load(published):
    def change_provenance(manifest):
        manifest["fingerprints"]["config_hash"] = "f" * 64
        manifest["fingerprints"]["sql_hash"] = "e" * 64
        manifest["library_versions"]["scikit-learn"] = "0.0.0"

    edit_manifest(published.run_dir, change_provenance)

    loaded = load_run(published.models_dir)

    assert loaded.manifest["fingerprints"]["config_hash"] == "f" * 64
    assert loaded.manifest["library_versions"]["scikit-learn"] == "0.0.0"


MANIFEST_ONLY_DAMAGE = {
    "input_schema_hash-missing": LOAD_BREAKS["manifest-lacks-schema-hash"][0],
    "input_schema-edited-after-publication": LOAD_BREAKS["schema-edited-after-publication"][0],
    "unknown-kind": LOAD_BREAKS["unknown-kind"][0],
}


@pytest.mark.parametrize(
    "damage", list(MANIFEST_ONLY_DAMAGE.values()), ids=list(MANIFEST_ONLY_DAMAGE)
)
def test_a_broken_manifest_is_refused_before_the_bundle_is_opened(
    published, damage, monkeypatch
):
    def bundle_must_not_be_opened(*args, **kwargs):
        pytest.fail("the bundle was opened although the manifest is broken")

    damage(published.run_dir)
    monkeypatch.setattr(persist.joblib, "load", bundle_must_not_be_opened)

    with pytest.raises(ModelLoadError, match=published.run_id):
        load_run(published.models_dir)
