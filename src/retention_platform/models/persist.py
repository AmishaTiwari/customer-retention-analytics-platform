"""Persistence for a trained model: provenance fingerprints, the manifest,
and publication of a model run as a directory holding bundle.joblib and
manifest.json.

Every file hash converts CRLF to LF in memory before hashing, so the same
content gives the same digest on every operating system. All digests are
lowercase 64-character SHA-256 hex strings with no prefix.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from retention_platform.config import REPO_ROOT
from retention_platform.pipeline.schema import (
    SQL_TYPE_TO_KIND,
    check_schema_matches_preprocessor,
)
from retention_platform.pipeline.schema import schema_hash as compute_schema_hash
from retention_platform.tracking import RUN_ID_FORMAT, config_snapshot, format_run_id

logger = logging.getLogger(__name__)

HASH_NORMALIZATION = "CRLF to LF before hashing"

# The sql/ subfolders in the order their scripts run: sql/01 to sql/04 by
# the data preparation steps, then sql/05 by the feature build.
SQL_FOLDERS = (
    "01_staging",
    "02_cleaning",
    "03_target",
    "04_modeling_view",
    "05_features",
)

_DEFAULT_CHUNK_SIZE = 1 << 20


def _feed_normalized(digest, path: Path, chunk_size: int) -> None:
    """Stream a file into digest with CRLF converted to LF.

    A CR at the end of a chunk is held back until the next chunk shows
    whether an LF follows; a CR left at the end of the file is fed as is.
    A CR not followed by LF is kept.
    """
    pending_cr = False
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            if pending_cr:
                chunk = b"\r" + chunk
                pending_cr = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_cr = True
            digest.update(chunk.replace(b"\r\n", b"\n"))
    if pending_cr:
        digest.update(b"\r")


def sha256_normalized(path: str | Path, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of a file with CRLF converted to LF in
    memory. The file on disk is not changed."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    digest = hashlib.sha256()
    _feed_normalized(digest, Path(path), chunk_size)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    """SHA-256 of the canonical JSON of the configuration without its
    "paths" section (sorted keys, (",", ":") separators, UTF-8)."""
    canonical = json.dumps(config_snapshot(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sql_hash(sql_dir: str | Path | None = None) -> str:
    """SHA-256 over the .sql scripts, folder by folder in SQL_FOLDERS order,
    and in sorted file-name order within a folder. Each script contributes
    its file name (UTF-8), one NUL byte, then its normalized contents. Other
    files, such as .gitkeep, are ignored."""
    root = Path(sql_dir) if sql_dir is not None else REPO_ROOT / "sql"
    digest = hashlib.sha256()

    for folder in SQL_FOLDERS:
        folder_path = root / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"SQL folder not found: {folder_path}")
        for script in sorted(folder_path.glob("*.sql")):
            digest.update(script.name.encode("utf-8"))
            digest.update(b"\x00")
            _feed_normalized(digest, script, _DEFAULT_CHUNK_SIZE)

    return digest.hexdigest()


def _check_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def data_fingerprint(raw_dir: str | Path, train_rows: int, test_rows: int) -> dict:
    """Hashes of every .csv file in raw_dir, keyed by file name and sorted
    by name, plus the row counts. The counts are only passed through."""
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {raw_path}")

    files = sorted((p for p in raw_path.glob("*.csv") if p.is_file()), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .csv files found in {raw_path}")

    return {
        "raw_files_sha256": {path.name: sha256_normalized(path) for path in files},
        "train_rows": _check_count("train_rows", train_rows),
        "test_rows": _check_count("test_rows", test_rows),
    }


def build_fingerprints(config: dict, train_rows: int, test_rows: int) -> dict:
    """The fingerprints block of a model manifest, as plain JSON data."""
    return {
        "config_hash": config_hash(config),
        "sql_hash": sql_hash(),
        "data_fingerprint": data_fingerprint(
            config["paths"]["raw_data"], train_rows, test_rows
        ),
        "hash_normalization": HASH_NORMALIZATION,
    }


BUNDLE_FILE = "bundle.joblib"
MANIFEST_FILE = "manifest.json"
TEMP_DIR_PREFIX = ".tmp-"
HELD_OUT_RESULTS_REFERENCE = "ADR-0014"

# The exact wording of the training_cv_metrics estimate_type field.
TRAINING_CV_ESTIMATE_TYPE = (
    "Five-fold out-of-fold estimates on the training split, "
    "using the same folds as tuning; potentially mildly optimistic."
)

_TRAINING_CV_METRIC_KEYS = {
    "k",
    "training_rows",
    "estimate_type",
    "precision_at_k",
    "lift_at_k",
    "pr_auc",
    "brier_score",
}

# Distribution names of the libraries that matter for loading a bundle.
_LIBRARIES = ("scikit-learn", "xgboost", "numpy", "pandas", "joblib")

_TRAINED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class PublicationError(Exception):
    """Raised when a model run cannot be published without risking a replace."""


def is_valid_run_id(text: object) -> bool:
    """True only for text in exactly the run id shape that is also a real
    date and time. Rejects anything that could build a different path."""
    if not isinstance(text, str):
        return False
    try:
        parsed = datetime.strptime(text, RUN_ID_FORMAT)
    except ValueError:
        return False
    return parsed.strftime(RUN_ID_FORMAT) == text


def package_version() -> str:
    """Version of the installed distribution that provides this package."""
    top_level = __name__.split(".")[0]
    names = set(importlib.metadata.packages_distributions().get(top_level, []))
    if len(names) != 1:
        raise importlib.metadata.PackageNotFoundError(
            f"expected exactly one installed distribution providing {top_level!r}, "
            f"found {sorted(names)}"
        )
    return importlib.metadata.version(names.pop())


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries that matter for loading a bundle."""
    return {name: importlib.metadata.version(name) for name in _LIBRARIES}


def _to_plain(value):
    """Convert NumPy scalars to plain Python values, recursively. Dict keys
    must be text."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ValueError(f"JSON object keys must be text, got {key!r}")
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def build_manifest(
    *,
    run_id: str,
    instant: datetime,
    schema: dict,
    hyperparameters: dict,
    training_cv_metrics: dict,
    fingerprints: dict,
) -> dict:
    """Build the manifest of a model run as plain JSON data.

    run_id must be a valid run id and equal the run id formatted from
    instant, so the id and trained_at_utc come from one moment. The held-out
    test results are not part of the manifest.
    """
    expected_run_id = format_run_id(instant)
    if not is_valid_run_id(run_id):
        raise ValueError(f"run_id is not a valid run id: {run_id!r}")
    if run_id != expected_run_id:
        raise ValueError(f"run_id {run_id!r} does not match the instant ({expected_run_id!r})")

    metrics = _to_plain(training_cv_metrics)
    if not isinstance(metrics, dict) or set(metrics) != _TRAINING_CV_METRIC_KEYS:
        got = sorted(metrics) if isinstance(metrics, dict) else metrics
        raise ValueError(
            f"training_cv_metrics must have exactly the keys "
            f"{sorted(_TRAINING_CV_METRIC_KEYS)}, got {got}"
        )
    if metrics["estimate_type"] != TRAINING_CV_ESTIMATE_TYPE:
        raise ValueError("training_cv_metrics estimate_type has the wrong wording")

    try:
        manifest = {
            "run_id": run_id,
            "input_schema": copy.deepcopy(schema),
            "input_schema_hash": compute_schema_hash(schema),
            "hyperparameters": _to_plain(hyperparameters),
            "library_versions": library_versions(),
            "trained_at_utc": instant.astimezone(timezone.utc).strftime(_TRAINED_AT_FORMAT),
            "package_version": package_version(),
            "training_cv_metrics": metrics,
            "held_out_results_reference": HELD_OUT_RESULTS_REFERENCE,
            "fingerprints": copy.deepcopy(fingerprints),
        }
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the manifest cannot be written as JSON: {exc}") from exc
    return manifest


def ensure_run_path_free(models_dir: str | Path, run_id: str) -> None:
    """Raise PublicationError if models_dir/run_id exists in any form (an
    empty or half-filled folder, a complete run, a plain file, a broken
    link). Writes nothing and does not create models_dir."""
    if not is_valid_run_id(run_id):
        raise ValueError(f"run_id is not a valid run id: {run_id!r}")
    path = Path(models_dir) / run_id
    if os.path.lexists(path):
        raise PublicationError(f"The model path already exists, nothing was changed: {path}")


def publish_run(
    models_dir: str | Path,
    run_id: str,
    manifest: dict,
    preprocessor,
    model,
    schema_hash: str,
) -> Path:
    """Publish models_dir/run_id/ holding bundle.joblib and manifest.json.

    Both files are written into a temporary directory inside models_dir and
    published with one os.rename, so the run directory appears complete or
    not at all. If the final path exists, the rename fails and the error
    says the run id collided and that nothing was replaced. The temporary
    directory is removed after any failure.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"run_id is not a valid run id: {run_id!r}")
    if manifest.get("run_id") != run_id:
        raise ValueError(f"manifest run_id {manifest.get('run_id')!r} does not match {run_id!r}")
    if manifest.get("input_schema_hash") != schema_hash:
        raise ValueError("manifest input_schema_hash does not match the schema hash")
    try:
        manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the manifest cannot be written as JSON: {exc}") from exc

    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    final_path = models / run_id
    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX, dir=models))

    try:
        joblib.dump(
            {"preprocessor": preprocessor, "model": model, "schema_hash": schema_hash},
            temp_dir / BUNDLE_FILE,
        )
        with open(temp_dir / MANIFEST_FILE, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(manifest_text)
        try:
            os.rename(temp_dir, final_path)
        except OSError as exc:
            if os.path.lexists(final_path):
                raise PublicationError(
                    f"The run id collided: {final_path} already exists, nothing was replaced"
                ) from exc
            raise
    except BaseException:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        raise

    return final_path


class ModelLoadError(Exception):
    """Raised when no model run can be selected, or a selected run cannot be
    loaded or fails a load-time check. The message says what failed and
    names the run id when a run is involved."""


@dataclass(frozen=True)
class LoadedRun:
    """A loaded model run: its manifest and schema, and the fitted
    preprocessor and model from its bundle."""

    run_id: str
    run_dir: Path
    manifest: dict
    schema: dict
    schema_hash: str
    preprocessor: object
    model: object


def _run_problem(run_dir: Path, name: str) -> str | None:
    """Return why a directory is not a valid model run, or None if it is.
    Never opens bundle.joblib."""
    if name.startswith(TEMP_DIR_PREFIX):
        return "temporary folder left by an unfinished publication"
    if not is_valid_run_id(name):
        return "the folder name is not a run id"
    if not (run_dir / BUNDLE_FILE).is_file():
        return f"{BUNDLE_FILE} is missing"
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return f"{MANIFEST_FILE} is missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{MANIFEST_FILE} cannot be read: {exc}"
    if not isinstance(manifest, dict):
        return f"{MANIFEST_FILE} is not a JSON object"
    if manifest.get("run_id") != name:
        return f"manifest run_id {manifest.get('run_id')!r} differs from the folder name"
    return None


def find_valid_run_ids(models_dir: str | Path) -> list[str]:
    """Return the valid run ids in models_dir in ascending order.

    Only directories are looked at; plain files are ignored. Every skipped
    directory is reported with one warning naming it and the reason. A
    missing models_dir gives an empty list.
    """
    root = Path(models_dir)
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ModelLoadError(f"Cannot list the models folder {root}: {exc}") from exc

    valid = []
    for entry in entries:
        if not entry.is_dir():
            continue
        problem = _run_problem(entry, entry.name)
        if problem is None:
            valid.append(entry.name)
        else:
            logger.warning("Skipping model folder %s: %s", entry.name, problem)
    return valid


def select_run_id(models_dir: str | Path, run_id: str | None = None) -> str:
    """Return the run id to use.

    With run_id given, that exact run is used or ModelLoadError is raised,
    never another run; the id is checked against the run id format before
    any path is built. Without it, the highest valid run id is chosen by
    name, not by modification time.
    """
    if run_id is not None:
        if not is_valid_run_id(run_id):
            raise ModelLoadError(f"The requested run id is not in the run id format: {run_id!r}")
        run_dir = Path(models_dir) / run_id
        if not run_dir.exists():
            raise ModelLoadError(f"Requested model run {run_id} was not found in {models_dir}")
        if not run_dir.is_dir():
            raise ModelLoadError(f"Requested model run {run_id} is not a folder")
        problem = _run_problem(run_dir, run_id)
        if problem is not None:
            raise ModelLoadError(f"Requested model run {run_id} is not valid: {problem}")
        selected = run_id
    else:
        valid = find_valid_run_ids(models_dir)
        if not valid:
            raise ModelLoadError(
                f"No valid model run found in {models_dir}. Run `make train` first."
            )
        selected = valid[-1]

    logger.info("Selected model run %s", selected)
    return selected


def _load_error(run_id: str, text: str) -> ModelLoadError:
    return ModelLoadError(f"Model run {run_id}: {text}")


def load_run(models_dir: str | Path, run_id: str | None = None) -> LoadedRun:
    """Select a run and load its manifest and bundle.

    Loading a joblib file runs code from the file, so a bundle must only be
    loaded if this repository produced it on a trusted machine.

    Only these checks are enforced: the bundle holds exactly a
    preprocessor, a model and a schema hash; the bundle's schema hash equals
    the manifest's; the manifest's schema hashes to the stored value; every
    column kind is a known kind; and the schema matches the fitted
    preprocessor. Fingerprints and library versions are never checked. A
    manifest that fails its own checks is refused before the bundle is
    opened.
    """
    selected = select_run_id(models_dir, run_id)
    run_dir = Path(models_dir) / selected

    try:
        manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _load_error(selected, f"{MANIFEST_FILE} cannot be read: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _load_error(selected, f"{MANIFEST_FILE} is not a JSON object")

    for key in ("input_schema", "input_schema_hash"):
        if key not in manifest:
            raise _load_error(selected, f"{MANIFEST_FILE} has no {key}")
    schema = manifest["input_schema"]
    stored_hash = manifest["input_schema_hash"]

    if not isinstance(schema, dict) or not isinstance(schema.get("columns"), list):
        raise _load_error(selected, "the manifest's input schema has no column list")
    if compute_schema_hash(schema) != stored_hash:
        raise _load_error(selected, "the manifest's input schema does not match its schema hash")

    known_kinds = set(SQL_TYPE_TO_KIND.values())
    for column in schema["columns"]:
        if not isinstance(column, dict) or column.get("kind") not in known_kinds:
            name = column.get("name") if isinstance(column, dict) else column
            kind = column.get("kind") if isinstance(column, dict) else None
            raise _load_error(
                selected,
                f"schema column {name!r} has kind {kind!r}, expected one of {sorted(known_kinds)}",
            )

    try:
        bundle = joblib.load(run_dir / BUNDLE_FILE)
    except Exception as exc:
        raise _load_error(selected, f"{BUNDLE_FILE} cannot be read: {exc}") from exc

    if not isinstance(bundle, dict) or set(bundle) != {"preprocessor", "model", "schema_hash"}:
        raise _load_error(
            selected, "the bundle must be a dict with exactly preprocessor, model and schema_hash"
        )
    if bundle["schema_hash"] != stored_hash:
        raise _load_error(selected, "the bundle's schema hash differs from the manifest's")

    try:
        check_schema_matches_preprocessor(schema, bundle["preprocessor"])
    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        raise _load_error(selected, f"the schema does not match the preprocessor: {exc}") from exc

    return LoadedRun(
        run_id=selected,
        run_dir=run_dir,
        manifest=manifest,
        schema=schema,
        schema_hash=stored_hash,
        preprocessor=bundle["preprocessor"],
        model=bundle["model"],
    )
