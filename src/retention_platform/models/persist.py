"""Provenance fingerprints for a trained model: hashes of the configuration,
the SQL scripts and the raw data files.

Every file hash converts CRLF to LF in memory before hashing, so the same
content gives the same digest on every operating system. All digests are
lowercase 64-character SHA-256 hex strings with no prefix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from retention_platform.config import REPO_ROOT
from retention_platform.tracking import config_snapshot

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
