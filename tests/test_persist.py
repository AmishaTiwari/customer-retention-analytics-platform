"""Tests for models/persist.py: the provenance fingerprints.

Synthetic files live under tmp_path; the real repository folders are only
read, never written.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

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
from retention_platform.models.persist import (
    HASH_NORMALIZATION,
    SQL_FOLDERS,
    build_fingerprints,
    config_hash,
    data_fingerprint,
    sha256_normalized,
    sql_hash,
)
from retention_platform.tracking import config_snapshot

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
