"""Tests for logging_setup's file handler and for tracking.py.

Everything is written under tmp_path; the real artifacts/ folder is never
touched. The root logger is saved and restored around each test, because
pytest installs its own handlers on it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from retention_platform import tracking
from retention_platform.config import load_config
from retention_platform.logging_setup import (
    attach_file_handler,
    detach_file_handler,
    setup_logging,
)
from retention_platform.tracking import (
    RUN_ID_FORMAT,
    TrackedRun,
    TrackingError,
    config_snapshot,
    format_run_id,
    track_run,
)

INSTANT = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
LOGGER = logging.getLogger("test_tracking")


@pytest.fixture
def root_logger():
    """The root logger at INFO level. Handlers are left alone, because
    pytest adds its own for each test phase, so tests compare the handler
    list before and after instead of assuming a fixed baseline. The level
    is restored afterwards, and any file handler a test leaked is removed
    and closed."""
    root = logging.getLogger()
    saved_level = root.level
    root.setLevel(logging.INFO)
    yield root
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {"runs_dir": tmp_path / "runs", "models_dir": tmp_path / "models"},
        "reproducibility": {"seed": 42},
        "data": {"test_size": 0.2, "cv_folds": 5, "cv_repeats": 1},
        "business": {"k": 0.10, "k_sensitivity": [0.05, 0.10]},
        "tuning": {"budget_per_model": 30, "objective_metric": "pr_auc"},
        "inference": {"input_schema": None, "output_path": None},
    }


def canonical_hash(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- File handler -----------------------------------------------------------


def test_attached_handler_writes_utf8_lines(root_logger, tmp_path):
    path = tmp_path / "run.log"
    handler = attach_file_handler(path)

    LOGGER.info("café → naïve ✓")
    detach_file_handler(handler)

    assert "café → naïve ✓" in path.read_text(encoding="utf-8")


def test_detached_handler_is_removed_closed_and_silent(root_logger, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    path = folder / "run.log"
    handler = attach_file_handler(path)
    LOGGER.info("while attached")

    detach_file_handler(handler)
    LOGGER.info("after detach")

    assert handler not in root_logger.handlers
    assert handler.stream is None
    content = path.read_text(encoding="utf-8")
    assert "while attached" in content
    assert "after detach" not in content
    shutil.rmtree(folder)
    assert not folder.exists()


def test_detach_twice_is_safe(root_logger, tmp_path):
    handler = attach_file_handler(tmp_path / "run.log")

    detach_file_handler(handler)
    detach_file_handler(handler)

    assert handler not in root_logger.handlers


def test_console_handler_is_unchanged_by_attach_and_detach(root_logger, tmp_path):
    # setup_logging does nothing while the root logger has handlers, so the
    # root logger is emptied for this test and put back in the finally block.
    saved = list(root_logger.handlers)
    for existing in saved:
        root_logger.removeHandler(existing)
    try:
        setup_logging()
        (console,) = root_logger.handlers
        formatter = console.formatter
        level = root_logger.level

        handler = attach_file_handler(tmp_path / "run.log")
        assert root_logger.handlers == [console, handler]
        assert console.formatter is formatter
        assert root_logger.level == level

        detach_file_handler(handler)
        assert root_logger.handlers == [console]
        assert console.formatter is formatter
        assert root_logger.level == level
    finally:
        for leftover in list(root_logger.handlers):
            root_logger.removeHandler(leftover)
        for existing in saved:
            root_logger.addHandler(existing)


def test_repeated_cycles_do_not_duplicate_lines_or_leak_handlers(root_logger, tmp_path):
    before = list(root_logger.handlers)

    for cycle in range(3):
        path = tmp_path / f"run{cycle}.log"
        handler = attach_file_handler(path)
        LOGGER.info("cycle %d", cycle)
        detach_file_handler(handler)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert f"cycle {cycle}" in lines[0]
        assert root_logger.handlers == before


# --- Run id and snapshot ----------------------------------------------------


def test_format_run_id_of_a_known_instant():
    assert format_run_id(INSTANT) == "20261011T120000Z"


def test_format_run_id_converts_an_aware_non_utc_instant():
    plus_two = timezone(timedelta(hours=2))

    assert format_run_id(datetime(2026, 10, 11, 14, 0, 0, tzinfo=plus_two)) == "20261011T120000Z"


def test_format_run_id_drops_sub_second_parts():
    assert format_run_id(INSTANT.replace(microsecond=999999)) == "20261011T120000Z"


def test_format_run_id_rejects_a_naive_instant():
    with pytest.raises(ValueError):
        format_run_id(datetime(2026, 10, 11, 12, 0, 0))


def test_format_run_id_parses_back_with_the_format():
    parsed = datetime.strptime(format_run_id(INSTANT), RUN_ID_FORMAT)

    assert parsed == INSTANT.replace(tzinfo=None)


def test_config_snapshot_drops_paths_and_keeps_the_rest(config):
    snapshot = config_snapshot(config)

    assert "paths" not in snapshot
    assert snapshot == {key: value for key, value in config.items() if key != "paths"}


def test_config_snapshot_does_not_change_or_share_its_input(config):
    original = {key: json.loads(json.dumps(value, default=str)) for key, value in config.items()}

    snapshot = config_snapshot(config)
    snapshot["business"]["k_sensitivity"].append(0.99)

    assert "paths" in config
    assert config["business"]["k_sensitivity"] == [0.05, 0.10]
    assert json.loads(json.dumps(config, default=str)) == original


def test_config_snapshot_of_the_real_config_is_json_serialisable():
    snapshot = config_snapshot(load_config())

    assert "paths" not in snapshot
    assert json.loads(json.dumps(snapshot)) == snapshot


# --- Starting a run ---------------------------------------------------------


def test_run_folder_and_files_exist_when_the_body_starts(root_logger, config):
    with track_run(config, "train", INSTANT) as run:
        expected = config["paths"]["runs_dir"] / "train" / "20261011T120000Z"
        assert run.folder == expected
        assert run.run_id == "20261011T120000Z"
        assert run.run_type == "train"
        assert (expected / "config.json").is_file()
        assert (expected / "run.log").is_file()
        assert not (expected / "metrics.json").exists()


def test_config_json_equals_the_snapshot_and_hashes_the_same(root_logger, config):
    with track_run(config, "score", INSTANT) as run:
        text = (run.folder / "config.json").read_text(encoding="utf-8")

    loaded = json.loads(text)
    assert loaded == config_snapshot(config)
    assert canonical_hash(loaded) == canonical_hash(config_snapshot(config))
    assert "\n  " in text


def test_tracked_run_is_frozen(root_logger, config):
    with track_run(config, "train", INSTANT) as run:
        assert isinstance(run, TrackedRun)
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.run_id = "other"


def test_existing_folder_is_refused_untouched_and_leaves_no_handler(root_logger, config):
    folder = config["paths"]["runs_dir"] / "train" / "20261011T120000Z"
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("keep", encoding="utf-8")
    before = list(root_logger.handlers)

    with pytest.raises(TrackingError, match="create the run folder"):
        with track_run(config, "train", INSTANT):
            pytest.fail("the body must not run")

    assert [path.name for path in folder.iterdir()] == ["keep.txt"]
    assert (folder / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert root_logger.handlers == before


def test_bad_run_type_is_a_value_error(root_logger, config):
    with pytest.raises(ValueError):
        with track_run(config, "other", INSTANT):
            pytest.fail("the body must not run")

    assert not config["paths"]["runs_dir"].exists()


def test_train_and_score_runs_at_the_same_instant_do_not_collide(root_logger, config):
    with track_run(config, "train", INSTANT) as train_run:
        pass
    with track_run(config, "score", INSTANT) as score_run:
        pass

    assert train_run.folder != score_run.folder
    assert train_run.folder.is_dir() and score_run.folder.is_dir()


def test_second_run_of_the_same_type_at_the_same_instant_fails(root_logger, config):
    with track_run(config, "train", INSTANT):
        pass

    with pytest.raises(TrackingError):
        with track_run(config, "train", INSTANT):
            pytest.fail("the body must not run")


def test_failure_to_create_the_folder_is_a_tracking_error(root_logger, config, tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file", encoding="utf-8")
    config["paths"]["runs_dir"] = blocker
    before = list(root_logger.handlers)

    with pytest.raises(TrackingError, match="create the run folder"):
        with track_run(config, "train", INSTANT):
            pytest.fail("the body must not run")

    assert root_logger.handlers == before


def test_failure_to_write_config_json_is_a_tracking_error(root_logger, config, monkeypatch):
    def broken_write(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(tracking, "_write_json", broken_write)
    before = list(root_logger.handlers)

    with pytest.raises(TrackingError, match=r"write config\.json.*disk full"):
        with track_run(config, "train", INSTANT):
            pytest.fail("the body must not run")

    assert root_logger.handlers == before


def test_failure_to_open_run_log_is_a_tracking_error(root_logger, config, monkeypatch):
    def broken_attach(path):
        raise OSError("cannot open")

    monkeypatch.setattr(tracking, "attach_file_handler", broken_attach)
    before = list(root_logger.handlers)

    with pytest.raises(TrackingError, match=r"open run\.log.*cannot open"):
        with track_run(config, "train", INSTANT):
            pytest.fail("the body must not run")

    assert root_logger.handlers == before


# --- Running and ending -----------------------------------------------------


def test_log_lines_inside_the_block_land_in_run_log_and_none_after(root_logger, config):
    with track_run(config, "train", INSTANT) as run:
        LOGGER.info("inside the run ✓")
    LOGGER.info("after the run")

    content = (run.folder / "run.log").read_text(encoding="utf-8")
    assert "inside the run ✓" in content
    assert "after the run" not in content


def test_handler_is_detached_after_success(root_logger, config):
    before = list(root_logger.handlers)

    with track_run(config, "train", INSTANT):
        assert len(root_logger.handlers) == len(before) + 1

    assert root_logger.handlers == before


def test_handler_is_detached_after_an_exception_which_propagates_unchanged(
    root_logger, config
):
    before = list(root_logger.handlers)
    failure = RuntimeError("body failed")

    with pytest.raises(RuntimeError) as exc_info:
        with track_run(config, "train", INSTANT) as run:
            raise failure

    assert exc_info.value is failure
    assert root_logger.handlers == before
    shutil.rmtree(run.folder)


def test_a_failed_body_leaves_config_and_log_but_no_metrics(root_logger, config):
    with pytest.raises(RuntimeError):
        with track_run(config, "train", INSTANT) as run:
            LOGGER.info("about to fail")
            raise RuntimeError("body failed")

    assert sorted(path.name for path in run.folder.iterdir()) == ["config.json", "run.log"]


# --- Metrics ----------------------------------------------------------------


def test_write_metrics_writes_indented_utf8_json_and_leaves_no_temporary_file(
    root_logger, config
):
    metrics = {"k": 0.10, "note": "café ✓", "values": [1, 2]}

    with track_run(config, "train", INSTANT) as run:
        run.write_metrics(metrics)
        names = sorted(path.name for path in run.folder.iterdir())

    text = (run.folder / "metrics.json").read_text(encoding="utf-8")
    assert json.loads(text) == metrics
    assert "café ✓" in text
    assert "\n  " in text
    assert names == ["config.json", "metrics.json", "run.log"]


def test_rename_failure_leaves_no_metrics_and_removes_the_temporary_file(
    root_logger, config, monkeypatch
):
    def broken_replace(source, target):
        raise OSError("rename failed")

    with track_run(config, "train", INSTANT) as run:
        monkeypatch.setattr(tracking.os, "replace", broken_replace)
        with pytest.raises(TrackingError, match=r"metrics\.json.*rename failed"):
            run.write_metrics({"k": 0.1})
        monkeypatch.undo()

        assert sorted(path.name for path in run.folder.iterdir()) == ["config.json", "run.log"]


def test_rename_and_removal_failure_leaves_no_metrics_file(
    root_logger, config, monkeypatch
):
    def broken(*args, **kwargs):
        raise OSError("failed")

    with track_run(config, "train", INSTANT) as run:
        monkeypatch.setattr(tracking.os, "replace", broken)
        monkeypatch.setattr(tracking.os, "remove", broken)
        with pytest.raises(TrackingError):
            run.write_metrics({"k": 0.1})
        monkeypatch.undo()

        assert not (run.folder / "metrics.json").exists()
        names = {path.name for path in run.folder.iterdir()}
        assert names == {"config.json", "run.log", ".tmp-metrics.json"}


@pytest.mark.parametrize(
    "metrics", [{"value": object()}, {"value": float("nan")}], ids=["object", "nan"]
)
def test_unserialisable_metrics_leave_no_metrics_and_no_temporary_file(
    root_logger, config, metrics
):
    with track_run(config, "train", INSTANT) as run:
        with pytest.raises(TrackingError, match=r"serialise metrics\.json"):
            run.write_metrics(metrics)

        assert sorted(path.name for path in run.folder.iterdir()) == ["config.json", "run.log"]


def test_a_second_write_fails_and_keeps_the_first_file(root_logger, config):
    with track_run(config, "train", INSTANT) as run:
        run.write_metrics({"first": 1})
        first = (run.folder / "metrics.json").read_bytes()

        with pytest.raises(TrackingError, match="already written"):
            run.write_metrics({"second": 2})

        assert (run.folder / "metrics.json").read_bytes() == first
