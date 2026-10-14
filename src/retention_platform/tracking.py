"""Run tracking for training and scoring runs: the run id, the config
snapshot, the run folder with config.json, run.log and metrics.json, and
the file handler that writes run.log.

A run folder is <runs_dir>/<run_type>/<run_id>/. It is created once and
never reused. metrics.json is written last, so a folder without it is a
failed or incomplete run. Every failure of a required tracking operation is
raised as TrackingError.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from retention_platform.logging_setup import attach_file_handler, detach_file_handler

# The one definition of the run id format (a UTC timestamp).
RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"

RUN_TYPES = ("train", "score")

CONFIG_FILE = "config.json"
LOG_FILE = "run.log"
METRICS_FILE = "metrics.json"
_TEMP_METRICS_FILE = ".tmp-metrics.json"


class TrackingError(Exception):
    """Raised when a required tracking operation fails."""


def _tracking_error(operation: str, path: Path, cause: object) -> TrackingError:
    return TrackingError(f"Tracking failed to {operation} at {path}: {cause}")


def format_run_id(instant: datetime) -> str:
    """Format a timezone-aware instant as a UTC run id such as
    20261011T120000Z. Sub-second parts are dropped."""
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("instant must be timezone-aware")
    return instant.astimezone(timezone.utc).strftime(RUN_ID_FORMAT)


def config_snapshot(config: dict) -> dict:
    """Return a deep copy of the resolved configuration without the
    "paths" section. The input is not changed."""
    return {key: copy.deepcopy(value) for key, value in config.items() if key != "paths"}


def _json_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    text = _json_text(payload)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@dataclass(frozen=True)
class TrackedRun:
    """One tracked run: its id, type and folder."""

    run_id: str
    run_type: str
    folder: Path

    def write_metrics(self, metrics: dict) -> None:
        """Write metrics.json last, through a temporary file in the run
        folder, so a failed write leaves no metrics.json. Metrics are
        written once per run."""
        target = self.folder / METRICS_FILE
        temp = self.folder / _TEMP_METRICS_FILE

        if target.exists():
            raise _tracking_error("write metrics.json", target, "metrics were already written")

        try:
            text = _json_text(metrics)
        except (TypeError, ValueError) as exc:
            raise _tracking_error("serialise metrics.json", target, exc) from exc

        try:
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
            os.replace(temp, target)
        except (OSError, UnicodeError) as exc:
            try:
                os.remove(temp)
            except OSError:
                pass
            raise _tracking_error("write metrics.json", target, exc) from exc


@contextmanager
def track_run(config: dict, run_type: str, instant: datetime) -> Iterator[TrackedRun]:
    """Start a tracked run and end it when the block exits.

    Creates <runs_dir>/<run_type>/<run_id>/ (it must not exist yet), writes
    config.json, then attaches the run.log file handler. The block starts
    only after all three exist. The handler is detached and its file closed
    when the block ends, on success and on failure; an exception raised in
    the block propagates unchanged. metrics.json is never written here.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got {run_type!r}")

    run_id = format_run_id(instant)
    folder = Path(config["paths"]["runs_dir"]) / run_type / run_id

    try:
        folder.parent.mkdir(parents=True, exist_ok=True)
        folder.mkdir()
    except OSError as exc:
        raise _tracking_error("create the run folder", folder, exc) from exc

    config_path = folder / CONFIG_FILE
    try:
        _write_json(config_path, config_snapshot(config))
    except (OSError, TypeError, ValueError) as exc:
        raise _tracking_error("write config.json", config_path, exc) from exc

    log_path = folder / LOG_FILE
    try:
        handler = attach_file_handler(log_path)
    except OSError as exc:
        raise _tracking_error("open run.log", log_path, exc) from exc

    try:
        yield TrackedRun(run_id=run_id, run_type=run_type, folder=folder)
    finally:
        detach_file_handler(handler)
