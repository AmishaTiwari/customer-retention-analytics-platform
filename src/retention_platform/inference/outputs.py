"""Build and publish the scoring outputs.

Two publication paths exist. publish_outputs replaces the four-column
outreach CSV and then the run summary, using a temporary file next to each
final file. publish_output_set writes the eleven-column outreach CSV, the
economics document, the recommendations document and the run summary to
temporary files first, replaces the CSV, the economics document and the
recommendations document in that order, checks the three files and the run
IDs read back from the two documents, and replaces the run summary last.

Every output is built as bytes in memory before any file is touched, so a
table, text or manifest that cannot be written fails before a replacement.
On a replacement failure nothing is rolled back and no backup is made.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retention_platform.inference.drivers import DIRECTION_LOWERS, DIRECTION_RAISES, DRIVER_LABELS
from retention_platform.inference.ranking import assign_risk_tiers
from retention_platform.models.persist import (
    HELD_OUT_RESULTS_REFERENCE,
    TRAINING_CV_ESTIMATE_TYPE,
    is_valid_run_id,
)

logger = logging.getLogger(__name__)

OUTREACH_FILE = "prioritized_outreach_list.csv"
SUMMARY_FILE = "run_summary.json"
CONTRACT_VERSION = "1"
CONTRACT_VERSION_V2 = "2"
ECONOMICS_FILE = "economics_summary.md"
RECOMMENDATIONS_FILE = "business_recommendations.md"
OUTPUT_FILES_PREFIX = "outputs/"
OUTPUT_FILES = {
    "outreach_list": OUTPUT_FILES_PREFIX + OUTREACH_FILE,
    "economics_summary": OUTPUT_FILES_PREFIX + ECONOMICS_FILE,
    "business_recommendations": OUTPUT_FILES_PREFIX + RECOMMENDATIONS_FILE,
}
SNAPSHOT_NOTICE = (
    "Historical snapshot: this list scores the held-out test split for "
    "demonstration. It is not a live outreach list."
)

SCORING_RUN_ID_LABEL = "Scoring run ID: "
MODEL_RUN_ID_LABEL = "Model run ID: "

_TEMP_PREFIX = ".tmp-"
_TABLE_COLUMNS = ["customer_id", "churn_probability", "rank", "in_top_k"]
EXTENDED_TABLE_COLUMNS = [
    *_TABLE_COLUMNS,
    "risk_tier",
    "driver_1",
    "driver_1_direction",
    "driver_2",
    "driver_2_direction",
    "driver_3",
    "driver_3_direction",
]
RISK_TIERS = ("High", "Medium", "Low")
_DRIVER_PAIRS = (
    ("driver_1", "driver_1_direction"),
    ("driver_2", "driver_2_direction"),
    ("driver_3", "driver_3_direction"),
)
_MODEL_HEALTH_KEYS = [
    "trained_at_utc",
    "package_version",
    "training_cv_metrics",
    "held_out_results_reference",
]
_METRIC_KEYS = [
    "k",
    "training_rows",
    "estimate_type",
    "precision_at_k",
    "lift_at_k",
    "pr_auc",
    "brier_score",
]
_TRAINED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class OutputGenerationError(Exception):
    """The outputs could not be built or written to temporary files.
    Nothing was replaced."""


class OutputPublicationError(Exception):
    """A file replacement or a check before the summary replacement failed.
    The flags say which final files this run replaced."""

    def __init__(
        self,
        message: str,
        *,
        csv_replaced: bool,
        summary_replaced: bool,
        economics_replaced: bool = False,
        recommendations_replaced: bool = False,
    ):
        super().__init__(message)
        self.csv_replaced = csv_replaced
        self.economics_replaced = economics_replaced
        self.recommendations_replaced = recommendations_replaced
        self.summary_replaced = summary_replaced


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_table(table: pd.DataFrame) -> None:
    """Raise OutputGenerationError unless the table is a usable ranked table."""
    if list(table.columns) != _TABLE_COLUMNS:
        raise OutputGenerationError(
            f"the outreach table must have exactly the columns {_TABLE_COLUMNS}, "
            f"got {list(table.columns)}"
        )
    if len(table) == 0:
        raise OutputGenerationError("the outreach table has no rows")
    for column in _TABLE_COLUMNS:
        if table[column].isna().any():
            raise OutputGenerationError(f"the outreach table has a missing value in {column}")

    ids = table["customer_id"].tolist()
    if not all(isinstance(value, str) and value.strip() for value in ids):
        raise OutputGenerationError("the outreach table has an empty or non-text customer_id")
    if not pd.api.types.is_integer_dtype(table["rank"]):
        raise OutputGenerationError("the outreach table's rank must be integers")
    if not all(isinstance(value, (bool, np.bool_)) for value in table["in_top_k"].tolist()):
        raise OutputGenerationError("the outreach table's in_top_k must be True or False")
    probabilities = pd.to_numeric(table["churn_probability"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise OutputGenerationError(
            "the outreach table's churn_probability must be finite and within 0 to 1"
        )


def _check_manifest_health(manifest: dict) -> dict:
    """Return a deep copy of the model_health block, after checking the
    manifest fields it is copied from."""
    for key in _MODEL_HEALTH_KEYS:
        if key not in manifest:
            raise OutputGenerationError(f"the manifest has no {key}")

    trained_at = manifest["trained_at_utc"]
    try:
        if not isinstance(trained_at, str):
            raise ValueError
        datetime.strptime(trained_at, _TRAINED_AT_FORMAT)
    except ValueError:
        raise OutputGenerationError(
            f"the manifest's trained_at_utc is not an ISO 8601 UTC time ending in Z: {trained_at!r}"
        ) from None
    if not isinstance(manifest["package_version"], str) or not manifest["package_version"]:
        raise OutputGenerationError("the manifest's package_version must be non-empty text")

    metrics = manifest["training_cv_metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(_METRIC_KEYS):
        got = sorted(metrics) if isinstance(metrics, dict) else metrics
        raise OutputGenerationError(
            f"the manifest's training_cv_metrics must have exactly the keys {_METRIC_KEYS}, got {got}"
        )
    for name in ("k", "precision_at_k", "lift_at_k", "pr_auc", "brier_score"):
        if not _is_number(metrics[name]) or not math.isfinite(metrics[name]):
            raise OutputGenerationError(
                f"the manifest's training_cv_metrics {name} must be a finite number"
            )
    rows = metrics["training_rows"]
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise OutputGenerationError("the manifest's training_cv_metrics training_rows must be an integer")
    if metrics["estimate_type"] != TRAINING_CV_ESTIMATE_TYPE:
        raise OutputGenerationError("the manifest's training_cv_metrics estimate_type has the wrong wording")

    if manifest["held_out_results_reference"] != HELD_OUT_RESULTS_REFERENCE:
        raise OutputGenerationError(
            f"the manifest's held_out_results_reference must be {HELD_OUT_RESULTS_REFERENCE!r}"
        )

    return copy.deepcopy({key: manifest[key] for key in _MODEL_HEALTH_KEYS})


def build_run_summary(
    table: pd.DataFrame,
    *,
    k_frac: float,
    model_run_id: str,
    scoring_run_id: str,
    manifest: dict,
) -> dict:
    """Build the run summary as plain Python data, with the keys and key
    order of the output contract. model_health is a deep copy of the
    manifest's values."""
    _check_table(table)
    if not _is_number(k_frac) or not 0 < k_frac <= 1:
        raise OutputGenerationError(f"k_frac must be a number with 0 < k_frac <= 1, got {k_frac!r}")
    for name, value in (("model_run_id", model_run_id), ("scoring_run_id", scoring_run_id)):
        if not is_valid_run_id(value):
            raise OutputGenerationError(f"{name} is not a valid run id: {value!r}")
    model_health = _check_manifest_health(manifest)

    return {
        "contract_version": CONTRACT_VERSION,
        "k": float(k_frac),
        "customers_scored": int(len(table)),
        "customers_flagged": int(table["in_top_k"].sum()),
        "model_run_id": model_run_id,
        "scoring_run_id": scoring_run_id,
        "snapshot_notice": SNAPSHOT_NOTICE,
        "model_health": model_health,
    }


def document_header(title: str, scoring_run_id: str, model_run_id: str) -> str:
    """The four fixed header lines of a Markdown document, each ending in a
    line feed: the title, an empty line, the scoring run ID and the model
    run ID."""
    if not isinstance(title, str) or not title.strip():
        raise OutputGenerationError(f"the document title must be non-empty text, got {title!r}")
    if title.splitlines() != [title]:
        raise OutputGenerationError(f"the document title must be a single line, got {title!r}")
    for name, value in (("scoring_run_id", scoring_run_id), ("model_run_id", model_run_id)):
        if not is_valid_run_id(value):
            raise OutputGenerationError(f"{name} is not a valid run id: {value!r}")
    return (
        f"# {title}\n"
        "\n"
        f"{SCORING_RUN_ID_LABEL}{scoring_run_id}\n"
        f"{MODEL_RUN_ID_LABEL}{model_run_id}\n"
    )


def outreach_csv_bytes(table: pd.DataFrame) -> bytes:
    """The outreach list as UTF-8 CSV bytes: one header row, no index, LF line
    endings, in_top_k as True or False, and each probability written with
    its shortest exact text."""
    _check_table(table)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_TABLE_COLUMNS)
    for customer_id, probability, rank, flagged in zip(
        table["customer_id"], table["churn_probability"], table["rank"], table["in_top_k"]
    ):
        writer.writerow([customer_id, repr(float(probability)), str(int(rank)), str(bool(flagged))])
    return buffer.getvalue().encode("utf-8")


def check_extended_outreach_table(table: pd.DataFrame, k_frac: float, medium_edge: float) -> None:
    """Raise OutputGenerationError unless the table is a usable eleven-column
    outreach table.

    The four v1 columns get the v1 checks. risk_tier must follow rank by the
    tier rule for k_frac and medium_edge, High exactly when in_top_k is True.
    Every driver is a label of the label map, a customer's three drivers are
    different, and every direction is one of the two allowed values.
    """
    if list(table.columns) != EXTENDED_TABLE_COLUMNS:
        raise OutputGenerationError(
            f"the outreach table must have exactly the columns {EXTENDED_TABLE_COLUMNS}, "
            f"got {list(table.columns)}"
        )
    _check_table(table[_TABLE_COLUMNS])
    if not (
        _is_number(k_frac)
        and _is_number(medium_edge)
        and 0 < k_frac < medium_edge <= 1
    ):
        raise OutputGenerationError(
            f"k_frac ({k_frac!r}) and medium_edge ({medium_edge!r}) must be numbers "
            "with 0 < k_frac < medium_edge <= 1"
        )

    ids = table["customer_id"].tolist()
    tiers = table["risk_tier"].tolist()
    for customer, tier in zip(ids, tiers):
        if not isinstance(tier, str) or tier not in RISK_TIERS:
            raise OutputGenerationError(
                f"risk_tier is not High, Medium or Low for customer {customer!r}: {tier!r}"
            )
    try:
        expected = assign_risk_tiers(table["rank"].reset_index(drop=True), k_frac, medium_edge)
    except ValueError as exc:
        raise OutputGenerationError(f"the ranks cannot be turned into risk tiers: {exc}") from exc
    for customer, tier, wanted, flagged in zip(ids, tiers, expected.tolist(), table["in_top_k"].tolist()):
        if (tier == "High") != bool(flagged):
            raise OutputGenerationError(
                f"risk_tier is {tier} but in_top_k is {bool(flagged)} for customer {customer!r}: "
                "High must be exactly the flagged customers"
            )
        if tier != wanted:
            raise OutputGenerationError(
                f"risk_tier does not follow rank for customer {customer!r}: expected {wanted}, got {tier}"
            )

    allowed_labels = set(DRIVER_LABELS.values())
    allowed_directions = (DIRECTION_RAISES, DIRECTION_LOWERS)
    drivers = [table[driver].tolist() for driver, _ in _DRIVER_PAIRS]
    directions = [table[direction].tolist() for _, direction in _DRIVER_PAIRS]
    for position, customer in enumerate(ids):
        shown = []
        for (driver_column, direction_column), labels, kinds in zip(_DRIVER_PAIRS, drivers, directions):
            label, direction = labels[position], kinds[position]
            if not isinstance(label, str) or not label.strip():
                raise OutputGenerationError(f"{driver_column} is empty or not text for customer {customer!r}")
            if label not in allowed_labels:
                raise OutputGenerationError(
                    f"{driver_column} is not a label of the label map for customer {customer!r}: {label!r}"
                )
            if direction not in allowed_directions:
                raise OutputGenerationError(
                    f"{direction_column} is not {DIRECTION_RAISES!r} or {DIRECTION_LOWERS!r} "
                    f"for customer {customer!r}: {direction!r}"
                )
            shown.append(label)
        if len(set(shown)) != len(shown):
            raise OutputGenerationError(f"the three drivers of customer {customer!r} are not all different")


def extended_outreach_csv_bytes(table: pd.DataFrame, k_frac: float, medium_edge: float) -> bytes:
    """The eleven-column outreach list as UTF-8 CSV bytes, written the same
    way as outreach_csv_bytes: one header row, no index, LF line endings,
    in_top_k as True or False, each probability with its shortest exact text."""
    check_extended_outreach_table(table, k_frac, medium_edge)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EXTENDED_TABLE_COLUMNS)
    columns = [table[name].tolist() for name in EXTENDED_TABLE_COLUMNS]
    for customer_id, probability, rank, flagged, *rest in zip(*columns):
        writer.writerow(
            [customer_id, repr(float(probability)), str(int(rank)), str(bool(flagged)), *rest]
        )
    return buffer.getvalue().encode("utf-8")


def summary_json_bytes(summary: dict) -> bytes:
    """The run summary as indented UTF-8 JSON bytes with a trailing newline."""
    try:
        text = json.dumps(summary, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OutputGenerationError(f"the run summary cannot be written as JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()


def _remove_quietly(*paths: Path) -> None:
    """Remove files where possible; a failure here is never reported."""
    for path in paths:
        try:
            os.remove(path)
        except Exception:
            pass


def publish_outputs(
    outputs_dir: str | Path, table: pd.DataFrame, summary: dict
) -> tuple[Path, Path]:
    """Replace the outreach CSV and then the run summary in outputs_dir.

    Each file is replaced as a whole through a temporary file in the same
    folder. The CSV is replaced first. If the summary replacement then
    fails, the new CSV stays, the previous summary is untouched, and the
    previous CSV is not restored. Returns (csv_path, summary_path).
    """
    csv_bytes = outreach_csv_bytes(table)
    summary_bytes = summary_json_bytes(summary)

    folder = Path(outputs_dir)
    csv_path = folder / OUTREACH_FILE
    summary_path = folder / SUMMARY_FILE
    csv_temp = folder / (_TEMP_PREFIX + OUTREACH_FILE)
    summary_temp = folder / (_TEMP_PREFIX + SUMMARY_FILE)

    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_file(csv_temp, csv_bytes)
        _write_file(summary_temp, summary_bytes)
    except OSError as exc:
        _remove_quietly(csv_temp, summary_temp)
        raise OutputGenerationError(
            f"the outputs could not be written to temporary files in {folder}: {exc}"
        ) from exc

    try:
        os.replace(csv_temp, csv_path)
    except OSError as exc:
        _remove_quietly(csv_temp, summary_temp)
        raise OutputPublicationError(
            f"Publication failed: {csv_path.name} was not replaced and {summary_path.name} "
            f"was not replaced. Both output files are as they were. Cause: {exc}",
            csv_replaced=False,
            summary_replaced=False,
        ) from exc

    try:
        os.replace(summary_temp, summary_path)
    except OSError as exc:
        _remove_quietly(summary_temp)
        raise OutputPublicationError(
            f"Publication failed part way: {csv_path.name} was replaced with the new list, "
            f"but {summary_path.name} was not replaced. The pair may be inconsistent until a "
            f"successful run. The previous {csv_path.name} was not restored and no copy of it "
            f"is kept. Cause: {exc}",
            csv_replaced=True,
            summary_replaced=False,
        ) from exc

    logger.info("Published %s and %s", csv_path, summary_path)
    return csv_path, summary_path


def build_extended_run_summary(
    table: pd.DataFrame,
    *,
    k_frac: float,
    medium_edge: float,
    model_run_id: str,
    scoring_run_id: str,
    manifest: dict,
) -> dict:
    """Build the contract v2 run summary as plain Python data, with the keys
    and key order of the contract. model_health is a deep copy of the
    manifest's values."""
    check_extended_outreach_table(table, k_frac, medium_edge)
    for name, value in (("model_run_id", model_run_id), ("scoring_run_id", scoring_run_id)):
        if not is_valid_run_id(value):
            raise OutputGenerationError(f"{name} is not a valid run id: {value!r}")
    model_health = _check_manifest_health(manifest)

    return {
        "contract_version": CONTRACT_VERSION_V2,
        "k": float(k_frac),
        "customers_scored": int(len(table)),
        "customers_flagged": int(table["in_top_k"].sum()),
        "model_run_id": model_run_id,
        "scoring_run_id": scoring_run_id,
        "snapshot_notice": SNAPSHOT_NOTICE,
        "model_health": model_health,
        "output_files": dict(OUTPUT_FILES),
    }


def _check_set_inputs(
    table: pd.DataFrame,
    economics_text: object,
    recommendations_text: object,
    summary: object,
    k_frac: float,
    medium_edge: float,
) -> None:
    check_extended_outreach_table(table, k_frac, medium_edge)
    for name, text in (("economics_text", economics_text), ("recommendations_text", recommendations_text)):
        if not isinstance(text, str) or not text.strip():
            raise OutputGenerationError(f"{name} must be non-empty text, got {text!r}")
    if not isinstance(summary, dict):
        raise OutputGenerationError(f"the run summary must be a dict, got {type(summary).__name__}")
    for name in ("scoring_run_id", "model_run_id"):
        if not is_valid_run_id(summary.get(name)):
            raise OutputGenerationError(f"summary {name} is not a valid run id: {summary.get(name)!r}")
    scored, flagged = summary.get("customers_scored"), summary.get("customers_flagged")
    if isinstance(scored, bool) or scored != len(table):
        raise OutputGenerationError(
            f"summary customers_scored ({scored!r}) does not match the {len(table)} rows of the table"
        )
    if isinstance(flagged, bool) or flagged != int(table["in_top_k"].sum()):
        raise OutputGenerationError(
            f"summary customers_flagged ({flagged!r}) does not match the {int(table['in_top_k'].sum())} "
            "flagged rows of the table"
        )
    if not _is_number(summary.get("k")) or summary["k"] != k_frac:
        raise OutputGenerationError(f"summary k ({summary.get('k')!r}) does not match k_frac ({k_frac!r})")


def _set_publication_error(replaced: dict, cause: str) -> OutputPublicationError:
    """The class 2b error: for each of the four files, whether this run replaced it."""
    status = "; ".join(
        f"{name} {'was replaced' if replaced[name] else 'was not replaced'}"
        for name in (OUTREACH_FILE, ECONOMICS_FILE, RECOMMENDATIONS_FILE, SUMMARY_FILE)
    )
    message = (
        f"Publication failed: {status}. The output set may be inconsistent until a successful "
        "run. Previous files were not restored and no copy of them is kept. "
        f"Cause: {cause}"
    )
    return OutputPublicationError(
        message,
        csv_replaced=replaced[OUTREACH_FILE],
        economics_replaced=replaced[ECONOMICS_FILE],
        recommendations_replaced=replaced[RECOMMENDATIONS_FILE],
        summary_replaced=replaced[SUMMARY_FILE],
    )


def _problem_before_summary(
    deliverables: tuple[Path, Path, Path], documents: tuple[Path, Path], summary: dict
) -> str | None:
    """Why the three deliverables or the run IDs in the two documents are not
    as expected, or None. The documents are read back from disk and only
    lines 3 and 4 count."""
    for path in deliverables:
        if not path.is_file():
            return f"{path.name} does not exist as a file in {path.parent}"

    expected = (
        (3, SCORING_RUN_ID_LABEL + summary["scoring_run_id"]),
        (4, MODEL_RUN_ID_LABEL + summary["model_run_id"]),
    )
    for path in documents:
        try:
            lines = path.read_bytes().decode("utf-8").split("\n")
        except (OSError, UnicodeDecodeError) as exc:
            return f"{path.name} could not be read back: {exc}"
        for number, wanted in expected:
            found = lines[number - 1] if len(lines) >= number else None
            if found != wanted:
                return f"{path.name} line {number} is {found!r}, expected {wanted!r}"
    return None


def publish_output_set(
    outputs_dir: str | Path,
    table: pd.DataFrame,
    economics_text: str,
    recommendations_text: str,
    summary: dict,
    *,
    k_frac: float,
    medium_edge: float,
) -> tuple[Path, Path, Path, Path]:
    """Replace the four outputs in outputs_dir and return their paths.

    All four are written to temporary files next to their final files first.
    The CSV, the economics document and the recommendations document are
    then replaced in that order, the three files and the run IDs read back
    from the two documents are checked, and the run summary is replaced
    last. The first failure stops publication: nothing is rolled back, and
    the error says for each of the four files whether this run replaced it.
    """
    _check_set_inputs(table, economics_text, recommendations_text, summary, k_frac, medium_edge)
    try:
        payloads = {
            OUTREACH_FILE: extended_outreach_csv_bytes(table, k_frac, medium_edge),
            ECONOMICS_FILE: economics_text.encode("utf-8"),
            RECOMMENDATIONS_FILE: recommendations_text.encode("utf-8"),
            SUMMARY_FILE: summary_json_bytes(summary),
        }
    except UnicodeEncodeError as exc:
        raise OutputGenerationError(f"a document cannot be encoded as UTF-8: {exc}") from exc

    folder = Path(outputs_dir)
    finals = {name: folder / name for name in payloads}
    temps = {name: folder / (_TEMP_PREFIX + name) for name in payloads}

    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            _write_file(temps[name], data)
    except OSError as exc:
        _remove_quietly(*temps.values())
        raise OutputGenerationError(
            f"the outputs could not be written to temporary files in {folder}: {exc}"
        ) from exc

    replaced = {name: False for name in payloads}
    for name in (OUTREACH_FILE, ECONOMICS_FILE, RECOMMENDATIONS_FILE):
        try:
            os.replace(temps[name], finals[name])
        except OSError as exc:
            _remove_quietly(*temps.values())
            raise _set_publication_error(replaced, str(exc)) from exc
        replaced[name] = True

    problem = _problem_before_summary(
        (finals[OUTREACH_FILE], finals[ECONOMICS_FILE], finals[RECOMMENDATIONS_FILE]),
        (finals[ECONOMICS_FILE], finals[RECOMMENDATIONS_FILE]),
        summary,
    )
    if problem is not None:
        _remove_quietly(*temps.values())
        raise _set_publication_error(replaced, f"check before the run summary failed: {problem}")

    try:
        os.replace(temps[SUMMARY_FILE], finals[SUMMARY_FILE])
    except OSError as exc:
        _remove_quietly(*temps.values())
        raise _set_publication_error(replaced, str(exc)) from exc
    replaced[SUMMARY_FILE] = True

    logger.info("Published %s in %s", ", ".join(payloads), folder)
    return (
        finals[OUTREACH_FILE],
        finals[ECONOMICS_FILE],
        finals[RECOMMENDATIONS_FILE],
        finals[SUMMARY_FILE],
    )
