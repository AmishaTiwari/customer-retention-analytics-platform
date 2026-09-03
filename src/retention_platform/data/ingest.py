"""Verification of committed raw source tables.

Checks that the raw CSV files in data/raw/ match an expected spec
(filenames, row counts, column counts, column names) and raises with a
complete list of mismatches if they do not. Does not download or modify
any files.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from retention_platform.config import load_config
from retention_platform.logging_setup import setup_logging

logger = logging.getLogger("retention_platform.data.ingest")


@dataclass(frozen=True)
class RawFileSpec:
    """Expected shape of a single committed raw source table."""

    filename: str
    expected_rows: int
    expected_columns: list[str]


RAW_FILE_SPECS: list[RawFileSpec] = [
    RawFileSpec(
        filename="telco_customer_churn_demographics.csv",
        expected_rows=7043,
        expected_columns=[
            "Customer ID",
            "Count",
            "Gender",
            "Age",
            "Under 30",
            "Senior Citizen",
            "Married",
            "Dependents",
            "Number of Dependents",
        ],
    ),
    RawFileSpec(
        filename="telco_customer_churn_location.csv",
        expected_rows=7043,
        expected_columns=[
            "Customer ID",
            "Count",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Lat Long",
            "Latitude",
            "Longitude",
        ],
    ),
    RawFileSpec(
        filename="telco_customer_churn_population.csv",
        expected_rows=1671,
        expected_columns=["ID", "Zip Code", "Population"],
    ),
    RawFileSpec(
        filename="telco_customer_churn_services.csv",
        expected_rows=7043,
        expected_columns=[
            "Customer ID",
            "Count",
            "Quarter",
            "Referred a Friend",
            "Number of Referrals",
            "Tenure in Months",
            "Offer",
            "Phone Service",
            "Avg Monthly Long Distance Charges",
            "Multiple Lines",
            "Internet Service",
            "Internet Type",
            "Avg Monthly GB Download",
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
            "Unlimited Data",
            "Contract",
            "Paperless Billing",
            "Payment Method",
            "Monthly Charge",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
            "Total Revenue",
        ],
    ),
    RawFileSpec(
        filename="telco_customer_churn_status.csv",
        expected_rows=7043,
        expected_columns=[
            "Customer ID",
            "Count",
            "Quarter",
            "Satisfaction Score",
            "Customer Status",
            "Churn Label",
            "Churn Value",
            "Churn Score",
            "CLTV",
            "Churn Category",
            "Churn Reason",
        ],
    ),
]


class RawDataVerificationError(Exception):
    """Raised when one or more raw files fail verification."""


def _read_header_and_row_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        row_count = sum(1 for _ in reader)
    return header, row_count


def _verify_file(path: Path, spec: RawFileSpec) -> list[str]:
    """Return a list of mismatch descriptions for a single file (empty if OK)."""
    if not path.exists():
        return [f"{spec.filename}: file not found at {path}"]

    actual_columns, actual_rows = _read_header_and_row_count(path)
    errors: list[str] = []

    if actual_rows != spec.expected_rows:
        errors.append(
            f"{spec.filename}: row count mismatch "
            f"(expected {spec.expected_rows}, got {actual_rows})"
        )

    if len(actual_columns) != len(spec.expected_columns):
        errors.append(
            f"{spec.filename}: column count mismatch "
            f"(expected {len(spec.expected_columns)}, got {len(actual_columns)})"
        )

    if actual_columns != spec.expected_columns:
        if sorted(actual_columns) == sorted(spec.expected_columns):
            errors.append(
                f"{spec.filename}: column order differs from expected "
                f"(expected {spec.expected_columns}, got {actual_columns})"
            )
        else:
            missing = [c for c in spec.expected_columns if c not in actual_columns]
            unexpected = [c for c in actual_columns if c not in spec.expected_columns]
            errors.append(
                f"{spec.filename}: column names mismatch "
                f"(expected {spec.expected_columns}, got {actual_columns}; "
                f"missing={missing}, unexpected={unexpected})"
            )

    return errors


def verify_raw_data(raw_dir: Path | None = None) -> None:
    """Verify every expected raw file against RAW_FILE_SPECS.

    Raises RawDataVerificationError listing every mismatch found across all
    files if any file fails verification.
    """
    raw_dir = raw_dir or load_config()["paths"]["raw_data"]
    all_errors: list[str] = []

    for spec in RAW_FILE_SPECS:
        path = raw_dir / spec.filename
        file_errors = _verify_file(path, spec)
        if file_errors:
            all_errors.extend(file_errors)
        else:
            logger.info(
                "Verified %s: %d rows, %d columns",
                spec.filename,
                spec.expected_rows,
                len(spec.expected_columns),
            )

    if all_errors:
        for error in all_errors:
            logger.error(error)
        raise RawDataVerificationError(
            f"Raw data verification failed with {len(all_errors)} issue(s):\n"
            + "\n".join(f"- {error}" for error in all_errors)
        )


def main() -> None:
    setup_logging()
    try:
        verify_raw_data()
    except RawDataVerificationError as exc:
        logger.error("Raw data verification failed:\n%s", exc)
        sys.exit(1)
    logger.info("Raw data verification passed for all %d expected files.", len(RAW_FILE_SPECS))


if __name__ == "__main__":
    main()
