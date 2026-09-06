"""Runs the versioned SQL scripts against the local DuckDB database.

Executes sql/01_staging/, sql/02_cleaning/, sql/03_target/, and
sql/04_modeling_view/, in order.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import duckdb

from retention_platform.config import ConfigValidationError, load_config
from retention_platform.logging_setup import setup_logging

logger = logging.getLogger("retention_platform.data.prepare")

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGING_SQL_DIR = REPO_ROOT / "sql" / "01_staging"
CLEANING_SQL_DIR = REPO_ROOT / "sql" / "02_cleaning"
TARGET_SQL_DIR = REPO_ROOT / "sql" / "03_target"
MODELING_VIEW_SQL_DIR = REPO_ROOT / "sql" / "04_modeling_view"

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+TABLE\s+(\w+)", re.IGNORECASE
)


class StagingError(Exception):
    """Raised when a staging SQL script fails to execute."""


class CleaningError(Exception):
    """Raised when a cleaning SQL script fails to execute."""


class TargetConstructionError(Exception):
    """Raised when a target-construction SQL script fails to execute."""


class ModelingViewError(Exception):
    """Raised when a modeling-view SQL script fails to execute."""


def _table_name_from_sql(sql_text: str, script_path: Path) -> str:
    match = _CREATE_TABLE_RE.search(sql_text)
    if not match:
        raise ValueError(f"Could not find a CREATE OR REPLACE TABLE statement in {script_path}")
    return match.group(1)


def run_staging(
    conn: duckdb.DuckDBPyConnection,
    raw_dir: Path | None = None,
    staging_dir: Path | None = None,
) -> None:
    """Execute every SQL script in sql/01_staging/, in filename order."""
    raw_dir = raw_dir or load_config()["paths"]["raw_data"]
    staging_dir = staging_dir or STAGING_SQL_DIR

    scripts = sorted(staging_dir.glob("*.sql"))
    for script_path in scripts:
        sql_text = script_path.read_text(encoding="utf-8")
        sql_text = sql_text.replace("{raw_dir}", raw_dir.resolve().as_posix())

        try:
            table_name = _table_name_from_sql(sql_text, script_path)
            conn.execute(sql_text)
        except Exception as exc:
            logger.error("Staging script %s failed: %s", script_path.name, exc)
            raise StagingError(f"Staging script {script_path.name} failed: {exc}") from exc

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("%s: %d rows", table_name, row_count)


def run_cleaning(
    conn: duckdb.DuckDBPyConnection,
    cleaning_dir: Path | None = None,
) -> None:
    """Execute every SQL script in sql/02_cleaning/, in filename order."""
    cleaning_dir = cleaning_dir or CLEANING_SQL_DIR

    scripts = sorted(cleaning_dir.glob("*.sql"))
    for script_path in scripts:
        sql_text = script_path.read_text(encoding="utf-8")

        try:
            table_name = _table_name_from_sql(sql_text, script_path)
            conn.execute(sql_text)
        except Exception as exc:
            logger.error("Cleaning script %s failed: %s", script_path.name, exc)
            raise CleaningError(f"Cleaning script {script_path.name} failed: {exc}") from exc

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("%s: %d rows", table_name, row_count)


def run_target(
    conn: duckdb.DuckDBPyConnection,
    target_dir: Path | None = None,
) -> None:
    """Execute every SQL script in sql/03_target/, in filename order."""
    target_dir = target_dir or TARGET_SQL_DIR

    scripts = sorted(target_dir.glob("*.sql"))
    for script_path in scripts:
        sql_text = script_path.read_text(encoding="utf-8")

        try:
            table_name = _table_name_from_sql(sql_text, script_path)
            conn.execute(sql_text)
        except Exception as exc:
            logger.error("Target-construction script %s failed: %s", script_path.name, exc)
            raise TargetConstructionError(
                f"Target-construction script {script_path.name} failed: {exc}"
            ) from exc

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("%s: %d rows", table_name, row_count)


def run_modeling_view(
    conn: duckdb.DuckDBPyConnection,
    modeling_view_dir: Path | None = None,
) -> None:
    """Execute every SQL script in sql/04_modeling_view/, in filename order."""
    modeling_view_dir = modeling_view_dir or MODELING_VIEW_SQL_DIR

    scripts = sorted(modeling_view_dir.glob("*.sql"))
    for script_path in scripts:
        sql_text = script_path.read_text(encoding="utf-8")

        try:
            table_name = _table_name_from_sql(sql_text, script_path)
            conn.execute(sql_text)
        except Exception as exc:
            logger.error("Modeling-view script %s failed: %s", script_path.name, exc)
            raise ModelingViewError(
                f"Modeling-view script {script_path.name} failed: {exc}"
            ) from exc

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("%s: %d rows", table_name, row_count)


def main() -> None:
    setup_logging()

    conn = None
    try:
        db_path = load_config()["paths"]["interim_db"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(db_path))
        run_staging(conn)
        run_cleaning(conn)
        run_target(conn)
        run_modeling_view(conn)
    except StagingError as exc:
        logger.error("Data preparation (staging) failed: %s", exc)
        sys.exit(1)
    except CleaningError as exc:
        logger.error("Data preparation (cleaning) failed: %s", exc)
        sys.exit(1)
    except TargetConstructionError as exc:
        logger.error("Data preparation (target construction) failed: %s", exc)
        sys.exit(1)
    except ModelingViewError as exc:
        logger.error("Data preparation (modeling view) failed: %s", exc)
        sys.exit(1)
    except ConfigValidationError as exc:
        logger.error("Configuration error:\n%s", exc)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    logger.info("Data preparation (staging + cleaning + target + modeling view) complete.")


if __name__ == "__main__":
    main()
