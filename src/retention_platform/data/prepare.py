"""Runs the versioned SQL scripts against the local DuckDB database.

Currently executes sql/01_staging/ only. Later stages (cleaning, target,
modeling view) will be added as additional steps in main() without changing
how staging runs.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import duckdb

from retention_platform.logging_setup import setup_logging

logger = logging.getLogger("retention_platform.data.prepare")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_INTERIM_DIR = REPO_ROOT / "data" / "interim"
DEFAULT_DB_PATH = DEFAULT_INTERIM_DIR / "retention_platform.duckdb"
STAGING_SQL_DIR = REPO_ROOT / "sql" / "01_staging"

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+TABLE\s+(\w+)", re.IGNORECASE
)


class StagingError(Exception):
    """Raised when a staging SQL script fails to execute."""


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
    raw_dir = raw_dir or DEFAULT_RAW_DIR
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


def main() -> None:
    setup_logging()

    DEFAULT_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        run_staging(conn)
    except StagingError as exc:
        logger.error("Data preparation (staging) failed: %s", exc)
        sys.exit(1)
    finally:
        conn.close()

    logger.info("Data preparation (staging) complete.")


if __name__ == "__main__":
    main()
