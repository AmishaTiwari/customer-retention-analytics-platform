"""Runs the versioned feature-engineering SQL scripts against the local
DuckDB database.

Independent from retention_platform.data.prepare: this module only builds
features on top of an already-prepared database (mv_churn must already
exist) and never re-runs staging/cleaning/target/modeling-view itself.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

from retention_platform.config import ConfigValidationError, load_config
from retention_platform.data.prepare import _table_name_from_sql
from retention_platform.logging_setup import setup_logging

logger = logging.getLogger("retention_platform.features.build")

REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_SQL_DIR = REPO_ROOT / "sql" / "05_features"


class FeatureBuildError(Exception):
    """Raised when a feature-engineering SQL script fails to execute."""


def run_features(
    conn: duckdb.DuckDBPyConnection,
    features_dir: Path | None = None,
) -> None:
    """Execute every SQL script in sql/05_features/, in filename order."""
    features_dir = features_dir or FEATURES_SQL_DIR

    scripts = sorted(features_dir.glob("*.sql"))
    for script_path in scripts:
        sql_text = script_path.read_text(encoding="utf-8")

        try:
            table_name = _table_name_from_sql(sql_text, script_path)
            conn.execute(sql_text)
        except Exception as exc:
            logger.error("Feature-engineering script %s failed: %s", script_path.name, exc)
            raise FeatureBuildError(
                f"Feature-engineering script {script_path.name} failed: {exc}"
            ) from exc

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info("%s: %d rows", table_name, row_count)


def main() -> None:
    setup_logging()

    conn = None
    try:
        db_path = load_config()["paths"]["interim_db"]
        conn = duckdb.connect(str(db_path))
        run_features(conn)
    except FeatureBuildError as exc:
        logger.error("Feature engineering failed: %s", exc)
        sys.exit(1)
    except duckdb.Error as exc:
        logger.error("Database error:\n%s", exc)
        sys.exit(1)
    except ConfigValidationError as exc:
        logger.error("Configuration error:\n%s", exc)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
