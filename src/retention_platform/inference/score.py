"""Scoring: the scoring core and the scoring command.

The core (score_batch) turns a validated customer table and a loaded model
run into the ranked outreach table (customer_id, churn_probability, rank,
in_top_k). The table is validated first; a column that is not a model
input is never given to the model, and no label column is read.

The command (main, run_scoring) selects the held-out test population,
scores it, builds and publishes the four outputs and tracks the run. Run it with
`python -m retention_platform.inference.score`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retention_platform.config import REPO_ROOT, ConfigValidationError, load_config
from retention_platform.inference.drivers import (
    DRIVER_COLUMNS,
    DRIVER_LABELS,
    DriverError,
    averaged_contributions,
    global_importance,
    select_top_drivers,
)
from retention_platform.inference.economics import (
    EconomicsAssumptions,
    EconomicsError,
    analyze_economics,
    render_economics_document,
)
from retention_platform.inference.outputs import (
    EXTENDED_TABLE_COLUMNS,
    OutputGenerationError,
    OutputPublicationError,
    build_extended_run_summary,
    publish_output_set,
)
from retention_platform.inference.ranking import assign_risk_tiers, top_k_count
from retention_platform.inference.recommendations import (
    RecommendationsError,
    compute_recommendation_facts,
    render_recommendations_document,
    select_recommendations,
)
from retention_platform.inference.validate import InputValidationError, validate_scoring_table
from retention_platform.logging_setup import setup_logging
from retention_platform.models.persist import LoadedRun, ModelLoadError, load_run
from retention_platform.pipeline.split import split_feat_churn
from retention_platform.tracking import TrackingError, track_run

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"


class ScoringError(Exception):
    """Raised when the model's output cannot be used to build the ranking."""


def _model_probabilities(loaded_run: LoadedRun, inputs: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """The model's churn probability for each row, checked to be usable."""
    matrix = loaded_run.preprocessor.transform(inputs)
    raw = np.asarray(loaded_run.model.predict_proba(matrix))
    n_rows = len(inputs)

    if raw.ndim != 2 or raw.shape[0] != n_rows or raw.shape[1] < 2:
        raise ScoringError(
            f"the model returned an output of shape {raw.shape} for {n_rows} rows"
        )
    probabilities = raw[:, 1].astype(float)

    non_finite = np.flatnonzero(~np.isfinite(probabilities))
    if non_finite.size:
        row = int(non_finite[0])
        raise ScoringError(
            f"the model returned a non-finite probability for {non_finite.size} row(s); "
            f"first at row {row} (customer_id '{ids[row]}')"
        )
    outside = np.flatnonzero((probabilities < 0) | (probabilities > 1))
    if outside.size:
        row = int(outside[0])
        raise ScoringError(
            f"the model returned a probability outside [0, 1] for {outside.size} row(s); "
            f"first at row {row} (customer_id '{ids[row]}', value {probabilities[row]!r})"
        )
    return probabilities


def score_batch(frame: pd.DataFrame, loaded_run: LoadedRun, k_frac: float) -> pd.DataFrame:
    """Score a customer table and return the ranked outreach table.

    The table is validated first and InputValidationError propagates
    unchanged. The model then sees only the schema's input columns, in the
    schema's order. Rows are ordered by churn probability, highest first,
    with ties broken by customer_id ascending as text. rank runs 1 to n
    and in_top_k is true for the first top_k_count(n, k_frac) rows. The
    caller's frame is not changed.
    """
    validate_scoring_table(frame, loaded_run.schema)

    identifier = loaded_run.schema["identifier_column"]
    input_columns = [column["name"] for column in loaded_run.schema["columns"]]
    inputs = frame[input_columns].copy()
    ids = [str(value) for value in frame[identifier].astype(object).tolist()]

    probabilities = _model_probabilities(loaded_run, inputs, ids)

    table = pd.DataFrame(
        {
            "customer_id": pd.Series(ids, dtype=object),
            "churn_probability": pd.Series(probabilities, dtype="float64"),
        }
    )
    table = table.sort_values(
        by=["churn_probability", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)

    n_rows = len(table)
    table["rank"] = np.arange(1, n_rows + 1, dtype="int64")
    table["in_top_k"] = table["rank"] <= top_k_count(n_rows, k_frac)
    return table[["customer_id", "churn_probability", "rank", "in_top_k"]]


def extend_outreach_table(
    table: pd.DataFrame, drivers: pd.DataFrame, k_frac: float, medium_edge: float
) -> pd.DataFrame:
    """The eleven-column outreach table: the score_batch table plus the risk
    tier and the three labelled drivers with their directions.

    drivers is the select_top_drivers output, indexed by customer_id; its
    rows are matched to the table by customer_id, not by position. Driver
    features are turned into labels with DRIVER_LABELS. The inputs are not
    changed.
    """
    matched = drivers.loc[table["customer_id"]]
    extended = table.copy()
    extended["risk_tier"] = assign_risk_tiers(table["rank"], k_frac, medium_edge).to_numpy()
    for column in DRIVER_COLUMNS:
        values = matched[column].tolist()
        if column.endswith("_direction"):
            extended[column] = values
        else:
            extended[column] = [DRIVER_LABELS[feature] for feature in values]
    return extended[EXTENDED_TABLE_COLUMNS]


def _economics_assumptions(config: dict) -> EconomicsAssumptions:
    """The base economics assumptions from business.economics."""
    economics = config["business"]["economics"]
    return EconomicsAssumptions(
        save_rate=economics["save_rate"],
        months_kept=economics["months_kept"],
        offer_cost=economics["offer_cost"],
        arpu_proxy=economics["arpu_proxy"],
    )


@dataclass(frozen=True)
class ScoringResult:
    """What a finished scoring run produced."""

    run_id: str
    model_run_id: str
    customers_scored: int
    customers_flagged: int
    outreach_path: Path
    economics_path: Path
    recommendations_path: Path
    summary_path: Path


class PostPublicationTrackingError(Exception):
    """All four outputs were published, but a required tracking write failed."""

    def __init__(self, scoring_run_id: str, message: str):
        super().__init__(message)
        self.scoring_run_id = scoring_run_id


def select_scoring_population(conn: duckdb.DuckDBPyConnection, schema: dict) -> pd.DataFrame:
    """The held-out test split, reduced to the identifier and the model's
    input columns. Columns are only selected, never converted, so missing
    values in the nullable text columns stay None."""
    test_split = split_feat_churn(conn).test
    columns = [schema["identifier_column"], *[column["name"] for column in schema["columns"]]]
    return test_split[columns].copy()


def run_scoring(
    config: dict,
    conn: duckdb.DuckDBPyConnection,
    instant: datetime,
    *,
    run_id: str | None = None,
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> ScoringResult:
    """Score the test population inside a tracked scoring run.

    run_id picks a model run; None means the newest valid one. The outputs
    go to outputs_dir. All four outputs are built in memory before the
    first file is replaced. metrics.json is written last, only after the
    four outputs are published; if that write fails the outputs are kept
    and PostPublicationTrackingError is raised.
    """
    with track_run(config, "score", instant) as run:
        try:
            loaded_run = load_run(config["paths"]["models_dir"], run_id)
            logger.info("Scoring run %s uses model run %s", run.run_id, loaded_run.run_id)

            population = select_scoring_population(conn, loaded_run.schema)
            k_frac = config["business"]["k"]
            table = score_batch(population, loaded_run, k_frac)

            medium_edge = config["business"]["risk_tiers"]["medium_edge"]
            identifier = loaded_run.schema["identifier_column"]
            input_columns = [column["name"] for column in loaded_run.schema["columns"]]
            ids = [str(value) for value in population[identifier].astype(object).tolist()]
            inputs = population[input_columns].set_axis(ids, axis=0)
            drivers = select_top_drivers(averaged_contributions(loaded_run, inputs, ids))
            extended = extend_outreach_table(table, drivers, k_frac, medium_edge)

            assumptions = _economics_assumptions(config)
            results = analyze_economics(
                table, inputs["monthly_charge"], k_frac, config["business"]["k_sensitivity"], assumptions
            )
            economics_text = render_economics_document(
                results, assumptions, k_frac, len(table), run.run_id, loaded_run.run_id
            )

            facts = compute_recommendation_facts(
                table, inputs, drivers, global_importance(loaded_run), k_frac, medium_edge
            )
            recommendations = select_recommendations(facts, results.base_case, DRIVER_LABELS)
            recommendations_text = render_recommendations_document(
                facts,
                recommendations,
                results,
                DRIVER_LABELS,
                k_frac,
                medium_edge,
                run.run_id,
                loaded_run.run_id,
            )

            summary = build_extended_run_summary(
                extended,
                k_frac=k_frac,
                medium_edge=medium_edge,
                model_run_id=loaded_run.run_id,
                scoring_run_id=run.run_id,
                manifest=loaded_run.manifest,
            )
            outreach_path, economics_path, recommendations_path, summary_path = publish_output_set(
                outputs_dir,
                extended,
                economics_text,
                recommendations_text,
                summary,
                k_frac=k_frac,
                medium_edge=medium_edge,
            )
        except Exception as exc:
            logger.error("Scoring run %s failed: %s: %s", run.run_id, type(exc).__name__, exc)
            raise

        customers_flagged = int(table["in_top_k"].sum())
        try:
            run.write_metrics(
                {
                    "k": float(k_frac),
                    "customers_scored": int(len(table)),
                    "customers_flagged": customers_flagged,
                    "model_run_id": loaded_run.run_id,
                }
            )
        except TrackingError as exc:
            message = (
                f"The outputs were generated and published, but tracking failed for "
                f"scoring run {run.run_id}: {exc}"
            )
            logger.error(message)
            raise PostPublicationTrackingError(run.run_id, message) from exc

        return ScoringResult(
            run_id=run.run_id,
            model_run_id=loaded_run.run_id,
            customers_scored=int(len(table)),
            customers_flagged=customers_flagged,
            outreach_path=outreach_path,
            economics_path=economics_path,
            recommendations_path=recommendations_path,
            summary_path=summary_path,
        )


def main(argv: list[str] | None = None) -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="score", description="Score the held-out test split and publish the four outputs."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="model run id to score with (default: the newest valid model run)",
    )
    args = parser.parse_args(argv)

    conn = None
    try:
        config = load_config()
        instant = datetime.now(timezone.utc)
        conn = duckdb.connect(str(config["paths"]["interim_db"]), read_only=True)
        result = run_scoring(
            config, conn, instant, run_id=args.run_id, outputs_dir=DEFAULT_OUTPUTS_DIR
        )
    except InputValidationError as exc:
        logger.error(
            "Input validation failed (class 1). Nothing was published and existing output "
            "files are unchanged. %s",
            exc,
        )
        sys.exit(1)
    except OutputPublicationError as exc:
        logger.error("Output publication failed (class 2b). %s", exc)
        sys.exit(1)
    except PostPublicationTrackingError as exc:
        logger.error("Tracking failed after publication (class 3). %s", exc)
        sys.exit(1)
    except (
        ConfigValidationError,
        TrackingError,
        ModelLoadError,
        ScoringError,
        OutputGenerationError,
        DriverError,
        EconomicsError,
        RecommendationsError,
        duckdb.Error,
    ) as exc:
        logger.error(
            "Scoring failed before anything was published (class 2a): %s: %s. "
            "Existing output files are unchanged.",
            type(exc).__name__,
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error(
            "Unexpected error before anything was published (class 2a): %s: %s. "
            "Existing output files are unchanged.",
            type(exc).__name__,
            exc,
        )
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    logger.info(
        "Scoring complete: run %s, model run %s, %d customers scored, %d flagged, outputs %s, %s, %s and %s",
        result.run_id,
        result.model_run_id,
        result.customers_scored,
        result.customers_flagged,
        result.outreach_path,
        result.economics_path,
        result.recommendations_path,
        result.summary_path,
    )


if __name__ == "__main__":
    main()
