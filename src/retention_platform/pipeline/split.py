"""Stratified train/test split of feat_churn.

Loads feat_churn from the DuckDB database and partitions it into a
stratified train/test split on is_voluntary_churn. test_size and
random_state come from config/config.yaml so the split is reproducible
without being persisted to disk -- it is recomputed from feat_churn each
time this is called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from retention_platform.config import load_config

logger = logging.getLogger("retention_platform.pipeline.split")


@dataclass(frozen=True)
class TrainTestSplit:
    """Train and test partitions of feat_churn, kept as separate DataFrames."""

    train: pd.DataFrame
    test: pd.DataFrame


def split_feat_churn(conn: duckdb.DuckDBPyConnection) -> TrainTestSplit:
    """Load feat_churn from conn and return a stratified train/test split."""
    config = load_config()
    test_size = config["data"]["test_size"]
    random_state = config["reproducibility"]["seed"]

    df = conn.execute("SELECT * FROM feat_churn").fetchdf()

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_voluntary_churn"],
    )

    logger.info(
        "Train split: %d rows, churn rate %.2f%%",
        len(train_df),
        train_df["is_voluntary_churn"].mean() * 100,
    )
    logger.info(
        "Test split: %d rows, churn rate %.2f%%",
        len(test_df),
        test_df["is_voluntary_churn"].mean() * 100,
    )

    return TrainTestSplit(train=train_df, test=test_df)
