"""Build train/val/test TensorDatasets and metadata from startup CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict
from torch.utils.data import TensorDataset

from startup_success_predictor.data.preprocessing import (
    encode_categorical,
    handle_missing_values,
    normalize_features,
    polars_to_tensor,
    temporal_split,
)

_SUCCESS_STATUSES = ("acquired", "ipo", "operating")
_NUMERICAL_DTYPES = (pl.Int64, pl.Int32, pl.Float64, pl.Float32)


def _add_binary_target(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Add binary 'target' column: 1 for success statuses, 0 otherwise."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    return df.with_columns(
        pl.when(pl.col(target_col).is_in(list(_SUCCESS_STATUSES)))
        .then(1)
        .otherwise(0)
        .alias("target")
    )


def _numerical_feature_columns(
    df: pl.DataFrame,
    exclude_cols: set[str],
    categorical_cols: list[str],
) -> list[str]:
    """Return column names that are numerical and not in categorical_cols."""
    candidate = set(df.columns) - exclude_cols
    return [
        col
        for col in candidate
        if col not in categorical_cols and df[col].dtype in _NUMERICAL_DTYPES
    ]


def _apply_label_encoding_to_splits(
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    categorical_cols: list[str],
    encoding_meta: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply label encoding mappings to val and test (unseen categories → -1)."""
    if encoding_meta.get("method") != "label":
        return val_df, test_df
    for col in categorical_cols:
        if col not in encoding_meta:
            continue
        mapping = encoding_meta[col]["mapping"]
        val_df = val_df.with_columns(
            pl.col(col).replace_strict(mapping, default=-1).alias(col)
        )
        test_df = test_df.with_columns(
            pl.col(col).replace_strict(mapping, default=-1).alias(col)
        )
    return val_df, test_df


def _dataframe_to_tensor_dataset(
    df: pl.DataFrame, feature_cols: list[str]
) -> TensorDataset:
    """Convert a single DataFrame to a (features, labels) TensorDataset."""
    features = polars_to_tensor(df, feature_cols)
    labels = polars_to_tensor(df, ["target"]).squeeze(-1)
    return TensorDataset(features, labels)


def _build_tensor_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Convert three DataFrames to (features, labels) TensorDatasets."""
    return (
        _dataframe_to_tensor_dataset(train_df, feature_cols),
        _dataframe_to_tensor_dataset(val_df, feature_cols),
        _dataframe_to_tensor_dataset(test_df, feature_cols),
    )


class ProcessedData(BaseModel):
    """Result of processing startup CSV: datasets and metadata for training/inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    feature_cols: list[str]
    normalization_stats: dict[str, dict[str, float]]
    encoding_meta: dict[str, Any]


def process_startup_dataset(
    data_path: Path | str,
    train_end: str = "2015-01-01",
    val_end: str = "2018-01-01",
    target_col: str = "status",
    date_col: str = "founded_at",
    categorical_cols: list[str] | None = None,
    handle_missing: str = "drop",
    encoding_method: str = "label",
) -> ProcessedData:
    """Load CSV, split temporally, encode and normalize features, return datasets and metadata."""
    path = Path(data_path)
    df = pl.read_csv(path)
    df = handle_missing_values(df, strategy=handle_missing)
    df = _add_binary_target(df, target_col)

    train_df, val_df, test_df = temporal_split(df, date_col, train_end, val_end)

    exclude_cols = {target_col, "target", date_col}
    cat_cols = categorical_cols or []
    numerical_cols = _numerical_feature_columns(df, exclude_cols, cat_cols)

    encoding_meta: dict[str, Any] = {}
    if cat_cols:
        train_df, encoding_meta = encode_categorical(
            train_df, cat_cols, method=encoding_method
        )
        val_df, test_df = _apply_label_encoding_to_splits(
            val_df, test_df, cat_cols, encoding_meta
        )

    normalization_stats: dict[str, dict[str, float]] = {}
    if numerical_cols:
        train_df, normalization_stats = normalize_features(train_df, numerical_cols)
        val_df, _ = normalize_features(val_df, numerical_cols, normalization_stats)
        test_df, _ = normalize_features(test_df, numerical_cols, normalization_stats)

    feature_cols = numerical_cols + cat_cols
    train_ds, val_ds, test_ds = _build_tensor_datasets(
        train_df, val_df, test_df, feature_cols
    )

    return ProcessedData(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        feature_cols=feature_cols,
        normalization_stats=normalization_stats,
        encoding_meta=encoding_meta,
    )


class DatasetProcessor:
    """Static processor for startup CSV to train/val/test TensorDatasets and metadata."""

    @staticmethod
    def process(
        data_path: Path | str,
        train_end: str = "2015-01-01",
        val_end: str = "2018-01-01",
        target_col: str = "status",
        date_col: str = "founded_at",
        categorical_cols: list[str] | None = None,
        handle_missing: str = "drop",
        encoding_method: str = "label",
    ) -> ProcessedData:
        """Delegate to process_startup_dataset."""
        return process_startup_dataset(
            data_path=data_path,
            train_end=train_end,
            val_end=val_end,
            target_col=target_col,
            date_col=date_col,
            categorical_cols=categorical_cols,
            handle_missing=handle_missing,
            encoding_method=encoding_method,
        )
