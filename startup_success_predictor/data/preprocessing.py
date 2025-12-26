"""Data preprocessing utilities using Polars."""

from datetime import date
from typing import Any

import polars as pl
import torch
from torch import Tensor


def encode_categorical(
    df: pl.DataFrame,
    categorical_cols: list[str],
    method: str = "onehot",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Encode categorical variables.

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        method: Encoding method ('onehot' or 'label')

    Returns:
        Tuple of (encoded DataFrame, encoding metadata)
    """
    encoding_meta: dict[str, Any] = {}

    if method == "onehot":
        # Use Polars to_dummies for one-hot encoding
        df_encoded = df.to_dummies(columns=categorical_cols, drop_first=False)
        encoding_meta["method"] = "onehot"
        encoding_meta["columns"] = categorical_cols
    elif method == "label":
        # Label encoding using categorical mapping
        df_encoded = df.clone()
        for col in categorical_cols:
            # Get unique categories
            categories = df[col].unique().sort().to_list()
            # Create mapping
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            encoding_meta[col] = {"categories": categories, "mapping": mapping}

            # Apply mapping
            df_encoded = df_encoded.with_columns(
                pl.col(col).replace_strict(mapping, default=None).alias(col)
            )
        encoding_meta["method"] = "label"
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df_encoded, encoding_meta


def normalize_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """
    Normalize numerical features using Z-score normalization.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        stats: Pre-computed statistics (mean, std). If None, compute from data.

    Returns:
        Tuple of (normalized DataFrame, statistics dict)
    """
    if stats is None:
        # Compute statistics
        stats = {}
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()

            if isinstance(mean, int | float):
                safe_mean = float(mean)
            else:
                safe_mean = 0.0

            if isinstance(std, int | float) and std > 0:
                safe_std = float(std)
            else:
                safe_std = 1.0
            stats[col] = {
                "mean": safe_mean,
                "std": safe_std,
            }

    # Apply normalization
    df_normalized = df.clone()
    for col in feature_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df_normalized = df_normalized.with_columns(
            ((pl.col(col) - mean) / std).alias(col)
        )

    return df_normalized, stats


def handle_missing_values(
    df: pl.DataFrame,
    strategy: str = "drop",
    fill_value: float | None = None,
) -> pl.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'fill')
        fill_value: Value to fill if strategy is 'fill'

    Returns:
        DataFrame with missing values handled
    """
    if strategy == "drop":
        return df.drop_nulls()
    elif strategy == "mean":
        return df.fill_null(strategy="mean")
    elif strategy == "median":
        return df.with_columns(pl.all().fill_null(pl.all().median()))
    elif strategy == "fill":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'fill'")
        return df.fill_null(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def temporal_split(
    df: pl.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data temporally into train, validation, and test sets.

    Args:
        df: Input DataFrame
        date_col: Name of date column
        train_end: End date for training set (exclusive)
        val_end: End date for validation set (exclusive)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure date column is datetime
    if df[date_col].dtype != pl.Date and df[date_col].dtype != pl.Datetime:
        df = df.with_columns(pl.col(date_col).str.to_date())

    # Convert string boundaries to native date objects for safe comparison
    train_end_date = date.fromisoformat(train_end)
    val_end_date = date.fromisoformat(val_end)

    train_df = df.filter(pl.col(date_col) < train_end_date)
    val_df = df.filter(
        (pl.col(date_col) >= train_end_date) & (pl.col(date_col) < val_end_date)
    )
    test_df = df.filter(pl.col(date_col) >= val_end_date)

    return train_df, val_df, test_df


def polars_to_tensor(df: pl.DataFrame, columns: list[str] | None = None) -> Tensor:
    """
    Convert Polars DataFrame to PyTorch Tensor.

    Args:
        df: Input DataFrame
        columns: List of columns to convert. If None, use all columns.

    Returns:
        PyTorch Tensor
    """
    if columns is not None:
        df = df.select(columns)

    # Convert to numpy then to tensor
    return torch.from_numpy(df.to_numpy()).float()
