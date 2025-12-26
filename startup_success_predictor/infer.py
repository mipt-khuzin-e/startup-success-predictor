"""Inference utilities for startup success prediction.

This module provides reusable functions for loading a model, preprocessing
inputs, and running predictions. CLI entrypoints are implemented in
``startup_success_predictor.cli`` (Typer-based).
"""

from pathlib import Path

import polars as pl
import torch

from startup_success_predictor.data.preprocessing import polars_to_tensor
from startup_success_predictor.models.classifier_module import ClassifierModule


def load_model(checkpoint_path: Path) -> ClassifierModule:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    model = ClassifierModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def preprocess_input(
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    normalization_stats: dict[str, dict[str, float]],
    encoding_meta: dict[str, dict[str, list[str]]],
) -> torch.Tensor:
    """
    Preprocess input data for inference.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        categorical_cols: List of categorical column names
        normalization_stats: Normalization statistics
        encoding_meta: Encoding metadata

    Returns:
        Preprocessed tensor
    """
    # Encode categorical variables
    if categorical_cols:
        for col in categorical_cols:
            if col in encoding_meta:
                mapping = encoding_meta[col]["mapping"]
                df = df.with_columns(
                    pl.col(col).replace_strict(mapping, default=None).alias(col)
                )

    # Normalize features
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    if numerical_cols:
        for col in numerical_cols:
            if col in normalization_stats:
                mean = normalization_stats[col]["mean"]
                std = normalization_stats[col]["std"]
                df = df.with_columns(((pl.col(col) - mean) / std).alias(col))

    # Convert to tensor
    return polars_to_tensor(df, feature_cols)


def predict(
    model: ClassifierModule,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions.

    Args:
        model: Trained model
        input_tensor: Input tensor

    Returns:
        Tuple of (predictions, probabilities)
    """
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

    return preds, probs


# NOTE:
# This module intentionally does not provide a standalone CLI. Use the
# Typer-based ``infer`` command in ``startup_success_predictor.cli`` for
# end-user inference from the command line.
