"""Inference utilities for startup success prediction."""

import logging
from pathlib import Path

import polars as pl
import torch

from startup_success_predictor.data.preprocessing import polars_to_tensor
from startup_success_predictor.models.classifier_module import ClassifierModule

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path) -> ClassifierModule:
    """Load trained model from checkpoint."""
    logger.info("Loading model from: %s", checkpoint_path)
    model = ClassifierModule.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()
    return model


def preprocess_input(
    df: pl.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    normalization_stats: dict[str, dict[str, float]],
    encoding_meta: dict[str, dict[str, list[str]]],
) -> torch.Tensor:
    """Preprocess input data for inference."""
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
    """Make predictions with model."""
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

    return preds, probs
