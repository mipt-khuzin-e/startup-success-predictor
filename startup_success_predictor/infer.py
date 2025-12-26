"""Inference script for startup success prediction."""

import argparse
from pathlib import Path

import polars as pl
import torch

from startup_success_predictor.data.preprocessing import (
    polars_to_tensor,
)
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
                df = df.with_columns(pl.col(col).replace_strict(mapping, default=None).alias(col))

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


def main() -> None:
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference on startup data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (default: predictions.csv)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    input_path = Path(args.input)

    if args.output is None:
        output_path = Path("predictions.csv")
    else:
        output_path = Path(args.output)

    # Load model
    model = load_model(checkpoint_path)

    # Load input data
    print(f"Loading input data from: {input_path}")
    df = pl.read_csv(input_path)
    print(f"Input shape: {df.shape}")

    # Note: In a real scenario, you would need to load preprocessing metadata
    # (feature_cols, categorical_cols, normalization_stats, encoding_meta)
    # from the training run. For now, this is a placeholder.
    print("\nWarning: Preprocessing metadata should be loaded from training run.")
    print("This is a simplified example.")

    # Make predictions (simplified - assumes preprocessed input)
    # In production, you would apply the same preprocessing as during training
    input_tensor = torch.from_numpy(df.to_numpy()).float()
    preds, probs = predict(model, input_tensor)

    # Add predictions to DataFrame
    df = df.with_columns(
        [
            pl.Series("prediction", preds.squeeze().numpy()),
            pl.Series("probability", probs.squeeze().numpy()),
        ]
    )

    # Save predictions
    print(f"\nSaving predictions to: {output_path}")
    df.write_csv(output_path)
    print("Predictions saved successfully!")

    # Print summary
    n_positive = (preds == 1).sum().item()
    print("\nPrediction summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Predicted successful: {n_positive} ({n_positive / len(df) * 100:.2f}%)")
    print(
        f"  Predicted unsuccessful: {len(df) - n_positive} ({(len(df) - n_positive) / len(df) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
