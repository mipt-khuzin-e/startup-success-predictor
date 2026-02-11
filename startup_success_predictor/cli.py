# ruff: noqa: B008
"""Central CLI for startup_success_predictor.

Provides Typer-based commands for training, ONNX export, inference, and data download.
"""

import os
import sys
from pathlib import Path

import polars as pl
import torch
import typer

from startup_success_predictor.config import get_settings
from startup_success_predictor.data.download import main as download_main
from startup_success_predictor.export_onnx import export_to_onnx
from startup_success_predictor.infer import load_model, predict
from startup_success_predictor.train import main as train_main

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

app = typer.Typer(help="CLI entrypoint for the GAN-Augmented Startup Success Predictor")


@app.command()
def train() -> None:
    """Run the full Hydra-based training pipeline."""

    # Clear argv so Hydra doesn't see Typer's subcommand as an override
    sys.argv = sys.argv[:1]
    train_main()


@app.command("export-onnx")
def export_onnx_command(
    checkpoint: Path | None = typer.Option(
        None,
        help="Path to Lightning checkpoint (.ckpt). Auto-detects best checkpoint if omitted.",
    ),
    output: Path | None = typer.Option(
        None,
        help="Path to save ONNX model (default: models/classifier.onnx from settings)",
    ),
    input_dim: int | None = typer.Option(
        None, help="Input feature dimension (auto-detected from checkpoint if omitted)"
    ),
    opset_version: int = typer.Option(18, help="ONNX opset version"),
) -> None:
    """Export trained classifier checkpoint to ONNX."""
    settings = get_settings()

    if checkpoint is None:
        checkpoint_dir = settings.models_dir / "classifier"
        ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
        if not ckpt_files:
            typer.echo(f"No checkpoint files found in {checkpoint_dir}", err=True)
            raise typer.Exit(1)
        # Pick the last checkpoint (best by val_auroc due to ModelCheckpoint sorting)
        checkpoint_path = ckpt_files[-1]
        typer.echo(f"Auto-detected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = checkpoint

    output_path = output or settings.models_dir / "classifier.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_dim=input_dim,
        opset_version=opset_version,
    )


@app.command()
def infer(
    checkpoint: Path | None = typer.Option(
        None, help="Path to Lightning checkpoint (.ckpt). Auto-detects best if omitted."
    ),
    input_csv: Path = typer.Option(..., help="Path to input CSV file"),
    output_csv: Path = typer.Option(
        Path("predictions.csv"), help="Path to save predictions CSV"
    ),
) -> None:
    """Run batch inference on a CSV file.

    NOTE: This currently assumes that the input CSV is already preprocessed
    to match the training features. In production, you should load and apply
    the same preprocessing metadata used during training.
    """
    if checkpoint is None:
        checkpoint_dir = get_settings().models_dir / "classifier"
        ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
        if not ckpt_files:
            typer.echo(f"No checkpoint files found in {checkpoint_dir}", err=True)
            raise typer.Exit(1)
        checkpoint = ckpt_files[-1]
        typer.echo(f"Auto-detected checkpoint: {checkpoint}")

    model = load_model(checkpoint)

    typer.echo(f"Loading input data from: {input_csv}")
    df = pl.read_csv(input_csv)
    typer.echo(f"Input shape: {df.shape}")

    # Simplified path: assumes numeric, preprocessed input
    input_tensor = torch.from_numpy(df.to_numpy()).float()
    preds, probs = predict(model, input_tensor)

    df = df.with_columns(
        [
            pl.Series("prediction", preds.squeeze().numpy()),
            pl.Series("probability", probs.squeeze().numpy()),
        ]
    )

    typer.echo(f"Saving predictions to: {output_csv}")
    df.write_csv(output_csv)
    typer.echo("Predictions saved successfully")

    n_positive = (preds == 1).sum().item()
    typer.echo("Prediction summary:")
    typer.echo(f"  Total samples: {len(df)}")
    typer.echo(
        f"  Predicted successful: {n_positive} ({n_positive / len(df) * 100:.2f}%)"
    )
    typer.echo(
        f"  Predicted unsuccessful: {len(df) - n_positive} ({(len(df) - n_positive) / len(df) * 100:.2f}%)"
    )


@app.command("download-data")
def download_data() -> None:
    """Download startup dataset from Kaggle using configured credentials."""

    download_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    app()
