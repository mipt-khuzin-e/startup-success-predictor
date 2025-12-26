"""Export trained model to ONNX format."""

import argparse
from pathlib import Path

import torch

from startup_success_predictor.config import get_settings
from startup_success_predictor.models.classifier_module import ClassifierModule


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int,
    opset_version: int = 14,
) -> None:
    """
    Export PyTorch Lightning model to ONNX format.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        input_dim: Input dimension for the model
        opset_version: ONNX opset version
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load model from checkpoint
    model = ClassifierModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model exported successfully to: {output_path}")

    # Validate ONNX model
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")


def main() -> None:
    """Main entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save ONNX model (default: models/classifier.onnx)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Input dimension of the model",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    settings = get_settings()
    checkpoint_path = Path(args.checkpoint)

    if args.output is None:
        output_path = settings.models_dir / "classifier.onnx"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_dim=args.input_dim,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
