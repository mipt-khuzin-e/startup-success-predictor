"""Export trained model to ONNX format.

This module exposes :func:`export_to_onnx` as a library function. CLI entrypoints
should be implemented in ``startup_success_predictor.cli`` (Typer-based).
"""

from pathlib import Path

import torch

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
        (dummy_input,),
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


# NOTE:
# CLI usage for this functionality is provided by ``startup_success_predictor.cli``
# via the Typer-based ``export-onnx`` command. This module is intentionally
# free of ``argparse`` and top-level side effects so it can be imported safely
# from other parts of the codebase.
