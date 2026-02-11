"""Export trained model to ONNX format."""

import logging
from pathlib import Path

import onnx
import torch

from startup_success_predictor.models.classifier_module import ClassifierModule

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int | None = None,
    opset_version: int = 18,
) -> None:
    """Export PyTorch Lightning checkpoint to ONNX."""
    logger.info("Loading model from: %s", checkpoint_path)

    # Load model from checkpoint and move to CPU for ONNX export
    model = ClassifierModule.load_from_checkpoint(
        checkpoint_path, weights_only=False, map_location="cpu"
    )
    model.cpu()
    model.eval()

    # Auto-detect input_dim from model if not provided
    if input_dim is None:
        input_dim = model.input_dim
    logger.info("Using input_dim=%d", input_dim)

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export to ONNX
    logger.info("Exporting to ONNX: %s", output_path)
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

    logger.info("Model exported successfully to: %s", output_path)

    # Validate ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed!")
