#!/bin/bash
# Convert ONNX model to TensorRT engine
# Requires: TensorRT installed and trtexec available

set -e

# Default paths
ONNX_MODEL="models/classifier.onnx"
TRT_ENGINE="models/classifier.engine"
BATCH_SIZE=1
WORKSPACE=1024  # MB

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx)
            ONNX_MODEL="$2"
            shift 2
            ;;
        --output)
            TRT_ENGINE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Converting ONNX model to TensorRT..."
echo "  Input: $ONNX_MODEL"
echo "  Output: $TRT_ENGINE"
echo "  Batch size: $BATCH_SIZE"
echo "  Workspace: ${WORKSPACE}MB"

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "Error: trtexec not found. Please install TensorRT."
    exit 1
fi

# Convert to TensorRT
# Note: we rely on the shapes embedded in the ONNX model and use explicit batch.
# If you need custom shapes, adjust the trtexec invocation accordingly.
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_ENGINE" \
    --explicitBatch \
    --workspace=$WORKSPACE \
    --fp16

echo "Conversion completed successfully!"
echo "TensorRT engine saved to: $TRT_ENGINE"
