#!/bin/bash

# Package PyTorch model for TorchServe
# Creates a .mar (Model Archive) file

set -e

echo "=========================================="
echo "Packaging Model for TorchServe"
echo "=========================================="

# Paths
MODEL_NAME="defect_classifier"
MODEL_PATH="../models/model_artifacts/resnet18_neu.pth"
HANDLER_PATH="handler.py"
EXPORT_PATH="."
VERSION="1.0"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Run 'python models/train_defect_classifier.py' first"
    exit 1
fi

# Check if torch-model-archiver is installed
if ! command -v torch-model-archiver &> /dev/null; then
    echo "Error: torch-model-archiver not found"
    echo "Install with: pip install torch-model-archiver"
    exit 1
fi

echo "Creating model archive..."
echo "  Model: $MODEL_PATH"
echo "  Handler: $HANDLER_PATH"
echo "  Version: $VERSION"

# Package model
torch-model-archiver \
    --model-name "$MODEL_NAME" \
    --version "$VERSION" \
    --serialized-file "$MODEL_PATH" \
    --handler "$HANDLER_PATH" \
    --export-path "$EXPORT_PATH" \
    --force

MAR_FILE="${EXPORT_PATH}/${MODEL_NAME}.mar"

if [ -f "$MAR_FILE" ]; then
    FILE_SIZE=$(du -h "$MAR_FILE" | cut -f1)
    echo ""
    echo "=========================================="
    echo "✓ Model archive created successfully!"
    echo "  File: $MAR_FILE"
    echo "  Size: $FILE_SIZE"
    echo "=========================================="
    echo ""
    echo "To serve the model:"
    echo "  torchserve --start --model-store . --models defect_classifier.mar --ts-config config.properties"
else
    echo "Error: Failed to create model archive"
    exit 1
fi