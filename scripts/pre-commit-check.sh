#!/bin/bash
# Pre-commit validation script
# Run before pushing: ./scripts/pre-commit-check.sh

set -e

echo "🔍 Running pre-commit checks..."
echo ""

# Check Python syntax
echo "1. Checking Python syntax..."
python -m py_compile models/*.py fastapi_app/*.py 2>/dev/null || {
    echo "✗ Python syntax errors found"
    exit 1
}
echo "✓ Python syntax OK"

# Check if model exists
echo ""
echo "2. Checking model artifacts..."
if [ ! -f "models/model_artifacts/resnet18_neu.onnx" ]; then
    echo "⚠️  ONNX model not found"
    echo "   Run: python models/export_to_onnx.py"
    exit 1
fi
echo "✓ Model artifacts present"

# Validate ONNX model
echo ""
echo "3. Validating ONNX model..."
python models/validate_onnx.py > /dev/null 2>&1 || {
    echo "✗ Model validation failed"
    exit 1
}
echo "✓ Model validation passed"

# Check Docker builds
echo ""
echo "4. Testing Docker builds..."
docker compose config > /dev/null 2>&1 || {
    echo "✗ Docker compose config invalid"
    exit 1
}
echo "✓ Docker config OK"

# Check HuggingFace credentials
echo ""
echo "5. Checking HuggingFace credentials..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set (CI/CD will fail)"
    echo "   Set with: export HF_TOKEN=your_token"
else
    echo "✓ HF_TOKEN configured"
fi

echo ""
echo "=========================================="
echo "✓ All pre-commit checks passed!"
echo "=========================================="
echo ""
echo "Ready to push. Pipeline will:"
echo "  1. Validate model from registry"
echo "  2. Build Docker images"
echo "  3. Run integration tests"
echo "  4. Update model registry"
echo ""