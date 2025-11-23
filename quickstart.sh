#!/bin/bash

set -e

echo "========================================"
echo "M1 Mac MLOps Setup - Quick Start"
echo "========================================"
echo ""

# Check if model exists
if [ ! -f "models/model_artifacts/resnet18_neu.pth" ]; then
    echo "📦 Training model..."
    python models/train_defect_classifier.py
    echo "✓ Model trained"
    echo ""
fi

# Export to ONNX
if [ ! -f "models/model_artifacts/resnet18_neu.onnx" ]; then
    echo "📤 Exporting to ONNX..."
    python models/export_to_onnx.py
    echo "✓ Model exported"
    echo ""
fi

# Validate
echo "✅ Validating model..."
python models/validate_onnx.py
echo ""

# Build Docker images
echo "🐳 Building Docker images..."
docker compose build
echo "✓ Images built"
echo ""

# Start services
echo "🚀 Starting services..."
docker compose up -d
echo "✓ Services started"
echo ""

# Wait for services to be ready
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Test endpoints
echo ""
echo "🧪 Testing endpoints..."
echo ""
echo "FastAPI Health:"
curl -s http://localhost:8000/health | jq || echo "FastAPI not ready yet"
echo ""
echo "TorchServe Health:"
curl -s http://localhost:8080/ping || echo "TorchServe not ready yet"
echo ""
echo ""

echo "========================================"
echo "✓ Setup Complete!"
echo "========================================"
echo ""
echo "Services running:"
echo "  FastAPI:    http://localhost:8000"
echo "  TorchServe: http://localhost:8080"
echo ""
echo "Next steps:"
echo "  1. Start test UI: python serve_ui.py"
echo "  2. Open http://localhost:3000/test_ui.html"
echo "  3. Upload an image to test"
echo ""
echo "View logs: docker compose logs -f"
echo "Stop:      docker compose down"
echo ""