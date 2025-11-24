#!/bin/bash
# Cleanup script - removes CI/CD, shell scripts, and fly.io configs

set -e

echo "🧹 Cleaning up repository..."
echo ""

# CI/CD
if [ -d ".github" ]; then
    rm -rf .github/
    echo "✓ Removed .github/"
fi

# Scripts directory
if [ -d "scripts" ]; then
    rm -rf scripts/
    echo "✓ Removed scripts/"
fi

# Root shell scripts
for file in quickstart.sh setup-flyio.sh setup-frontend.sh; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "✓ Removed $file"
    fi
done

# TorchServe package script
if [ -f "torchserve/package_model.sh" ]; then
    rm torchserve/package_model.sh
    echo "✓ Removed torchserve/package_model.sh"
fi

# Fly.io configs
for file in fly.fastapi.toml fly.torchserve.toml fly.frontend.toml; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "✓ Removed $file"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining structure:"
echo "  - models/ (training, export, validation scripts)"
echo "  - fastapi_app/ (API service)"
echo "  - torchserve/ (handler and config)"
echo "  - k8s/ (Kubernetes manifests)"
echo "  - Docker files and compose"
echo "  - Makefile for local development"