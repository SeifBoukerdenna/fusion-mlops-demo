#!/bin/bash
set -e

echo "🚀 Fly.io Deployment Setup"
echo "================================"

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Installing..."
    curl -L https://fly.io/install.sh | sh
    echo "✓ flyctl installed"
fi

# Check if logged in
echo ""
echo "Checking Fly.io authentication..."
if ! flyctl auth whoami &> /dev/null; then
    echo "Please login to Fly.io:"
    flyctl auth login
fi

echo "✓ Authenticated"
echo ""

# Create FastAPI app
echo "1️⃣  Setting up FastAPI service..."
if flyctl apps list | grep -q "fusion-mlops-fastapi"; then
    echo "   App already exists"
else
    flyctl apps create fusion-mlops-fastapi --org personal
    echo "✓ Created fusion-mlops-fastapi"
fi

# Create TorchServe app
echo ""
echo "2️⃣  Setting up TorchServe service..."
if flyctl apps list | grep -q "fusion-mlops-torchserve"; then
    echo "   App already exists"
else
    flyctl apps create fusion-mlops-torchserve --org personal
    echo "✓ Created fusion-mlops-torchserve"
fi

# Get API token
echo ""
echo "3️⃣  Getting Fly.io API token..."
FLY_TOKEN=$(flyctl auth token)

echo ""
echo "================================"
echo "✓ Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Add this secret to GitHub:"
echo "   Go to: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/settings/secrets/actions"
echo "   Name: FLY_API_TOKEN"
echo "   Value: $FLY_TOKEN"
echo ""
echo "2. Copy fly.io configs to your repo:"
echo "   cp fly.fastapi.toml /path/to/repo/"
echo "   cp fly.torchserve.toml /path/to/repo/"
echo ""
echo "3. Update CI/CD workflow:"
echo "   cp ml-cicd-flyio.yml .github/workflows/ml-cicd.yml"
echo ""
echo "4. Push to GitHub:"
echo "   git add fly.*.toml .github/workflows/ml-cicd.yml"
echo "   git commit -m 'feat: add Fly.io deployment'"
echo "   git push"
echo ""
echo "Your apps will be available at:"
echo "  FastAPI: https://fusion-mlops-fastapi.fly.dev"
echo "  TorchServe: https://fusion-mlops-torchserve.fly.dev"
echo ""