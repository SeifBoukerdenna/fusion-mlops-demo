#!/bin/bash
set -e

echo "🎨 Setting up Frontend on Fly.io"

# Create app
flyctl apps create fusion-mlops-ui --org personal || echo "App exists"

# Get token (if needed)
echo ""
echo "Add to GitHub Secrets if not already done:"
echo "Name: FLY_API_TOKEN"
echo "Value: $(flyctl auth token)"
echo ""
echo "Frontend will be at: https://fusion-mlops-ui.fly.dev"