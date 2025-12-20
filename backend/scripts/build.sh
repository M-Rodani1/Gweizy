#!/bin/bash
# Build script for Render deployment
# This ensures packages install correctly

set -e

echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential python3-dev libpq-dev

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing Python packages..."
pip install --no-cache-dir -r requirements.txt

echo "Training ML models (this may take 2-3 minutes)..."
cd /opt/render/project/src/backend
python3 scripts/train_directional_optimized.py || echo "⚠️  Model training failed, will use fallback predictions"

echo "Build complete!"

