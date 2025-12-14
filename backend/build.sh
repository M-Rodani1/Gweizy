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

echo "Build complete!"

