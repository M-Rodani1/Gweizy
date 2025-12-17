#!/bin/bash
# Build script to train spike detection models on Render

echo "🤖 Training spike detection models..."

cd /opt/render/project/src/backend || exit 1

# Train spike detectors
python3 scripts/train_spike_detector.py

if [ $? -eq 0 ]; then
    echo "✅ Spike detection models trained successfully"
    ls -lh models/saved_models/spike_detector*.pkl
else
    echo "❌ Model training failed"
    exit 1
fi
