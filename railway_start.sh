#!/bin/bash

echo "=== Railway Start Script ==="
echo "Checking for libgomp..."
ldconfig -p | grep gomp || echo "libgomp not in ldconfig"
ls -la /usr/lib/x86_64-linux-gnu/libgomp* 2>/dev/null || echo "No libgomp in /usr/lib"

# Ensure /data/models directory exists for persistent model storage
echo "Setting up model directories..."
mkdir -p /data/models

# Download models if not present
echo "Checking/downloading ML models..."
cd /app/backend
python scripts/download_models.py || echo "Model download completed (may have warnings)"

# Create fallback spike detectors if models don't exist
if [ ! -f /data/models/spike_detector_1h.pkl ]; then
    echo "Creating fallback spike detectors..."
    python scripts/create_fallback_spike_detectors.py || echo "Fallback creation completed"
    # Copy to persistent storage
    cp -v models/saved_models/spike_detector_*.pkl /data/models/ 2>/dev/null || true
fi

echo "Starting application with WebSocket support..."
exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    --workers 1 \
    --timeout 120 \
    --log-level info
