#!/bin/bash
# Railway startup script - Single service with background data collection

cd /app/backend || exit 1

echo "=== Starting Gas Fees ML Service ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"

# Check if model training should be triggered on this deploy
# Support both TRAIN_MODELS (new) and RETRAIN_ON_DEPLOY (legacy) for backward compatibility
if [ "$TRAIN_MODELS" = "true" ] || [ "$RETRAIN_ON_DEPLOY" = "true" ]; then
    echo "=== Model Training Triggered (TRAIN_MODELS or RETRAIN_ON_DEPLOY) ==="
    echo "=== Note: Training will run in background thread after app starts ==="
    echo "=== Training progress will be visible in application logs ==="
fi

# Start gunicorn with config that includes background data collection
# The app.py will check TRAIN_MODELS and trigger training automatically
exec gunicorn app:app --config gunicorn_config.py
