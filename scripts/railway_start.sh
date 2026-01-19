#!/bin/bash
# Railway startup script - Single service with background data collection

cd /app/backend || exit 1

echo "=== Starting Gas Fees ML Service ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"

# Check if retraining should be triggered on this deploy
if [ "$RETRAIN_ON_DEPLOY" = "true" ]; then
    echo "=== Triggering Model Retraining ==="
    python3 scripts/retrain_models_simple.py || echo "Retraining failed, continuing with existing models"
    echo "=== Retraining Complete ==="
fi

# Start gunicorn with config that includes background data collection
exec gunicorn app:app --config gunicorn_config.py
