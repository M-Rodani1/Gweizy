#!/bin/bash
# Railway startup script

# Debug: Check if files exist
echo "Checking files..."
ls -la /app/backend/app.py || echo "app.py not found!"
pwd

# Change to backend directory and ensure it's in sys.path
cd /app/backend
export PYTHONPATH=/app/backend:${PYTHONPATH}

# Debug: Show python path
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"
python3 -c "import sys; print('Python path:', sys.path)"

# Start gunicorn with current directory as module base
exec python3 -m gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --preload
