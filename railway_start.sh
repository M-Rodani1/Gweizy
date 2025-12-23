#!/bin/bash
# Railway startup script

# Change to backend directory
cd /app/backend

# Add current directory to PYTHONPATH so Python can find app module
export PYTHONPATH=/app/backend:$PYTHONPATH

# Start gunicorn
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --preload
