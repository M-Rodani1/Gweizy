#!/bin/bash
# Start script for Railway deployment

cd backend

# Start worker in background with nohup to ensure it keeps running
nohup python3 -u worker.py > /tmp/worker.log 2>&1 &

# Give worker a moment to start
sleep 2

# Start gunicorn web server
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
