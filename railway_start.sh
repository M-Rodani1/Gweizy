#!/bin/bash

echo "=== Railway Start Script ==="
echo "Checking for libgomp..."
ldconfig -p | grep gomp || echo "libgomp not in ldconfig"
ls -la /usr/lib/x86_64-linux-gnu/libgomp* 2>/dev/null || echo "No libgomp in /usr/lib"

echo "Starting application with WebSocket support..."
cd /app/backend
exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    --workers 1 \
    --timeout 120 \
    --log-level info
