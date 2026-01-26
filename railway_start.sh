#!/bin/bash

echo "=== Railway Start Script ==="
echo "Checking for libgomp..."
ldconfig -p | grep gomp || echo "libgomp not in ldconfig"
ls -la /usr/lib/x86_64-linux-gnu/libgomp* 2>/dev/null || echo "No libgomp in /usr/lib"

echo "Starting application..."
cd /app/backend
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 2 --timeout 120
