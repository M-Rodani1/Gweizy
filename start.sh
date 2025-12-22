#!/bin/bash
# Start script for Railway deployment

cd backend

# Use gunicorn with preload to ensure background threads start
exec gunicorn app:app \
  --bind 0.0.0.0:$PORT \
  --workers 1 \
  --threads 4 \
  --timeout 120 \
  --preload
