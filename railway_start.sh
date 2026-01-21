#!/bin/bash
# Railway startup script
# This is called by Railway if configured to use a custom start script

cd backend

# Use gunicorn with config file
exec gunicorn app:app --config gunicorn_config.py
