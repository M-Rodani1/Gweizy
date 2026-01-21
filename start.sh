#!/bin/bash
# Start script for Railway deployment
# Railway looks for this in the root directory

cd backend

# Use gunicorn with config file
exec gunicorn app:app --config gunicorn_config.py
