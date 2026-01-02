#!/bin/bash
# Start script for Railway deployment

cd backend

# Use gunicorn with config file
exec gunicorn app:app --config gunicorn_config.py
