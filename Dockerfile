# Use Python 3.11 slim as base
FROM python:3.11-slim

# Install system dependencies including libgomp
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create data directory for SQLite
RUN mkdir -p /data

# Expose port
EXPOSE 8080

# Start command
WORKDIR /app/backend
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "120"]
