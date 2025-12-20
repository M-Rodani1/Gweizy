#!/bin/bash

# Test script for all API endpoints

BASE_URL="http://localhost:5000"

echo "🧪 Testing Base Gas Optimizer API Endpoints"
echo "============================================"
echo ""

# Health check
echo "1. Testing /api/health"
curl -s "$BASE_URL/api/health" | python3 -m json.tool
echo ""
echo ""

# Current gas
echo "2. Testing /api/current"
curl -s "$BASE_URL/api/current" | python3 -m json.tool | head -20
echo ""
echo ""

# Predictions
echo "3. Testing /api/predictions"
curl -s "$BASE_URL/api/predictions" | python3 -m json.tool | head -30
echo ""
echo ""

# Historical
echo "4. Testing /api/historical?hours=24"
curl -s "$BASE_URL/api/historical?hours=24" | python3 -m json.tool | head -20
echo ""
echo ""

# Transactions
echo "5. Testing /api/transactions"
curl -s "$BASE_URL/api/transactions?limit=5" | python3 -m json.tool
echo ""
echo ""

# Stats
echo "6. Testing /api/stats"
curl -s "$BASE_URL/api/stats?hours=24" | python3 -m json.tool
echo ""
echo ""

# Config
echo "7. Testing /api/config"
curl -s "$BASE_URL/api/config" | python3 -m json.tool
echo ""
echo ""

# Accuracy
echo "8. Testing /api/accuracy"
curl -s "$BASE_URL/api/accuracy" | python3 -m json.tool
echo ""
echo ""

echo "✅ All endpoint tests completed!"
echo ""
echo "💡 Check logs/ directory for request logs"
echo "💡 Check for 'Cache HIT' messages in logs to verify caching"

