#!/bin/bash
# Quick script to check data collection status
# Usage: ./check_data.sh [API_URL]

API_URL="${1:-https://basegasfeesml-production.up.railway.app}"

echo "🔍 Checking data collection status..."
echo "API URL: $API_URL"
echo ""

# Check data quality
echo "📊 Data Quality:"
curl -s "$API_URL/api/pipeline/data-quality" | python3 -m json.tool 2>/dev/null || curl -s "$API_URL/api/pipeline/data-quality"

echo ""
echo ""

# Check training data
echo "🎯 Training Data Status:"
curl -s "$API_URL/api/retraining/check-data" | python3 -m json.tool 2>/dev/null || curl -s "$API_URL/api/retraining/check-data"

echo ""
echo ""

# Check pipeline status
echo "🤖 Pipeline Status:"
curl -s "$API_URL/api/pipeline/status" | python3 -m json.tool 2>/dev/null || curl -s "$API_URL/api/pipeline/status"

