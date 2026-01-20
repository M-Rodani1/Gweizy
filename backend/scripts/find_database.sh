#!/bin/bash
# Find the database file on Railway

echo "🔍 Searching for gas_data.db on Railway..."
echo ""

# Try multiple locations
echo "Checking common locations:"
railway run "ls -lh /data/gas_data.db 2>/dev/null && echo '✅ Found at /data/gas_data.db'" || echo "❌ Not at /data/gas_data.db"
railway run "ls -lh /app/backend/gas_data.db 2>/dev/null && echo '✅ Found at /app/backend/gas_data.db'" || echo "❌ Not at /app/backend/gas_data.db"
railway run "ls -lh /app/gas_data.db 2>/dev/null && echo '✅ Found at /app/gas_data.db'" || echo "❌ Not at /app/gas_data.db"
railway run "ls -lh backend/gas_data.db 2>/dev/null && echo '✅ Found at backend/gas_data.db'" || echo "❌ Not at backend/gas_data.db"

echo ""
echo "Searching entire filesystem for gas_data.db:"
railway run "find / -name 'gas_data.db' -type f 2>/dev/null | head -10"

echo ""
echo "Checking /data directory contents:"
railway run "ls -lah /data/ 2>/dev/null || echo 'Cannot access /data'"

echo ""
echo "Checking current working directory:"
railway run "pwd && ls -lah . 2>/dev/null"
