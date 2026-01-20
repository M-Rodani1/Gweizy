#!/bin/bash
# Download gas_data.db from Railway volume

echo "📥 Downloading database from Railway..."
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found!"
    echo ""
    echo "Install it with:"
    echo "  npm i -g @railway/cli"
    echo ""
    echo "Or use one of these methods:"
    echo ""
    echo "Method 1: Railway Dashboard"
    echo "  1. Go to https://railway.app"
    echo "  2. Select your project"
    echo "  3. Go to your service → Volumes"
    echo "  4. Click on the volume"
    echo "  5. Navigate to backend/gas_data.db"
    echo "  6. Download the file"
    echo ""
    echo "Method 2: Railway CLI (after installing)"
    echo "  1. railway login"
    echo "  2. railway link"
    echo "  3. railway run cat backend/gas_data.db > gas_data.db"
    echo ""
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "⚠️  Not logged in to Railway"
    echo "Running: railway login"
    railway login
fi

# Check if project is linked
if [ ! -f .railway.json ] && [ ! -f railway.json ]; then
    echo "⚠️  Project not linked to Railway"
    echo "Running: railway link"
    railway link
fi

# Download the database
echo "Downloading backend/gas_data.db..."
railway run cat backend/gas_data.db > gas_data.db

if [ -f gas_data.db ]; then
    file_size=$(du -h gas_data.db | cut -f1)
    echo ""
    echo "✅ Successfully downloaded gas_data.db"
    echo "   File size: $file_size"
    echo "   Location: $(pwd)/gas_data.db"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Upload this file to Google Colab"
    echo "   2. Or use it locally for training"
else
    echo ""
    echo "❌ Failed to download database"
    echo "   Check that:"
    echo "   - You're in the correct project"
    echo "   - The volume is mounted correctly"
    echo "   - The file path is backend/gas_data.db"
    exit 1
fi
