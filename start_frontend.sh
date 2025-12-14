#!/bin/bash

# Start Frontend Development Server
cd "$(dirname "$0")"

echo "Starting Base Gas Optimizer Frontend..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo ""
fi

echo "Starting Vite development server..."
echo "   Frontend will be available at: http://localhost:3000"
echo ""
echo "Backend should be running on: http://localhost:5001"
echo ""

npm run dev

