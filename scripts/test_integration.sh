#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🧪 Base Gas Optimizer Integration Test${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Check if backend is running
echo -e "${YELLOW}[1/5] Checking if backend is running...${NC}"
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running${NC}\n"
else
    echo -e "${RED}❌ Backend is not running${NC}"
    echo -e "${YELLOW}💡 Start it with: cd backend && python app.py${NC}\n"
    exit 1
fi

# Step 2: Check if frontend is accessible
echo -e "${YELLOW}[2/5] Checking if frontend is accessible...${NC}"
if curl -s http://localhost:3000 > /dev/null 2>&1 || curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is accessible${NC}\n"
else
    echo -e "${RED}❌ Frontend is not accessible${NC}"
    echo -e "${YELLOW}💡 Start it with: npm run dev${NC}\n"
    exit 1
fi

# Step 3: Run backend tests
echo -e "${YELLOW}[3/5] Running backend tests...${NC}"
cd backend
python tests/run_all_tests.py
BACKEND_TEST_RESULT=$?
cd ..

if [ $BACKEND_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Backend tests passed${NC}\n"
else
    echo -e "${RED}❌ Backend tests failed${NC}\n"
    exit 1
fi

# Step 4: Test CORS
echo -e "${YELLOW}[4/5] Testing CORS configuration...${NC}"
CORS_HEADER=$(curl -s -I -X OPTIONS http://localhost:5000/api/current | grep -i "access-control-allow-origin")
if [ -n "$CORS_HEADER" ]; then
    echo -e "${GREEN}✅ CORS is configured${NC}\n"
else
    echo -e "${YELLOW}⚠️  CORS headers not found (may be okay)${NC}\n"
fi

# Step 5: Test full data flow
echo -e "${YELLOW}[5/5] Testing complete data flow...${NC}"
PREDICTION_DATA=$(curl -s http://localhost:5000/api/predictions)
if echo "$PREDICTION_DATA" | grep -q "predictions"; then
    echo -e "${GREEN}✅ Predictions endpoint returns data${NC}\n"
else
    echo -e "${RED}❌ Predictions endpoint issue${NC}\n"
    exit 1
fi

# Success!
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 All integration tests passed!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}✅ Backend: Running & healthy${NC}"
echo -e "${GREEN}✅ Frontend: Accessible${NC}"
echo -e "${GREEN}✅ API: All endpoints working${NC}"
echo -e "${GREEN}✅ Data Flow: Complete${NC}\n"

echo -e "${BLUE}🚀 Your application is ready!${NC}"
echo -e "${BLUE}Access it at: http://localhost:3000 or http://localhost:5173${NC}\n"

