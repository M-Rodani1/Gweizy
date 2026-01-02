#!/usr/bin/env python3
"""
Test all API endpoints to ensure they return valid responses
Run: python tests/test_api_endpoints.py
"""

import requests
import json
import sys
from datetime import datetime

API_BASE = 'http://localhost:5001/api'

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, passed, message=""):
    """Print test result"""
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - {name}")
    if message:
        print(f"     {message}")

def test_health():
    """Test /api/health endpoint"""
    try:
        response = requests.get(f'{API_BASE}/health', timeout=5)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            data.get('status') == 'ok' and
            'timestamp' in data
        )
        
        print_test('Health Check', passed, f"Status: {data.get('status')}")
        return passed
        
    except Exception as e:
        print_test('Health Check', False, f"Error: {str(e)}")
        return False

def test_current_gas():
    """Test /api/current endpoint"""
    try:
        response = requests.get(f'{API_BASE}/current', timeout=5)
        data = response.json()
        
        required_fields = ['current_gas', 'base_fee', 'priority_fee', 'block_number']
        has_fields = all(field in data for field in required_fields)
        
        # Validate data types
        valid_data = (
            isinstance(data.get('current_gas'), (int, float)) and
            data.get('current_gas') > 0 and
            data.get('current_gas') < 1000  # Sanity check
        )
        
        passed = response.status_code == 200 and has_fields and valid_data
        
        print_test(
            'Current Gas', 
            passed, 
            f"Gas: {data.get('current_gas')} gwei, Block: {data.get('block_number')}"
        )
        return passed
        
    except Exception as e:
        print_test('Current Gas', False, f"Error: {str(e)}")
        return False

def test_predictions():
    """Test /api/predictions endpoint"""
    try:
        response = requests.get(f'{API_BASE}/predictions', timeout=10)
        data = response.json()
        
        # Check structure
        has_current = 'current' in data
        has_predictions = 'predictions' in data
        
        if has_predictions:
            predictions = data['predictions']
            has_all_horizons = all(h in predictions for h in ['1h', '4h', '24h'])
        else:
            has_all_horizons = False
        
        passed = (
            response.status_code == 200 and
            has_current and
            has_predictions and
            has_all_horizons
        )
        
        if passed and isinstance(predictions.get('1h'), list):
            pred_1h = predictions['1h'][0].get('predictedGwei', 0) if predictions['1h'] else 0
            print_test('Predictions', passed, f"1h prediction: {pred_1h} gwei")
        else:
            print_test('Predictions', passed)
        
        return passed
        
    except Exception as e:
        print_test('Predictions', False, f"Error: {str(e)}")
        return False

def test_historical():
    """Test /api/historical endpoint"""
    try:
        response = requests.get(f'{API_BASE}/historical?hours=24', timeout=5)
        data = response.json()
        
        has_data = 'data' in data
        has_count = 'count' in data
        
        if has_data:
            data_count = len(data['data'])
            data_valid = data_count > 0
        else:
            data_count = 0
            data_valid = False
        
        passed = (
            response.status_code == 200 and
            has_data and
            has_count and
            data_valid
        )
        
        print_test('Historical Data', passed, f"Records: {data_count}")
        return passed
        
    except Exception as e:
        print_test('Historical Data', False, f"Error: {str(e)}")
        return False

def test_transactions():
    """Test /api/transactions endpoint"""
    try:
        response = requests.get(f'{API_BASE}/transactions?limit=5', timeout=5)
        data = response.json()
        
        has_transactions = 'transactions' in data
        
        if has_transactions:
            tx_count = len(data['transactions'])
            tx_valid = tx_count > 0
        else:
            tx_count = 0
            tx_valid = False
        
        passed = (
            response.status_code == 200 and
            has_transactions and
            tx_valid
        )
        
        print_test('Transactions', passed, f"Count: {tx_count}")
        return passed
        
    except Exception as e:
        print_test('Transactions', False, f"Error: {str(e)}")
        return False

def test_config():
    """Test /api/config endpoint"""
    try:
        response = requests.get(f'{API_BASE}/config', timeout=5)
        data = response.json()
        
        required_fields = ['name', 'chainId', 'version']
        has_fields = all(field in data for field in required_fields)
        
        correct_chain = data.get('chainId') == 8453
        
        passed = (
            response.status_code == 200 and
            has_fields and
            correct_chain
        )
        
        print_test('Config', passed, f"Chain ID: {data.get('chainId')}")
        return passed
        
    except Exception as e:
        print_test('Config', False, f"Error: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🧪 Testing Base Gas Optimizer API Endpoints{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    print(f"{Colors.YELLOW}Testing API at: {API_BASE}{Colors.END}\n")
    
    # Run all tests
    tests = [
        test_health,
        test_current_gas,
        test_predictions,
        test_historical,
        test_transactions,
        test_config
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()  # Blank line between tests
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}✅ All tests passed! ({passed}/{total}){Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ {total - passed} test(s) failed ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.YELLOW}💡 Tips:{Colors.END}")
        print(f"  - Make sure backend is running: python app.py")
        print(f"  - Check models are trained: python scripts/train_model.py")
        print(f"  - Verify database has data: Check data collection")
        sys.exit(1)

if __name__ == "__main__":
    main()

