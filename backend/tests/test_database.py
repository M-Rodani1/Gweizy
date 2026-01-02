#!/usr/bin/env python3
"""
Test database operations
Run: python tests/test_database.py
"""

import sys
sys.path.append('.')

from data.database import DatabaseManager
from datetime import datetime, timedelta

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, passed, message=""):
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - {name}")
    if message:
        print(f"     {message}")

def test_database_connection():
    """Test database connection"""
    try:
        db = DatabaseManager()
        passed = db.session is not None
        print_test('Database Connection', passed)
        return passed
    except Exception as e:
        print_test('Database Connection', False, f"Error: {str(e)}")
        return False

def test_data_exists():
    """Test if historical data exists"""
    try:
        db = DatabaseManager()
        data = db.get_historical_data(hours=24)
        
        count = len(data)
        passed = count > 0
        
        print_test('Historical Data Exists', passed, f"{count} records in last 24h")
        return passed
    except Exception as e:
        print_test('Historical Data Exists', False, f"Error: {str(e)}")
        return False

def test_data_quality():
    """Test data quality"""
    try:
        db = DatabaseManager()
        data = db.get_historical_data(hours=168)  # 7 days
        
        if len(data) < 10:
            print_test('Data Quality', False, "Not enough data (need at least 10 records)")
            return False
        
        # Check for reasonable gas prices (0 < gas < 1000 gwei)
        valid_prices = all(
            0 < d.current_gas < 1000 
            for d in data
        )
        
        # Check timestamps are sequential
        timestamps = [d.timestamp for d in data]
        is_sequential = all(
            timestamps[i] <= timestamps[i+1] 
            for i in range(len(timestamps)-1)
        )
        
        passed = valid_prices and is_sequential
        
        print_test(
            'Data Quality', 
            passed, 
            f"Valid prices: {valid_prices}, Sequential: {is_sequential}"
        )
        return passed
        
    except Exception as e:
        print_test('Data Quality', False, f"Error: {str(e)}")
        return False

def test_sufficient_training_data():
    """Test if we have enough data for ML training"""
    try:
        db = DatabaseManager()
        data = db.get_historical_data(hours=720)  # 30 days
        
        count = len(data)
        # Need at least 100 records for basic training
        passed = count >= 100
        
        print_test(
            'Sufficient Training Data', 
            passed, 
            f"{count} records (need ≥100 for training)"
        )
        return passed
        
    except Exception as e:
        print_test('Sufficient Training Data', False, f"Error: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🗄️  Testing Database{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    tests = [
        test_database_connection,
        test_data_exists,
        test_data_quality,
        test_sufficient_training_data
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    if passed == total:
        print(f"{Colors.GREEN}✅ All database tests passed! ({passed}/{total}){Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ {total - passed} test(s) failed ({passed}/{total}){Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()

