#!/usr/bin/env python3
"""
Master test runner - runs all tests
Run: python tests/run_all_tests.py
"""

import subprocess
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def run_test_file(filepath, name):
    """Run a test file and return pass/fail"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Running: {name}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    import os
    import sys
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    test_path = os.path.join(backend_dir, filepath)
    
    result = subprocess.run(['python3', test_path], cwd=backend_dir)
    passed = result.returncode == 0
    
    return passed

def main():
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}🚀 Base Gas Optimizer - Complete Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
    
    tests = [
        ('tests/test_database.py', 'Database Tests'),
        ('tests/test_models.py', 'ML Model Tests'),
        ('tests/test_api_endpoints.py', 'API Endpoint Tests'),
    ]
    
    results = []
    
    for filepath, name in tests:
        try:
            passed = run_test_file(filepath, name)
            results.append((name, passed))
        except FileNotFoundError:
            print(f"{Colors.RED}❌ Test file not found: {filepath}{Colors.END}")
            results.append((name, False))
        except Exception as e:
            print(f"{Colors.RED}❌ Error running {name}: {str(e)}{Colors.END}")
            results.append((name, False))
    
    # Final summary
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}📊 Final Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
    
    for name, passed in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"{status} - {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    
    if total_passed == total_tests:
        print(f"{Colors.GREEN}🎉 All test suites passed! ({total_passed}/{total_tests}){Colors.END}")
        print(f"{Colors.GREEN}✅ Your application is ready to deploy!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ {total_tests - total_passed} test suite(s) failed{Colors.END}")
        print(f"\n{Colors.YELLOW}Next steps:{Colors.END}")
        print(f"  1. Review failed tests above")
        print(f"  2. Fix issues")
        print(f"  3. Run: python tests/run_all_tests.py")
        sys.exit(1)

if __name__ == "__main__":
    main()

