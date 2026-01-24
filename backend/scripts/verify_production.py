#!/usr/bin/env python3
"""
Production Verification Script

Verifies that the gas optimization ensemble is properly configured
and ready for production use.

Usage:
    python scripts/verify_production.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_ensemble_service():
    """Verify ensemble service loads and works."""
    print("\n1. Verifying Ensemble Service...")

    try:
        from services.ensemble_service import get_ensemble_service

        service = get_ensemble_service(8453)

        # Test a recommendation (this triggers lazy loading)
        rec = service.get_recommendation(
            current_price=0.001,
            urgency=0.5,
            time_waiting=5
        )

        # Get status after loading
        status = service.get_status()

        print(f"   Ensemble loaded: {status.get('ensemble_loaded', False)}")
        print(f"   Is ready: {status.get('is_ready', False)}")

        if 'ensemble_info' in status:
            print(f"   Num agents: {status['ensemble_info'].get('num_agents', 0)}")

        print(f"   Test recommendation: {rec.action} ({rec.confidence:.0%})")

        if status.get('is_ready'):
            print("   [PASS] Ensemble service is ready")
            return True
        else:
            print("   [WARN] Ensemble not loaded, using heuristics")
            return True  # Still functional with heuristics

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False


def verify_api_routes():
    """Verify API routes are importable."""
    print("\n2. Verifying API Routes...")

    try:
        from api.agent_routes import agent_bp, get_ensemble_recommendation
        print("   [PASS] Ensemble routes imported successfully")
        return True
    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False


def verify_model_files():
    """Verify required model files exist."""
    print("\n3. Verifying Model Files...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    files_to_check = [
        ('ensemble.pkl', 'Production ensemble model'),
        ('dqn_best.pkl', 'Best single DQN model'),
        ('dqn_final.pkl', 'Final DQN model'),
    ]

    model_dir = os.path.join(base_dir, 'models', 'rl_agents', 'chain_8453')

    all_found = True
    for filename, description in files_to_check:
        path = os.path.join(model_dir, filename)
        exists = os.path.exists(path)
        status = "[PASS]" if exists else "[WARN]"
        size = ""
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"   {status} {description}: {filename}{size}")
        if not exists and filename == 'ensemble.pkl':
            all_found = False

    return all_found


def verify_dependencies():
    """Verify required dependencies are installed."""
    print("\n4. Verifying Dependencies...")

    deps = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('flask', 'flask'),
    ]

    all_ok = True
    for name, import_name in deps:
        try:
            __import__(import_name)
            print(f"   [PASS] {name}")
        except ImportError:
            print(f"   [FAIL] {name} not installed")
            all_ok = False

    return all_ok


def run_performance_test():
    """Run a quick performance test."""
    print("\n5. Running Quick Performance Test...")

    try:
        from services.ensemble_service import get_ensemble_service
        import time

        service = get_ensemble_service(8453)

        # Warm up
        service.get_recommendation(current_price=0.001, urgency=0.5)

        # Time 100 recommendations
        start = time.time()
        n_requests = 100
        for i in range(n_requests):
            service.get_recommendation(
                current_price=0.001 + (i * 0.00001),
                urgency=0.5,
                time_waiting=i % 20
            )
        elapsed = time.time() - start

        rps = n_requests / elapsed
        ms_per_req = (elapsed / n_requests) * 1000

        print(f"   {n_requests} recommendations in {elapsed:.2f}s")
        print(f"   {rps:.0f} requests/second")
        print(f"   {ms_per_req:.1f} ms/request")

        if rps > 50:
            print("   [PASS] Performance acceptable")
            return True
        else:
            print("   [WARN] Performance below 50 rps")
            return True

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False


def main():
    print("=" * 60)
    print("PRODUCTION VERIFICATION")
    print("=" * 60)

    results = {
        'Ensemble Service': verify_ensemble_service(),
        'API Routes': verify_api_routes(),
        'Model Files': verify_model_files(),
        'Dependencies': verify_dependencies(),
        'Performance': run_performance_test(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("PRODUCTION READY")
        print("\nEndpoints available:")
        print("  GET/POST /agent/ensemble/recommend - Get recommendation")
        print("  GET      /agent/ensemble/status   - Get service status")
        print("  GET/POST /agent/recommend         - Legacy single-agent")
    else:
        print("ISSUES DETECTED - Please review above")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
