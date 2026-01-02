"""
Quick Start - Run Your First Model Test

This script makes it easy to test your current model performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from datetime import datetime


def check_prerequisites():
    """Check if everything is ready for testing"""
    print("🔍 Checking prerequisites...\n")

    issues = []

    # Check if models exist
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_files = list(model_dir.glob('gas_predictor_ensemble_*.joblib'))

    if not model_files:
        issues.append("❌ No trained models found!")
        issues.append("   Run: python scripts/train_ensemble_final.py")
    else:
        print(f"✅ Found {len(model_files)} trained model(s)")
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"   Latest: {latest.name}")

    # Check if database has data
    try:
        from data.database import get_db_connection
        import pandas as pd

        conn = get_db_connection()
        df = pd.read_sql("SELECT COUNT(*) as count FROM gas_prices", conn)
        count = df['count'].iloc[0]
        conn.close()

        if count < 1000:
            issues.append(f"⚠️  Only {count} data points in database")
            issues.append("   Recommend: At least 1,000 points for testing")
        else:
            print(f"✅ Database has {count:,} data points")

    except Exception as e:
        issues.append(f"❌ Database error: {e}")

    # Check dependencies
    try:
        import sklearn
        import matplotlib
        print("✅ All Python dependencies installed")
    except ImportError as e:
        issues.append(f"❌ Missing dependency: {e}")
        issues.append("   Run: pip install -r requirements.txt")

    if issues:
        print(f"\n❌ Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(issue)
        return False

    print("\n✅ All prerequisites met! Ready to test.\n")
    return True


def run_quick_test():
    """Run a quick backtest"""
    print("="*70)
    print("  🚀 QUICK MODEL TEST")
    print("="*70)
    print("\nThis will test your model on the last 48 hours of data.")
    print("Estimated time: 1-2 minutes\n")

    try:
        from comprehensive_backtester import ComprehensiveBacktester

        # Create backtester (test on 48 hours)
        print("📊 Initializing backtester...")
        backtester = ComprehensiveBacktester(lookback_hours=48)

        # Load data
        df = backtester.load_historical_data()

        if len(df) < 100:
            print(f"\n⚠️  Warning: Only {len(df)} data points found")
            print("   For best results, collect more data over time")
            print("   Continuing with available data...\n")

        # Prepare features
        df = backtester.prepare_features(df)

        # Run backtest
        results = backtester.backtest_predictions(df, horizons=['1h'])

        # Show results
        print("\n" + "="*70)
        print("  📊 YOUR CURRENT MODEL PERFORMANCE")
        print("="*70)

        if '1h' in results:
            metrics = results['1h']['metrics']

            print(f"\n1-Hour Predictions ({results['1h']['n_samples']} samples tested):")
            print(f"  MAE:        {metrics['mae']:.6f} Gwei")
            print(f"  R² Score:   {metrics['r2']:.4f}")
            print(f"  Dir. Acc:   {metrics['directional_accuracy']:.2%}")

            # Rating
            print(f"\n🎯 Overall Rating:")
            if metrics['r2'] > 0.7 and metrics['directional_accuracy'] > 0.75:
                print("  ✅ GOOD - Model is performing well!")
            elif metrics['r2'] > 0.5 and metrics['directional_accuracy'] > 0.6:
                print("  ⚠️  FAIR - Model works but has room for improvement")
            else:
                print("  ❌ NEEDS IMPROVEMENT - See ML_IMPROVEMENT_PLAN.md")

            # Save results
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            report_path = f'quick_test_{timestamp}.json'

            report = {
                'test_date': datetime.utcnow().isoformat(),
                'lookback_hours': 48,
                'metrics': {k: float(v) if v is not None else None
                           for k, v in metrics.items()}
            }

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\n📄 Results saved to: {report_path}")

        else:
            print("\n⚠️  Not enough data to test 1-hour predictions")

        # Next steps
        print(f"\n" + "="*70)
        print("  📝 NEXT STEPS")
        print("="*70)
        print("\n1. Run full backtest (test on 7 days):")
        print("   python testing/comprehensive_backtester.py")
        print("\n2. Review improvement plan:")
        print("   cat ../ML_IMPROVEMENT_PLAN.md")
        print("\n3. Monitor live performance:")
        print("   python testing/live_performance_monitor.py")
        print("\n4. Read testing guide:")
        print("   cat ../TESTING_GUIDE.md")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure model is trained: python scripts/train_ensemble_final.py")
        print("2. Check data collection is running")
        print("3. Review error message above")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  🧪 FIRST-TIME MODEL TESTING")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check if everything is set up correctly")
    print("  2. Run a quick test on recent data")
    print("  3. Show you how good your model is")
    print("  4. Guide you on next steps\n")

    input("Press Enter to start... ")

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Please fix the issues above before testing.")
        return

    input("\nPress Enter to run quick test... ")

    # Run test
    run_quick_test()

    print("\n✅ Testing complete!\n")


if __name__ == '__main__':
    main()
