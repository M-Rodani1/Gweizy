#!/usr/bin/env python3
"""
Monitor Data Collection and Auto-Retrain at Optimal Threshold

This script monitors onchain_features collection and automatically retrains
models when 1,000+ records are available for optimal performance.
"""

import sys
import os
import time
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager


def check_data_availability():
    """Check if we have optimal data for training"""
    db = DatabaseManager()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Check recent onchain features (last 24 hours)
    cursor.execute("""
        SELECT COUNT(*) FROM onchain_features 
        WHERE timestamp > datetime('now', '-24 hours')
    """)
    recent_onchain = cursor.fetchone()[0]
    
    # Check total onchain features
    cursor.execute("SELECT COUNT(*) FROM onchain_features")
    total_onchain = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'recent_onchain': recent_onchain,
        'total_onchain': total_onchain,
        'ready_for_optimal_training': recent_onchain >= 1000
    }


def main():
    print("="*70)
    print("🎯 Optimal Retraining Monitor")
    print("="*70)
    print("\n📊 Target: 1,000+ recent onchain_features records")
    print("   This ensures:")
    print("   • Enhanced features have real values (not zeros)")
    print("   • Full coverage of network conditions")
    print("   • Maximum model quality (R² > 0.50 expected)")
    print("   • Best directional accuracy (> 65% expected)")
    
    check_interval = 300  # Check every 5 minutes
    start_time = time.time()
    last_count = 0
    
    print("\n📊 Starting monitoring...")
    print("   (Press Ctrl+C to stop and check status)\n")
    
    try:
        while True:
            stats = check_data_availability()
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            
            # Calculate progress
            progress_pct = min(100, (stats['recent_onchain'] / 1000) * 100)
            remaining = max(0, 1000 - stats['recent_onchain'])
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status:")
            print(f"   ⏱️  Monitoring for: {hours:.1f} hours")
            print(f"   📊 Recent (24h): {stats['recent_onchain']:,} / 1,000 records")
            print(f"   📊 Total: {stats['total_onchain']:,} records")
            print(f"   📈 Progress: {progress_pct:.1f}%")
            
            if stats['recent_onchain'] > last_count:
                rate = stats['recent_onchain'] - last_count
                print(f"   ⚡ Collection rate: {rate} records/5min")
            
            if remaining > 0:
                if stats['recent_onchain'] > 0:
                    rate_per_hour = stats['recent_onchain'] / (hours if hours > 0 else 1)
                    if rate_per_hour > 0:
                        hours_remaining = remaining / rate_per_hour
                        print(f"   ⏳ Estimated time remaining: ~{hours_remaining:.1f} hours")
                else:
                    print(f"   ⏳ Estimated time remaining: ~16-17 hours (at 1 record/min)")
            else:
                print(f"   ✅ READY FOR OPTIMAL TRAINING!")
            
            print()
            
            # Check if ready
            if stats['ready_for_optimal_training']:
                print("="*70)
                print("🎉 OPTIMAL DATA THRESHOLD REACHED!")
                print("="*70)
                print(f"\n✅ Collected {stats['recent_onchain']:,} recent onchain_features records")
                print("   This is sufficient for optimal model training.")
                print("\n🚀 Starting automatic retraining...")
                print()
                
                # Run training script
                train_script = os.path.join(os.path.dirname(__file__), 'train_with_current_data.py')
                result = subprocess.run(
                    [sys.executable, train_script],
                    cwd=os.path.dirname(os.path.dirname(__file__))
                )
                
                if result.returncode == 0:
                    print("\n" + "="*70)
                    print("✅ OPTIMAL RETRAINING COMPLETE!")
                    print("="*70)
                    print("\n📊 Next Steps:")
                    print("   1. Check model performance metrics")
                    print("   2. Run backtest: python3 testing/backtest_current_models.py")
                    print("   3. Expected improvements:")
                    print("      • R²: Should be positive (target > 0.50)")
                    print("      • Directional Accuracy: Should be > 65%")
                    print("      • Enhanced features: Now have real values")
                else:
                    print("\n⚠️  Training completed with warnings. Check output above.")
                
                break
            
            last_count = stats['recent_onchain']
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        stats = check_data_availability()
        print(f"\n📊 Final Status:")
        print(f"   Recent (24h): {stats['recent_onchain']:,} / 1,000 records")
        print(f"   Progress: {(stats['recent_onchain'] / 1000) * 100:.1f}%")
        print(f"   Status: {'✅ Ready for optimal training!' if stats['ready_for_optimal_training'] else '⏳ Still collecting...'}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
