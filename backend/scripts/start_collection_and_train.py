#!/usr/bin/env python3
"""
Start Data Collection and Retrain Models

This script:
1. Starts the onchain collector service in background
2. Monitors data collection progress
3. Automatically retrains models once sufficient data is collected
"""

import sys
import os
import time
import subprocess
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager
from services.onchain_collector_service import OnChainCollectorService
import threading


def check_data_availability():
    """Check if we have enough data for training"""
    db = DatabaseManager()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Check gas prices
    cursor.execute("SELECT COUNT(*) FROM gas_prices")
    gas_count = cursor.fetchone()[0]
    
    # Check onchain features
    cursor.execute("SELECT COUNT(*) FROM onchain_features")
    onchain_count = cursor.fetchone()[0]
    
    # Check recent onchain features (last 24 hours)
    cursor.execute("""
        SELECT COUNT(*) FROM onchain_features 
        WHERE timestamp > datetime('now', '-24 hours')
    """)
    recent_onchain = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'gas_prices': gas_count,
        'onchain_features': onchain_count,
        'recent_onchain': recent_onchain,
        'ready_for_training': recent_onchain >= 1000  # Optimal: Need 1,000+ recent records for best results
    }


def monitor_collection(collector_service, check_interval=300):  # Check every 5 minutes
    """Monitor data collection progress"""
    print("\n📊 Monitoring data collection...")
    print("   (Press Ctrl+C to stop and retrain with current data)\n")
    
    start_time = time.time()
    last_count = 0
    
    try:
        while collector_service.running:
            time.sleep(check_interval)
            
            stats = check_data_availability()
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection Status:")
            print(f"   ⏱️  Running for: {hours:.1f} hours")
            print(f"   📊 Total onchain_features: {stats['onchain_features']}")
            print(f"   📊 Recent (24h): {stats['recent_onchain']}")
            print(f"   📊 Collection rate: {stats['recent_onchain'] - last_count} records/5min")
            print(f"   ✅ Ready for training: {'Yes' if stats['ready_for_training'] else 'No (need 1,000+ records for optimal)'}")
            print()
            
            last_count = stats['recent_onchain']
            
            # Auto-retrain if we have enough data
            if stats['ready_for_training']:
                print("🎉 Sufficient data collected! Ready to retrain models.")
                print("   Stopping collector and starting training...")
                collector_service.stop()
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        collector_service.stop()


def start_collection():
    """Start the onchain collector service"""
    print("="*60)
    print("🚀 Starting Enhanced Features Collection Service")
    print("="*60)
    
    # Check current data
    stats = check_data_availability()
    print(f"\n📊 Current Data Status:")
    print(f"   Gas Prices: {stats['gas_prices']:,} records")
    print(f"   OnChain Features: {stats['onchain_features']:,} records")
    print(f"   Recent (24h): {stats['recent_onchain']:,} records")
    print(f"   Ready for Training: {'✅ Yes' if stats['ready_for_training'] else '❌ No (need 1,000+ records for optimal)'}")
    
    if stats['ready_for_training']:
        print("\n✅ Already have enough data! Proceeding to training...")
        return None
    
    # Start collector service
    print(f"\n📦 Starting collector service...")
    print(f"   Interval: {os.getenv('COLLECTION_INTERVAL', '60')} seconds (1 minute)")
    print(f"   This will collect enhanced congestion features")
    print(f"   Target: 1,000+ records (approximately 16-17 hours for optimal training)")
    
    collector = OnChainCollectorService(interval_seconds=60)  # 1 minute
    
    # Start in background thread
    collector_thread = threading.Thread(target=collector.start, daemon=True)
    collector_thread.start()
    
    # Give it a moment to start
    time.sleep(2)
    
    if collector.running:
        print("✅ Collector service started successfully!")
        return collector
    else:
        print("❌ Failed to start collector service")
        return None


def retrain_models():
    """Retrain models with new improvements"""
    print("\n" + "="*60)
    print("🎯 Retraining Models with Week 1 Quick Wins")
    print("="*60)
    print("\nImprovements being used:")
    print("  ✅ 1-minute sampling")
    print("  ✅ Enhanced congestion features (27% variance explained)")
    print("  ✅ RobustScaler (outlier handling)")
    
    try:
        # Import and run training
        from models.feature_engineering import GasFeatureEngineer
        from models.model_trainer import GasModelTrainer
        
        print("\n📊 Step 1: Preparing training data...")
        engineer = GasFeatureEngineer()
        
        try:
            df = engineer.prepare_training_data(hours_back=720)  # 30 days
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("💡 Need more data. Continue collection for a few more hours.")
            return False
        
        if len(df) == 0:
            print("❌ No training samples generated")
            print("💡 Need more onchain_features data. Continue collection.")
            return False
        
        print(f"✅ Prepared {len(df)} training samples")
        
        print("\n📊 Step 2: Extracting features and targets...")
        feature_cols = engineer.get_feature_columns(df)
        X = df[feature_cols]
        y_1h = df['target_1h']
        y_4h = df['target_4h']
        y_24h = df['target_24h']
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        
        # Check enhanced features
        enhanced_features = [
            'pending_tx_count', 'unique_addresses', 'tx_per_second',
            'gas_utilization_ratio', 'congestion_level'
        ]
        found = [f for f in enhanced_features if f in feature_cols]
        print(f"   Enhanced features: {len(found)}/{len(enhanced_features)}")
        
        print("\n📊 Step 3: Training models with RobustScaler...")
        trainer = GasModelTrainer()
        results = trainer.train_all_models(X, y_1h, y_4h, y_24h)
        
        print("\n📊 Step 4: Saving models...")
        trainer.save_models()
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        
        # Print summary
        print("\n📊 Model Performance Summary:")
        for horizon, result in results.items():
            best = result['best']
            print(f"\n{horizon}:")
            print(f"  Model: {best['name']}")
            print(f"  MAE: {best['metrics']['mae']:.6f} gwei")
            print(f"  RMSE: {best['metrics']['rmse']:.6f} gwei")
            print(f"  R²: {best['metrics']['r2']:.4f}")
            print(f"  Directional Accuracy: {best['metrics']['directional_accuracy']*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 Week 1 Quick Wins - Data Collection & Retraining")
    print("="*60)
    
    # Step 1: Check if we already have enough data
    stats = check_data_availability()
    
    if stats['ready_for_training']:
        print("\n✅ Already have sufficient data!")
        print("   Proceeding directly to training...")
        retrain_models()
        return
    
    # Step 2: Start collection
    collector = start_collection()
    
    if not collector:
        print("❌ Could not start collector. Exiting.")
        return
    
    # Step 3: Monitor collection
    try:
        monitor_collection(collector)
    except KeyboardInterrupt:
        print("\n⚠️  Collection stopped by user")
        collector.stop()
    
    # Step 4: Retrain models
    print("\n" + "="*60)
    print("🎯 Starting Model Retraining")
    print("="*60)
    
    retrain_models()
    
    print("\n" + "="*60)
    print("✅ Process Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run backtest to measure improvement:")
    print("   python3 testing/backtest_current_models.py")
    print("\n2. Compare with baseline results")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
