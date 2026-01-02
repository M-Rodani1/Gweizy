#!/usr/bin/env python3
"""
Quick Model Training Script
Trains ML models and feature selector for Base chain (8453) and optionally other chains.

Usage:
    python train_models_now.py                    # Train Base chain only
    python train_models_now.py --chain 1          # Train Ethereum
    python train_models_now.py --all-chains      # Train all supported chains
    python train_models_now.py --chain 8453,1,137  # Train specific chains
"""

import sys
import os
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain_models_simple import fetch_training_data, prepare_features, train_model, save_model
from models.feature_selector import SHAPFeatureSelector
from data.multichain_collector import CHAINS
from utils.logger import logger


def train_for_chain(chain_id: int = 8453, hours: int = 720):
    """
    Train models for a specific chain.
    
    Args:
        chain_id: Chain ID to train for (default: 8453 for Base)
        hours: Hours of historical data to use
    """
    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
    
    print(f"\n{'='*60}")
    print(f"🎯 Training Models for {chain_name} (Chain ID: {chain_id})")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Fetch data
        print("📊 Step 1: Fetching training data...")
        data = fetch_training_data(hours=hours)
        
        # Filter data for this chain if chain_id is specified
        if chain_id != 8453:  # Base is default, others need filtering
            from data.database import DatabaseManager
            db = DatabaseManager()
            data = db.get_historical_data(hours=hours, chain_id=chain_id)
            if not data:
                print(f"❌ No data found for {chain_name}. Make sure data collection is running.")
                return False
        
        if len(data) < 100:
            print(f"❌ Not enough data: {len(data)} records. Need at least 100.")
            print("💡 Wait for more data to be collected, or reduce --hours parameter.")
            return False
        
        print(f"✅ Fetched {len(data)} records\n")
        
        # Step 2: Prepare features
        print("📊 Step 2: Preparing features...")
        X, y_1h, y_4h, y_24h, feature_meta = prepare_features(data)
        print(f"✅ Prepared {len(X)} samples with {X.shape[1]} features\n")
        
        # Step 3: Train models for each horizon
        horizons = {
            '1h': y_1h,
            '4h': y_4h,
            '24h': y_24h
        }
        
        trained_models = {}
        feature_selector = None
        
        for horizon, y in horizons.items():
            print(f"\n{'='*60}")
            print(f"🎯 Training {horizon} model")
            print(f"{'='*60}")
            
            # Train model (includes feature selection if needed)
            model_data = train_model(
                X, (y, y_1h if horizon == '1h' else y_4h if horizon == '4h' else y_24h),
                horizon,
                feature_meta=feature_meta,
                use_feature_selection=(horizon == '1h')  # Only do feature selection once
            )
            
            if model_data:
                trained_models[horizon] = model_data
                if model_data.get('feature_selector') and feature_selector is None:
                    feature_selector = model_data['feature_selector']
                print(f"✅ {horizon} model trained successfully")
            else:
                print(f"❌ Failed to train {horizon} model")
        
        # Step 4: Save models
        if trained_models:
            print(f"\n{'='*60}")
            print("💾 Saving models...")
            print(f"{'='*60}")
            
            # Determine output directory based on chain
            if chain_id == 8453:
                output_dir = 'backend/models/saved_models'
            else:
                output_dir = f'backend/models/saved_models/chain_{chain_id}'
                os.makedirs(output_dir, exist_ok=True)
            
            for horizon, model_data in trained_models.items():
                save_model(model_data, horizon, output_dir=output_dir)
                print(f"✅ Saved {horizon} model")
            
            # Save feature selector separately (only once, for Base chain)
            if feature_selector and chain_id == 8453:
                selector_path = os.path.join(output_dir, 'feature_selector.pkl')
                import joblib
                joblib.dump(feature_selector, selector_path)
                print(f"✅ Saved feature selector to {selector_path}")
            
            print(f"\n✅ All models saved successfully for {chain_name}!")
            return True
        else:
            print("\n❌ No models were trained successfully")
            return False
            
    except Exception as e:
        print(f"\n❌ Error training models for {chain_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Train ML models for gas price prediction')
    parser.add_argument('--chain', type=str, help='Chain ID(s) to train (comma-separated, e.g., 8453,1,137)')
    parser.add_argument('--all-chains', action='store_true', help='Train all supported chains')
    parser.add_argument('--hours', type=int, default=720, help='Hours of historical data to use (default: 720)')
    
    args = parser.parse_args()
    
    # Determine which chains to train
    if args.all_chains:
        chains = list(CHAINS.keys())
    elif args.chain:
        chains = [int(c.strip()) for c in args.chain.split(',')]
    else:
        chains = [8453]  # Default to Base chain
    
    print(f"\n{'='*60}")
    print("🚀 MODEL TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Chains to train: {[CHAINS.get(c, {}).get('name', f'Chain {c}') for c in chains]}")
    print(f"Hours of data: {args.hours}")
    print(f"{'='*60}\n")
    
    results = {}
    for chain_id in chains:
        success = train_for_chain(chain_id, args.hours)
        results[chain_id] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    for chain_id, success in results.items():
        chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
        status = "✅ Success" if success else "❌ Failed"
        print(f"{chain_name} (Chain {chain_id}): {status}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\n✅ {successful}/{len(results)} chains trained successfully")
    
    if successful > 0:
        print("\n💡 Models are now ready to use!")
        print("   The API will automatically load these models on next request.")


if __name__ == '__main__':
    main()

