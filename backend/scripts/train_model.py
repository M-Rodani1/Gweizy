#!/usr/bin/env python3
"""
Script to train gas price prediction models
Run this daily or when you want to retrain models
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_engineering import GasFeatureEngineer
from models.model_trainer import GasModelTrainer


def main():
    print("="*70)
    print("🚀 Base Gas Price Model Training Pipeline")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n📊 Step 1: Preparing training data...")
    engineer = GasFeatureEngineer()
    
    try:
        df = engineer.prepare_training_data(hours_back=720)  # 30 days
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure data collection has been running for at least a few days")
        return
    
    # Step 2: Extract features and targets
    print("\n📊 Step 2: Extracting features and targets...")
    feature_cols = engineer.get_feature_columns(df)
    X = df[feature_cols]
    y_1h = df['target_1h']
    y_4h = df['target_4h']
    y_24h = df['target_24h']
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    
    # Step 3: Train models
    print("\n📊 Step 3: Training models...")
    trainer = GasModelTrainer()
    results = trainer.train_all_models(X, y_1h, y_4h, y_24h)
    
    # Step 4: Save models
    print("\n📊 Step 4: Saving models...")
    trainer.save_models()
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    
    # Print summary
    print("\n📊 Model Summary:")
    for horizon, result in results.items():
        best = result['best']
        print(f"\n{horizon}:")
        print(f"  Model: {best['name']}")
        print(f"  MAE: {best['metrics']['mae']:.6f} gwei")
        print(f"  RMSE: {best['metrics']['rmse']:.6f} gwei")
        print(f"  R²: {best['metrics']['r2']:.4f}")
        print(f"  Directional Accuracy: {best['metrics']['directional_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()

