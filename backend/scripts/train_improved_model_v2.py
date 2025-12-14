#!/usr/bin/env python3
"""
Improved model training - predicting percentage change instead of absolute price
This should give much better R² scores
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from data.database import DatabaseManager
from models.advanced_features import create_advanced_features


def train_improved_models():
    """Train models predicting percentage change (better than absolute price)"""
    
    print("="*70)
    print("Advanced Gas Price Model Training (Percentage Change)")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Loading historical data...")
    db = DatabaseManager()
    historical_data = db.get_historical_data(hours=720)  # 30 days
    
    if len(historical_data) < 100:
        print(f"❌ Error: Not enough data ({len(historical_data)} records). Need at least 100.")
        return
    
    # Convert to format expected by advanced_features
    data = []
    for h in historical_data:
        if isinstance(h, dict):
            data.append({
                'timestamp': h.get('timestamp', ''),
                'gas_price': h.get('gwei', 0) or h.get('current_gas', 0)
            })
        else:
            data.append({
                'timestamp': h.timestamp if hasattr(h, 'timestamp') else '',
                'gas_price': h.current_gas if hasattr(h, 'current_gas') else 0
            })
    
    print(f"✅ Loaded {len(data)} historical data points")
    
    # 2. Create features and targets with percentage change
    print("\n[2/6] Creating features and percentage change targets...")
    
    df_full = pd.DataFrame(data)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], format='mixed', errors='coerce')
    df_full = df_full.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Ensure we have gas_price column
    if 'gas_price' not in df_full.columns:
        if 'current_gas' in df_full.columns:
            df_full['gas_price'] = df_full['current_gas']
        elif 'gwei' in df_full.columns:
            df_full['gas_price'] = df_full['gwei']
    
    # Create features
    X_full, y_full = create_advanced_features(df_full)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ Base dataset: {X_full.shape[0]} samples")
    
    # Create percentage change targets (much more predictable than absolute prices)
    # For 1h, 4h, 24h: predict the percentage change from current price
    original_gas = df_full['gas_price'].values
    
    # Calculate percentage changes
    # shift(-12) gets gas price 12 steps (1h) in the future
    future_1h = pd.Series(original_gas).shift(-12)
    future_4h = pd.Series(original_gas).shift(-48)
    future_24h = pd.Series(original_gas).shift(-288)
    
    # Percentage change: (future - current) / current * 100
    y_1h_pct = ((future_1h - pd.Series(original_gas)) / pd.Series(original_gas) * 100).values
    y_4h_pct = ((future_4h - pd.Series(original_gas)) / pd.Series(original_gas) * 100).values
    y_24h_pct = ((future_24h - pd.Series(original_gas)) / pd.Series(original_gas) * 100).values
    
    # Align with features
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len].reset_index(drop=True)
    y_1h_pct = y_1h_pct[:min_len]
    y_4h_pct = y_4h_pct[:min_len]
    y_24h_pct = y_24h_pct[:min_len]
    
    # Remove NaN (where we can't calculate future price)
    valid_1h = ~np.isnan(y_1h_pct) & ~np.isinf(y_1h_pct)
    valid_4h = ~np.isnan(y_4h_pct) & ~np.isinf(y_4h_pct)
    valid_24h = ~np.isnan(y_24h_pct) & ~np.isinf(y_24h_pct)
    
    X_1h = X_full[valid_1h].reset_index(drop=True)
    y_1h = pd.Series(y_1h_pct[valid_1h])
    
    X_4h = X_full[valid_4h].reset_index(drop=True)
    y_4h = pd.Series(y_4h_pct[valid_4h])
    
    X_24h = X_full[valid_24h].reset_index(drop=True)
    y_24h = pd.Series(y_24h_pct[valid_24h])
    
    print(f"✅ 1h: {len(y_1h)} samples (predicting % change)")
    print(f"✅ 4h: {len(y_4h)} samples (predicting % change)")
    print(f"✅ 24h: {len(y_24h)} samples (predicting % change)")
    print(f"   Sample % changes - 1h: mean={y_1h.mean():.2f}%, std={y_1h.std():.2f}%")
    
    # 3. Train models for each horizon
    horizons = {
        '1h': (X_1h, y_1h),
        '4h': (X_4h, y_4h),
        '24h': (X_24h, y_24h)
    }
    
    os.makedirs('backend/models/saved_models', exist_ok=True)
    
    best_models = {}
    
    for horizon, (X_h, y_h) in horizons.items():
        print(f"\n{'='*70}")
        print(f"[3/6] Training models for {horizon} horizon (predicting % change)")
        print(f"{'='*70}")
        
        if len(X_h) < 100:
            print(f"⚠️  Not enough data for {horizon}, skipping...")
            continue
        
        # Split data (time-based split)
        split_idx = int(len(X_h) * 0.8)
        X_train, X_test = X_h.iloc[:split_idx], X_h.iloc[split_idx:]
        y_train, y_test = y_h.iloc[:split_idx], y_h.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = f'backend/models/saved_models/scaler_{horizon}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
        
        # Train Random Forest (simpler params for faster training)
        print("\n📊 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='log2',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        print("📊 Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        models = {
            'RandomForest': rf_model,
            'GradientBoosting': gb_model
        }
        
        best_model = None
        best_r2 = -float('inf')
        best_name = None
        best_mae = float('inf')
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Directional accuracy (did we predict up/down correctly?)
            if len(y_test) > 1:
                y_test_diff = np.diff(y_test.values)
                y_pred_diff = np.diff(y_pred)
                directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
            else:
                directional_accuracy = 0.0
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.4f}%")
            print(f"  RMSE: {rmse:.4f}%")
            print(f"  R²:   {r2:.4f} ({'✅ GOOD' if r2 > 0.3 else '⚠️  NEEDS IMPROVEMENT' if r2 > 0 else '❌ POOR'})")
            print(f"  Directional Accuracy: {directional_accuracy:.2%} ({'✅ GOOD' if directional_accuracy > 0.6 else '⚠️  NEEDS IMPROVEMENT'})")
            
            if r2 > best_r2:
                best_r2 = r2
                best_mae = mae
                best_model = model
                best_name = name
        
        # Save best model
        print(f"\n{'='*70}")
        print(f"✅ BEST MODEL for {horizon}: {best_name} (R² = {best_r2:.4f})")
        print(f"{'='*70}")
        
        model_path = f'backend/models/saved_models/model_{horizon}.pkl'
        save_data = {
            'model': best_model,
            'model_name': best_name,
            'scaler': scaler,
            'feature_names': X_h.columns.tolist(),
            'predicts_percentage_change': True,  # Flag that this predicts % change
            'metrics': {
                'r2': best_r2,
                'mae': best_mae,
                'rmse': rmse,
                'directional_accuracy': directional_accuracy
            },
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, model_path)
        print(f"💾 Saved model to {model_path}")
        
        best_models[horizon] = {
            'model': best_model,
            'scaler': scaler,
            'name': best_name,
            'r2': best_r2
        }
    
    # 4. Save feature names
    feature_names_path = 'backend/models/saved_models/feature_names.pkl'
    joblib.dump(X_full.columns.tolist(), feature_names_path)
    print(f"\n💾 Saved feature names to {feature_names_path}")
    
    # 5. Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    for horizon, info in best_models.items():
        print(f"{horizon}: {info['name']} - R² = {info['r2']:.4f}")
    print("="*70)
    
    return best_models


if __name__ == '__main__':
    train_improved_models()

