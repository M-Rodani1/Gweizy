#!/usr/bin/env python3
"""
Improved model training with advanced features and hyperparameter tuning
Run this to retrain models with 100+ features for better accuracy
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
import joblib
from datetime import datetime
from data.database import DatabaseManager
from models.advanced_features import prepare_training_data


def train_improved_models():
    """Train models with advanced features and hyperparameter tuning"""
    
    print("="*70)
    print("Advanced Gas Price Model Training")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Loading historical data...")
    db = DatabaseManager()
    historical_data = db.get_historical_data(hours=720)  # 30 days
    
    if len(historical_data) < 100:
        print(f"❌ Error: Not enough data ({len(historical_data)} records). Need at least 100.")
        print("💡 Make sure data collection has been running for at least a few days")
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
    
    # 2. Create advanced features (we'll do this in step 3 with target alignment)
    print("\n[2/6] Preparing data structure...")
    print(f"✅ Loaded {len(data)} data points for feature engineering")
    
    # 3. Create target variables and features with proper alignment
    print("\n[3/6] Creating features and target variables...")
    
    from models.advanced_features import create_advanced_features
    import pandas as pd
    
    # Reconstruct full dataframe with gas_price for target creation
    df_full = pd.DataFrame(data)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], format='mixed', errors='coerce')
    df_full = df_full.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Ensure we have gas_price column
    if 'gas_price' not in df_full.columns:
        if 'current_gas' in df_full.columns:
            df_full['gas_price'] = df_full['current_gas']
        elif 'gwei' in df_full.columns:
            df_full['gas_price'] = df_full['gwei']
    
    # Create features first (this will use current gas_price values)
    X_full, y_full = create_advanced_features(df_full)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ Base dataset: {X_full.shape[0]} samples")
    
    # For 1h, 4h, 24h predictions, we need to shift the target FORWARD
    # Assuming 5-minute intervals: 1h = 12 steps, 4h = 48 steps, 24h = 288 steps
    # shift(-12) means: value at index i is the gas price 12 steps (1 hour) in the future
    # We need to get the original gas prices from df_full
    original_gas = df_full['gas_price'].values
    
    y_1h_raw = pd.Series(original_gas).shift(-12)
    y_4h_raw = pd.Series(original_gas).shift(-48)
    y_24h_raw = pd.Series(original_gas).shift(-288)
    
    # Align: remove rows where target is NaN (can't predict future for last N rows)
    # Also need to account for rows dropped during feature engineering
    # X_full and original_gas should have same length after feature engineering
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len]
    y_1h_raw = y_1h_raw.iloc[:min_len]
    y_4h_raw = y_4h_raw.iloc[:min_len]
    y_24h_raw = y_24h_raw.iloc[:min_len]
    
    valid_1h = ~y_1h_raw.isna()
    valid_4h = ~y_4h_raw.isna()
    valid_24h = ~y_24h_raw.isna()
    
    X_1h = X_full[valid_1h].reset_index(drop=True)
    y_1h = y_1h_raw[valid_1h].reset_index(drop=True)
    
    X_4h = X_full[valid_4h].reset_index(drop=True)
    y_4h = y_4h_raw[valid_4h].reset_index(drop=True)
    
    X_24h = X_full[valid_24h].reset_index(drop=True)
    y_24h = y_24h_raw[valid_24h].reset_index(drop=True)
    
    print(f"✅ 1h: {len(y_1h)} samples (features aligned with future targets)")
    print(f"✅ 4h: {len(y_4h)} samples (features aligned with future targets)")
    print(f"✅ 24h: {len(y_24h)} samples (features aligned with future targets)")
    
    # 4. Train models for each horizon
    horizons = {
        '1h': (X_1h, y_1h),
        '4h': (X_4h, y_4h),
        '24h': (X_24h, y_24h)
    }
    
    os.makedirs('backend/models/saved_models', exist_ok=True)
    
    best_models = {}
    
    for horizon, (X_h, y_h) in horizons.items():
        print(f"\n{'='*70}")
        print(f"[4/6] Training models for {horizon} horizon")
        print(f"{'='*70}")
        
        if len(X_h) < 100:
            print(f"⚠️  Not enough data for {horizon}, skipping...")
            continue
        
        # Split data (use time-based split, don't shuffle!)
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
        
        # Train Random Forest with hyperparameter tuning
        print("\n📊 Training Random Forest with Grid Search...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        rf_grid = GridSearchCV(
            rf_model,
            rf_params,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid.fit(X_train_scaled, y_train)
        
        print(f"✅ Best RF Parameters: {rf_grid.best_params_}")
        print(f"✅ Best CV Score: {rf_grid.best_score_:.4f}")
        
        # Train Gradient Boosting
        print("\n📊 Training Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        
        gb_grid = GridSearchCV(
            gb_model,
            gb_params,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        gb_grid.fit(X_train_scaled, y_train)
        
        print(f"✅ Best GB Parameters: {gb_grid.best_params_}")
        print(f"✅ Best CV Score: {gb_grid.best_score_:.4f}")
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        models = {
            'RandomForest': rf_grid.best_estimator_,
            'GradientBoosting': gb_grid.best_estimator_
        }
        
        best_model = None
        best_r2 = -float('inf')
        best_name = None
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Directional accuracy
            if len(y_test) > 1:
                y_test_diff = np.diff(y_test.values)
                y_pred_diff = np.diff(y_pred)
                directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
            else:
                directional_accuracy = 0.0
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  R²:   {r2:.4f} ({'✅ GOOD' if r2 > 0.7 else '⚠️  NEEDS IMPROVEMENT'})")
            print(f"  Directional Accuracy: {directional_accuracy:.2%} ({'✅ GOOD' if directional_accuracy > 0.75 else '⚠️  NEEDS IMPROVEMENT'})")
            
            if r2 > best_r2:
                best_r2 = r2
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
            'metrics': {
                'r2': best_r2,
                'mae': mae,
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
    
    # 5. Save feature names
    feature_names_path = 'backend/models/saved_models/feature_names.pkl'
    joblib.dump(X.columns.tolist(), feature_names_path)
    print(f"\n💾 Saved feature names to {feature_names_path}")
    
    # 6. Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    for horizon, info in best_models.items():
        print(f"{horizon}: {info['name']} - R² = {info['r2']:.4f}")
    print("="*70)
    
    return best_models


if __name__ == '__main__':
    train_improved_models()

