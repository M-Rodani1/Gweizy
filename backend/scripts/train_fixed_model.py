#!/usr/bin/env python3
"""
Train improved model with proper scaling and outlier removal
Target: R² > 70%, Directional Accuracy > 75%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from data.database import DatabaseManager
from models.advanced_features import create_advanced_features


def remove_outliers_robust(X, y):
    """Remove outliers using robust IQR method"""
    
    # Use RobustScaler approach - less sensitive to outliers
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    
    # Use 2.5 IQR for moderate outlier removal
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr
    
    # Ensure bounds are reasonable
    lower_bound = max(lower_bound, y.min() * 0.1)  # Don't go below 10% of min
    upper_bound = min(upper_bound, y.max() * 10)    # Don't go above 10x max
    
    # Filter
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    removed = (~mask).sum()
    if removed > 0:
        print(f"Removed {removed} outliers ({removed/len(y)*100:.1f}%)")
        print(f"Outlier bounds: < {lower_bound:.6f} or > {upper_bound:.6f}")
    
    return X[mask], y[mask]


def train_fixed_model():
    """Train model with proper scaling and outlier handling"""
    
    print("="*70)
    print("TRAINING FIXED MODEL (Target: R² > 70%)")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=720)  # 30 days
    
    if len(historical) < 100:
        print(f"❌ Not enough data: {len(historical)} records")
        return
    
    # Convert to format
    data = []
    for h in historical:
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
    
    print(f"✅ Loaded {len(data)} data points")
    
    # 2. Create features
    print("\n[2/6] Creating advanced features...")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
    
    X_full, y_full = create_advanced_features(df)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ {X_full.shape[0]} samples")
    
    # 3. Create target variables (predicting absolute price)
    print("\n[3/6] Creating target variables...")
    original_gas = df['gas_price'].values
    
    # Align with features
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len].reset_index(drop=True)
    original_gas = original_gas[:min_len]
    
    # Create 1h target (shift forward by 12 steps)
    y_1h_raw = pd.Series(original_gas).shift(-12)
    
    # Align
    valid = ~y_1h_raw.isna()
    X_1h = X_full[valid].reset_index(drop=True)
    y_1h = y_1h_raw[valid].reset_index(drop=True)
    
    print(f"✅ 1h targets: {len(y_1h)} samples")
    
    # 4. Remove outliers
    print("\n[4/6] Removing outliers...")
    X_clean, y_clean = remove_outliers_robust(X_1h, y_1h)
    
    print(f"✅ Clean data: {len(X_clean)} samples")
    
    # 5. Split and scale
    print("\n[5/6] Splitting and scaling data...")
    split_idx = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    # Use RobustScaler for features (less sensitive to outliers)
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target variable
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    print("✅ Features and targets scaled")
    
    # 6. Train models
    print("\n[6/6] Training models...")
    
    models_to_try = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            random_state=42
        )
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = None
    best_metrics = {}
    
    for name, model in models_to_try.items():
        print(f"   Training {name}...")
        model.fit(X_train_scaled, y_train_scaled)
        
        # Predict
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        if len(y_test) > 1:
            y_test_diff = np.diff(y_test.values)
            y_pred_diff = np.diff(y_pred)
            dir_acc = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
        else:
            dir_acc = 0.0
        
        print(f"      R²: {r2:.4f} ({r2*100:.2f}%)")
        print(f"      Directional: {dir_acc:.2%}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            best_metrics = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'directional_accuracy': dir_acc
            }
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*70}")
    print(f"MAE:  {best_metrics['mae']:.6f} gwei")
    print(f"RMSE: {best_metrics['rmse']:.6f} gwei")
    print(f"R²:   {best_metrics['r2']:.4f} ({best_metrics['r2']*100:.2f}%) {'✅ GOOD' if best_metrics['r2'] > 0.7 else '⚠️ NEEDS IMPROVEMENT' if best_metrics['r2'] > 0.5 else '❌ POOR'}")
    print(f"Directional Accuracy: {best_metrics['directional_accuracy']:.2%} {'✅ GOOD' if best_metrics['directional_accuracy'] > 0.7 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*70}")
    
    # Save model
    os.makedirs('backend/models/saved_models', exist_ok=True)
    
    save_data = {
        'model': best_model,
        'model_name': best_name,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': X_clean.columns.tolist(),
        'predicts_percentage_change': False,  # Predicts absolute price
        'metrics': best_metrics,
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(save_data, 'backend/models/saved_models/model_1h.pkl')
    joblib.dump(feature_scaler, 'backend/models/saved_models/scaler_1h.pkl')
    joblib.dump(X_clean.columns.tolist(), 'backend/models/saved_models/feature_names.pkl')
    
    print("\n✅ Model trained and saved!")
    
    return best_model, feature_scaler, target_scaler


if __name__ == '__main__':
    train_fixed_model()

