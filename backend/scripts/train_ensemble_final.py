#!/usr/bin/env python3
"""
Train ensemble model with proper scaling - FINAL VERSION
Uses ensemble of RandomForest + GradientBoosting with voting
Target: R² > 70%, Directional Accuracy > 75%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
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
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr
    lower_bound = max(lower_bound, y.min() * 0.1)
    upper_bound = min(upper_bound, y.max() * 10)
    mask = (y >= lower_bound) & (y <= upper_bound)
    removed = (~mask).sum()
    if removed > 0:
        print(f"Removed {removed} outliers ({removed/len(y)*100:.1f}%)")
    return X[mask], y[mask]


def train_ensemble_final():
    """Train ensemble model - final optimized version"""
    
    print("="*70)
    print("TRAINING ENSEMBLE MODEL (FINAL - Target: R² > 70%)")
    print("="*70)
    
    # 1. Load MORE data
    print("\n[1/7] Loading data...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=1440)  # 60 days for more data
    
    if len(historical) < 100:
        print(f"❌ Not enough data: {len(historical)} records")
        return
    
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
    print("\n[2/7] Creating advanced features...")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
    
    # Filter out zero/negative prices
    df = df[df['gas_price'] > 0].reset_index(drop=True)
    
    X_full, y_full = create_advanced_features(df)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ {X_full.shape[0]} samples")
    
    # 3. Create target variables (predicting absolute price)
    print("\n[3/7] Creating target variables...")
    original_gas = df['gas_price'].values
    
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len].reset_index(drop=True)
    original_gas = original_gas[:min_len]
    
    # Create 1h target (shift forward by 12 steps)
    y_1h_raw = pd.Series(original_gas).shift(-12)
    
    valid = ~y_1h_raw.isna()
    X_1h = X_full[valid].reset_index(drop=True)
    y_1h = y_1h_raw[valid].reset_index(drop=True)
    
    print(f"✅ 1h targets: {len(y_1h)} samples")
    
    # 4. Remove outliers
    print("\n[4/7] Removing outliers...")
    X_clean, y_clean = remove_outliers_robust(X_1h, y_1h)
    
    print(f"✅ Clean data: {len(X_clean)} samples")
    
    # 5. Split and scale
    print("\n[5/7] Splitting and scaling data...")
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
    
    # 6. Train individual models
    print("\n[6/7] Training individual models...")
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    gb_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.85,
        random_state=42
    )
    
    print("   Training RandomForest...")
    rf_model.fit(X_train_scaled, y_train_scaled)
    
    print("   Training GradientBoosting...")
    gb_model.fit(X_train_scaled, y_train_scaled)
    
    # 7. Create ensemble and evaluate
    print("\n[7/7] Creating ensemble and evaluating...")
    
    # Ensemble with equal weights - fit it
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('gb', gb_model)
    ], weights=[0.5, 0.5])
    
    # Fit ensemble (it will use already-fitted models)
    ensemble.fit(X_train_scaled, y_train_scaled)
    
    # Predict with ensemble
    y_pred_scaled = ensemble.predict(X_test_scaled)
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
    
    # Also evaluate individual models for comparison
    rf_pred_scaled = rf_model.predict(X_test_scaled)
    rf_pred = target_scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()
    rf_r2 = r2_score(y_test, rf_pred)
    
    gb_pred_scaled = gb_model.predict(X_test_scaled)
    gb_pred = target_scaler.inverse_transform(gb_pred_scaled.reshape(-1, 1)).ravel()
    gb_r2 = r2_score(y_test, gb_pred)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"RandomForest R²:     {rf_r2:.4f} ({rf_r2*100:.2f}%)")
    print(f"GradientBoosting R²: {gb_r2:.4f} ({gb_r2*100:.2f}%)")
    print(f"ENSEMBLE R²:         {r2:.4f} ({r2*100:.2f}%) {'✅ GOOD' if r2 > 0.7 else '⚠️ NEEDS IMPROVEMENT' if r2 > 0.5 else '❌ POOR'}")
    print(f"ENSEMBLE MAE:        {mae:.6f} gwei")
    print(f"ENSEMBLE RMSE:       {rmse:.6f} gwei")
    print(f"Directional Accuracy: {dir_acc:.2%} {'✅ GOOD' if dir_acc > 0.7 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*70}")
    
    # Save ensemble model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_data = {
        'model': ensemble,
        'model_name': 'Ensemble (RF+GB)',
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': X_clean.columns.tolist(),
        'predicts_percentage_change': False,
        'uses_log_scale': False,
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
        },
        'trained_at': datetime.now().isoformat()
    }
    
    model_path = os.path.join(models_dir, 'model_1h.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_1h.pkl')
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    
    joblib.dump(save_data, model_path)
    joblib.dump(feature_scaler, scaler_path)
    joblib.dump(X_clean.columns.tolist(), features_path)
    
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    
    print("\n✅ Ensemble model trained and saved!")
    
    return ensemble, feature_scaler, target_scaler


if __name__ == '__main__':
    train_ensemble_final()

