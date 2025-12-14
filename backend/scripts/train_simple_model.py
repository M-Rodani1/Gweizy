#!/usr/bin/env python3
"""
Train simpler model for testing - focuses on getting R² > 50% first
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from data.database import DatabaseManager


def create_simple_features(df):
    """Create only the most important features"""
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Only most important lags
    for lag in [1, 2, 3, 6, 12, 24, 48]:  # Last few values + 1hr + 2hr + 4hr
        df[f'lag_{lag}'] = df['gas_price'].shift(lag)
    
    # Only essential rolling stats
    for window in [12, 24, 72]:  # 1hr, 2hr, 6hr
        df[f'ma_{window}'] = df['gas_price'].rolling(window, min_periods=1).mean()
        df[f'std_{window}'] = df['gas_price'].rolling(window, min_periods=1).std()
    
    # Recent trend
    df['trend_1h'] = df['gas_price'].diff(12)
    df['trend_3h'] = df['gas_price'].diff(36)
    df['pct_change_1h'] = df['gas_price'].pct_change(12)
    
    # Fill NaN with 0
    df = df.fillna(0)
    
    features = [col for col in df.columns if col not in ['timestamp', 'gas_price', 'gas', 'current_gas', 'gwei']]
    
    return df[features], df['gas_price']


def remove_outliers(X, y):
    """Remove outliers using IQR method"""
    
    # Calculate IQR for target variable
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 3 * iqr  # Use 3 IQR for less aggressive removal
    upper_bound = q3 + 3 * iqr
    
    # Filter
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    removed = (~mask).sum()
    print(f"Removed {removed} outliers ({removed/len(y)*100:.1f}%)")
    if removed > 0:
        print(f"Outlier range: < {lower_bound:.6f} or > {upper_bound:.6f}")
    
    return X[mask], y[mask]


def train_simple_model():
    """Train simpler model for testing"""
    
    print("="*70)
    print("TRAINING SIMPLE MODEL (Testing Approach)")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
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
    
    # Create simple features
    print("\n[2/5] Creating simple features...")
    df = pd.DataFrame(data)
    X, y = create_simple_features(df)
    
    print(f"✅ Created {X.shape[1]} features")
    print(f"✅ {X.shape[0]} samples")
    
    # Remove outliers
    print("\n[3/5] Removing outliers...")
    X_clean, y_clean = remove_outliers(X, y)
    
    # Create target variables (predicting absolute price, not percentage change)
    # For 1h: shift forward by 12 steps (1 hour at 5min intervals)
    y_1h_raw = y_clean.shift(-12)
    
    # Align
    valid = ~y_1h_raw.isna()
    X_1h = X_clean[valid].reset_index(drop=True)
    y_1h = y_1h_raw[valid].reset_index(drop=True)
    
    print(f"✅ 1h targets: {len(y_1h)} samples")
    
    # Split
    print("\n[4/5] Splitting data...")
    split_idx = int(len(X_1h) * 0.8)
    X_train, X_test = X_1h.iloc[:split_idx], X_1h.iloc[split_idx:]
    y_train, y_test = y_1h.iloc[:split_idx], y_1h.iloc[split_idx:]
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target variable too!
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    print("✅ Features and targets scaled")
    
    # Train model
    print("\n[5/5] Training model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict on scaled data
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform back to original scale
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
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"MAE:  {mae:.6f} gwei")
    print(f"RMSE: {rmse:.6f} gwei")
    print(f"R²:   {r2:.4f} ({r2*100:.2f}%) {'✅ GOOD' if r2 > 0.5 else '⚠️ NEEDS IMPROVEMENT' if r2 > 0.2 else '❌ POOR'}")
    print(f"Directional Accuracy: {dir_acc:.2%} {'✅ GOOD' if dir_acc > 0.65 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*70}")
    
    # Save model
    os.makedirs('backend/models/saved_models', exist_ok=True)
    
    save_data = {
        'model': model,
        'model_name': 'RandomForest',
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': X_1h.columns.tolist(),
        'predicts_percentage_change': False,  # This predicts absolute price
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
        },
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(save_data, 'backend/models/saved_models/model_1h.pkl')
    joblib.dump(feature_scaler, 'backend/models/saved_models/scaler_1h.pkl')
    joblib.dump(X_1h.columns.tolist(), 'backend/models/saved_models/feature_names.pkl')
    
    print("\n✅ Simple model trained and saved!")
    print(f"   Model: backend/models/saved_models/model_1h.pkl")
    print(f"   Scaler: backend/models/saved_models/scaler_1h.pkl")
    
    return model, feature_scaler, target_scaler


if __name__ == '__main__':
    train_simple_model()

