#!/usr/bin/env python3
"""
Train improved model with feature selection, hyperparameter tuning, and external features
Target: R² > 70%, Directional Accuracy > 75%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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


def add_external_features(df):
    """Add external features like block times and network congestion estimates"""
    df = df.copy()
    
    # Estimate block time (Base L2 typically ~2 seconds)
    df['estimated_block_time'] = 2.0
    
    # Network congestion estimate (based on recent volatility)
    df['recent_volatility'] = df['gas_price'].rolling(window=12, min_periods=1).std()
    df['congestion_score'] = (df['recent_volatility'] / df['gas_price'].rolling(window=12, min_periods=1).mean()).fillna(0)
    
    # Time since last significant change
    df['gas_change'] = df['gas_price'].diff().abs()
    df['time_since_spike'] = 0
    spike_threshold = df['gas_price'].quantile(0.9)
    for i in range(1, len(df)):
        if df.iloc[i]['gas_price'] > spike_threshold:
            df.iloc[i, df.columns.get_loc('time_since_spike')] = 0
        else:
            df.iloc[i, df.columns.get_loc('time_since_spike')] = df.iloc[i-1, df.columns.get_loc('time_since_spike')] + 1
    
    # Market momentum (rate of change)
    df['momentum_1h'] = df['gas_price'].pct_change(12)
    df['momentum_4h'] = df['gas_price'].pct_change(48)
    
    # Fill NaN
    df = df.fillna(0)
    
    return df


def select_best_features(X, y, k=50, method='mutual_info'):
    """Select top k features using feature selection"""
    print(f"\n   Selecting top {k} features using {method}...")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = dict(zip(X.columns, selector.scores_))
    
    print(f"   Selected {len(selected_features)} features")
    print(f"   Top 10 features: {sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features, selector


def train_improved_v3():
    """Train improved model with all optimizations"""
    
    print("="*70)
    print("TRAINING IMPROVED MODEL V3 (Target: R² > 70%)")
    print("="*70)
    
    # 1. Load MORE data
    print("\n[1/8] Loading data...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=2160)  # 90 days for more data
    
    if len(historical) < 100:
        print(f"⚠️ Only {len(historical)} records, generating more data...")
        # Generate synthetic data if needed
        from scripts.generate_synthetic_data import generate_synthetic_data
        generate_synthetic_data()
        historical = db.get_historical_data(hours=2160)
    
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
    print("\n[2/8] Creating advanced features...")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
    
    df = df[df['gas_price'] > 0].reset_index(drop=True)
    
    X_full, y_full = create_advanced_features(df)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ {X_full.shape[0]} samples")
    
    # 3. Add external features
    print("\n[3/8] Adding external features...")
    df_with_external = add_external_features(df)
    
    # Merge external features with X_full
    external_features = ['estimated_block_time', 'recent_volatility', 'congestion_score', 
                        'time_since_spike', 'momentum_1h', 'momentum_4h']
    
    for feat in external_features:
        if feat in df_with_external.columns:
            X_full[feat] = df_with_external[feat].values[:len(X_full)]
    
    X_full = X_full.fillna(0)
    print(f"✅ Added {len(external_features)} external features")
    print(f"✅ Total features: {X_full.shape[1]}")
    
    # 4. Create target variables
    print("\n[4/8] Creating target variables...")
    original_gas = df['gas_price'].values
    
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len].reset_index(drop=True)
    original_gas = original_gas[:min_len]
    
    y_1h_raw = pd.Series(original_gas).shift(-12)
    
    valid = ~y_1h_raw.isna()
    X_1h = X_full[valid].reset_index(drop=True)
    y_1h = y_1h_raw[valid].reset_index(drop=True)
    
    print(f"✅ 1h targets: {len(y_1h)} samples")
    
    # 5. Remove outliers
    print("\n[5/8] Removing outliers...")
    X_clean, y_clean = remove_outliers_robust(X_1h, y_1h)
    
    print(f"✅ Clean data: {len(X_clean)} samples")
    
    # 6. Feature selection
    print("\n[6/8] Feature selection...")
    # Select top 60 features (reduce from 110+ to most important)
    X_selected, selected_features, feature_selector = select_best_features(
        X_clean, y_clean, k=60, method='mutual_info'
    )
    
    # 7. Split and scale
    print("\n[7/8] Splitting and scaling data...")
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    print("✅ Features and targets scaled")
    
    # 8. Train with hyperparameter tuning
    print("\n[8/8] Training models with hyperparameter tuning...")
    
    # Use TimeSeriesSplit for cross-validation (more splits for better validation)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # RandomForest with grid search (more conservative to reduce overfitting)
    print("   Tuning RandomForest...")
    rf_param_grid = {
        'n_estimators': [400, 500],
        'max_depth': [15, 20, 25],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt']
    }
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid, cv=tscv, 
        scoring='r2', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_scaled, y_train_scaled)
    rf_model = rf_grid.best_estimator_
    print(f"      Best RF params: {rf_grid.best_params_}")
    print(f"      Best RF CV score: {rf_grid.best_score_:.4f}")
    
    # GradientBoosting with grid search (more conservative)
    print("   Tuning GradientBoosting...")
    gb_param_grid = {
        'n_estimators': [400, 500],
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 0.85],
        'min_samples_split': [5, 10]
    }
    
    gb_base = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(
        gb_base, gb_param_grid, cv=tscv,
        scoring='r2', n_jobs=-1, verbose=0
    )
    gb_grid.fit(X_train_scaled, y_train_scaled)
    gb_model = gb_grid.best_estimator_
    print(f"      Best GB params: {gb_grid.best_params_}")
    print(f"      Best GB CV score: {gb_grid.best_score_:.4f}")
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('gb', gb_model)
    ], weights=[0.5, 0.5])
    ensemble.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate
    y_pred_scaled = ensemble.predict(X_test_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
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
    
    # Individual model performance
    rf_pred_scaled = rf_model.predict(X_test_scaled)
    rf_pred = target_scaler.inverse_transform(rf_pred_scaled.reshape(-1, 1)).ravel()
    rf_r2 = r2_score(y_test, rf_pred)
    
    gb_pred_scaled = gb_model.predict(X_test_scaled)
    gb_pred = target_scaler.inverse_transform(gb_pred_scaled.reshape(-1, 1)).ravel()
    gb_r2 = r2_score(y_test, gb_pred)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"{'='*70}")
    print(f"RandomForest R²:     {rf_r2:.4f} ({rf_r2*100:.2f}%)")
    print(f"GradientBoosting R²: {gb_r2:.4f} ({gb_r2*100:.2f}%)")
    print(f"ENSEMBLE R²:         {r2:.4f} ({r2*100:.2f}%) {'✅ EXCELLENT' if r2 > 0.7 else '✅ GOOD' if r2 > 0.6 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"ENSEMBLE MAE:        {mae:.6f} gwei")
    print(f"ENSEMBLE RMSE:       {rmse:.6f} gwei")
    print(f"Directional Accuracy: {dir_acc:.2%} {'✅ EXCELLENT' if dir_acc > 0.75 else '✅ GOOD' if dir_acc > 0.65 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*70}")
    
    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_data = {
        'model': ensemble,
        'model_name': 'Ensemble (RF+GB) - Optimized',
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': selected_features,  # Save selected features only
        'feature_selector': feature_selector,  # Save selector for prediction
        'predicts_percentage_change': False,
        'uses_log_scale': False,
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
        },
        'trained_at': datetime.now().isoformat(),
        'hyperparameters': {
            'rf': rf_grid.best_params_,
            'gb': gb_grid.best_params_
        }
    }
    
    model_path = os.path.join(models_dir, 'model_1h.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_1h.pkl')
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    
    joblib.dump(save_data, model_path)
    joblib.dump(feature_scaler, scaler_path)
    joblib.dump(selected_features, features_path)
    
    print(f"\n✅ Model saved!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Features: {len(selected_features)} selected features")
    
    return ensemble, feature_scaler, target_scaler


if __name__ == '__main__':
    train_improved_v3()

