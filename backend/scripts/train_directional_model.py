#!/usr/bin/env python3
"""
Train model optimized for directional accuracy (predicting up/down)
Target: Directional Accuracy > 75%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def add_directional_features(df):
    """Add features specifically for direction prediction"""
    df = df.copy()
    
    # Momentum indicators (strong predictors of direction)
    df['momentum_5min'] = df['gas_price'].pct_change(1).fillna(0)
    df['momentum_15min'] = df['gas_price'].pct_change(3).fillna(0)
    df['momentum_30min'] = df['gas_price'].pct_change(6).fillna(0)
    df['momentum_1h'] = df['gas_price'].pct_change(12).fillna(0)
    df['momentum_2h'] = df['gas_price'].pct_change(24).fillna(0)
    
    # Rate of change acceleration
    df['acceleration'] = df['momentum_1h'].diff().fillna(0)
    
    # Volatility (high volatility = more direction changes)
    df['volatility_1h'] = df['gas_price'].rolling(window=12, min_periods=1).std().fillna(0)
    df['volatility_4h'] = df['gas_price'].rolling(window=48, min_periods=1).std().fillna(0)
    df['volatility_ratio'] = (df['volatility_1h'] / (df['volatility_4h'] + 1e-8)).fillna(0)
    
    # Price position relative to recent range
    df['price_position'] = 0.0
    for i in range(12, len(df)):
        window = df['gas_price'].iloc[i-12:i]
        if len(window) > 0 and window.max() > window.min():
            df.iloc[i, df.columns.get_loc('price_position')] = (
                (df.iloc[i]['gas_price'] - window.min()) / (window.max() - window.min())
            )
    
    # Trend strength (how consistent is the direction)
    df['trend_strength'] = df['gas_price'].rolling(window=12, min_periods=1).apply(
        lambda x: 1 if len(x) < 2 else np.corrcoef(range(len(x)), x)[0, 1] if not np.isnan(np.corrcoef(range(len(x)), x)[0, 1]) else 0
    ).fillna(0)
    
    # Mean reversion indicator
    ma_12 = df['gas_price'].rolling(window=12, min_periods=1).mean()
    ma_24 = df['gas_price'].rolling(window=24, min_periods=1).mean()
    df['mean_reversion'] = ((df['gas_price'] - ma_12) / (ma_12 + 1e-8)).fillna(0)
    df['ma_cross'] = ((ma_12 - ma_24) / (ma_24 + 1e-8)).fillna(0)
    
    # External features
    df['estimated_block_time'] = 2.0
    df['recent_volatility'] = df['volatility_1h']
    df['congestion_score'] = (df['recent_volatility'] / (ma_12 + 1e-8)).fillna(0)
    
    return df


def train_directional_model():
    """Train model optimized for directional accuracy"""
    
    print("="*70)
    print("TRAINING DIRECTIONAL MODEL (Target: >75% Accuracy)")
    print("="*70)
    
    # 1. Load data
    print("\n[1/8] Loading data...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=2160)  # 90 days
    
    if len(historical) < 100:
        print(f"⚠️ Only {len(historical)} records, generating more data...")
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
    print("\n[2/8] Creating features...")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
    
    df = df[df['gas_price'] > 0].reset_index(drop=True)
    
    X_full, _ = create_advanced_features(df)
    
    # Add directional-specific features
    df_with_directional = add_directional_features(df)
    
    # Merge directional features
    directional_features = [
        'momentum_5min', 'momentum_15min', 'momentum_30min', 'momentum_1h', 'momentum_2h',
        'acceleration', 'volatility_1h', 'volatility_4h', 'volatility_ratio',
        'price_position', 'trend_strength', 'mean_reversion', 'ma_cross',
        'estimated_block_time', 'recent_volatility', 'congestion_score'
    ]
    
    for feat in directional_features:
        if feat in df_with_directional.columns:
            X_full[feat] = df_with_directional[feat].values[:len(X_full)]
    
    X_full = X_full.fillna(0)
    
    print(f"✅ Created {X_full.shape[1]} features")
    print(f"✅ {X_full.shape[0]} samples")
    
    # 3. Create DIRECTIONAL target (1 = up, 0 = down)
    print("\n[3/8] Creating directional targets...")
    original_gas = df['gas_price'].values
    
    min_len = min(len(X_full), len(original_gas))
    X_full = X_full.iloc[:min_len].reset_index(drop=True)
    original_gas = original_gas[:min_len]
    
    # Create 1h target - predict direction of change
    y_1h_future = pd.Series(original_gas).shift(-12)
    y_1h_current = pd.Series(original_gas)
    
    # Direction: 1 if going up, 0 if going down
    y_1h_direction = (y_1h_future > y_1h_current).astype(int)
    
    valid = ~y_1h_direction.isna()
    X_1h = X_full[valid].reset_index(drop=True)
    y_1h = y_1h_direction[valid].reset_index(drop=True)
    
    print(f"✅ 1h directional targets: {len(y_1h)} samples")
    print(f"   Up: {y_1h.sum()} ({y_1h.sum()/len(y_1h)*100:.1f}%)")
    print(f"   Down: {(~y_1h.astype(bool)).sum()} ({(~y_1h.astype(bool)).sum()/len(y_1h)*100:.1f}%)")
    
    # 4. Remove outliers (based on price, not direction)
    print("\n[4/8] Removing outliers...")
    price_target = pd.Series(original_gas)[valid].reset_index(drop=True)
    X_clean, _ = remove_outliers_robust(X_1h, price_target)
    y_clean = y_1h[X_clean.index].reset_index(drop=True)
    
    print(f"✅ Clean data: {len(X_clean)} samples")
    
    # 5. Feature selection for classification
    print("\n[5/8] Feature selection for direction...")
    selector = SelectKBest(score_func=mutual_info_classif, k=min(70, X_clean.shape[1]))
    X_selected = selector.fit_transform(X_clean, y_clean)
    selected_features = X_clean.columns[selector.get_support()].tolist()
    
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index)
    
    print(f"✅ Selected {len(selected_features)} features")
    
    # 6. Split and scale
    print("\n[6/8] Splitting and scaling data...")
    split_idx = int(len(X_selected_df) * 0.8)
    X_train, X_test = X_selected_df.iloc[:split_idx], X_selected_df.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    print("✅ Features scaled")
    
    # 7. Train classification models
    print("\n[7/8] Training classification models...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # RandomForest Classifier
    print("   Tuning RandomForest Classifier...")
    rf_param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [15, 20, 25],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced', None]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_scaled, y_train)
    rf_model = rf_grid.best_estimator_
    print(f"      Best RF CV accuracy: {rf_grid.best_score_:.4f}")
    
    # GradientBoosting Classifier
    print("   Tuning GradientBoosting Classifier...")
    gb_param_grid = {
        'n_estimators': [300, 400],
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 0.85]
    }
    
    gb_base = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(
        gb_base, gb_param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    gb_grid.fit(X_train_scaled, y_train)
    gb_model = gb_grid.best_estimator_
    print(f"      Best GB CV accuracy: {gb_grid.best_score_:.4f}")
    
    # Ensemble
    ensemble = VotingClassifier([
        ('rf', rf_model),
        ('gb', gb_model)
    ], voting='soft', weights=[0.5, 0.5])
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Also get probabilities for confidence
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n{'='*70}")
    print("DIRECTIONAL ACCURACY RESULTS:")
    print(f"{'='*70}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) {'✅ EXCELLENT' if accuracy > 0.75 else '✅ GOOD' if accuracy > 0.65 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*70}")
    
    # 8. Save model
    print("\n[8/8] Saving model...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_data = {
        'model': ensemble,
        'model_name': 'Directional Classifier (RF+GB)',
        'feature_scaler': feature_scaler,
        'feature_names': selected_features,
        'feature_selector': selector,
        'predicts_direction': True,  # Flag that this predicts direction
        'metrics': {
            'accuracy': accuracy,
            'rf_cv_score': rf_grid.best_score_,
            'gb_cv_score': gb_grid.best_score_
        },
        'trained_at': datetime.now().isoformat(),
        'hyperparameters': {
            'rf': rf_grid.best_params_,
            'gb': gb_grid.best_params_
        }
    }
    
    model_path = os.path.join(models_dir, 'model_directional_1h.pkl')
    joblib.dump(save_data, model_path)
    joblib.dump(feature_scaler, os.path.join(models_dir, 'scaler_directional_1h.pkl'))
    joblib.dump(selected_features, os.path.join(models_dir, 'feature_names_directional_1h.pkl'))
    
    print(f"✅ Model saved to {model_path}")
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("1. Update API to use directional model for direction prediction")
    print("2. Combine with regression model for magnitude")
    print("3. Use direction + magnitude for final predictions")
    print(f"{'='*70}")
    
    return ensemble, feature_scaler


if __name__ == '__main__':
    train_directional_model()

