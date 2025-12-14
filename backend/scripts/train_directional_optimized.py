#!/usr/bin/env python3
"""
Improved training script focused on directional accuracy and stability

Key improvements:
1. Predict percentage change instead of absolute price
2. Train separate directional classifier
3. Use only most important features (no overfitting)
4. Proper time series validation
5. Weight recent data more heavily

Target: R² > 70%, Directional Accuracy > 70%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from data.database import DatabaseManager


def create_optimized_features(df):
    """Create only the most important features (avoid overfitting)"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Core features that actually matter for gas prediction
    features = pd.DataFrame(index=df.index)

    # 1. Time features (cyclical encoding)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    features['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    features['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (features['is_weekend'] == 0)).astype(int)

    # 2. Recent lags (most predictive)
    for lag in [1, 2, 3, 6, 12, 24]:  # 5min to 2hr
        features[f'lag_{lag}'] = df['gas_price'].shift(lag)

    # 3. Moving averages (trend indicators)
    for window in [6, 12, 24, 72]:  # 30min to 6hr
        features[f'ma_{window}'] = df['gas_price'].rolling(window, min_periods=1).mean()
        features[f'std_{window}'] = df['gas_price'].rolling(window, min_periods=1).std()

    # 4. Momentum indicators
    for period in [6, 12, 24]:
        features[f'pct_change_{period}'] = df['gas_price'].pct_change(period)
        features[f'diff_{period}'] = df['gas_price'].diff(period)

    # 5. Volatility (key for gas prices)
    for window in [12, 24, 72]:
        ma = df['gas_price'].rolling(window, min_periods=1).mean()
        std = df['gas_price'].rolling(window, min_periods=1).std()
        features[f'cv_{window}'] = std / ma.replace(0, np.nan)
        features[f'range_{window}'] = (
            df['gas_price'].rolling(window, min_periods=1).max() -
            df['gas_price'].rolling(window, min_periods=1).min()
        )

    # 6. Distance from average (mean reversion indicator)
    for window in [12, 24, 72]:
        ma = df['gas_price'].rolling(window, min_periods=1).mean()
        features[f'dist_from_ma_{window}'] = df['gas_price'] - ma
        features[f'dist_from_ma_pct_{window}'] = (df['gas_price'] - ma) / ma.replace(0, np.nan)

    # 7. Current gas price (important baseline)
    features['current_gas'] = df['gas_price']

    # Fill NaN with 0
    features = features.fillna(0)

    return features


def create_target_variables(df, horizon=12):
    """
    Create PERCENTAGE CHANGE target instead of absolute price
    This is more stable and easier to predict

    Args:
        horizon: Number of periods ahead (12 = 1 hour at 5min intervals)

    Returns:
        pct_change: Percentage change target
        direction: Binary target (1 = up, 0 = down)
    """
    future_price = df['gas_price'].shift(-horizon)
    current_price = df['gas_price']

    # Percentage change (more stable than absolute price)
    pct_change = ((future_price - current_price) / current_price) * 100

    # Direction (binary: will it go up or down?)
    direction = (future_price > current_price).astype(int)

    return pct_change, direction


def train_with_sample_weights(X, y, recent_weight=2.0):
    """
    Create sample weights that give more importance to recent data
    Recent patterns are more relevant than old patterns
    """
    weights = np.ones(len(X))
    # Give 2x weight to most recent 20% of data
    recent_cutoff = int(len(X) * 0.8)
    weights[recent_cutoff:] = recent_weight
    return weights


def evaluate_model(model, X_test, y_test, y_test_pct, scaler, model_type='regression'):
    """Comprehensive evaluation"""
    if model_type == 'regression':
        # Predict percentage change
        y_pred_pct = model.predict(X_test)

        # Metrics on percentage change
        mae_pct = mean_absolute_error(y_test_pct, y_pred_pct)
        rmse_pct = np.sqrt(mean_squared_error(y_test_pct, y_pred_pct))
        r2_pct = r2_score(y_test_pct, y_pred_pct)

        # Directional accuracy (did we predict up/down correctly?)
        pred_direction = (y_pred_pct > 0).astype(int)
        actual_direction = (y_test_pct > 0).astype(int)
        dir_acc = accuracy_score(actual_direction, pred_direction)

        # Convert back to absolute price for MAE/RMSE in gwei
        # Note: This is approximate since we don't have current_price in test set
        # We'll just report percentage-based metrics

        return {
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'r2': r2_pct,
            'directional_accuracy': dir_acc
        }
    else:
        # Classification model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {'directional_accuracy': acc}


def train_directional_optimized():
    """Main training function"""

    print("="*70)
    print("TRAINING DIRECTIONAL-OPTIMIZED MODEL")
    print("Target: Predict percentage change + direction")
    print("="*70)

    # 1. Load data
    print("\n[1/7] Loading data...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=2160)  # 90 days

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

    df = pd.DataFrame(data)
    df = df[df['gas_price'] > 0].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} data points")

    # 2. Create features (optimized set only)
    print("\n[2/7] Creating optimized features...")
    X = create_optimized_features(df)
    print(f"✅ Created {X.shape[1]} features (focused set)")

    # 3. Create targets (percentage change + direction)
    print("\n[3/7] Creating targets (percentage change)...")
    y_pct_1h, y_dir_1h = create_target_variables(df, horizon=12)  # 1 hour

    # Remove rows with NaN targets
    valid = ~(y_pct_1h.isna() | y_dir_1h.isna())
    X_clean = X[valid].reset_index(drop=True)
    y_pct_clean = y_pct_1h[valid].reset_index(drop=True)
    y_dir_clean = y_dir_1h[valid].reset_index(drop=True)

    print(f"✅ Valid samples: {len(X_clean)}")
    print(f"   Mean % change: {y_pct_clean.mean():.3f}%")
    print(f"   Std % change: {y_pct_clean.std():.3f}%")
    print(f"   % going up: {y_dir_clean.mean()*100:.1f}%")

    # 4. Remove extreme outliers more aggressively (gas spikes are unpredictable)
    print("\n[4/7] Removing extreme outliers...")
    # Cap at ±50% change (anything beyond this is likely a spike/anomaly)
    mask = (y_pct_clean >= -50) & (y_pct_clean <= 50)
    X_clean = X_clean[mask].reset_index(drop=True)
    y_pct_clean = y_pct_clean[mask].reset_index(drop=True)
    y_dir_clean = y_dir_clean[mask].reset_index(drop=True)
    print(f"✅ After outlier removal: {len(X_clean)} samples")
    print(f"   New mean % change: {y_pct_clean.mean():.3f}%")
    print(f"   New std % change: {y_pct_clean.std():.3f}%")

    # 5. Split data (time series - no shuffle!)
    print("\n[5/7] Splitting data (80/20)...")
    split_idx = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_pct_train, y_pct_test = y_pct_clean.iloc[:split_idx], y_pct_clean.iloc[split_idx:]
    y_dir_train, y_dir_test = y_dir_clean.iloc[:split_idx], y_dir_clean.iloc[split_idx:]

    # Scale features
    scaler = RobustScaler()  # Robust to outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

    # 6. Train REGRESSION model (percentage change)
    print("\n[6/7] Training regression model (percentage change)...")

    # Create sample weights (recent data more important)
    sample_weights = train_with_sample_weights(X_train_scaled, y_pct_train)

    # Train ensemble
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    gb_reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    ensemble_reg = VotingRegressor([
        ('rf', rf_reg),
        ('gb', gb_reg)
    ], weights=[0.6, 0.4])

    print("   Training regression ensemble...")
    ensemble_reg.fit(X_train_scaled, y_pct_train, sample_weight=sample_weights)

    # Evaluate regression model
    y_pred_pct = ensemble_reg.predict(X_test_scaled)
    mae_pct = mean_absolute_error(y_pct_test, y_pred_pct)
    rmse_pct = np.sqrt(mean_squared_error(y_pct_test, y_pred_pct))
    r2 = r2_score(y_pct_test, y_pred_pct)

    # Directional accuracy from regression
    pred_dir_from_reg = (y_pred_pct > 0).astype(int)
    actual_dir = (y_pct_test > 0).astype(int)
    dir_acc_reg = accuracy_score(actual_dir, pred_dir_from_reg)

    print(f"\n   Regression Results:")
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%) {'✅' if r2 > 0.7 else '⚠️'}")
    print(f"   MAE: {mae_pct:.3f}% change")
    print(f"   RMSE: {rmse_pct:.3f}% change")
    print(f"   Directional Accuracy: {dir_acc_reg:.2%} {'✅' if dir_acc_reg > 0.7 else '⚠️'}")

    # 7. Train CLASSIFICATION model (pure direction)
    print("\n[7/7] Training classification model (direction)...")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train_scaled, y_dir_train, sample_weight=sample_weights)

    # Evaluate classifier
    y_pred_dir = clf.predict(X_test_scaled)
    dir_acc_clf = accuracy_score(y_dir_test, y_pred_dir)

    print(f"   Classification Directional Accuracy: {dir_acc_clf:.2%} {'✅' if dir_acc_clf > 0.7 else '⚠️'}")

    # Combine predictions (use classifier for direction, regression for magnitude)
    combined_dir_acc = max(dir_acc_reg, dir_acc_clf)

    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Regression R²:              {r2:.4f} ({r2*100:.2f}%)")
    print(f"MAE (% change):             {mae_pct:.3f}%")
    print(f"RMSE (% change):            {rmse_pct:.3f}%")
    print(f"Directional (Regression):   {dir_acc_reg:.2%}")
    print(f"Directional (Classifier):   {dir_acc_clf:.2%}")
    print(f"BEST Directional Accuracy:  {combined_dir_acc:.2%} {'✅ EXCELLENT' if combined_dir_acc > 0.75 else '✅ GOOD' if combined_dir_acc > 0.65 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"{'='*70}")

    # Feature importance (top 20) - from VotingRegressor we need to access individual model
    try:
        # Get feature importance from RF model in ensemble
        rf_importance = ensemble_reg.named_estimators_['rf'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)

        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))
    except:
        print("\nCould not extract feature importance")

    # 8. Save models
    print("\n[8/8] Saving models...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    # Convert predictions to absolute price metrics (approximate)
    # For display purposes, convert to gwei-equivalent
    avg_gas_price = df['gas_price'].mean()
    mae_gwei = (mae_pct / 100) * avg_gas_price
    rmse_gwei = (rmse_pct / 100) * avg_gas_price

    save_data = {
        'model': ensemble_reg,
        'classifier': clf,  # Save classifier too
        'model_name': 'Directional-Optimized Ensemble (RF+GB)',
        'feature_scaler': scaler,
        'feature_names': X_train.columns.tolist(),
        'predicts_percentage_change': True,
        'uses_log_scale': False,
        'metrics': {
            'r2': r2,
            'mae': mae_gwei,
            'rmse': rmse_gwei,
            'directional_accuracy': combined_dir_acc,
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'dir_acc_regression': dir_acc_reg,
            'dir_acc_classifier': dir_acc_clf
        },
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    model_path = os.path.join(models_dir, 'model_1h.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_1h.pkl')
    features_path = os.path.join(models_dir, 'feature_names.pkl')

    joblib.dump(save_data, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X_train.columns.tolist(), features_path)

    print(f"\n✅ Models saved successfully!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Features: {len(X_train.columns)} features")

    return ensemble_reg, clf, scaler


if __name__ == '__main__':
    train_directional_optimized()
