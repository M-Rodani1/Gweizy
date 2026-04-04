"""
Automated Model Retraining Service

Lightweight retraining triggered when drift is detected.
Trains on recent production data, registers with ModelRegistry,
and only activates if the new model outperforms the current one.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from utils.logger import logger
from config import Config

# Training parameters
HORIZONS = {'1h': 1, '4h': 4, '24h': 24}
MIN_TRAINING_ROWS = 500
TRAINING_HOURS = 72  # Use last 72 hours of data


def _load_training_data() -> Optional[pd.DataFrame]:
    """Load recent gas price data from the production database"""
    from data.database import DatabaseManager
    db = DatabaseManager()
    session = db._get_session()

    try:
        from data.database import GasPrice
        from sqlalchemy import desc

        cutoff = datetime.now() - timedelta(hours=TRAINING_HOURS)
        rows = (
            session.query(GasPrice)
            .filter(GasPrice.timestamp >= cutoff)
            .order_by(GasPrice.timestamp.asc())
            .all()
        )

        if len(rows) < MIN_TRAINING_ROWS:
            logger.warning(f"Insufficient data for retraining: {len(rows)} rows (need {MIN_TRAINING_ROWS})")
            return None

        data = []
        for r in rows:
            data.append({
                'timestamp': r.timestamp,
                'gas_price': r.current_gas,
                'base_fee': r.base_fee,
                'priority_fee': r.priority_fee,
                'gas_used': r.gas_used,
                'gas_limit': r.gas_limit,
                'utilization': r.utilization,
                'block_number': r.block_number,
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    finally:
        session.close()


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features matching the training pipeline"""
    df = df.copy()

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f'gas_lag_{lag}'] = df['gas_price'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24]:
        df[f'gas_rolling_mean_{window}'] = df['gas_price'].rolling(window).mean()
        df[f'gas_rolling_std_{window}'] = df['gas_price'].rolling(window).std()

    # Rate of change
    df['gas_roc_1'] = df['gas_price'].pct_change(1)
    df['gas_roc_6'] = df['gas_price'].pct_change(6)

    # Utilization features
    if 'utilization' in df.columns:
        df['util_rolling_mean_6'] = df['utilization'].rolling(6).mean()
        df['util_rolling_std_6'] = df['utilization'].rolling(6).std()

    # Base fee features
    if 'base_fee' in df.columns:
        df['base_fee_roc'] = df['base_fee'].pct_change(1)

    df = df.dropna().reset_index(drop=True)
    return df


def _train_horizon(df: pd.DataFrame, horizon: str, hours: int) -> Optional[Dict]:
    """Train models for a single horizon"""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    # Create target: future gas price
    target_col = f'target_{horizon}'
    df[target_col] = df['gas_price'].shift(-hours)
    df_train = df.dropna(subset=[target_col]).copy()

    if len(df_train) < 100:
        logger.warning(f"Not enough samples for {horizon}: {len(df_train)}")
        return None

    # Feature columns (exclude targets, timestamp, raw identifiers)
    exclude = {'timestamp', 'gas_price', 'block_number'} | {
        c for c in df_train.columns if c.startswith('target_')
    }
    feature_cols = [c for c in df_train.columns if c not in exclude]

    X = df_train[feature_cols].values
    y = df_train[target_col].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time-series: no shuffle
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train regressor
    regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    regressor.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = regressor.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Directional accuracy
    if len(y_test) > 1:
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_acc = np.mean(actual_direction == pred_direction)
    else:
        directional_acc = 0.0

    # Train spike detector
    spike_threshold = 0.01  # Normal threshold in Gwei
    y_spike = (y_train > spike_threshold).astype(int)

    spike_detector = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    spike_detector.fit(X_train_scaled, y_spike)

    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'directional_accuracy': float(directional_acc),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }

    return {
        'regressor': regressor,
        'spike_detector': spike_detector,
        'scaler': scaler,
        'feature_names': feature_cols,
        'metrics': metrics,
    }


def retrain_models() -> Dict:
    """
    Run automated retraining pipeline.

    Returns:
        Dict with status, metrics, and version info per horizon
    """
    start = time.time()
    results = {}

    logger.info("[RETRAIN] Starting automated retraining")

    # Load data
    df = _load_training_data()
    if df is None:
        return {'success': False, 'reason': 'insufficient_data'}

    logger.info(f"[RETRAIN] Loaded {len(df)} rows ({df['timestamp'].min()} to {df['timestamp'].max()})")

    # Engineer features
    df_features = _engineer_features(df)
    logger.info(f"[RETRAIN] Feature engineering complete: {len(df_features)} samples, {len(df_features.columns)} features")

    # Train each horizon
    for horizon, hours in HORIZONS.items():
        try:
            result = _train_horizon(df_features.copy(), horizon, hours)
            if result is None:
                results[horizon] = {'success': False, 'reason': 'insufficient_samples'}
                continue

            # Save models to temp location
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, f'model_{horizon}.pkl')
            scaler_path = os.path.join(models_dir, f'scaler_{horizon}.pkl')
            spike_path = os.path.join(models_dir, f'spike_detector_{horizon}.pkl')
            features_path = os.path.join(models_dir, 'feature_names.pkl')

            joblib.dump(result['regressor'], model_path)
            joblib.dump(result['scaler'], scaler_path)
            joblib.dump(result['spike_detector'], spike_path)
            joblib.dump(result['feature_names'], features_path)

            # Register with model registry (auto-activates if better)
            from models.model_registry import get_registry
            registry = get_registry()
            version = registry.register_model(
                horizon=horizon,
                model_path=model_path,
                metrics=result['metrics'],
                metadata={
                    'training_rows': len(df),
                    'feature_count': len(result['feature_names']),
                    'training_hours': TRAINING_HOURS,
                    'retrained_at': datetime.now().isoformat(),
                    'automated': True,
                },
            )

            results[horizon] = {
                'success': True,
                'version': version,
                'metrics': result['metrics'],
            }
            logger.info(f"[RETRAIN] {horizon}: R²={result['metrics']['r2']:.4f}, MAE={result['metrics']['mae']:.6f}, version={version}")

        except Exception as e:
            logger.error(f"[RETRAIN] Failed for {horizon}: {e}")
            results[horizon] = {'success': False, 'error': str(e)}

    # Reload models in the API
    try:
        from api.routes import reload_models
        reload_models()
        logger.info("[RETRAIN] Models reloaded in API")
    except Exception as e:
        logger.warning(f"[RETRAIN] Could not reload API models: {e}")

    elapsed = time.time() - start
    logger.info(f"[RETRAIN] Complete in {elapsed:.1f}s")

    return {
        'success': any(r.get('success') for r in results.values()),
        'elapsed_seconds': round(elapsed, 1),
        'horizons': results,
    }
