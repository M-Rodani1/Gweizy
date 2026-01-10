#!/usr/bin/env python3
"""
Spike Detector Training Script

Trains classification models to predict upcoming gas price conditions:
- Normal: < 0.01 Gwei (optimal for transactions)
- Elevated: 0.01 - 0.05 Gwei (moderate activity)
- Spike: > 0.05 Gwei (high activity, recommend waiting)

These models are used by HybridPredictor for classification-based predictions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# Thresholds (must match HybridPredictor)
NORMAL_THRESHOLD = 0.01
ELEVATED_THRESHOLD = 0.05

# Railway environment detection
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None


def fetch_training_data(hours=2160):
    """Fetch data from database (default: 90 days)"""
    print(f"📊 Fetching {hours} hours of data for spike detector training...", flush=True)

    from data.database import DatabaseManager
    db = DatabaseManager()
    data = db.get_historical_data(hours=hours)

    if not data:
        raise ValueError("No data available in database")

    print(f"✅ Fetched {len(data):,} records", flush=True)
    return data


def prepare_dataframe(data):
    """Convert raw data to DataFrame with proper types"""
    df = pd.DataFrame(data)

    # Normalize column names
    if 'gas_price' not in df.columns and 'current_gas' in df.columns:
        df['gas_price'] = df['current_gas']

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Fill missing gas_price with base_fee if available
    if 'base_fee' in df.columns:
        df['gas_price'] = df['gas_price'].fillna(df['base_fee'])

    # Remove rows with missing gas_price
    df = df.dropna(subset=['gas_price'])

    print(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)
    print(f"   Gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei", flush=True)

    return df


def create_spike_features(df):
    """
    Create features for spike detection.
    Must match HybridPredictor.create_spike_features() exactly.
    """
    df = df.copy()
    df = df.sort_values('timestamp')

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # Recent volatility
    for window in [6, 12, 24, 48]:
        df[f'volatility_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).std()
        df[f'range_{window}'] = (
            df['gas_price'].rolling(window=window, min_periods=1).max() -
            df['gas_price'].rolling(window=window, min_periods=1).min()
        )
        df[f'mean_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).mean()
        df[f'is_rising_{window}'] = (
            df['gas_price'] > df[f'mean_{window}']
        ).astype(int)

    # Rate of change
    for lag in [1, 2, 3, 6, 12]:
        df[f'pct_change_{lag}'] = df['gas_price'].pct_change(lag).fillna(0)
        df[f'diff_{lag}'] = df['gas_price'].diff(lag).fillna(0)

    # Recent spike indicator
    df['recent_spike'] = (
        df['gas_price'].rolling(window=24, min_periods=1).max() > ELEVATED_THRESHOLD
    ).astype(int)

    # Replace inf/-inf with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df


def get_feature_names():
    """Get list of feature names used by spike detector"""
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours'
    ]

    for window in [6, 12, 24, 48]:
        features.extend([
            f'volatility_{window}',
            f'range_{window}',
            f'mean_{window}',
            f'is_rising_{window}'
        ])

    for lag in [1, 2, 3, 6, 12]:
        features.extend([
            f'pct_change_{lag}',
            f'diff_{lag}'
        ])

    features.append('recent_spike')

    return features


def classify_price(price):
    """Classify gas price into Normal/Elevated/Spike"""
    if price < NORMAL_THRESHOLD:
        return 0  # Normal
    elif price < ELEVATED_THRESHOLD:
        return 1  # Elevated
    else:
        return 2  # Spike


def create_targets(df, steps_per_hour=12):
    """
    Create classification targets for each horizon.
    Target = class of gas price N hours in the future.
    """
    targets = {}

    for horizon, hours in [('1h', 1), ('4h', 4), ('24h', 24)]:
        steps = hours * steps_per_hour
        # Shift gas price forward to get future price
        future_price = df['gas_price'].shift(-steps)
        # Classify future price
        targets[horizon] = future_price.apply(lambda x: classify_price(x) if pd.notna(x) else np.nan)

    return targets


def train_spike_detector(X, y, horizon):
    """Train a single spike detector model"""
    print(f"\n{'='*60}", flush=True)
    print(f"🎯 Training Spike Detector for {horizon}", flush=True)
    print(f"{'='*60}", flush=True)

    # Remove NaN targets
    valid_idx = ~y.isna()
    X_clean = X[valid_idx]
    y_clean = y[valid_idx].astype(int)

    print(f"   Valid samples: {len(X_clean):,}", flush=True)

    if len(X_clean) < 100:
        print(f"❌ Insufficient data for {horizon}", flush=True)
        return None

    # Class distribution
    class_counts = y_clean.value_counts().sort_index()
    print(f"\n   Class Distribution:", flush=True)
    class_names = ['Normal', 'Elevated', 'Spike']
    for cls, count in class_counts.items():
        pct = count / len(y_clean) * 100
        print(f"      {class_names[cls]}: {count:,} ({pct:.1f}%)", flush=True)

    # Time-series split (80/20)
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_train = y_clean.iloc[:split_idx]
    y_test = y_clean.iloc[split_idx:]

    print(f"\n   Train: {len(X_train):,}, Test: {len(X_test):,}", flush=True)

    # Train GradientBoosting classifier
    print(f"\n   Training GradientBoosting classifier...", flush=True)

    # Use fewer estimators on Railway to save memory
    n_estimators = 100 if IS_RAILWAY else 200

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n✅ Model Performance:", flush=True)
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)", flush=True)
    print(f"   F1 Score (weighted): {f1:.4f}", flush=True)

    # Detailed classification report
    print(f"\n   Classification Report:", flush=True)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    for line in report.split('\n'):
        print(f"      {line}", flush=True)

    # Feature importance
    feature_names = list(X_clean.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n   Top 10 Features:", flush=True)
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"      {i+1}. {feature_names[idx]}: {importances[idx]:.4f}", flush=True)

    return {
        'model': model,
        'feature_names': feature_names,
        'metrics': {
            'accuracy': accuracy,
            'f1_score': f1
        },
        'class_distribution': class_counts.to_dict(),
        'trained_at': datetime.now().isoformat()
    }


def save_spike_detector(detector_data, horizon, output_dir=None):
    """Save trained spike detector to persistent storage"""
    if output_dir is None:
        from config import Config
        output_dir = Config.MODELS_DIR

    if detector_data is None:
        return False

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f'spike_detector_{horizon}.pkl')

    # Save the detector data (model + feature_names + metadata)
    joblib.dump(detector_data, filepath)

    print(f"💾 Saved spike detector to {filepath}", flush=True)
    return filepath


def main():
    """Main training function"""
    print("="*70, flush=True)
    print("🎯 Spike Detector Training", flush=True)
    print("="*70, flush=True)

    if IS_RAILWAY:
        print("🚂 Railway environment detected", flush=True)

    try:
        # Step 1: Fetch data
        data = fetch_training_data(hours=2160)

        # Step 2: Prepare DataFrame
        df = prepare_dataframe(data)

        # Determine sample rate
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            median_diff = time_diffs.median().total_seconds() / 60
            steps_per_hour = max(1, int(60 / median_diff))
            print(f"   Sample rate: ~{median_diff:.1f} min ({steps_per_hour} steps/hour)", flush=True)
        else:
            steps_per_hour = 12

        # Step 3: Create features
        print(f"\n📊 Creating spike detection features...", flush=True)
        df = create_spike_features(df)

        feature_names = get_feature_names()
        X = df[feature_names]

        print(f"   Features: {len(feature_names)}", flush=True)
        print(f"   Samples: {len(X):,}", flush=True)

        # Step 4: Create targets
        targets = create_targets(df, steps_per_hour)

        # Step 5: Train models for each horizon
        results = {}

        for horizon in ['1h', '4h', '24h']:
            y = targets[horizon]
            detector_data = train_spike_detector(X, y, horizon)

            if detector_data:
                results[horizon] = detector_data
                save_spike_detector(detector_data, horizon)

        if not results:
            print("\n❌ No spike detectors were trained successfully", flush=True)
            return False

        # Summary
        print("\n" + "="*70, flush=True)
        print("✅ Spike Detector Training Complete!", flush=True)
        print("="*70, flush=True)

        print("\n📊 Summary:", flush=True)
        for horizon, data in results.items():
            metrics = data['metrics']
            print(f"\n   {horizon}:", flush=True)
            print(f"      Accuracy: {metrics['accuracy']*100:.1f}%", flush=True)
            print(f"      F1 Score: {metrics['f1_score']:.4f}", flush=True)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
