"""
Train a spike detection classifier for Base gas prices

This classifier predicts whether gas prices will spike above normal levels,
which is more useful than predicting exact prices on Base L2.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import sqlite3
from datetime import datetime

# Spike thresholds (in Gwei)
NORMAL_THRESHOLD = 0.01    # Below this = Normal
ELEVATED_THRESHOLD = 0.05  # Above this = Spike


def create_spike_features(df):
    """Create features specifically for spike prediction"""
    df = df.copy()

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # Recent volatility (key predictor for spikes)
    for window in [6, 12, 24, 48]:  # 30min, 1h, 2h, 4h (5-min intervals)
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

    # Recent spike indicator (was there a recent spike?)
    df['recent_spike'] = (
        df['gas_price'].rolling(window=24, min_periods=1).max() > ELEVATED_THRESHOLD
    ).astype(int)

    return df


def create_spike_labels(df, horizon=12):
    """
    Create labels: Will there be a spike in the next {horizon} periods?

    Returns 3 classes:
    - 0: Normal (< 0.01 Gwei)
    - 1: Elevated (0.01 - 0.05 Gwei)
    - 2: Spike (> 0.05 Gwei)
    """
    future_max = df['gas_price'].rolling(window=horizon, min_periods=1).max().shift(-horizon)

    labels = pd.cut(
        future_max,
        bins=[0, NORMAL_THRESHOLD, ELEVATED_THRESHOLD, float('inf')],
        labels=[0, 1, 2],
        include_lowest=True
    )

    # Convert to int, keeping NaN as NaN
    labels = labels.cat.codes.replace(-1, np.nan)

    return labels


def train_spike_classifier(horizon_name='1h', horizon_steps=12):
    """Train spike detection classifier for a given horizon"""

    print(f"\n{'='*60}")
    print(f"Training Spike Detector for {horizon_name} horizon")
    print(f"{'='*60}\n")

    # Load data
    db_path = os.path.join(os.path.dirname(__file__), '..', 'gas_data.db')
    conn = sqlite3.connect(db_path)

    query = """
    SELECT timestamp, current_gas as gas_price, base_fee, priority_fee
    FROM gas_prices
    ORDER BY timestamp ASC
    """

    df = pd.read_sql_query(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    conn.close()

    print(f"✓ Loaded {len(df):,} records")

    # Create features and labels
    df = create_spike_features(df)
    df['spike_label'] = create_spike_labels(df, horizon=horizon_steps)

    # Remove rows with NaN labels (last {horizon} rows)
    df = df.dropna(subset=['spike_label'])

    # Check class distribution
    print(f"\nClass Distribution:")
    print(df['spike_label'].value_counts().sort_index())
    print(f"\nClass Percentages:")
    print((df['spike_label'].value_counts(normalize=True) * 100).sort_index())

    # Prepare features
    feature_cols = [col for col in df.columns if col not in [
        'timestamp', 'gas_price', 'base_fee', 'priority_fee', 'spike_label'
    ]]

    X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = df['spike_label'].values

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"✓ Features: {len(feature_cols)}")

    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Elevated', 'Spike']))

    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}\n")
    print("Rows = Actual, Columns = Predicted")
    print("           Normal  Elevated  Spike")
    cm = confusion_matrix(y_test, y_pred)
    for i, label in enumerate(['Normal   ', 'Elevated ', 'Spike    ']):
        print(f"{label} {cm[i]}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*60}\n")
    print(feature_importance.head(10).to_string(index=False))

    # Calculate spike detection metrics
    # True if predicted spike (class 2) when actual spike (class 2)
    spike_precision = np.sum((y_pred == 2) & (y_test == 2)) / max(np.sum(y_pred == 2), 1)
    spike_recall = np.sum((y_pred == 2) & (y_test == 2)) / max(np.sum(y_test == 2), 1)

    print(f"\n{'='*60}")
    print("SPIKE DETECTION PERFORMANCE")
    print(f"{'='*60}\n")
    print(f"Spike Precision: {spike_precision:.2%} (of predicted spikes, how many were real)")
    print(f"Spike Recall:    {spike_recall:.2%} (of real spikes, how many we caught)")

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f'spike_detector_{horizon_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': clf,
            'feature_names': feature_cols,
            'thresholds': {
                'normal': NORMAL_THRESHOLD,
                'elevated': ELEVATED_THRESHOLD
            },
            'training_date': datetime.now().isoformat(),
            'horizon': horizon_name,
            'horizon_steps': horizon_steps
        }, f)

    print(f"\n✓ Saved spike detector to {model_path}")

    return clf, feature_cols


if __name__ == "__main__":
    print("Base Gas Fee Spike Detection Classifier Training")
    print("="*60)

    horizons = {
        '1h': 12,   # 12 * 5min = 1 hour
        '4h': 48,   # 48 * 5min = 4 hours
        '24h': 288  # 288 * 5min = 24 hours
    }

    for horizon_name, horizon_steps in horizons.items():
        train_spike_classifier(horizon_name, horizon_steps)

    print(f"\n{'='*60}")
    print("✓ All spike detectors trained successfully!")
    print(f"{'='*60}")
