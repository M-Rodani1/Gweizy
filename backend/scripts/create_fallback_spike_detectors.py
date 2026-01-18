#!/usr/bin/env python3
"""
Create Fallback Spike Detector Models

Creates basic spike detector models using synthetic data patterns
when insufficient historical data is available. These models provide
reasonable baseline predictions until more real data is collected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from config import Config

# Thresholds (must match HybridPredictor)
NORMAL_THRESHOLD = 0.01
ELEVATED_THRESHOLD = 0.05


def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic gas price data that mimics typical patterns.
    Uses realistic distributions based on L2 gas behavior.
    """
    np.random.seed(42)

    # Generate timestamps (5-minute intervals)
    hours = np.arange(n_samples) % 24
    days_of_week = np.arange(n_samples) // (24 * 12) % 7

    # Base gas price with daily patterns
    # Lower at night (hours 0-6), higher during peak hours (12-20)
    base_pattern = 0.005 + 0.003 * np.sin(2 * np.pi * hours / 24 - np.pi/2)

    # Weekend effect (slightly lower)
    weekend_effect = np.where((days_of_week == 5) | (days_of_week == 6), 0.8, 1.0)

    # Random noise
    noise = np.random.lognormal(mean=-1, sigma=0.5, size=n_samples) * 0.01

    # Occasional spikes (10% of data)
    spike_mask = np.random.random(n_samples) < 0.10
    spikes = np.where(spike_mask, np.random.exponential(0.05, n_samples), 0)

    gas_prices = (base_pattern * weekend_effect + noise + spikes).clip(0.001, 0.5)

    # Create DataFrame
    df = pd.DataFrame({
        'gas_price': gas_prices,
        'hour': hours,
        'day_of_week': days_of_week
    })

    return df


def create_features(df):
    """Create features for spike detection (simplified version)."""
    features = pd.DataFrame()

    # Rolling statistics
    for window in [6, 12, 24]:  # 30min, 1h, 2h
        features[f'rolling_mean_{window}'] = df['gas_price'].rolling(window, min_periods=1).mean()
        features[f'rolling_std_{window}'] = df['gas_price'].rolling(window, min_periods=1).std().fillna(0)
        features[f'rolling_max_{window}'] = df['gas_price'].rolling(window, min_periods=1).max()

    # Lag features
    for lag in [1, 3, 6, 12]:
        features[f'lag_{lag}'] = df['gas_price'].shift(lag).fillna(df['gas_price'].iloc[0])

    # Time features
    features['hour'] = df['hour']
    features['day_of_week'] = df['day_of_week']
    features['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    features['is_peak_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 20)).astype(int)

    # Rate of change
    features['price_change'] = df['gas_price'].diff().fillna(0)
    features['price_pct_change'] = df['gas_price'].pct_change().fillna(0).clip(-1, 1)

    # Current price (important feature)
    features['current_price'] = df['gas_price']

    return features.fillna(0)


def create_labels(gas_prices, horizon_steps):
    """Create classification labels based on future gas price."""
    future_prices = gas_prices.shift(-horizon_steps)

    labels = pd.Series(0, index=gas_prices.index)  # Default: normal
    labels[future_prices >= NORMAL_THRESHOLD] = 1  # Elevated
    labels[future_prices >= ELEVATED_THRESHOLD] = 2  # Spike

    # Fill NaN with most common class
    labels = labels.fillna(0).astype(int)

    return labels


def train_spike_detector(X, y, horizon_name):
    """Train a single spike detector model."""
    print(f"   Training {horizon_name} spike detector...", flush=True)

    # Use GradientBoosting for consistency with main script
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        min_samples_split=20
    )

    model.fit(X, y)

    # Calculate basic metrics
    accuracy = model.score(X, y)
    print(f"      Training accuracy: {accuracy:.2%}", flush=True)

    return model


def main():
    print("=" * 60)
    print("🔧 Creating Fallback Spike Detector Models")
    print("=" * 60)
    print()

    # Generate synthetic data
    print("📊 Generating synthetic training data...", flush=True)
    df = generate_synthetic_data(n_samples=5000)
    print(f"   Generated {len(df):,} samples", flush=True)
    print(f"   Gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei", flush=True)
    print()

    # Create features
    print("🔧 Creating features...", flush=True)
    X = create_features(df)
    feature_names = X.columns.tolist()
    print(f"   Created {len(feature_names)} features", flush=True)
    print()

    # Define horizons (steps = 5-minute intervals)
    horizons = {
        '1h': 12,   # 12 * 5min = 1 hour
        '4h': 48,   # 48 * 5min = 4 hours
        '24h': 288  # 288 * 5min = 24 hours
    }

    # Output directory (use absolute path from script location)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, 'models', 'saved_models')
    os.makedirs(output_dir, exist_ok=True)

    print("🎯 Training spike detectors...", flush=True)
    for horizon_name, steps in horizons.items():
        # Create labels
        y = create_labels(df['gas_price'], steps)

        # Train model
        model = train_spike_detector(X, y, horizon_name)

        # Save model
        model_path = os.path.join(output_dir, f'spike_detector_{horizon_name}.pkl')

        # Save as dict with metadata (matching expected format)
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'horizon': horizon_name,
            'thresholds': {
                'normal': NORMAL_THRESHOLD,
                'elevated': ELEVATED_THRESHOLD
            },
            'is_fallback': True,
            'created_at': pd.Timestamp.now().isoformat()
        }

        joblib.dump(model_data, model_path)
        print(f"   ✅ Saved: {model_path}", flush=True)

    print()
    print("=" * 60)
    print("✅ Fallback spike detectors created successfully!")
    print("=" * 60)
    print()
    print("Note: These are basic models trained on synthetic data.")
    print("They will be replaced with better models once enough")
    print("real data has been collected (run train_spike_detectors.py).")


if __name__ == '__main__':
    main()
