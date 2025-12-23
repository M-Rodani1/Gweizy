#!/usr/bin/env python3
"""
Simple Model Retraining Script

Fetches current data from the database and retrains ML models
with the correct feature set to match production.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.advanced_features import create_advanced_features
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def fetch_training_data(hours=720):
    """Fetch data from database"""
    print(f"📊 Fetching {hours} hours of data from database...")

    from data.database import DatabaseManager
    db = DatabaseManager()

    # Get historical data
    data = db.get_historical_data(hours=hours)

    if not data:
        raise ValueError(f"No data available in database")

    print(f"✅ Fetched {len(data)} records")
    return data


def prepare_features(data):
    """Prepare features using the same pipeline as production"""
    print("\n📊 Creating features (same as production)...")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Parse timestamps and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Handle gas price column names
    if 'gas_price' not in df.columns:
        if 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
        elif 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        else:
            raise ValueError("No gas price column found")

    print(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei")

    # Create advanced features using the same function as production
    X, y = create_advanced_features(df)

    print(f"✅ Created {X.shape[1]} features from {len(df)} records")

    # Create targets for different horizons
    # 1h = 12 steps (at 5min intervals), 4h = 48 steps, 24h = 288 steps
    y_1h = y.shift(-12)
    y_4h = y.shift(-48)
    y_24h = y.shift(-288)

    return X, y_1h, y_4h, y_24h


def train_model(X, y, horizon, min_samples=100):
    """Train a single model for given horizon"""
    print(f"\n{'='*60}")
    print(f"🎯 Training model for {horizon} horizon")
    print(f"{'='*60}")

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]

    print(f"   Valid samples: {len(X_clean)}")

    if len(X_clean) < min_samples:
        print(f"⚠️  Not enough data ({len(X_clean)} < {min_samples}), skipping...")
        return None

    # Split: 80% train, 20% test (maintain temporal order)
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_train = y_clean.iloc[:split_idx]
    y_test = y_clean.iloc[split_idx:]

    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Scale features with RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    print(f"📊 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Directional accuracy
    if len(y_test) > 1:
        y_diff_actual = np.diff(y_test.values)
        y_diff_pred = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(y_diff_pred))
    else:
        directional_accuracy = 0.0

    print(f"\n✅ Model Performance:")
    print(f"   MAE: {mae:.6f} gwei")
    print(f"   RMSE: {rmse:.6f} gwei")
    print(f"   R²: {r2:.4f}")
    print(f"   Directional Accuracy: {directional_accuracy*100:.1f}%")

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X_clean.columns),
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    }


def save_model(model_data, horizon, output_dir='backend/models/saved_models'):
    """Save trained model"""
    if model_data is None:
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Save main model file
    filepath = os.path.join(output_dir, f'model_{horizon}.pkl')
    save_data = {
        'model': model_data['model'],
        'model_name': 'RandomForest',
        'metrics': model_data['metrics'],
        'trained_at': datetime.now().isoformat(),
        'feature_names': model_data['feature_names'],
        'scaler_type': 'RobustScaler'
    }
    joblib.dump(save_data, filepath)
    print(f"💾 Saved model to {filepath}")

    # Save scaler separately
    scaler_path = os.path.join(output_dir, f'scaler_{horizon}.pkl')
    joblib.dump(model_data['scaler'], scaler_path)
    print(f"💾 Saved scaler to {scaler_path}")

    # Save feature names separately for reference
    feature_names_path = os.path.join(output_dir, f'feature_names_{horizon}.txt')
    with open(feature_names_path, 'w') as f:
        for feat in model_data['feature_names']:
            f.write(f"{feat}\n")
    print(f"💾 Saved feature names to {feature_names_path}")

    return True


def main():
    print("="*70)
    print("🎯 Simple Model Retraining")
    print("="*70)

    try:
        # Step 1: Fetch data
        data = fetch_training_data(hours=720)  # 30 days

        # Step 2: Prepare features
        X, y_1h, y_4h, y_24h = prepare_features(data)

        # Step 3: Train models for each horizon
        results = {}
        for horizon, y in [('1h', y_1h), ('4h', y_4h), ('24h', y_24h)]:
            model_data = train_model(X, y, horizon)
            if model_data:
                results[horizon] = model_data
                save_model(model_data, horizon)

        if not results:
            print("\n❌ No models were trained successfully")
            return False

        # Step 4: Summary
        print("\n" + "="*70)
        print("✅ Retraining Complete!")
        print("="*70)

        print("\n📊 Model Performance Summary:")
        for horizon, model_data in results.items():
            metrics = model_data['metrics']
            print(f"\n{horizon}:")
            print(f"  MAE: {metrics['mae']:.6f} gwei")
            print(f"  RMSE: {metrics['rmse']:.6f} gwei")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']*100:.1f}%")
            print(f"  Features: {len(model_data['feature_names'])}")

        print("\n" + "="*70)
        print("📋 Next Steps:")
        print("="*70)
        print("1. The new models are saved in backend/models/saved_models/")
        print("2. Commit and push to Railway to deploy the new models")
        print("3. The prediction endpoint will now use these retrained models")
        print("4. Monitor performance and retrain again as more data is collected")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
