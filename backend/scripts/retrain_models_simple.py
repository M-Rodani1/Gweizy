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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib

# Hyperparameter search space for RandomForest
RF_PARAM_DISTRIBUTIONS = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
}

# Whether to use hyperparameter tuning (set False for faster training)
USE_HYPERPARAMETER_TUNING = True
TUNING_ITERATIONS = 15  # Number of random combinations to try
CV_FOLDS = 3  # Cross-validation folds


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

    # IMPROVEMENT 1: Outlier Detection and Filtering
    # Use IQR method to identify extreme outliers
    Q1 = df['gas_price'].quantile(0.25)
    Q3 = df['gas_price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries (using 3x IQR for extreme outliers only)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = (df['gas_price'] < lower_bound) | (df['gas_price'] > upper_bound)
    outlier_count = outliers.sum()

    if outlier_count > 0:
        print(f"   Found {outlier_count} extreme outliers (>{upper_bound:.4f} or <{lower_bound:.4f} gwei)")
        print(f"   Median: {df['gas_price'].median():.6f}, Q1: {Q1:.6f}, Q3: {Q3:.6f}")

        # Cap outliers instead of removing them (preserve time series continuity)
        df.loc[df['gas_price'] > upper_bound, 'gas_price'] = upper_bound
        df.loc[df['gas_price'] < lower_bound, 'gas_price'] = lower_bound

        print(f"   Capped extreme outliers to bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # IMPROVEMENT 2: Log-scale transformation for better handling of wide ranges
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    df['gas_price_log'] = np.log(df['gas_price'] + epsilon)

    print(f"   Using log-scale predictions for better outlier handling")

    # Create advanced features using the same function as production
    X, y_original = create_advanced_features(df)

    # Use log-scaled target
    y = df['gas_price_log']

    print(f"✅ Created {X.shape[1]} features from {len(df)} records")

    # Create targets for different horizons
    # 1h = 12 steps (at 5min intervals), 4h = 48 steps, 24h = 288 steps
    y_1h = y.shift(-12)
    y_4h = y.shift(-48)
    y_24h = y.shift(-288)

    # Also return original scale targets for metrics calculation
    y_1h_original = y_original.shift(-12)
    y_4h_original = y_original.shift(-48)
    y_24h_original = y_original.shift(-288)

    return X, (y_1h, y_1h_original), (y_4h, y_4h_original), (y_24h, y_24h_original)


def train_model(X, y_tuple, horizon, min_samples=100):
    """
    Train a single model for given horizon

    Args:
        X: Features
        y_tuple: (y_log, y_original) - log-scale and original scale targets
        horizon: Prediction horizon
        min_samples: Minimum samples required
    """
    print(f"\n{'='*60}")
    print(f"🎯 Training model for {horizon} horizon")
    print(f"{'='*60}")

    y_log, y_original = y_tuple

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y_log.isna() | y_original.isna())
    X_clean = X[valid_idx]
    y_log_clean = y_log[valid_idx]
    y_original_clean = y_original[valid_idx]

    print(f"   Valid samples: {len(X_clean)}")

    if len(X_clean) < min_samples:
        print(f"⚠️  Not enough data ({len(X_clean)} < {min_samples}), skipping...")
        return None

    # Split: 80% train, 20% test (maintain temporal order)
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_log_train = y_log_clean.iloc[:split_idx]
    y_log_test = y_log_clean.iloc[split_idx:]
    y_original_test = y_original_clean.iloc[split_idx:]

    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Scale features with RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model on log-scale targets
    if USE_HYPERPARAMETER_TUNING and len(X_train) >= 1000:
        print(f"📊 Training Random Forest with hyperparameter tuning...")
        print(f"   Testing {TUNING_ITERATIONS} parameter combinations with {CV_FOLDS}-fold CV")

        # Use TimeSeriesSplit for proper time series cross-validation
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        search = RandomizedSearchCV(
            base_model,
            RF_PARAM_DISTRIBUTIONS,
            n_iter=TUNING_ITERATIONS,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        search.fit(X_train_scaled, y_log_train)
        model = search.best_estimator_

        print(f"   Best parameters found:")
        for param, value in search.best_params_.items():
            print(f"     {param}: {value}")
        print(f"   Best CV MAE: {-search.best_score_:.6f}")
    else:
        print(f"📊 Training Random Forest (log-scale)...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_log_train)

    # Evaluate on log scale
    y_log_pred = model.predict(X_test_scaled)

    # Convert predictions back to original scale
    epsilon = 1e-8
    y_pred_original = np.exp(y_log_pred) - epsilon

    # IMPROVEMENT 3: Better Metrics - Calculate on original scale
    mae = mean_absolute_error(y_original_test, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_original_test, y_pred_original))
    r2 = r2_score(y_original_test, y_pred_original)

    # MAPE (Mean Absolute Percentage Error) - better for relative errors
    mape = np.mean(np.abs((y_original_test - y_pred_original) / (y_original_test + 1e-8))) * 100

    # Directional accuracy (on original scale)
    if len(y_original_test) > 1:
        y_diff_actual = np.diff(y_original_test.values)
        y_diff_pred = np.diff(y_pred_original)
        directional_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(y_diff_pred))
    else:
        directional_accuracy = 0.0

    print(f"\n✅ Model Performance (on original scale):")
    print(f"   MAE: {mae:.6f} gwei")
    print(f"   RMSE: {rmse:.6f} gwei")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Directional Accuracy: {directional_accuracy*100:.1f}%")

    # Additional insight: show median prediction vs actual
    median_actual = np.median(y_original_test)
    median_pred = np.median(y_pred_original)
    print(f"   Median Actual: {median_actual:.6f} gwei")
    print(f"   Median Predicted: {median_pred:.6f} gwei")

    # Feature importance analysis
    feature_names = list(X_clean.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n📈 Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Store best hyperparameters if tuning was used
    best_params = model.get_params() if USE_HYPERPARAMETER_TUNING else None

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X_clean.columns),
        'uses_log_scale': True,  # Flag for prediction inference
        'best_params': best_params,
        'feature_importances': dict(zip(feature_names, importances.tolist())),
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'median_actual': median_actual,
            'median_pred': median_pred
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
        'model_name': 'RandomForest_LogScale_Tuned' if USE_HYPERPARAMETER_TUNING else 'RandomForest_LogScale',
        'metrics': model_data['metrics'],
        'trained_at': datetime.now().isoformat(),
        'feature_names': model_data['feature_names'],
        'scaler_type': 'RobustScaler',
        'uses_log_scale': True,  # IMPORTANT: Predictions need exp() transformation
        'predicts_percentage_change': False,
        'best_params': model_data.get('best_params'),
        'feature_importances': model_data.get('feature_importances'),
        'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING
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

        if USE_HYPERPARAMETER_TUNING:
            print("\n🔧 Hyperparameter tuning was ENABLED")
            print(f"   Tested {TUNING_ITERATIONS} combinations with {CV_FOLDS}-fold TimeSeriesSplit CV")
        else:
            print("\n🔧 Hyperparameter tuning was DISABLED (using defaults)")

        print("\n📊 Model Performance Summary:")
        for horizon, model_data in results.items():
            metrics = model_data['metrics']
            print(f"\n{horizon}:")
            print(f"  MAE: {metrics['mae']:.6f} gwei")
            print(f"  RMSE: {metrics['rmse']:.6f} gwei")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']*100:.1f}%")
            print(f"  Features: {len(model_data['feature_names'])}")
            print(f"  Median Actual: {metrics['median_actual']:.6f} gwei")
            print(f"  Median Predicted: {metrics['median_pred']:.6f} gwei")
            if model_data.get('best_params'):
                print(f"  Best n_estimators: {model_data['best_params'].get('n_estimators')}")
                print(f"  Best max_depth: {model_data['best_params'].get('max_depth')}")

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
