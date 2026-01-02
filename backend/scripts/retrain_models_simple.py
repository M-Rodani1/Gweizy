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
from models.feature_pipeline import build_feature_matrix, build_horizon_targets, normalize_gas_dataframe
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import joblib

# Hyperparameter search space for RandomForest
RF_PARAM_DISTRIBUTIONS = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [10, 15, 20, 30, None],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 8],
    'model__max_features': ['sqrt', 'log2', 0.5, 0.7],
}

# Whether to use hyperparameter tuning (set False for faster training)
# Note: Hyperparameter tuning is memory-intensive. On Railway (limited RAM),
# we use conservative settings to avoid OOM kills.
import os
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

USE_HYPERPARAMETER_TUNING = True
TUNING_ITERATIONS = 8 if IS_RAILWAY else 15  # Fewer iterations on Railway
CV_FOLDS = 2 if IS_RAILWAY else 3  # Fewer folds on Railway
N_JOBS_TUNING = 1 if IS_RAILWAY else -1  # Sequential on Railway to save memory


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

    df = normalize_gas_dataframe(data)

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

    # Build features using shared pipeline
    X, feature_meta, df = build_feature_matrix(df, include_external_features=True)

    # Log-scale transformation for better handling of wide ranges
    epsilon = 1e-8
    y = np.log(df['gas_price'] + epsilon)

    print(f"✅ Created {X.shape[1]} features from {len(df)} records")
    if feature_meta.get('sample_rate_minutes'):
        print(f"   Detected sample rate: {feature_meta['sample_rate_minutes']:.2f} minutes")

    steps_per_hour = feature_meta.get('steps_per_hour', 12)

    targets_log = build_horizon_targets(y, steps_per_hour)
    targets_original = build_horizon_targets(df['gas_price'], steps_per_hour)

    return (
        X,
        (targets_log['1h'], targets_original['1h']),
        (targets_log['4h'], targets_original['4h']),
        (targets_log['24h'], targets_original['24h']),
        feature_meta
    )


def train_model(X, y_tuple, horizon, min_samples=100, feature_meta=None, use_feature_selection=True):
    import sys
    # Force immediate output flushing for Railway logs
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Helper function for flushed printing
    def print_flush(*args, **kwargs):
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)
    """
    Train a single model for given horizon

    Args:
        X: Features
        y_tuple: (y_log, y_original) - log-scale and original scale targets
        horizon: Prediction horizon
        min_samples: Minimum samples required
        use_feature_selection: Whether to use SHAP feature selection
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"🎯 Training model for {horizon} horizon")
    print_flush(f"{'='*60}")

    y_log, y_original = y_tuple

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y_log.isna() | y_original.isna())
    X_clean = X[valid_idx]
    y_log_clean = y_log[valid_idx]
    y_original_clean = y_original[valid_idx]

    print_flush(f"   Valid samples: {len(X_clean)}")

    if len(X_clean) < min_samples:
        print_flush(f"⚠️  Not enough data ({len(X_clean)} < {min_samples}), skipping...")
        return None

    # Apply SHAP feature selection to reduce features
    feature_selector = None
    if use_feature_selection and X_clean.shape[1] > 40:
        try:
            from models.feature_selector import SHAPFeatureSelector
            n_features = 30 if not IS_RAILWAY else 25  # Fewer features on Railway
            feature_selector = SHAPFeatureSelector(n_features=n_features)
            feature_selector.fit(X_clean, y_log_clean, verbose=True)
            X_clean = feature_selector.transform(X_clean)
            print_flush(f"   ✅ Reduced to {X_clean.shape[1]} features using SHAP selection")
            
            # Save feature selector to persistent storage
            try:
                feature_selector.save()  # Uses Config.MODELS_DIR
                print_flush(f"   ✅ Saved feature selector to persistent storage")
            except Exception as save_err:
                print_flush(f"   ⚠️ Could not save feature selector: {save_err}")
        except Exception as e:
            print_flush(f"   ⚠️ Feature selection failed: {e}, using all features")
            feature_selector = None

    # Split: 80% train, 20% test (maintain temporal order)
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_log_train = y_log_clean.iloc[:split_idx]
    y_log_test = y_log_clean.iloc[split_idx:]
    y_original_test = y_original_clean.iloc[split_idx:]

    print_flush(f"   Train samples: {len(X_train)}")
    print_flush(f"   Test samples: {len(X_test)}")

    # Train Random Forest model on log-scale targets
    search = None
    if USE_HYPERPARAMETER_TUNING and len(X_train) >= 1000:
        print_flush(f"📊 Training Random Forest with hyperparameter tuning...")
        print_flush(f"   Testing {TUNING_ITERATIONS} parameter combinations with {CV_FOLDS}-fold CV")

        # Use TimeSeriesSplit for proper time series cross-validation
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', base_model)
        ])

        search = RandomizedSearchCV(
            pipeline,
            RF_PARAM_DISTRIBUTIONS,
            n_iter=TUNING_ITERATIONS,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=N_JOBS_TUNING,  # Use 1 on Railway to avoid OOM
            verbose=0
        )

        search.fit(X_train, y_log_train)
        best_pipeline = search.best_estimator_
        scaler = best_pipeline.named_steps['scaler']
        model = best_pipeline.named_steps['model']

        print_flush(f"   Best parameters found:")
        for param, value in search.best_params_.items():
            print_flush(f"     {param}: {value}")
        print_flush(f"   Best CV MAE: {-search.best_score_:.6f}")
    else:
        print_flush(f"📊 Training Random Forest (log-scale)...")
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_log_train)
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['model']

    # Evaluate on log scale
    X_test_scaled = scaler.transform(X_test)
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

    print_flush(f"\n✅ Model Performance (on original scale):")
    print_flush(f"   MAE: {mae:.6f} gwei")
    print_flush(f"   RMSE: {rmse:.6f} gwei")
    print_flush(f"   R²: {r2:.4f}")
    print_flush(f"   MAPE: {mape:.2f}%")
    print_flush(f"   Directional Accuracy: {directional_accuracy*100:.1f}%")

    # Additional insight: show median prediction vs actual
    median_actual = np.median(y_original_test)
    median_pred = np.median(y_pred_original)
    print_flush(f"   Median Actual: {median_actual:.6f} gwei")
    print_flush(f"   Median Predicted: {median_pred:.6f} gwei")

    # Feature importance analysis
    feature_names = list(X_clean.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print_flush(f"\n📈 Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print_flush(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Store best hyperparameters if tuning was used
    best_params = search.best_params_ if USE_HYPERPARAMETER_TUNING and len(X_train) >= 1000 else None

    return {
        'model': model,
        'scaler': scaler,
        'feature_selector': feature_selector,  # SHAP-based feature selector
        'feature_names': list(X_clean.columns),
        'uses_log_scale': True,  # Flag for prediction inference
        'best_params': best_params,
        'feature_importances': dict(zip(feature_names, importances.tolist())),
        'feature_pipeline': feature_meta or {},
        'sample_rate_minutes': (feature_meta or {}).get('sample_rate_minutes'),
        'steps_per_hour': (feature_meta or {}).get('steps_per_hour'),
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


def save_model(model_data, horizon, output_dir=None, training_samples=None):
    """Save trained model to persistent storage"""
    if output_dir is None:
        # Use persistent storage on Railway, fallback to local
        from config import Config
        output_dir = Config.MODELS_DIR
    
    if model_data is None:
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Save main model file
    filepath = os.path.join(output_dir, f'model_{horizon}.pkl')
    save_data = {
        'model': model_data['model'],
        'model_name': 'RandomForest_LogScale_SHAP' if model_data.get('feature_selector') else 'RandomForest_LogScale',
        'metrics': model_data['metrics'],
        'trained_at': datetime.now().isoformat(),
        'feature_names': model_data['feature_names'],
        'feature_selector': model_data.get('feature_selector'),  # SHAP selector for inference
        'feature_scaler': model_data['scaler'],  # Include scaler in main file
        'feature_pipeline': model_data.get('feature_pipeline', {}),
        'sample_rate_minutes': model_data.get('sample_rate_minutes'),
        'steps_per_hour': model_data.get('steps_per_hour'),
        'scaler_type': 'RobustScaler',
        'uses_log_scale': True,  # IMPORTANT: Predictions need exp() transformation
        'predicts_percentage_change': False,
        'best_params': model_data.get('best_params'),
        'feature_importances': model_data.get('feature_importances'),
        'hyperparameter_tuning_used': USE_HYPERPARAMETER_TUNING,
        'shap_feature_selection_used': model_data.get('feature_selector') is not None
    }
    joblib.dump(save_data, filepath)
    print(f"💾 Saved model to {filepath}", flush=True)

    # Save scaler separately
    scaler_path = os.path.join(output_dir, f'scaler_{horizon}.pkl')
    joblib.dump(model_data['scaler'], scaler_path)
    print(f"💾 Saved scaler to {scaler_path}", flush=True)

    # Save feature names separately for reference
    feature_names_path = os.path.join(output_dir, f'feature_names_{horizon}.txt')
    with open(feature_names_path, 'w') as f:
        for feat in model_data['feature_names']:
            f.write(f"{feat}\n")
    print(f"💾 Saved feature names to {feature_names_path}", flush=True)

    # Register model with ModelRegistry
    try:
        from models.model_registry import get_registry
        registry = get_registry()
        registry.register_model(
            horizon=horizon,
            model_path=filepath,
            metrics=model_data['metrics'],
            metadata={
                'model_name': save_data['model_name'],
                'feature_count': len(model_data['feature_names']),
                'uses_feature_selection': model_data.get('feature_selector') is not None,
                'hyperparameter_tuning': USE_HYPERPARAMETER_TUNING,
                'best_params': model_data.get('best_params'),
                'training_samples': training_samples,
                'scaler_path': scaler_path,
                'feature_selector_path': os.path.join(output_dir, 'feature_selector.pkl') if model_data.get('feature_selector') else None
            }
        )
        print(f"✅ Registered model version for {horizon}", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to register model version: {e}", flush=True)

    return filepath  # Return the model path instead of True


def main():
    import sys
    # Force unbuffered output for Railway logs
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("="*70, flush=True)
    print("🎯 Simple Model Retraining", flush=True)
    print("="*70, flush=True)

    if IS_RAILWAY:
        print("🚂 Railway environment detected - using memory-efficient settings")
        print(f"   Tuning iterations: {TUNING_ITERATIONS}, CV folds: {CV_FOLDS}, Jobs: {N_JOBS_TUNING}")

    try:
        # Step 1: Fetch data
        data = fetch_training_data(hours=720)  # 30 days

        # Step 2: Prepare features
        X, y_1h, y_4h, y_24h, feature_meta = prepare_features(data)

        # Step 3: Train models for each horizon
        results = {}
        registry = None
        try:
            from models.model_registry import get_registry
            registry = get_registry()
        except Exception as e:
            print(f"⚠️ Model registry not available: {e}", flush=True)
        
        for horizon, y in [('1h', y_1h), ('4h', y_4h), ('24h', y_24h)]:
            print(f"\n{'='*70}", flush=True)
            print(f"🚀 Starting training for {horizon} horizon", flush=True)
            print(f"{'='*70}\n", flush=True)
            model_data = train_model(X, y, horizon, feature_meta=feature_meta)
            if model_data:
                results[horizon] = model_data
                model_path = save_model(model_data, horizon, training_samples=len(X))
                # Model registration is now handled inside save_model()

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
