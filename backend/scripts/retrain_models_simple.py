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

# Disable hyperparameter tuning on Railway - too slow with large datasets
USE_HYPERPARAMETER_TUNING = False if IS_RAILWAY else True
TUNING_ITERATIONS = 8 if IS_RAILWAY else 15  # Fewer iterations on Railway
CV_FOLDS = 2 if IS_RAILWAY else 3  # Fewer folds on Railway
N_JOBS_TUNING = 1 if IS_RAILWAY else -1  # Sequential on Railway to save memory


def fetch_training_data(hours=2160):
    """Fetch data from database (default: 90 days = 2160 hours)"""
    print(f"📊 Fetching {hours} hours of data from database...", flush=True)

    from data.database import DatabaseManager
    db = DatabaseManager()

    # Get historical data
    data = db.get_historical_data(hours=hours)

    if not data:
        raise ValueError(f"No data available in database")

    print(f"✅ Fetched {len(data):,} total records from database", flush=True)
    
    # Log data range if available
    if data:
        try:
            from dateutil import parser
            timestamps = []
            for d in data:
                ts = d.get('timestamp', '')
                if isinstance(ts, str):
                    try:
                        timestamps.append(parser.parse(ts))
                    except:
                        try:
                            from datetime import datetime
                            timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        except:
                            pass
                elif hasattr(ts, 'year'):
                    timestamps.append(ts)
            
            if timestamps:
                timestamps.sort()
                days_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                print(f"   📅 Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')} ({days_span:.1f} days)", flush=True)
        except Exception as e:
            print(f"   ⚠️  Could not determine date range: {e}", flush=True)
    
    return data


def prepare_features(data):
    """Prepare features using the same pipeline as production"""
    def print_flush(*args, **kwargs):
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)
    
    print_flush("\n📊 Creating features (same as production)...")
    print_flush(f"   Input records: {len(data):,}")

    df = normalize_gas_dataframe(data)
    
    print_flush(f"   After normalization: {len(df):,} records")
    print_flush(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print_flush(f"   Gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei")

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
        print_flush(f"   ⚠️  Outlier Analysis:")
        print_flush(f"      Total outliers: {outlier_count:,} ({outlier_count/len(df)*100:.1f}%)")
        below_count = (df['gas_price'] < lower_bound).sum()
        above_count = (df['gas_price'] > upper_bound).sum()
        print_flush(f"      Below {lower_bound:.6f}: {below_count:,}")
        print_flush(f"      Above {upper_bound:.6f}: {above_count:,}")
        print_flush(f"      Max outlier: {df['gas_price'].max():.6f} gwei")
        print_flush(f"      Min outlier: {df['gas_price'].min():.6f} gwei")
        print_flush(f"      Median: {df['gas_price'].median():.6f}, Q1: {Q1:.6f}, Q3: {Q3:.6f}")

        # Cap outliers instead of removing them (preserve time series continuity)
        df.loc[df['gas_price'] > upper_bound, 'gas_price'] = upper_bound
        df.loc[df['gas_price'] < lower_bound, 'gas_price'] = lower_bound

        print_flush(f"      Capped extreme outliers to bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")

    # Build features using shared pipeline
    X, feature_meta, df = build_feature_matrix(df, include_external_features=True)
    
    print_flush(f"   After feature engineering: {len(X):,} samples, {X.shape[1]} features")
    
    # Check for NaN values before processing
    nan_counts = X.isna().sum()
    nan_rows = X.isna().any(axis=1).sum()
    if nan_rows > 0:
        print_flush(f"   ⚠️  Found {nan_rows:,} rows with NaN values ({nan_rows/len(X)*100:.1f}%)")
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            print_flush(f"   ⚠️  Features with NaN: {len(nan_features)} features")
            top_nan = nan_features.nlargest(5)
            for feat, count in top_nan.items():
                print_flush(f"      - {feat}: {count:,} NaN ({count/len(X)*100:.1f}%)")

    # IMPROVEMENT: Remove log transformation - use original gas_price directly
    # This avoids amplifying errors on very small values (0.001-0.01 gwei)
    print_flush(f"✅ Created {X.shape[1]} features from {len(df):,} records")
    if feature_meta.get('sample_rate_minutes'):
        print_flush(f"   Detected sample rate: {feature_meta['sample_rate_minutes']:.2f} minutes")

    steps_per_hour = feature_meta.get('steps_per_hour', 12)

    # IMPROVEMENT: Predict percentage change instead of absolute price
    # Calculate percentage change: (future_price - current_price) / current_price * 100
    gas_price = df['gas_price']
    targets_original = build_horizon_targets(gas_price, steps_per_hour)
    
    # Calculate percentage change targets
    targets_pct_change = {}
    for horizon in ['1h', '4h', '24h']:
        future_price = targets_original[horizon]
        current_price = gas_price
        # Percentage change: (future - current) / current * 100
        pct_change = ((future_price - current_price) / (current_price + 1e-8)) * 100
        targets_pct_change[horizon] = pct_change
    
    # Log target quality
    for horizon in ['1h', '4h', '24h']:
        target_orig = targets_original[horizon]
        target_pct = targets_pct_change[horizon]
        valid_targets = (~target_orig.isna() & ~target_pct.isna()).sum()
        if valid_targets > 0:
            pct_stats = target_pct.dropna()
            print_flush(f"   {horizon} horizon: {valid_targets:,} valid targets ({valid_targets/len(target_orig)*100:.1f}%)")
            print_flush(f"      Pct change range: {pct_stats.min():.2f}% to {pct_stats.max():.2f}% (median: {pct_stats.median():.2f}%)")

    # Store current prices for later use in evaluation
    # This ensures we can convert percentage change predictions back to absolute prices
    current_prices = gas_price.copy()
    
    return (
        X,
        (targets_pct_change['1h'], targets_original['1h'], current_prices),
        (targets_pct_change['4h'], targets_original['4h'], current_prices),
        (targets_pct_change['24h'], targets_original['24h'], current_prices),
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
        y_tuple: (y_pct_change, y_original, current_prices) - percentage change, original scale targets, and current prices
        horizon: Prediction horizon
        min_samples: Minimum samples required
        use_feature_selection: Whether to use SHAP feature selection
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"🎯 Training model for {horizon} horizon")
    print_flush(f"{'='*60}")

    y_pct_change, y_original, current_prices = y_tuple

    # Log initial counts
    print_flush(f"\n   📊 Data Quality Check:")
    print_flush(f"      Input samples: {len(X):,}")
    print_flush(f"      Features: {X.shape[1]}")
    
    # Check NaN distribution
    feature_nan_count = X.isna().any(axis=1).sum()
    target_nan_count = (y_pct_change.isna() | y_original.isna()).sum()
    print_flush(f"      Rows with NaN in features: {feature_nan_count:,} ({feature_nan_count/len(X)*100:.1f}%)")
    print_flush(f"      Rows with NaN in targets: {target_nan_count:,} ({target_nan_count/len(y_pct_change)*100:.1f}%)")

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y_pct_change.isna() | y_original.isna())
    X_clean = X[valid_idx]
    y_pct_change_clean = y_pct_change[valid_idx]
    y_original_clean = y_original[valid_idx]
    current_prices_clean = current_prices[valid_idx]
    
    removed_count = len(X) - len(X_clean)
    print_flush(f"      ✅ Valid samples after cleaning: {len(X_clean):,}")
    if removed_count > 0:
        print_flush(f"      ⚠️  Removed {removed_count:,} invalid samples ({removed_count/len(X)*100:.1f}%)")

    if len(X_clean) < min_samples:
        print_flush(f"\n❌ INSUFFICIENT DATA: {len(X_clean):,} valid samples < {min_samples:,} minimum required")
        print_flush(f"   Need {min_samples - len(X_clean):,} more valid samples to train")
        return None
    
    print_flush(f"   ✅ Sufficient data: {len(X_clean):,} valid samples (minimum: {min_samples:,})")

    # Apply enhanced feature selection with multiple methods
    feature_selector = None
    if use_feature_selection and X_clean.shape[1] > 40:
        try:
            from models.feature_selector import SHAPFeatureSelector
            n_features = 30 if not IS_RAILWAY else 25  # Fewer features on Railway
            # Use multiple selection methods for better feature selection
            feature_selector = SHAPFeatureSelector(
                n_features=n_features, 
                use_multiple_methods=True  # Enable multi-method feature selection
            )
            feature_selector.fit(X_clean, y_pct_change_clean, verbose=True)
            X_clean = feature_selector.transform(X_clean)
            print_flush(f"   ✅ Reduced to {X_clean.shape[1]} features using multi-method selection")
            
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
    y_pct_change_train = y_pct_change_clean.iloc[:split_idx]
    y_pct_change_test = y_pct_change_clean.iloc[split_idx:]
    y_original_test = y_original_clean.iloc[split_idx:]
    current_prices_test = current_prices_clean.iloc[split_idx:]

    print_flush(f"\n   📊 Train/Test Split:")
    print_flush(f"      Training set: {len(X_train):,} samples ({len(X_train)/len(X_clean)*100:.1f}%)")
    print_flush(f"      Test set: {len(X_test):,} samples ({len(X_test)/len(X_clean)*100:.1f}%)")
    print_flush(f"      Total used: {len(X_clean):,} samples")
    
    # Log target statistics
    y_original_train = y_original_clean.iloc[:split_idx]
    y_pct_change_train_stats = y_pct_change_train.dropna()
    y_pct_change_test_stats = y_pct_change_test.dropna()
    print_flush(f"\n   📈 Target Statistics (percentage change):")
    print_flush(f"      Train - Min: {y_pct_change_train_stats.min():.2f}%, Max: {y_pct_change_train_stats.max():.2f}%, Median: {y_pct_change_train_stats.median():.2f}%")
    print_flush(f"      Test  - Min: {y_pct_change_test_stats.min():.2f}%, Max: {y_pct_change_test_stats.max():.2f}%, Median: {y_pct_change_test_stats.median():.2f}%")
    print_flush(f"\n   📈 Target Statistics (original scale for reference):")
    print_flush(f"      Train - Min: {y_original_train.min():.6f}, Max: {y_original_train.max():.6f}, Median: {y_original_train.median():.6f} gwei")
    print_flush(f"      Test  - Min: {y_original_test.min():.6f}, Max: {y_original_test.max():.6f}, Median: {y_original_test.median():.6f} gwei")
    
    # Distribution shift detection
    train_median = y_original_train.median()
    test_median = y_original_test.median()
    train_std = y_original_train.std()
    test_std = y_original_test.std()
    median_shift_pct = ((test_median - train_median) / train_median * 100) if train_median > 0 else 0
    std_shift_pct = ((test_std - train_std) / train_std * 100) if train_std > 0 else 0
    
    print_flush(f"\n   📊 Distribution Shift Check:")
    print_flush(f"      Train median: {train_median:.6f}, std: {train_std:.6f}")
    print_flush(f"      Test median: {test_median:.6f}, std: {test_std:.6f}")
    print_flush(f"      Median shift: {median_shift_pct:+.1f}%")
    print_flush(f"      Std shift: {std_shift_pct:+.1f}%")
    if abs(median_shift_pct) > 10:
        print_flush(f"      ⚠️  WARNING: Significant distribution shift detected (>10%)")
    elif abs(median_shift_pct) > 5:
        print_flush(f"      ⚠️  Note: Moderate distribution shift detected (5-10%)")

    # Train multiple models on percentage change targets
    models_trained = {}
    
    # 1. Train Random Forest model on percentage change targets
    search = None
    if USE_HYPERPARAMETER_TUNING and len(X_train) >= 1000:
        print_flush(f"📊 Training Random Forest with hyperparameter tuning...")
        print_flush(f"   Testing {TUNING_ITERATIONS} parameter combinations with {CV_FOLDS}-fold CV")
        print_flush(f"   Target: Percentage change (not log-scale)")

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

        search.fit(X_train, y_pct_change_train)
        best_pipeline = search.best_estimator_
        scaler = best_pipeline.named_steps['scaler']
        model = best_pipeline.named_steps['model']

        print_flush(f"   Best parameters found:")
        for param, value in search.best_params_.items():
            print_flush(f"     {param}: {value}")
        print_flush(f"   Best CV MAE: {-search.best_score_:.4f}%")
    else:
        print_flush(f"📊 Training Random Forest (percentage change target)...")
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
        pipeline.fit(X_train, y_pct_change_train)
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['model']

    # Evaluate on percentage change
    X_test_scaled = scaler.transform(X_test)
    y_pct_change_pred = model.predict(X_test_scaled)

    # Convert percentage change predictions back to absolute price for evaluation
    # current_prices_test contains the current prices at the time of prediction
    # (before the shift to future prices)
    y_pred_original = current_prices_test.values * (1 + y_pct_change_pred / 100)

    # Calculate metrics on both percentage change and original scale
    # Percentage change metrics
    mae_pct = mean_absolute_error(y_pct_change_test, y_pct_change_pred)
    rmse_pct = np.sqrt(mean_squared_error(y_pct_change_test, y_pct_change_pred))
    r2_pct = r2_score(y_pct_change_test, y_pct_change_pred)

    # Original scale metrics (converted from percentage change)
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

    print_flush(f"\n✅ Model Performance (percentage change target):")
    print_flush(f"   MAE: {mae_pct:.4f}%")
    print_flush(f"   RMSE: {rmse_pct:.4f}%")
    print_flush(f"   R²: {r2_pct:.4f}")
    print_flush(f"\n✅ Model Performance (converted to original scale):")
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

    # Feature importance analysis (enhanced logging)
    feature_names = list(X_clean.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print_flush(f"\n📈 Model Feature Importance Analysis:")
    print_flush(f"   Total features used: {len(feature_names)}")
    print_flush(f"   Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        importance_pct = (importances[idx] / importances.sum() * 100) if importances.sum() > 0 else 0
        print_flush(f"      {i+1}. {feature_names[idx]}: {importances[idx]:.6f} ({importance_pct:.1f}% of total)")
    
    # Log cumulative importance
    top_10_importance = importances[indices[:10]].sum()
    top_10_pct = (top_10_importance / importances.sum() * 100) if importances.sum() > 0 else 0
    print_flush(f"   Top 10 features account for {top_10_pct:.1f}% of total importance")
    
    # Log least important features (potential candidates for removal)
    if len(feature_names) > 10:
        bottom_5_importance = importances[indices[-5:]].sum()
        bottom_5_pct = (bottom_5_importance / importances.sum() * 100) if importances.sum() > 0 else 0
        print_flush(f"   Bottom 5 features account for {bottom_5_pct:.1f}% of total importance")
        print_flush(f"   Least important features:")
        for i in range(max(0, len(feature_names) - 5), len(feature_names)):
            idx = indices[i]
            print_flush(f"      - {feature_names[idx]}: {importances[idx]:.6f}")

    # Store Random Forest model
    models_trained['RandomForest'] = {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'r2_pct': r2_pct
        },
        'best_params': search.best_params_ if search else None
    }
    
    # 2. Train LightGBM (if available)
    try:
        import lightgbm as lgb
        print_flush(f"\n📊 Training LightGBM...")
        lgbm = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            objective='regression',
            metric='rmse'
        )
        lgbm.fit(X_train_scaled, y_pct_change_train)
        
        # Evaluate
        y_pct_change_pred_lgbm = lgbm.predict(X_test_scaled)
        y_pred_original_lgbm = current_prices_test.values * (1 + y_pct_change_pred_lgbm / 100)
        
        mae_lgbm = mean_absolute_error(y_original_test, y_pred_original_lgbm)
        rmse_lgbm = np.sqrt(mean_squared_error(y_original_test, y_pred_original_lgbm))
        r2_lgbm = r2_score(y_original_test, y_pred_original_lgbm)
        r2_pct_lgbm = r2_score(y_pct_change_test, y_pct_change_pred_lgbm)
        
        models_trained['LightGBM'] = {
            'model': lgbm,
            'scaler': scaler,
            'metrics': {
                'mae': mae_lgbm,
                'rmse': rmse_lgbm,
                'r2': r2_lgbm,
                'r2_pct': r2_pct_lgbm
            }
        }
        print_flush(f"   ✅ LightGBM R²: {r2_lgbm:.4f} (better than RF: {r2_lgbm > r2})")
    except ImportError:
        print_flush(f"   ⚠️ LightGBM not available - skipping")
    except Exception as e:
        print_flush(f"   ⚠️ LightGBM training failed: {e}")
    
    # 3. Train XGBoost (if available)
    try:
        import xgboost as xgb
        print_flush(f"\n📊 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        xgb_model.fit(X_train_scaled, y_pct_change_train)
        
        # Evaluate
        y_pct_change_pred_xgb = xgb_model.predict(X_test_scaled)
        y_pred_original_xgb = current_prices_test.values * (1 + y_pct_change_pred_xgb / 100)
        
        mae_xgb = mean_absolute_error(y_original_test, y_pred_original_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_original_test, y_pred_original_xgb))
        r2_xgb = r2_score(y_original_test, y_pred_original_xgb)
        r2_pct_xgb = r2_score(y_pct_change_test, y_pct_change_pred_xgb)
        
        models_trained['XGBoost'] = {
            'model': xgb_model,
            'scaler': scaler,
            'metrics': {
                'mae': mae_xgb,
                'rmse': rmse_xgb,
                'r2': r2_xgb,
                'r2_pct': r2_pct_xgb
            }
        }
        print_flush(f"   ✅ XGBoost R²: {r2_xgb:.4f} (better than RF: {r2_xgb > r2})")
    except ImportError:
        print_flush(f"   ⚠️ XGBoost not available - skipping")
    except Exception as e:
        print_flush(f"   ⚠️ XGBoost training failed: {e}")
    
    # Select best model based on R² score
    best_model_name = 'RandomForest'
    best_r2 = r2
    for model_name, model_data in models_trained.items():
        if model_data['metrics']['r2'] > best_r2:
            best_r2 = model_data['metrics']['r2']
            best_model_name = model_name
    
    print_flush(f"\n🏆 Best model: {best_model_name} (R²: {best_r2:.4f})")
    
    # Use best model for return
    best_model_data = models_trained[best_model_name]
    model = best_model_data['model']
    scaler = best_model_data['scaler']
    best_params = best_model_data.get('best_params')

    return {
        'model': model,
        'model_name': best_model_name,
        'all_models': models_trained,  # Include all trained models
        'scaler': scaler,
        'feature_selector': feature_selector,  # Enhanced multi-method feature selector
        'feature_names': list(X_clean.columns),
        'uses_log_scale': False,  # No longer using log scale
        'predicts_percentage_change': True,  # IMPORTANT: Model predicts percentage change
        'best_params': best_params,
        'feature_importances': dict(zip(feature_names, model.feature_importances_ if hasattr(model, 'feature_importances_') else {})),
        'feature_pipeline': feature_meta or {},
        'sample_rate_minutes': (feature_meta or {}).get('sample_rate_minutes'),
        'steps_per_hour': (feature_meta or {}).get('steps_per_hour'),
        'metrics': {
            'mae': best_model_data['metrics']['mae'],
            'rmse': best_model_data['metrics']['rmse'],
            'r2': best_model_data['metrics']['r2'],
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'median_actual': median_actual,
            'median_pred': median_pred,
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'r2_pct': r2_pct
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
        'model_name': 'RandomForest_PctChange_SHAP' if model_data.get('feature_selector') else 'RandomForest_PctChange',
        'metrics': model_data['metrics'],
        'trained_at': datetime.now().isoformat(),
        'feature_names': model_data['feature_names'],
        'feature_selector': model_data.get('feature_selector'),  # SHAP selector for inference
        'feature_scaler': model_data['scaler'],  # Include scaler in main file
        'feature_pipeline': model_data.get('feature_pipeline', {}),
        'sample_rate_minutes': model_data.get('sample_rate_minutes'),
        'steps_per_hour': model_data.get('steps_per_hour'),
        'scaler_type': 'RobustScaler',
        'uses_log_scale': False,  # No longer using log scale
        'predicts_percentage_change': True,  # IMPORTANT: Model predicts percentage change, not absolute price
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
        print(f"\n{'='*70}", flush=True)
        print(f"🎯 SIMPLE MODEL RETRAINING - DATA PIPELINE", flush=True)
        print(f"{'='*70}", flush=True)
        data = fetch_training_data(hours=2160)  # 90 days (increased from 30 days)

        # Step 2: Prepare features
        print(f"\n{'='*70}", flush=True)
        print(f"🔧 FEATURE ENGINEERING", flush=True)
        print(f"{'='*70}", flush=True)
        X, y_1h, y_4h, y_24h, feature_meta = prepare_features(data)
        
        print(f"\n✅ Feature engineering complete:", flush=True)
        print(f"   Total samples: {len(X):,}", flush=True)
        print(f"   Total features: {X.shape[1]}", flush=True)

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
