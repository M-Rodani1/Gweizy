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
import time

# =============================================================================
# VERBOSE LOGGING HELPER
# =============================================================================
_start_time = time.time()

def log(msg, level="INFO"):
    """Print timestamped log message with elapsed time"""
    elapsed = time.time() - _start_time
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{elapsed:6.1f}s] [{level}] {msg}", flush=True)

def log_progress(step, total, desc=""):
    """Log progress percentage"""
    pct = (step / total * 100) if total > 0 else 0
    log(f"Progress: {step}/{total} ({pct:.1f}%) {desc}")

# =============================================================================

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

# Max records for training - Railway can handle more with 234k records available
# Use up to 100k records on Railway for better model quality
# With 234k records available, we can afford to use more for training
MAX_TRAINING_RECORDS = 100000 if IS_RAILWAY else 50000


def fetch_training_data(hours=2160, max_records=None):
    """Fetch data from database (default: 90 days = 2160 hours)
    
    Args:
        hours: Number of hours of historical data to fetch
        max_records: Maximum records to use for training (for performance)
    """
    if max_records is None:
        max_records = MAX_TRAINING_RECORDS
    
    log(f"📊 STEP: Fetching {hours} hours of data from database...")
    log(f"   Max records limit: {max_records:,}")

    from data.database import DatabaseManager
    log("   Creating DatabaseManager...")
    db = DatabaseManager()

    # Get historical data
    log("   Querying historical data...")
    data = db.get_historical_data(hours=hours)

    if not data:
        raise ValueError(f"No data available in database")

    log(f"✅ Fetched {len(data):,} total records from database")
    
    # Sample data if too large (for training performance)
    if len(data) > max_records:
        log(f"⚠️  SAMPLING: {len(data):,} records → {max_records:,} for faster training")
        # Keep most recent data, sample evenly from the rest
        import random
        random.seed(42)  # Reproducible sampling
        
        log("   Sorting by timestamp...")
        data_sorted = sorted(data, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Take most recent 20% + random sample of older data
        recent_count = max_records // 5  # 20% most recent
        older_count = max_records - recent_count
        log(f"   Taking {recent_count:,} recent + {older_count:,} sampled older records")
        
        recent_data = data_sorted[:recent_count]
        older_data = random.sample(data_sorted[recent_count:], min(older_count, len(data_sorted) - recent_count))
        data = recent_data + older_data
        
        # Re-sort by timestamp for proper time series
        log("   Re-sorting for time series order...")
        data = sorted(data, key=lambda x: x.get('timestamp', ''))
        log(f"✅ Using {len(data):,} records for training")
    
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
                            timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        except:
                            pass
                elif hasattr(ts, 'year'):
                    timestamps.append(ts)
            
            if timestamps:
                timestamps.sort()
                days_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                log(f"📅 Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')} ({days_span:.1f} days)")
        except Exception as e:
            log(f"⚠️  Could not determine date range: {e}", "WARN")
    
    return data


def prepare_features(data):
    """Prepare features using the same pipeline as production"""
    log("=" * 60)
    log("📊 STEP: FEATURE ENGINEERING")
    log("=" * 60)
    log(f"   Input records: {len(data):,}")

    log("   Normalizing gas dataframe...")
    df = normalize_gas_dataframe(data)
    
    log(f"✅ After normalization: {len(df):,} records")
    log(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log(f"   Gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei")

    # IMPROVEMENT 1: Outlier Detection and Filtering
    log("   Detecting outliers using IQR method...")
    Q1 = df['gas_price'].quantile(0.25)
    Q3 = df['gas_price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries (using 3x IQR for extreme outliers only)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = (df['gas_price'] < lower_bound) | (df['gas_price'] > upper_bound)
    outlier_count = outliers.sum()

    if outlier_count > 0:
        log(f"⚠️  Outlier Analysis:", "WARN")
        log(f"   Total outliers: {outlier_count:,} ({outlier_count/len(df)*100:.1f}%)")
        below_count = (df['gas_price'] < lower_bound).sum()
        above_count = (df['gas_price'] > upper_bound).sum()
        log(f"   Below {lower_bound:.6f}: {below_count:,}")
        log(f"   Above {upper_bound:.6f}: {above_count:,}")

        # Cap outliers instead of removing them (preserve time series continuity)
        df.loc[df['gas_price'] > upper_bound, 'gas_price'] = upper_bound
        df.loc[df['gas_price'] < lower_bound, 'gas_price'] = lower_bound
        log(f"   Capped outliers to bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
    else:
        log("✅ No extreme outliers detected")

    # Build features using shared pipeline
    # Disable external features on Railway - they make slow web3 API calls
    include_external = not IS_RAILWAY
    log(f"   Building feature matrix (external features: {include_external})...")
    log("   ⏳ Feature engineering in progress...")
    X, feature_meta, df = build_feature_matrix(df, include_external_features=include_external)
    
    log(f"✅ Feature matrix built: {len(X):,} samples, {X.shape[1]} features")
    
    # Check for NaN values before processing
    log("   Checking for NaN values...")
    nan_counts = X.isna().sum()
    nan_rows = X.isna().any(axis=1).sum()
    if nan_rows > 0:
        log(f"⚠️  Found {nan_rows:,} rows with NaN values ({nan_rows/len(X)*100:.1f}%)", "WARN")
    else:
        log("✅ No NaN values in feature matrix")

    if feature_meta.get('sample_rate_minutes'):
        log(f"   Detected sample rate: {feature_meta['sample_rate_minutes']:.2f} minutes")

    steps_per_hour = feature_meta.get('steps_per_hour', 12)

    # Build horizon targets
    log("   Building horizon targets (1h, 4h, 24h)...")
    gas_price = df['gas_price']
    targets_original = build_horizon_targets(gas_price, steps_per_hour)
    
    # Calculate percentage change targets
    log("   Calculating percentage change targets...")
    targets_pct_change = {}
    for horizon in ['1h', '4h', '24h']:
        future_price = targets_original[horizon]
        current_price = gas_price
        pct_change = ((future_price - current_price) / (current_price + 1e-8)) * 100
        targets_pct_change[horizon] = pct_change
    
    # Log target quality
    log("   Target quality summary:")
    for horizon in ['1h', '4h', '24h']:
        target_orig = targets_original[horizon]
        target_pct = targets_pct_change[horizon]
        valid_targets = (~target_orig.isna() & ~target_pct.isna()).sum()
        if valid_targets > 0:
            pct_stats = target_pct.dropna()
            log(f"   {horizon}: {valid_targets:,} valid targets, pct range: {pct_stats.min():.1f}% to {pct_stats.max():.1f}%")

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
    """
    Train a single model for given horizon

    Args:
        X: Features
        y_tuple: (y_pct_change, y_original, current_prices) - percentage change, original scale targets, and current prices
        horizon: Prediction horizon
        min_samples: Minimum samples required
        use_feature_selection: Whether to use SHAP feature selection
    """
    log("=" * 60)
    log(f"🎯 TRAINING MODEL FOR {horizon.upper()} HORIZON")
    log("=" * 60)

    y_pct_change, y_original, current_prices = y_tuple

    # Log initial counts
    log(f"📊 Data Quality Check:")
    log(f"   Input samples: {len(X):,}, Features: {X.shape[1]}")
    
    # Check NaN distribution
    feature_nan_count = X.isna().any(axis=1).sum()
    target_nan_count = (y_pct_change.isna() | y_original.isna()).sum()
    log(f"   NaN in features: {feature_nan_count:,}, NaN in targets: {target_nan_count:,}")

    # Remove NaN values
    log("   Cleaning data (removing NaN rows)...")
    valid_idx = ~(X.isna().any(axis=1) | y_pct_change.isna() | y_original.isna())
    X_clean = X[valid_idx]
    y_pct_change_clean = y_pct_change[valid_idx]
    y_original_clean = y_original[valid_idx]
    current_prices_clean = current_prices[valid_idx]
    
    removed_count = len(X) - len(X_clean)
    log(f"✅ Valid samples: {len(X_clean):,} (removed {removed_count:,})")

    if len(X_clean) < min_samples:
        log(f"❌ INSUFFICIENT DATA: {len(X_clean):,} < {min_samples:,} minimum required", "ERROR")
        return None

    # Apply enhanced feature selection with multiple methods
    feature_selector = None
    if use_feature_selection and X_clean.shape[1] > 40:
        try:
            log("   Running SHAP feature selection...")
            from models.feature_selector import SHAPFeatureSelector
            n_features = 30 if not IS_RAILWAY else 25  # Fewer features on Railway
            feature_selector = SHAPFeatureSelector(
                n_features=n_features, 
                use_multiple_methods=True
            )
            feature_selector.fit(X_clean, y_pct_change_clean, verbose=True)
            X_clean = feature_selector.transform(X_clean)
            log(f"✅ Reduced to {X_clean.shape[1]} features")
            
            try:
                feature_selector.save()
                log(f"✅ Saved feature selector")
            except Exception as save_err:
                log(f"⚠️ Could not save feature selector: {save_err}", "WARN")
        except Exception as e:
            log(f"⚠️ Feature selection failed: {e}, using all features", "WARN")
            feature_selector = None

    # Split: 80% train, 20% test (maintain temporal order)
    log("   Splitting data (80% train, 20% test)...")
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_pct_change_train = y_pct_change_clean.iloc[:split_idx]
    y_pct_change_test = y_pct_change_clean.iloc[split_idx:]
    y_original_test = y_original_clean.iloc[split_idx:]
    current_prices_test = current_prices_clean.iloc[split_idx:]

    log(f"✅ Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    # Log target statistics
    y_original_train = y_original_clean.iloc[:split_idx]
    
    # Distribution shift detection
    train_median = y_original_train.median()
    test_median = y_original_test.median()
    train_std = y_original_train.std()
    test_std = y_original_test.std()
    median_shift_pct = ((test_median - train_median) / train_median * 100) if train_median > 0 else 0
    std_shift_pct = ((test_std - train_std) / train_std * 100) if train_std > 0 else 0
    
    if abs(median_shift_pct) > 10:
        log(f"⚠️  Distribution shift: {median_shift_pct:+.1f}% (significant)", "WARN")
    elif abs(median_shift_pct) > 5:
        log(f"⚠️  Distribution shift: {median_shift_pct:+.1f}% (moderate)", "WARN")
    else:
        log(f"✅ Distribution shift: {median_shift_pct:+.1f}% (acceptable)")

    # Train multiple models on percentage change targets
    models_trained = {}
    
    # 1. Train Random Forest model on percentage change targets
    search = None
    if USE_HYPERPARAMETER_TUNING and len(X_train) >= 1000:
        log(f"🌲 Training RandomForest with hyperparameter tuning...")
        log(f"   {TUNING_ITERATIONS} param combos, {CV_FOLDS}-fold CV")

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
            n_jobs=N_JOBS_TUNING,
            verbose=0
        )

        search.fit(X_train, y_pct_change_train)
        best_pipeline = search.best_estimator_
        scaler = best_pipeline.named_steps['scaler']
        model = best_pipeline.named_steps['model']

        log(f"✅ Best CV MAE: {-search.best_score_:.4f}%")
    else:
        log(f"🌲 Training RandomForest (no hyperparameter tuning)...")
        log(f"   n_estimators=100, max_depth=15")
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
        log("   ⏳ Fitting model...")
        pipeline.fit(X_train, y_pct_change_train)
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['model']
        log("✅ RandomForest trained")

    # Evaluate on percentage change
    log("   Evaluating model performance...")
    X_test_scaled = scaler.transform(X_test)
    y_pct_change_pred = model.predict(X_test_scaled)

    # Convert percentage change predictions back to absolute price
    y_pred_original = current_prices_test.values * (1 + y_pct_change_pred / 100)

    # Calculate metrics
    mae_pct = mean_absolute_error(y_pct_change_test, y_pct_change_pred)
    rmse_pct = np.sqrt(mean_squared_error(y_pct_change_test, y_pct_change_pred))
    r2_pct = r2_score(y_pct_change_test, y_pct_change_pred)

    mae = mean_absolute_error(y_original_test, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_original_test, y_pred_original))
    r2 = r2_score(y_original_test, y_pred_original)

    mape = np.mean(np.abs((y_original_test - y_pred_original) / (y_original_test + 1e-8))) * 100

    if len(y_original_test) > 1:
        y_diff_actual = np.diff(y_original_test.values)
        y_diff_pred = np.diff(y_pred_original)
        directional_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(y_diff_pred))
    else:
        directional_accuracy = 0.0

    log(f"✅ RF Performance: R²={r2:.4f}, MAE={mae:.6f} gwei, MAPE={mape:.2f}%")
    log(f"   Directional accuracy: {directional_accuracy*100:.1f}%")

    median_actual = np.median(y_original_test)
    median_pred = np.median(y_pred_original)

    # Feature importance analysis (enhanced logging)
    feature_names = list(X_clean.columns)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    log(f"\n📈 Model Feature Importance Analysis:")
    log(f"   Total features used: {len(feature_names)}")
    log(f"   Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        importance_pct = (importances[idx] / importances.sum() * 100) if importances.sum() > 0 else 0
        log(f"      {i+1}. {feature_names[idx]}: {importances[idx]:.6f} ({importance_pct:.1f}% of total)")
    
    # Log cumulative importance
    top_10_importance = importances[indices[:10]].sum()
    top_10_pct = (top_10_importance / importances.sum() * 100) if importances.sum() > 0 else 0
    log(f"   Top 10 features account for {top_10_pct:.1f}% of total importance")
    
    # Log least important features (potential candidates for removal)
    if len(feature_names) > 10:
        bottom_5_importance = importances[indices[-5:]].sum()
        bottom_5_pct = (bottom_5_importance / importances.sum() * 100) if importances.sum() > 0 else 0
        log(f"   Bottom 5 features account for {bottom_5_pct:.1f}% of total importance")
        log(f"   Least important features:")
        for i in range(max(0, len(feature_names) - 5), len(feature_names)):
            idx = indices[i]
            log(f"      - {feature_names[idx]}: {importances[idx]:.6f}")

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
        log(f"\n📊 Training LightGBM...")
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
        log(f"   ✅ LightGBM R²: {r2_lgbm:.4f} (better than RF: {r2_lgbm > r2})")
    except ImportError:
        log(f"   ⚠️ LightGBM not available - skipping")
    except Exception as e:
        log(f"   ⚠️ LightGBM training failed: {e}")
    
    # 3. Train XGBoost (if available)
    try:
        import xgboost as xgb
        log(f"\n📊 Training XGBoost...")
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
        log(f"   ✅ XGBoost R²: {r2_xgb:.4f} (better than RF: {r2_xgb > r2})")
    except ImportError:
        log(f"   ⚠️ XGBoost not available - skipping")
    except Exception as e:
        log(f"   ⚠️ XGBoost training failed: {e}")
    
    # Select best model based on R² score
    best_model_name = 'RandomForest'
    best_r2 = r2
    for model_name, model_data in models_trained.items():
        if model_data['metrics']['r2'] > best_r2:
            best_r2 = model_data['metrics']['r2']
            best_model_name = model_name
    
    log(f"\n🏆 Best model: {best_model_name} (R²: {best_r2:.4f})")
    
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
    log(f"💾 Saved model to {filepath}")

    # Save scaler separately
    scaler_path = os.path.join(output_dir, f'scaler_{horizon}.pkl')
    joblib.dump(model_data['scaler'], scaler_path)
    log(f"💾 Saved scaler to {scaler_path}")

    # Save feature names separately for reference
    feature_names_path = os.path.join(output_dir, f'feature_names_{horizon}.txt')
    with open(feature_names_path, 'w') as f:
        for feat in model_data['feature_names']:
            f.write(f"{feat}\n")
    log(f"💾 Saved feature names")

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
        log(f"✅ Registered model version for {horizon}")
    except Exception as e:
        log(f"⚠️ Failed to register model version: {e}", "WARN")

    return filepath


def main():
    global _start_time
    _start_time = time.time()  # Reset start time
    
    log("=" * 70)
    log("🎯 SIMPLE MODEL RETRAINING SCRIPT STARTED")
    log("=" * 70)
    log(f"   Max training records: {MAX_TRAINING_RECORDS:,}")
    log(f"   Hyperparameter tuning: {'ENABLED' if USE_HYPERPARAMETER_TUNING else 'DISABLED'}")

    if IS_RAILWAY:
        log("🚂 Railway environment detected - memory-efficient mode")

    try:
        # Step 1: Fetch data
        log("=" * 60)
        log("PHASE 1: DATA FETCHING")
        log("=" * 60)
        data = fetch_training_data(hours=2160)

        # Step 2: Prepare features
        log("=" * 60)
        log("PHASE 2: FEATURE ENGINEERING")
        log("=" * 60)
        X, y_1h, y_4h, y_24h, feature_meta = prepare_features(data)
        
        log(f"✅ Feature engineering complete: {len(X):,} samples, {X.shape[1]} features")

        # Step 3: Train models for each horizon
        log("=" * 60)
        log("PHASE 3: MODEL TRAINING")
        log("=" * 60)
        
        results = {}
        registry = None
        try:
            from models.model_registry import get_registry
            registry = get_registry()
        except Exception as e:
            log(f"⚠️ Model registry not available: {e}", "WARN")
        
        for horizon, y in [('1h', y_1h), ('4h', y_4h), ('24h', y_24h)]:
            model_data = train_model(X, y, horizon, feature_meta=feature_meta)
            if model_data:
                results[horizon] = model_data
                model_path = save_model(model_data, horizon, training_samples=len(X))

        if not results:
            log("❌ No models were trained successfully", "ERROR")
            return False

        # Step 4: Summary
        log("=" * 60)
        log("🎉 TRAINING COMPLETE!")
        log("=" * 60)

        log("📊 Final Model Performance Summary:")
        for horizon, model_data in results.items():
            metrics = model_data['metrics']
            log(f"   {horizon}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.6f} gwei, MAPE={metrics['mape']:.2f}%")

        elapsed_total = time.time() - _start_time
        log(f"⏱️  Total training time: {elapsed_total/60:.1f} minutes")
        log("✅ Models saved and ready for use!")

        return True

    except Exception as e:
        log(f"❌ Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log("⚠️ Training interrupted by user", "WARN")
        sys.exit(1)
