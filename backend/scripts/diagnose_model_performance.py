#!/usr/bin/env python3
"""
Diagnostic script to investigate poor model performance.
Checks data quality, feature quality, and identifies potential issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.database import DatabaseManager
from models.feature_pipeline import build_feature_matrix, build_horizon_targets, normalize_gas_dataframe

def diagnose_data_quality():
    """Check data quality and quantity"""
    print("="*60)
    print("📊 DATA QUALITY DIAGNOSTICS")
    print("="*60)
    
    db = DatabaseManager()
    data = db.get_historical_data(hours=720)
    
    if not data:
        print("❌ No data available!")
        return
    
    print(f"\n✅ Fetched {len(data)} records")
    
    # Normalize to DataFrame
    df = normalize_gas_dataframe(data)
    
    print(f"\n📈 Gas Price Statistics:")
    print(f"   Min: {df['gas_price'].min():.6f} gwei")
    print(f"   Max: {df['gas_price'].max():.6f} gwei")
    print(f"   Median: {df['gas_price'].median():.6f} gwei")
    print(f"   Mean: {df['gas_price'].mean():.6f} gwei")
    print(f"   Std: {df['gas_price'].std():.6f} gwei")
    print(f"   Q1: {df['gas_price'].quantile(0.25):.6f} gwei")
    print(f"   Q3: {df['gas_price'].quantile(0.75):.6f} gwei")
    
    # Check for outliers
    Q1 = df['gas_price'].quantile(0.25)
    Q3 = df['gas_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = (df['gas_price'] < lower_bound) | (df['gas_price'] > upper_bound)
    print(f"\n🔍 Outlier Analysis:")
    print(f"   Outliers (3x IQR): {outliers.sum()} ({outliers.sum()/len(df)*100:.1f}%)")
    print(f"   Lower bound: {lower_bound:.6f} gwei")
    print(f"   Upper bound: {upper_bound:.6f} gwei")
    
    # Check data continuity
    df = df.sort_values('timestamp')
    time_diffs = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
    median_interval = time_diffs.median()
    print(f"\n⏱️  Data Continuity:")
    print(f"   Median interval: {median_interval:.2f} minutes")
    print(f"   Expected: ~5 minutes")
    print(f"   Gaps > 10 min: {(time_diffs > 10).sum()}")
    print(f"   Gaps > 30 min: {(time_diffs > 30).sum()}")
    
    # Check log transformation
    epsilon = 1e-8
    y_log = np.log(df['gas_price'] + epsilon)
    print(f"\n📊 Log Transformation Analysis:")
    print(f"   Original scale - Min: {df['gas_price'].min():.6f}, Max: {df['gas_price'].max():.6f}")
    print(f"   Log scale - Min: {y_log.min():.4f}, Max: {y_log.max():.4f}")
    print(f"   Log scale - Mean: {y_log.mean():.4f}, Std: {y_log.std():.4f}")
    
    # Check if log transformation is appropriate
    # For very small values, log can be problematic
    very_small = (df['gas_price'] < 0.01).sum()
    print(f"   Values < 0.01 gwei: {very_small} ({very_small/len(df)*100:.1f}%)")
    
    # Build features
    print(f"\n🔧 Feature Engineering:")
    X, feature_meta, df_features = build_feature_matrix(df, include_external_features=True)
    print(f"   Features created: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Check for NaN values
    nan_counts = X.isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        print(f"\n⚠️  Features with NaN values:")
        for feat, count in nan_features.head(10).items():
            print(f"   {feat}: {count} ({count/len(X)*100:.1f}%)")
    else:
        print(f"   ✅ No NaN values in features")
    
    # Check feature variance (low variance = not useful)
    feature_variance = X.var()
    low_variance = feature_variance[feature_variance < 1e-6]
    if len(low_variance) > 0:
        print(f"\n⚠️  Features with very low variance (< 1e-6): {len(low_variance)}")
        print(f"   These features may not be useful for prediction")
    
    # Check target quality
    steps_per_hour = feature_meta.get('steps_per_hour', 12)
    targets_log = build_horizon_targets(y_log, steps_per_hour)
    targets_original = build_horizon_targets(df['gas_price'], steps_per_hour)
    
    print(f"\n🎯 Target Quality (1h horizon):")
    y_1h_log = targets_log['1h']
    y_1h_original = targets_original['1h']
    
    # Remove NaN
    valid_idx = ~(X.isna().any(axis=1) | y_1h_log.isna() | y_1h_original.isna())
    valid_count = valid_idx.sum()
    
    print(f"   Valid samples: {valid_count}")
    print(f"   Target range (original): {y_1h_original[valid_idx].min():.6f} to {y_1h_original[valid_idx].max():.6f} gwei")
    print(f"   Target range (log): {y_1h_log[valid_idx].min():.4f} to {y_1h_log[valid_idx].max():.4f}")
    
    # Check if we have enough data
    if valid_count < 100:
        print(f"\n❌ INSUFFICIENT DATA: Only {valid_count} valid samples (need ≥100)")
    elif valid_count < 500:
        print(f"\n⚠️  LIMITED DATA: {valid_count} valid samples (recommend ≥500 for good performance)")
    else:
        print(f"\n✅ Sufficient data: {valid_count} valid samples")
    
    # Check feature-target correlation
    print(f"\n🔗 Feature-Target Correlations (top 10):")
    correlations = X[valid_idx].corrwith(y_1h_log[valid_idx]).abs().sort_values(ascending=False)
    for feat, corr in correlations.head(10).items():
        print(f"   {feat}: {corr:.4f}")
    
    if correlations.max() < 0.1:
        print(f"\n⚠️  WARNING: No strong feature correlations (< 0.1)")
        print(f"   This suggests features may not be predictive")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if valid_count < 100:
        print(f"   1. ❌ Collect more data (need at least 100 valid samples)")
    
    if very_small / len(df) > 0.5:
        print(f"   2. ⚠️  Consider not using log transformation for very small values")
        print(f"      Current: {very_small/len(df)*100:.1f}% of values < 0.01 gwei")
    
    if correlations.max() < 0.1:
        print(f"   3. ⚠️  Improve feature engineering - features show weak correlation with target")
    
    if len(low_variance) > X.shape[1] * 0.3:
        print(f"   4. ⚠️  Remove low-variance features ({len(low_variance)} features)")
    
    if median_interval > 10:
        print(f"   5. ⚠️  Data collection gaps too large (median: {median_interval:.1f} min)")
        print(f"      This can hurt time-series model performance")
    
    print(f"\n" + "="*60)

if __name__ == "__main__":
    try:
        diagnose_data_quality()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

