"""
Quick Model Performance Test
Adapted to work with your current setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle

print("="*70)
print("  🧪 QUICK MODEL PERFORMANCE TEST")
print("="*70)

# 1. Check Database
print("\n📊 Checking database...")
db_path = Path("gas_data.db")

if not db_path.exists():
    print("❌ Database not found at:", db_path)
    sys.exit(1)

conn = sqlite3.connect(str(db_path))

# Get record count
df_count = pd.read_sql("SELECT COUNT(*) as count FROM gas_prices", conn)
total_records = df_count['count'].iloc[0]

print(f"✅ Found {total_records:,} gas price records")

# Get date range
df_range = pd.read_sql(
    "SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest FROM gas_prices",
    conn
)
print(f"   Date range: {df_range['earliest'].iloc[0]} to {df_range['latest'].iloc[0]}")

# 2. Load recent data for testing
print("\n📈 Loading last 48 hours of data for testing...")

cutoff = (datetime.utcnow() - timedelta(hours=48)).isoformat()

query = """
SELECT timestamp, current_gas, base_fee, priority_fee
FROM gas_prices
WHERE timestamp >= ?
ORDER BY timestamp ASC
"""

df = pd.read_sql(query, conn, params=(cutoff,), parse_dates=['timestamp'])
conn.close()

print(f"✅ Loaded {len(df)} data points")

if len(df) < 50:
    print(f"⚠️  Warning: Only {len(df)} recent data points")
    print("   For best results, need 100+ points (about 8 hours at 5-min intervals)")
    if len(df) < 20:
        print("❌ Not enough data for meaningful test. Exiting.")
        sys.exit(1)

# 3. Check for models
print("\n🤖 Checking for trained models...")

model_dir = Path("models/saved_models")
model_files = []

# Check for .pkl files
for horizon in ['1h', '4h', '24h']:
    model_path = model_dir / f"model_{horizon}.pkl"
    if model_path.exists():
        model_files.append((horizon, model_path))
        print(f"✅ Found {horizon} model")

# Check for .joblib files
joblib_files = list(model_dir.glob("*.joblib"))
if joblib_files:
    print(f"✅ Found {len(joblib_files)} .joblib model(s)")

if not model_files:
    print("⚠️  No .pkl models found for quick test")
    print("   Models available:", list(model_dir.glob("*")))

# 4. Quick baseline test (without ML model)
print("\n" + "="*70)
print("  📊 BASELINE PERFORMANCE ANALYSIS")
print("="*70)

# Calculate basic statistics
current_gas = df['current_gas'].values

print(f"\n📈 Recent Gas Price Statistics (last {len(df)} points):")
print(f"   Mean:    {np.mean(current_gas):.6f} Gwei")
print(f"   Median:  {np.median(current_gas):.6f} Gwei")
print(f"   Std Dev: {np.std(current_gas):.6f} Gwei")
print(f"   Min:     {np.min(current_gas):.6f} Gwei")
print(f"   Max:     {np.max(current_gas):.6f} Gwei")

# Calculate volatility
price_changes = np.diff(current_gas)
volatility = np.std(price_changes)
avg_change = np.mean(np.abs(price_changes))

print(f"\n📊 Volatility Metrics:")
print(f"   Volatility (std):      {volatility:.6f} Gwei")
print(f"   Avg absolute change:   {avg_change:.6f} Gwei")
print(f"   Max price swing:       {np.max(np.abs(price_changes)):.6f} Gwei")

# Trend analysis
if len(current_gas) > 12:
    recent_12 = current_gas[-12:]
    older_12 = current_gas[-24:-12] if len(current_gas) > 24 else current_gas[:12]

    trend = "UP" if np.mean(recent_12) > np.mean(older_12) else "DOWN"
    trend_pct = ((np.mean(recent_12) - np.mean(older_12)) / np.mean(older_12)) * 100

    print(f"\n📉 Recent Trend:")
    print(f"   Direction: {trend}")
    print(f"   Change:    {trend_pct:+.2f}%")

# Naive forecast baseline
print(f"\n🎯 Naive Forecast Baseline:")
print("   (Predicting 'no change' from current price)")

# How well would "predict current price" work?
if len(current_gas) > 12:
    # Simulate predicting 1 hour ahead (12 intervals) by using current price
    y_true_1h = current_gas[12:]
    y_pred_naive = current_gas[:-12]

    mae_naive = np.mean(np.abs(y_true_1h - y_pred_naive))

    print(f"   Naive MAE (1h):  {mae_naive:.6f} Gwei")
    print(f"   (This is what your ML model needs to beat)")

# Data quality check
nan_count = df[['current_gas', 'base_fee']].isna().sum().sum()
if nan_count > 0:
    print(f"\n⚠️  Warning: Found {nan_count} missing values in data")

# 5. Summary
print("\n" + "="*70)
print("  ✅ TEST COMPLETE")
print("="*70)

print(f"\n📝 Summary:")
print(f"   ✅ Database: {total_records:,} total records")
print(f"   ✅ Recent data: {len(df)} points (last 48h)")
print(f"   ✅ Data quality: {'Good' if nan_count == 0 else 'Has missing values'}")

if model_files:
    print(f"   ✅ Models found: {len(model_files)} horizons")
else:
    print(f"   ⚠️  No .pkl models found (may need to retrain)")

print(f"\n🎯 Next Steps:")

if total_records < 10000:
    print(f"   1. Collect more data (currently {total_records:,}, recommend 10,000+)")
else:
    print(f"   1. ✅ Good data volume ({total_records:,} records)")

if not model_files:
    print(f"   2. Train models: python3 scripts/train_ensemble_final.py")
else:
    print(f"   2. ✅ Models available")

print(f"   3. Run full backtest: python3 testing/comprehensive_backtester.py")
print(f"   4. Monitor live: python3 testing/live_performance_monitor.py")
print(f"   5. Review improvements: cat ../ML_IMPROVEMENT_PLAN.md")

print("\n" + "="*70)
