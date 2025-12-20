"""
Full Backtest for Your Current Models
Tests on 1 week of recent historical data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("  🚀 COMPREHENSIVE MODEL BACKTEST")
print("="*80)

# Load data
print("\n📊 Loading historical data...")
conn = sqlite3.connect("gas_data.db")

# Get last 1000 records (about 3.5 days at 5-min intervals)
query = """
SELECT timestamp, current_gas, base_fee, priority_fee, block_number
FROM gas_prices
ORDER BY timestamp DESC
LIMIT 1000
"""

df = pd.read_sql(query, conn, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
conn.close()

print(f"✅ Loaded {len(df)} records")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Calculate basic features
print("\n🔧 Creating features...")

df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Lag features
for lag in [1, 3, 6, 12, 24]:
    df[f'gas_lag_{lag}'] = df['current_gas'].shift(lag)

# Rolling features
for window in [12, 24, 72]:
    df[f'gas_ma_{window}'] = df['current_gas'].rolling(window).mean()
    df[f'gas_std_{window}'] = df['current_gas'].rolling(window).std()

# Drop NaN
df = df.dropna()

print(f"✅ {len(df)} samples after feature engineering")

# Load models
print("\n🤖 Loading models...")

model_dir = Path("models/saved_models")
models = {}

for horizon in ['1h', '4h', '24h']:
    model_path = model_dir / f"model_{horizon}.pkl"
    scaler_path = model_dir / f"scaler_{horizon}.pkl"

    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                models[horizon] = pickle.load(f)
            print(f"✅ Loaded {horizon} model")

            # Try to load scaler
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    models[f'{horizon}_scaler'] = pickle.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {horizon} model: {e}")

if not models:
    print("❌ No models loaded!")
    sys.exit(1)

# Run backtest
print("\n" + "="*80)
print("  📈 RUNNING BACKTEST")
print("="*80)

results = {}

for horizon in ['1h', '4h', '24h']:
    if horizon not in models:
        continue

    print(f"\n{'='*80}")
    print(f"  Testing {horizon.upper()} Predictions")
    print(f"{'='*80}")

    # Create target (future gas price)
    if horizon == '1h':
        shift = 12  # 12 * 5 min = 1 hour
    elif horizon == '4h':
        shift = 48
    elif horizon == '24h':
        shift = 288

    # Create target
    test_df = df.copy()
    test_df['target'] = test_df['current_gas'].shift(-shift)
    test_df = test_df.dropna()

    if len(test_df) < 50:
        print(f"⚠️  Not enough data for {horizon} (need {shift} future points)")
        continue

    print(f"\n📊 Test samples: {len(test_df)}")

    # Prepare features
    feature_cols = [col for col in test_df.columns
                   if col not in ['timestamp', 'target', 'current_gas', 'block_number']]

    X = test_df[feature_cols].values
    y_true = test_df['target'].values

    # Get model
    model = models[horizon]

    try:
        # Make predictions
        y_pred = model.predict(X)

        # Handle different output formats
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Directional accuracy
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction)

        # Comparison to naive
        naive_pred = test_df['current_gas'].values
        naive_mae = mean_absolute_error(y_true, naive_pred)
        improvement = ((naive_mae - mae) / naive_mae) * 100

        # Store results
        results[horizon] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'naive_mae': naive_mae,
            'improvement_vs_naive': improvement,
            'n_samples': len(y_true)
        }

        # Print results
        print(f"\n🎯 Performance Metrics:")
        print(f"   MAE:                  {mae:.6f} Gwei")
        print(f"   RMSE:                 {rmse:.6f} Gwei")
        print(f"   R² Score:             {r2:.4f} {_get_r2_rating(r2)}")
        print(f"   MAPE:                 {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.2%} {_get_dir_rating(directional_accuracy)}")

        print(f"\n🆚 vs Naive Forecast:")
        print(f"   Naive MAE:            {naive_mae:.6f} Gwei")
        print(f"   Improvement:          {improvement:+.1f}%")

        # Error analysis
        errors = y_pred - y_true
        print(f"\n📊 Error Distribution:")
        print(f"   Median error:         {np.median(errors):.6f} Gwei")
        print(f"   Std dev:              {np.std(errors):.6f} Gwei")
        print(f"   P95 error:            {np.percentile(np.abs(errors), 95):.6f} Gwei")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def _get_r2_rating(r2):
    if r2 > 0.8:
        return "✅ EXCELLENT"
    elif r2 > 0.7:
        return "✅ GOOD"
    elif r2 > 0.5:
        return "⚠️  NEEDS IMPROVEMENT"
    else:
        return "❌ POOR"

def _get_dir_rating(acc):
    if acc > 0.8:
        return "✅ EXCELLENT"
    elif acc > 0.7:
        return "✅ GOOD"
    elif acc > 0.6:
        return "⚠️  NEEDS IMPROVEMENT"
    else:
        return "❌ POOR"

# Summary
print("\n" + "="*80)
print("  📊 BACKTEST SUMMARY")
print("="*80)

if results:
    print(f"\n{'Horizon':<10} {'MAE':<12} {'R²':<10} {'Dir Acc':<12} {'vs Naive':<12}")
    print("-"*80)

    for horizon, metrics in results.items():
        print(f"{horizon.upper():<10} "
              f"{metrics['mae']:.6f}   "
              f"{metrics['r2']:.4f}    "
              f"{metrics['directional_accuracy']:.2%}      "
              f"{metrics['improvement_vs_naive']:+.1f}%")

    # Overall assessment
    print(f"\n🎯 Overall Assessment:")

    avg_r2 = np.mean([m['r2'] for m in results.values()])
    avg_dir = np.mean([m['directional_accuracy'] for m in results.values()])
    avg_improve = np.mean([m['improvement_vs_naive'] for m in results.values()])

    print(f"   Average R²:              {avg_r2:.4f}")
    print(f"   Average Dir. Accuracy:   {avg_dir:.2%}")
    print(f"   Average Improvement:     {avg_improve:+.1f}%")

    if avg_r2 > 0.7 and avg_dir > 0.75:
        print(f"\n   ✅ GOOD - Models are performing well!")
    elif avg_r2 > 0.5 and avg_dir > 0.6:
        print(f"\n   ⚠️  FAIR - Models work but have room for improvement")
    else:
        print(f"\n   ❌ NEEDS IMPROVEMENT - See ML_IMPROVEMENT_PLAN.md")

    # Save results
    import json

    report = {
        'test_date': datetime.now().isoformat(),
        'n_records': len(df),
        'horizons': {h: {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in m.items()}
                    for h, m in results.items()}
    }

    report_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved to: {report_file}")

else:
    print("\n❌ No results to display")

print("\n" + "="*80)
print("  ✅ BACKTEST COMPLETE")
print("="*80)

print(f"\n🚀 Next Steps:")
print(f"   1. Review results above")
print(f"   2. Compare with targets (R² > 0.70, Dir Acc > 75%)")
print(f"   3. See improvement plan: cat ../ML_IMPROVEMENT_PLAN.md")
print(f"   4. Monitor live: python3 testing/live_performance_monitor.py")

print()
