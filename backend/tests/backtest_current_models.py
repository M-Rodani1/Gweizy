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
import joblib  # Use joblib instead of pickle (Week 1 Quick Win #3)
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("  🚀 COMPREHENSIVE MODEL BACKTEST")
print("="*80)

# Load data using feature engineering pipeline (Week 1 Quick Wins)
print("\n📊 Loading historical data...")
from models.feature_engineering import GasFeatureEngineer

engineer = GasFeatureEngineer()

try:
    # Use same pipeline as training (includes enhanced features)
    df = engineer.prepare_training_data(hours_back=168)  # 1 week
    
    if len(df) == 0:
        print("❌ No training samples generated")
        print("💡 Using fallback: direct database query")
        
        # Fallback to direct query
        conn = sqlite3.connect("gas_data.db")
        query = """
        SELECT timestamp, current_gas as gas, base_fee, priority_fee, block_number
        FROM gas_prices
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        df = pd.read_sql(query, conn, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        conn.close()
        
        # Basic features only
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df = df.dropna()
    
    print(f"✅ {len(df)} samples after feature engineering")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
except Exception as e:
    print(f"⚠️  Feature engineering failed: {e}")
    print("   Using fallback method...")
    
    conn = sqlite3.connect("gas_data.db")
    query = """
    SELECT timestamp, current_gas as gas, base_fee, priority_fee, block_number
    FROM gas_prices
    ORDER BY timestamp DESC
    LIMIT 1000
    """
    df = pd.read_sql(query, conn, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    conn.close()
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = df.dropna()
    
    print(f"✅ {len(df)} samples (basic features only)")

# Load models
print("\n🤖 Loading models...")

model_dir = Path("models/saved_models")
models = {}

for horizon in ['1h', '4h', '24h']:
    model_path = model_dir / f"model_{horizon}.pkl"
    
    if not model_path.exists():
        # Try backend path
        model_path = Path("backend") / model_dir / f"model_{horizon}.pkl"

    if model_path.exists():
        try:
            # Load with joblib (Week 1 Quick Win #3: models saved with joblib)
            model_data = joblib.load(model_path)
            
            # Handle new format (dict with 'model' key) or old format (model object)
            if isinstance(model_data, dict):
                models[horizon] = model_data.get('model')
                # Get scaler from model_data (Week 1 Quick Win #3: RobustScaler)
                scaler = model_data.get('feature_scaler') or model_data.get('scaler')
                if scaler:
                    models[f'{horizon}_scaler'] = scaler
                    print(f"✅ Loaded {horizon} model with RobustScaler")
                else:
                    print(f"✅ Loaded {horizon} model (no scaler)")
            else:
                # Old format - model is the object itself
                models[horizon] = model_data
                print(f"✅ Loaded {horizon} model (legacy format)")
                
        except Exception as e:
            print(f"⚠️  Error loading {horizon} model: {e}")
            import traceback
            traceback.print_exc()

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

    # Prepare features using same method as training
    # Get feature columns (exclude targets and metadata)
    exclude_cols = ['timestamp', 'target', 'target_1h', 'target_4h', 'target_24h', 
                    'gas', 'current_gas', 'block_number']
    
    # Try to use feature engineer's method
    try:
        feature_cols = engineer.get_feature_columns(test_df)
    except:
        # Fallback: manual feature selection
        feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    # Ensure we have the features the model expects
    # If model was trained with enhanced features, we need them
    X = test_df[feature_cols].values
    y_true = test_df['target'].values

    # Get model and scaler
    model = models[horizon]
    scaler = models.get(f'{horizon}_scaler')

    try:
        # Check feature count match
        if scaler:
            expected_features = scaler.n_features_in_
            actual_features = X.shape[1]
            
            if expected_features != actual_features:
                print(f"⚠️  Feature mismatch: model expects {expected_features}, got {actual_features}")
                print(f"   Available features: {len(feature_cols)}")
                print(f"   This is expected if enhanced features are sparse")
                print(f"   Skipping {horizon} backtest (need matching features)")
                continue
        
        # Scale features if scaler available (Week 1 Quick Win #3)
        if scaler:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
        else:
            # No scaler - predict directly
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
