#!/usr/bin/env python3
"""
Debug model performance issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from data.database import DatabaseManager
from datetime import datetime, timedelta

def debug_model():
    """Debug model performance issues"""
    
    print("="*60)
    print("MODEL DEBUGGING")
    print("="*60)
    
    # 1. Load historical data
    print("\n1. CHECKING DATA...")
    db = DatabaseManager()
    historical = db.get_historical_data(hours=168)  # Last week
    
    if len(historical) == 0:
        print("❌ ERROR: No historical data found!")
        return
    
    # Convert to list format
    gas_prices = []
    for h in historical:
        if isinstance(h, dict):
            gas_prices.append(h.get('gwei', 0) or h.get('current_gas', 0))
        else:
            gas_prices.append(h.current_gas if hasattr(h, 'current_gas') else 0)
    
    gas_prices = [p for p in gas_prices if p > 0]  # Filter zeros
    
    if len(gas_prices) == 0:
        print("❌ ERROR: No valid gas prices found!")
        return
    
    print(f"   Data points: {len(gas_prices)}")
    print(f"   Min gas: {min(gas_prices):.6f} gwei")
    print(f"   Max gas: {max(gas_prices):.6f} gwei")
    print(f"   Mean gas: {np.mean(gas_prices):.6f} gwei")
    print(f"   Std dev: {np.std(gas_prices):.6f} gwei")
    
    # Check for outliers
    q1 = np.percentile(gas_prices, 25)
    q3 = np.percentile(gas_prices, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [p for p in gas_prices if p < lower_bound or p > upper_bound]
    print(f"   Outliers detected: {len(outliers)} ({len(outliers)/len(gas_prices)*100:.1f}%)")
    
    if outliers:
        print(f"   Outlier range: {min(outliers):.6f} to {max(outliers):.6f}")
        print(f"   ⚠️ WARNING: High number of outliers may affect model!")
    
    # 2. Check model files
    print("\n2. CHECKING MODEL FILES...")
    
    try:
        model_path = 'backend/models/saved_models/model_1h.pkl'
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data.get('model')
            print("   ✅ Model loaded (dict format)")
            print(f"   Model type: {model_data.get('model_name', 'Unknown')}")
            print(f"   Metrics: R²={model_data.get('metrics', {}).get('r2', 0):.4f}, Dir={model_data.get('metrics', {}).get('directional_accuracy', 0):.4f}")
            predicts_pct = model_data.get('predicts_percentage_change', False)
            print(f"   Predicts percentage change: {predicts_pct}")
        else:
            model = model_data
            print("   ✅ Model loaded (direct format)")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    try:
        scaler_path = 'backend/models/saved_models/scaler_1h.pkl'
        scaler = joblib.load(scaler_path)
        print("   ✅ Scaler loaded")
        
        # Check scaler properties
        if hasattr(scaler, 'mean_') and len(scaler.mean_) > 0:
            print(f"   Scaler mean (first 5): {scaler.mean_[:5]}")
            print(f"   Scaler scale (first 5): {scaler.scale_[:5]}")
        
    except Exception as e:
        print(f"   ⚠️ Scaler not found: {e}")
        scaler = None
    
    try:
        feature_names_path = 'backend/models/saved_models/feature_names.pkl'
        feature_names = joblib.load(feature_names_path)
        print(f"   ✅ Feature names loaded ({len(feature_names)} features)")
    except Exception as e:
        print(f"   ⚠️ Feature names not found: {e}")
        feature_names = None
    
    # 3. Make sample prediction and check
    print("\n3. TESTING PREDICTION...")
    
    # Get recent data for prediction
    recent_data = []
    for h in historical[-100:]:  # Last 100 points
        if isinstance(h, dict):
            recent_data.append({
                'timestamp': h.get('timestamp', ''),
                'gas_price': h.get('gwei', 0) or h.get('current_gas', 0)
            })
        else:
            recent_data.append({
                'timestamp': h.timestamp if hasattr(h, 'timestamp') else '',
                'gas_price': h.current_gas if hasattr(h, 'current_gas') else 0
            })
    
    if len(recent_data) == 0:
        print("   ❌ No recent data for prediction")
        return
    
    current_gas = recent_data[-1]['gas_price']
    print(f"   Current gas: {current_gas:.6f} gwei")
    
    # Try to make prediction
    try:
        from models.advanced_features import create_advanced_features
        
        # Create features
        df = pd.DataFrame(recent_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if 'gas_price' not in df.columns:
            if 'current_gas' in df.columns:
                df['gas_price'] = df['current_gas']
            elif 'gwei' in df.columns:
                df['gas_price'] = df['gwei']
        
        X, _ = create_advanced_features(df)
        
        print(f"   Features shape: {X.shape}")
        print(f"   Feature ranges:")
        print(f"     Min: {X.min().min():.6f}")
        print(f"     Max: {X.max().max():.6f}")
        print(f"     Mean: {X.mean().mean():.6f}")
        
        # Get last features (most recent)
        X_last = X.iloc[[-1]]
        
        # Check feature alignment
        if feature_names:
            missing = [f for f in feature_names if f not in X.columns]
            extra = [f for f in X.columns if f not in feature_names]
            if missing:
                print(f"   ⚠️ Missing features: {missing[:5]}...")
            if extra:
                print(f"   ⚠️ Extra features: {extra[:5]}...")
            
            # Reorder to match
            if missing:
                for f in missing:
                    X_last[f] = 0
            X_last = X_last[feature_names]
        
        # Scale if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X_last)
            print(f"   Scaled features:")
            print(f"     Min: {X_scaled.min():.6f}")
            print(f"     Max: {X_scaled.max():.6f}")
            print(f"     Mean: {X_scaled.mean():.6f}")
        else:
            X_scaled = X_last
        
        # Predict
        prediction_raw = model.predict(X_scaled)[0]
        
        # Check if model predicts percentage change
        predicts_pct = isinstance(model_data, dict) and model_data.get('predicts_percentage_change', False)
        
        if predicts_pct:
            # Convert percentage change to absolute price
            prediction = current_gas * (1 + prediction_raw / 100)
            print(f"   Raw prediction (pct change): {prediction_raw:.2f}%")
            print(f"   Converted to absolute: {prediction:.6f} gwei")
        else:
            prediction = prediction_raw
            print(f"   Raw prediction: {prediction:.6f} gwei")
        
        print(f"   Current gas: {current_gas:.6f} gwei")
        print(f"   Difference: {abs(prediction - current_gas):.6f} gwei")
        print(f"   Percent error: {abs(prediction - current_gas)/current_gas*100:.1f}%")
        
        # Check if prediction is reasonable
        if prediction < 0:
            print("   ❌ ERROR: Negative prediction!")
        elif prediction > current_gas * 10:
            print("   ❌ ERROR: Prediction is 10x higher than current!")
        elif prediction < current_gas * 0.1:
            print("   ❌ ERROR: Prediction is 10x lower than current!")
        else:
            print("   ✅ Prediction seems reasonable")
        
    except Exception as e:
        print(f"   ❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Check for common issues
    print("\n4. CHECKING FOR COMMON ISSUES...")
    
    # Check if target variable was scaled
    print("   Checking data ranges...")
    if max(gas_prices) - min(gas_prices) > 1.0:
        print("   ⚠️ WARNING: Large range in gas prices!")
        print(f"      Range: {max(gas_prices) - min(gas_prices):.6f} gwei")
    
    # Check coefficient of variation
    cv = np.std(gas_prices) / np.mean(gas_prices) if np.mean(gas_prices) > 0 else 0
    print(f"   Coefficient of variation: {cv:.2f}")
    if cv > 1.0:
        print("   ⚠️ WARNING: High variability in gas prices!")
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    debug_model()

