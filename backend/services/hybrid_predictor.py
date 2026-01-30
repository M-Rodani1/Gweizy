import os
import sys
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime

class HybridPredictor:
    def __init__(self):
        # Path configuration (adjusted for Railway structure)
        # Assuming models are deployed to backend/models/saved_models/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(base_dir, 'models', 'saved_models')
        
        self.model_4h_path = os.path.join(self.model_dir, 'model_4h.pkl')
        self.model_1h_path = os.path.join(self.model_dir, 'model_1h.pkl')
        
        # Load models on initialization
        self.load_models()
        
        # Database connection - use same config as DatabaseManager
        # Use /data for persistent storage on Railway, fallback to local for development
        self.db_url = os.getenv("DATABASE_URL",
                                "sqlite:////data/gas_data.db" if os.path.exists('/data')
                                else "sqlite:///gas_data.db")
        # Fix for SQLAlchemy requiring postgresql:// instead of postgres://
        if self.db_url and self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)

        self.engine = create_engine(self.db_url)

    def load_models(self):
        try:
            self.data_4h = joblib.load(self.model_4h_path)
            self.data_1h = joblib.load(self.model_1h_path)
            
            # Extract threshold from saved model config, default to 0.8 (Conservative)
            self.threshold = self.data_1h.get('inference_config', {}).get('threshold', 0.8)
            print(f"✅ Hybrid Models loaded. Threshold: {self.threshold}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.data_4h = None
            self.data_1h = None

    def fetch_recent_data(self):
        """Fetch last 24h of data for feature engineering"""
        query = """
        SELECT timestamp, current_gas as gas_price, block_number, 
               base_fee, priority_fee, gas_used, gas_limit, utilization
        FROM gas_prices
        WHERE timestamp >= NOW() - INTERVAL '24 HOURS'
        ORDER BY timestamp ASC
        """
        # Adjust query for SQLite if running locally
        if 'sqlite' in self.db_url:
            query = """
            SELECT timestamp, current_gas as gas_price, block_number, 
                   base_fee, priority_fee, gas_used, gas_limit, utilization
            FROM gas_prices
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp ASC
            """
            
        try:
            df = pd.read_sql(query, self.engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"❌ DB Error: {e}")
            return pd.DataFrame()

    def generate_features(self, df):
        """Replicates training feature engineering exactly"""
        if df.empty: return None
        
        df = df.copy().sort_values('timestamp').reset_index(drop=True)
        
        # --- minimal feature set required by models ---
        # 1. Time
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 2. Pressure / EIP-1559
        if 'utilization' in df.columns and df['utilization'].notna().any():
            df['utilization'] = df['utilization'].fillna(50.0)  # Default to 50% if missing
            df['utilization_frac'] = df['utilization'] / 100.0
            df['pressure_index'] = df['utilization_frac'] - 0.50
            df['pressure_velocity'] = df['pressure_index'].diff().fillna(0)
            df['pressure_acceleration'] = df['pressure_velocity'].diff().fillna(0)
            for window in [10, 30, 60]:
                df[f'pressure_cum_{window}m'] = df['pressure_index'].rolling(window=window, min_periods=1).sum()
        else:
            # Create dummy pressure features if utilization is missing
            df['utilization_frac'] = 0.5
            df['pressure_index'] = 0.0
            df['pressure_velocity'] = 0.0
            df['pressure_acceleration'] = 0.0
            for window in [10, 30, 60]:
                df[f'pressure_cum_{window}m'] = 0.0
        
        # 3. Rolling Stats & Lags
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'gas_lag_{lag}'] = df['gas_price'].shift(lag)
            
        for window in [6, 12, 24, 48]:
            df[f'gas_ma_{window}'] = df['gas_price'].rolling(window).mean()
            df[f'gas_std_{window}'] = df['gas_price'].rolling(window).std()
            df[f'gas_min_{window}'] = df['gas_price'].rolling(window).min()

            # Volatility-normalized features (Important for 4h model)
            rolling_std = df[f'gas_std_{window}']
            rolling_mean = df[f'gas_ma_{window}']
            df[f'price_zscore_{window}'] = (df['gas_price'] - rolling_mean) / (rolling_std + 1e-8)
            df[f'price_rel_mean_{window}'] = df['gas_price'] / (rolling_mean + 1e-8)

            # Mean reversion features (required by model)
            df[f'distance_from_mean_{window}'] = df['gas_price'] - rolling_mean
            df[f'mean_reversion_{window}'] = -df[f'distance_from_mean_{window}'] / (rolling_std + 1e-8)
            df[f'detrended_{window}'] = df['gas_price'] - rolling_mean
            df[f'price_ma_ratio_{window}'] = df['gas_price'] / (rolling_mean + 1e-8)

            # Trend strength features (required by 1h model)
            price_change = df['gas_price'].diff(window)
            df[f'trend_strength_{window}'] = price_change / (rolling_std + 1e-8)

        # 4. Micro-structure
        if 'base_fee' in df.columns:
            df['gas_base_divergence'] = df['gas_price'] - df['base_fee']
            df['gas_base_ratio'] = df['gas_price'] / (df['base_fee'] + 1e-8)

        # 5. Interactions
        for window in [6, 12, 24]:
            vol = df['gas_price'].rolling(window).std()
            mom = df['gas_price'] - df['gas_price'].shift(window)
            price_level = df['gas_price'] / (df['gas_price'].rolling(window*2).mean() + 1e-8)
            
            df[f'vol_momentum_interact_{window}'] = vol * mom.abs()
            df[f'price_vol_interact_{window}'] = price_level * vol

        # Fill NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df.iloc[[-1]] # Return only the current state row

    def predict(self):
        if not self.data_4h or not self.data_1h:
            return {"error": "Models not loaded"}
            
        df = self.fetch_recent_data()
        if len(df) < 30:
            return {"status": "warming_up", "message": "Insufficient data"}
            
        current_features = self.generate_features(df)
        if current_features is None:
            return {"error": "Feature generation failed"}

        # STAGE 1: 4h Regressor
        model_4h = self.data_4h['regressor']
        scaler_4h = self.data_4h['scaler']
        cols_4h = self.data_4h['feature_names']
        
        X_4h = current_features[cols_4h]
        trend_signal = model_4h.predict(scaler_4h.transform(X_4h))[0]
        
        # STAGE 2: 1h Classifier
        model_1h = self.data_1h['classifier']
        scaler_1h = self.data_1h['scaler']
        cols_1h = self.data_1h['feature_names']
        
        # Inject Signal
        current_features['trend_signal_4h'] = trend_signal
        
        X_1h = current_features[cols_1h]
        probs = model_1h.predict_proba(scaler_1h.transform(X_1h))[0]
        # probs order: [Wait, Normal, Urgent]
        
        # Decision Logic (Probability Gating)
        p_wait, p_normal, p_urgent = probs[0], probs[1], probs[2]
        
        action = "NORMAL"
        confidence = p_normal
        
        if p_urgent > self.threshold:
            action = "URGENT"
            confidence = p_urgent
        elif p_wait > self.threshold:
            action = "WAIT"
            confidence = p_wait
            
        return {
            "action": action,
            "confidence": float(confidence),
            "gas_price": float(current_features['gas_price'].iloc[0]),
            "trend_signal_4h": float(trend_signal),
            "probabilities": {
                "wait": float(p_wait),
                "normal": float(p_normal),
                "urgent": float(p_urgent)
            }
        }
