import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.database import DatabaseManager


class GasFeatureEngineer:
    def __init__(self):
        self.db = DatabaseManager()
    
    def prepare_training_data(self, hours_back=720):
        """
        Fetch historical data and engineer features
        Returns: X (features), y (targets for 1h, 4h, 24h)
        """
        # Get raw data from database
        raw_data = self.db.get_historical_data(hours=hours_back)
        
        if len(raw_data) < 100:
            raise ValueError(f"Not enough data: only {len(raw_data)} records. Need at least 100.")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'gas': d.current_gas,
            'base_fee': d.base_fee,
            'priority_fee': d.priority_fee,
            'block_number': d.block_number
        } for d in raw_data])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Feature Engineering
        df = self._add_time_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_target_variables(df)
        
        # Remove NaN rows (from lag/rolling operations)
        df = df.dropna()
        
        print(f"✅ Prepared {len(df)} training samples with {len(df.columns)} features")
        
        return df
    
    def _add_time_features(self, df):
        """Extract time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lag_features(self, df):
        """Add lagged gas prices (past values)"""
        # Lag features: gas prices from 1h, 3h, 6h, 12h, 24h ago
        # Assuming data is collected every 5 minutes: 12 records per hour
        lags = [12, 36, 72, 144, 288]  # 1h, 3h, 6h, 12h, 24h
        
        for lag in lags:
            df[f'gas_lag_{lag//12}h'] = df['gas'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df):
        """Add rolling statistics"""
        windows = [12, 36, 72, 144]  # 1h, 3h, 6h, 12h
        
        for window in windows:
            hours = window // 12
            df[f'gas_rolling_mean_{hours}h'] = df['gas'].rolling(window).mean()
            df[f'gas_rolling_std_{hours}h'] = df['gas'].rolling(window).std()
            df[f'gas_rolling_min_{hours}h'] = df['gas'].rolling(window).min()
            df[f'gas_rolling_max_{hours}h'] = df['gas'].rolling(window).max()
        
        # Rate of change
        df['gas_change_1h'] = df['gas'].pct_change(12)
        df['gas_change_3h'] = df['gas'].pct_change(36)
        
        return df
    
    def _add_target_variables(self, df):
        """Add target variables (future gas prices)"""
        # Targets: gas prices 1h, 4h, 24h in the future
        df['target_1h'] = df['gas'].shift(-12)   # 1 hour ahead
        df['target_4h'] = df['gas'].shift(-48)   # 4 hours ahead
        df['target_24h'] = df['gas'].shift(-288) # 24 hours ahead
        
        return df
    
    def get_feature_columns(self, df):
        """Return list of feature column names"""
        exclude = ['timestamp', 'gas', 'block_number', 'target_1h', 'target_4h', 'target_24h']
        return [col for col in df.columns if col not in exclude]
    
    def prepare_prediction_features(self, recent_data):
        """
        Prepare features for making predictions on new data
        recent_data: List of recent gas price records (last 24+ hours)
        
        This method now uses advanced features if available, otherwise falls back to basic features
        """
        try:
            # Try to use advanced features
            from models.advanced_features import create_advanced_features
            
            df = pd.DataFrame(recent_data)
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure we have gas_price column
            if 'gas_price' not in df.columns:
                if 'current_gas' in df.columns:
                    df['gas_price'] = df['current_gas']
                elif 'gwei' in df.columns:
                    df['gas_price'] = df['gwei']
                elif 'gas' in df.columns:
                    df['gas_price'] = df['gas']
            
            X, _ = create_advanced_features(df)
            
            # Get the latest row (most recent data)
            return X.iloc[[-1]]
        except Exception as e:
            # Fallback to basic features
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Advanced features failed, using basic features: {e}")
            
            df = pd.DataFrame(recent_data)
            
            # Ensure timestamp is datetime type
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df = self._add_time_features(df)
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)
            
            # Get the latest row (most recent data)
            latest = df.iloc[-1:]
            feature_cols = self.get_feature_columns(df)
            
            return latest[feature_cols]


# Test the feature engineer
if __name__ == "__main__":
    engineer = GasFeatureEngineer()
    df = engineer.prepare_training_data(hours_back=720)
    print(f"\n📊 Data shape: {df.shape}")
    print(f"📊 Features: {engineer.get_feature_columns(df)}")
    print(f"\n📊 Sample data:\n{df.head()}")

