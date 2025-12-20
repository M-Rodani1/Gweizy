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
        # Handle both dict and object formats
        df = pd.DataFrame([{
            'timestamp': d.get('timestamp') if isinstance(d, dict) else d.timestamp,
            'gas': d.get('gwei') or d.get('current_gas') if isinstance(d, dict) else d.current_gas,
            'base_fee': d.get('baseFee') or d.get('base_fee') if isinstance(d, dict) else d.base_fee,
            'priority_fee': d.get('priorityFee') or d.get('priority_fee') if isinstance(d, dict) else d.priority_fee,
            'block_number': d.get('block_number') if isinstance(d, dict) else getattr(d, 'block_number', None)
        } for d in raw_data])
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Join enhanced congestion features (Week 1 Quick Win #2)
        df = self._join_onchain_features(df)
        
        # Feature Engineering
        df = self._add_time_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_target_variables(df)
        
        # Remove NaN rows (from lag/rolling operations)
        df = df.dropna()
        
        print(f"✅ Prepared {len(df)} training samples with {len(df.columns)} features")
        
        return df
    
    def _join_onchain_features(self, df):
        """
        Join enhanced congestion features from onchain_features table
        
        These features explain 27% of gas price variance and are critical
        for prediction accuracy (Week 1 Quick Win #2).
        """
        try:
            session = self.db._get_session()
            from data.database import OnChainFeatures
            from datetime import timedelta
            
            # Get onchain features for the same time period
            if 'timestamp' in df.columns and len(df) > 0:
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                
                # Query onchain features
                onchain_data = session.query(OnChainFeatures).filter(
                    OnChainFeatures.timestamp >= min_time - timedelta(minutes=5),
                    OnChainFeatures.timestamp <= max_time + timedelta(minutes=5)
                ).all()
                
                if onchain_data:
                    # Create DataFrame from onchain features
                    onchain_df = pd.DataFrame([{
                        'block_number': f.block_number,
                        'timestamp': f.timestamp,
                        'pending_tx_count': f.pending_tx_count if hasattr(f, 'pending_tx_count') else None,
                        'unique_addresses': f.unique_addresses if hasattr(f, 'unique_addresses') else None,
                        'tx_per_second': f.tx_per_second if hasattr(f, 'tx_per_second') else None,
                        'gas_utilization_ratio': f.gas_utilization_ratio if hasattr(f, 'gas_utilization_ratio') else None,
                        'contract_call_ratio': f.contract_call_ratio,
                        'avg_tx_gas': f.avg_tx_gas if hasattr(f, 'avg_tx_gas') else None,
                        'large_tx_ratio': f.large_tx_ratio if hasattr(f, 'large_tx_ratio') else None,
                        'congestion_level': f.congestion_level if hasattr(f, 'congestion_level') else None,
                        'is_highly_congested': f.is_highly_congested if hasattr(f, 'is_highly_congested') else None,
                    } for f in onchain_data])
                    
                    # Convert timestamp to datetime
                    if not pd.api.types.is_datetime64_any_dtype(onchain_df['timestamp']):
                        onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])
                    
                    # Merge on block_number (closest match)
                    # Use merge_asof for time-based join (finds closest timestamp)
                    df = pd.merge_asof(
                        df.sort_values('timestamp'),
                        onchain_df.sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest',
                        tolerance=pd.Timedelta(minutes=5),
                        suffixes=('', '_onchain')
                    )
                    
                    # Fill missing values with forward fill (use last known value)
                    enhanced_cols = [
                        'pending_tx_count', 'unique_addresses', 'tx_per_second',
                        'gas_utilization_ratio', 'avg_tx_gas', 'large_tx_ratio',
                        'congestion_level', 'is_highly_congested'
                    ]
                    for col in enhanced_cols:
                        if col in df.columns:
                            df[col] = df[col].ffill().fillna(0)
                    
                    print(f"✅ Joined {len(onchain_data)} onchain feature records")
                else:
                    print("⚠️  No onchain features found - using basic features only")
                    # Add empty columns so feature columns are consistent
                    enhanced_cols = [
                        'pending_tx_count', 'unique_addresses', 'tx_per_second',
                        'gas_utilization_ratio', 'avg_tx_gas', 'large_tx_ratio',
                        'congestion_level', 'is_highly_congested'
                    ]
                    for col in enhanced_cols:
                        df[col] = 0
            
            session.close()
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not join onchain features: {e}")
            # Add empty columns so feature columns are consistent
            enhanced_cols = [
                'pending_tx_count', 'unique_addresses', 'tx_per_second',
                'gas_utilization_ratio', 'avg_tx_gas', 'large_tx_ratio',
                'congestion_level', 'is_highly_congested'
            ]
            for col in enhanced_cols:
                if col not in df.columns:
                    df[col] = 0
        
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
        # Updated for 1-minute sampling (60 records per hour) - Week 1 Quick Win #1
        # Fallback to 12 records/hour if data is sparse (backward compatibility)
        sample_rate = self._detect_sample_rate(df)
        
        if sample_rate >= 50:  # ~1 minute sampling (60/hour)
            lags = [60, 180, 360, 720, 1440]  # 1h, 3h, 6h, 12h, 24h at 1-min intervals
            lag_hours = [1, 3, 6, 12, 24]
        else:  # ~5 minute sampling (12/hour) - legacy data
            lags = [12, 36, 72, 144, 288]  # 1h, 3h, 6h, 12h, 24h at 5-min intervals
            lag_hours = [1, 3, 6, 12, 24]
        
        for lag, hours in zip(lags, lag_hours):
            df[f'gas_lag_{hours}h'] = df['gas'].shift(lag)
        
        return df
    
    def _detect_sample_rate(self, df):
        """Detect sampling rate (records per hour)"""
        if len(df) < 2 or 'timestamp' not in df.columns:
            return 12  # Default to 5-minute sampling
        
        # Calculate average time between records
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return 12
        
        # Convert to minutes
        avg_diff_minutes = time_diffs.mean().total_seconds() / 60
        
        if avg_diff_minutes <= 0:
            return 12
        
        # Records per hour
        records_per_hour = 60 / avg_diff_minutes
        
        return records_per_hour
    
    def _add_rolling_features(self, df):
        """Add rolling statistics"""
        # Detect sample rate and adjust windows accordingly
        sample_rate = self._detect_sample_rate(df)
        
        if sample_rate >= 50:  # ~1 minute sampling (60/hour)
            windows = [60, 180, 360, 720]  # 1h, 3h, 6h, 12h at 1-min intervals
            window_hours = [1, 3, 6, 12]
            change_periods = [60, 180]  # 1h, 3h
        else:  # ~5 minute sampling (12/hour) - legacy data
            windows = [12, 36, 72, 144]  # 1h, 3h, 6h, 12h at 5-min intervals
            window_hours = [1, 3, 6, 12]
            change_periods = [12, 36]  # 1h, 3h
        
        for window, hours in zip(windows, window_hours):
            df[f'gas_rolling_mean_{hours}h'] = df['gas'].rolling(window).mean()
            df[f'gas_rolling_std_{hours}h'] = df['gas'].rolling(window).std()
            df[f'gas_rolling_min_{hours}h'] = df['gas'].rolling(window).min()
            df[f'gas_rolling_max_{hours}h'] = df['gas'].rolling(window).max()
        
        # Rate of change
        df['gas_change_1h'] = df['gas'].pct_change(change_periods[0])
        df['gas_change_3h'] = df['gas'].pct_change(change_periods[1])
        
        return df
    
    def _add_target_variables(self, df):
        """Add target variables (future gas prices)"""
        # Detect sample rate and adjust target shifts accordingly
        sample_rate = self._detect_sample_rate(df)
        
        if sample_rate >= 50:  # ~1 minute sampling (60/hour)
            df['target_1h'] = df['gas'].shift(-60)   # 1 hour ahead (60 records)
            df['target_4h'] = df['gas'].shift(-240)  # 4 hours ahead (240 records)
            df['target_24h'] = df['gas'].shift(-1440) # 24 hours ahead (1440 records)
        else:  # ~5 minute sampling (12/hour) - legacy data
            df['target_1h'] = df['gas'].shift(-12)   # 1 hour ahead (12 records)
            df['target_4h'] = df['gas'].shift(-48)   # 4 hours ahead (48 records)
            df['target_24h'] = df['gas'].shift(-288) # 24 hours ahead (288 records)
        
        return df
    
    def get_feature_columns(self, df):
        """Return list of feature column names"""
        exclude = ['timestamp', 'gas', 'block_number', 'target_1h', 'target_4h', 'target_24h']
        # Also exclude duplicate columns from merge
        exclude.extend([col for col in df.columns if col.endswith('_onchain')])
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

