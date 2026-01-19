"""
Feature Engineering for Gas Price Prediction.

This module provides feature engineering capabilities for the gas price prediction
ML pipeline. It transforms raw gas price data into a rich feature set that captures:

- Time-based patterns (hour of day, day of week, cyclical encodings)
- Historical price patterns (lag features at 1h, 3h, 6h, 12h, 24h)
- Rolling statistics (mean, std, min, max over various windows)
- On-chain network features (pending transactions, gas utilization, congestion)
- Rate of change indicators

The engineered features are used to train models that predict gas prices
at 1-hour, 4-hour, and 24-hour horizons.

Example:
    >>> engineer = GasFeatureEngineer()
    >>> df = engineer.prepare_training_data(hours_back=720, chain_id=8453)
    >>> feature_cols = engineer.get_feature_columns(df)
    >>> X = df[feature_cols]
    >>> y = df[['target_1h', 'target_4h', 'target_24h']]
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.database import DatabaseManager


class GasFeatureEngineer:
    """
    Transforms raw gas price data into ML-ready features.

    This class handles the complete feature engineering pipeline:
    1. Fetches historical gas price data from the database
    2. Joins on-chain features (network congestion, transaction counts)
    3. Engineers time-based, lag, and rolling statistical features
    4. Creates target variables for different prediction horizons

    The feature set is designed to capture both short-term volatility
    and longer-term patterns in gas prices, supporting accurate predictions
    across multiple time horizons.

    Attributes:
        db (DatabaseManager): Database connection for fetching historical data.

    Features Generated:
        Time Features:
            - hour, day_of_week, day_of_month, is_weekend
            - hour_sin, hour_cos (cyclical encoding)
            - dow_sin, dow_cos (cyclical encoding)

        Lag Features:
            - gas_lag_1h, gas_lag_3h, gas_lag_6h, gas_lag_12h, gas_lag_24h

        Rolling Features:
            - gas_rolling_mean_{1,3,6,12}h
            - gas_rolling_std_{1,3,6,12}h
            - gas_rolling_min_{1,3,6,12}h
            - gas_rolling_max_{1,3,6,12}h
            - gas_change_1h, gas_change_3h

        On-Chain Features (if available):
            - pending_tx_count, unique_addresses, tx_per_second
            - gas_utilization_ratio, contract_call_ratio
            - avg_tx_gas, large_tx_ratio
            - congestion_level, is_highly_congested
    """
    def __init__(self):
        self.db = DatabaseManager()
    
    def prepare_training_data(self, hours_back=720, chain_id=8453):
        """
        Fetch historical data and engineer features for a specific chain.
        
        Args:
            hours_back: Hours of historical data to fetch
            chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
        
        Returns: X (features), y (targets for 1h, 4h, 24h)
        """
        # Get raw data from database for specific chain
        raw_data = self.db.get_historical_data(hours=hours_back, chain_id=chain_id)
        
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
        df = self._add_momentum_features(df)
        df = self._add_spike_features(df)
        df = self._add_interaction_features(df)
        df = self._add_target_variables(df)
        
        # Remove NaN rows (from lag/rolling operations)
        # But be lenient with enhanced features - they can be zero if not available
        # Only drop rows where core features are missing
        core_features = ['gas', 'base_fee', 'priority_fee']
        df = df.dropna(subset=core_features)
        
        # Fill any remaining NaN in enhanced features and contract_call_ratio with 0
        enhanced_cols = [
            'pending_tx_count', 'unique_addresses', 'tx_per_second',
            'gas_utilization_ratio', 'avg_tx_gas', 'large_tx_ratio',
            'congestion_level', 'is_highly_congested', 'contract_call_ratio'
        ]
        for col in enhanced_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill NaN in lag/rolling features with forward fill then zero to avoid future leakage
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'change' in col]
        for col in lag_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
        
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
                        'contract_call_ratio': f.contract_call_ratio if hasattr(f, 'contract_call_ratio') else None,
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
                        direction='backward',
                        tolerance=pd.Timedelta(minutes=5),
                        suffixes=('', '_onchain')
                    )
                    
                    # Fill missing values with forward fill, then zero to avoid future leakage
                    # This allows training even when enhanced features are sparse
                    enhanced_cols = [
                        'pending_tx_count', 'unique_addresses', 'tx_per_second',
                        'gas_utilization_ratio', 'avg_tx_gas', 'large_tx_ratio',
                        'congestion_level', 'is_highly_congested'
                    ]
                    for col in enhanced_cols:
                        if col in df.columns:
                            # Forward fill, then zero
                            df[col] = df[col].ffill().fillna(0).infer_objects(copy=False)
                    
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
        """
        Extract time-based features from timestamp.

        Creates features that capture temporal patterns in gas prices:
        - Direct time components (hour, day of week, etc.)
        - Cyclical encodings using sin/cos transformations

        Cyclical encoding is used because time features are inherently cyclic
        (e.g., hour 23 is close to hour 0). Sin/cos encoding preserves this
        relationship, improving model performance.

        Args:
            df (pd.DataFrame): DataFrame with 'timestamp' column.

        Returns:
            pd.DataFrame: DataFrame with added time features:
                - hour: Hour of day (0-23)
                - day_of_week: Day of week (0=Monday, 6=Sunday)
                - day_of_month: Day of month (1-31)
                - is_weekend: Binary flag for Saturday/Sunday
                - hour_sin, hour_cos: Cyclical hour encoding
                - dow_sin, dow_cos: Cyclical day-of-week encoding
        """
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
        """
        Add lagged gas price features (historical values).

        Lag features capture the autoregressive nature of gas prices -
        past prices are strong predictors of future prices. This method
        creates features representing gas prices at various points in the past.

        The lag windows are automatically adjusted based on the detected
        sampling rate of the data (1-minute vs 5-minute intervals).

        Args:
            df (pd.DataFrame): DataFrame with 'gas' and 'timestamp' columns.

        Returns:
            pd.DataFrame: DataFrame with added lag features:
                - gas_lag_1h: Gas price 1 hour ago
                - gas_lag_3h: Gas price 3 hours ago
                - gas_lag_6h: Gas price 6 hours ago
                - gas_lag_12h: Gas price 12 hours ago
                - gas_lag_24h: Gas price 24 hours ago

        Note:
            Lag features will have NaN values for the first N rows where
            historical data isn't available. These are handled in the
            prepare_training_data method.
        """
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
        """
        Add rolling window statistical features with enhanced volatility metrics.

        Rolling features capture the recent behavior and volatility of gas prices.
        Enhanced with volatility-specific features for better pattern recognition.

        Args:
            df (pd.DataFrame): DataFrame with 'gas' column.

        Returns:
            pd.DataFrame: DataFrame with added rolling features including volatility metrics.
        """
        # Detect sample rate and adjust windows accordingly
        sample_rate = self._detect_sample_rate(df)
        
        if sample_rate >= 50:  # ~1 minute sampling (60/hour)
            windows = [60, 180, 360, 720, 1440]  # 1h, 3h, 6h, 12h, 24h at 1-min intervals
            window_hours = [1, 3, 6, 12, 24]
            change_periods = [60, 180]  # 1h, 3h
        else:  # ~5 minute sampling (12/hour) - legacy data
            windows = [12, 36, 72, 144, 288]  # 1h, 3h, 6h, 12h, 24h at 5-min intervals
            window_hours = [1, 3, 6, 12, 24]
            change_periods = [12, 36]  # 1h, 3h
        
        for window, hours in zip(windows, window_hours):
            # Basic rolling statistics
            rolling_mean = df['gas'].rolling(window).mean()
            rolling_std = df['gas'].rolling(window).std()
            
            df[f'gas_rolling_mean_{hours}h'] = rolling_mean
            df[f'gas_rolling_std_{hours}h'] = rolling_std
            df[f'gas_rolling_min_{hours}h'] = df['gas'].rolling(window).min()
            df[f'gas_rolling_max_{hours}h'] = df['gas'].rolling(window).max()
            
            # Enhanced volatility features
            # Coefficient of variation (volatility/mean ratio)
            df[f'gas_cv_{hours}h'] = rolling_std / (rolling_mean + 1e-8)  # Avoid division by zero
            
            # Realized volatility (historical price swings)
            # Calculate as the standard deviation of returns
            returns = df['gas'].pct_change().rolling(window).std()
            df[f'gas_realized_vol_{hours}h'] = returns
            
            # Volatility percentiles - is current volatility high/low?
            # Compare current rolling std to historical distribution
            std_percentile = rolling_std.rolling(window * 2, min_periods=window).apply(
                lambda x: (x.iloc[-1] - x.iloc[:-1].min()) / (x.iloc[:-1].max() - x.iloc[:-1].min() + 1e-8)
                if len(x) > 1 and x.iloc[:-1].max() > x.iloc[:-1].min() else 0.5
            )
            df[f'gas_vol_percentile_{hours}h'] = std_percentile
        
        # Rate of change over multiple horizons
        df['gas_change_1h'] = df['gas'].pct_change(change_periods[0])
        df['gas_change_3h'] = df['gas'].pct_change(change_periods[1])
        # Additional rate of change features
        if sample_rate >= 50:
            df['gas_change_6h'] = df['gas'].pct_change(360)
            df['gas_change_12h'] = df['gas'].pct_change(720)
            df['gas_change_24h'] = df['gas'].pct_change(1440)
        else:
            df['gas_change_6h'] = df['gas'].pct_change(72)
            df['gas_change_12h'] = df['gas'].pct_change(144)
            df['gas_change_24h'] = df['gas'].pct_change(288)
        
        return df
    
    def _add_momentum_features(self, df):
        """
        Add momentum and velocity features to capture rate of change and acceleration.
        
        Momentum features help identify trends and the strength of price movements:
        - First derivative (velocity): rate of price change
        - Second derivative (acceleration): rate of velocity change
        - Momentum strength indicators
        
        Args:
            df (pd.DataFrame): DataFrame with 'gas' column.
            
        Returns:
            pd.DataFrame: DataFrame with added momentum features.
        """
        # Detect sample rate
        sample_rate = self._detect_sample_rate(df)
        
        # Calculate first derivative (velocity) - rate of change
        if sample_rate >= 50:
            velocity_periods = [1, 60, 180, 360]  # 1min, 1h, 3h, 6h
            velocity_hours = [1, 1, 3, 6]
        else:
            velocity_periods = [1, 12, 36, 72]  # 1 period, 1h, 3h, 6h
            velocity_hours = [0, 1, 3, 6]
        
        for period, hours in zip(velocity_periods, velocity_hours):
            if hours == 0:
                df['gas_velocity_1period'] = df['gas'].diff(period)
            else:
                df[f'gas_velocity_{hours}h'] = df['gas'].diff(period)
        
        # Calculate second derivative (acceleration) - rate of velocity change
        if 'gas_velocity_1h' in df.columns:
            df['gas_acceleration_1h'] = df['gas_velocity_1h'].diff()
        if 'gas_velocity_3h' in df.columns:
            df['gas_acceleration_3h'] = df['gas_velocity_3h'].diff()
        
        # Percentage change momentum over different windows
        if sample_rate >= 50:
            momentum_windows = [60, 180, 360, 720]  # 1h, 3h, 6h, 12h
            momentum_hours = [1, 3, 6, 12]
        else:
            momentum_windows = [12, 36, 72, 144]  # 1h, 3h, 6h, 12h
            momentum_hours = [1, 3, 6, 12]
        
        for window, hours in zip(momentum_windows, momentum_hours):
            pct_change = df['gas'].pct_change(window)
            df[f'momentum_{hours}h'] = pct_change
            
            # Momentum strength: absolute value indicates strong vs weak trends
            df[f'momentum_strength_{hours}h'] = pct_change.abs()
            
            # Momentum direction: +1 for up, -1 for down, 0 for flat
            df[f'momentum_direction_{hours}h'] = np.sign(pct_change)
        
        # Momentum consistency: how many recent periods moved in same direction?
        if 'momentum_direction_1h' in df.columns:
            # Count consecutive same-direction moves
            momentum_consistency = []
            window_size = 6
            for i in range(len(df)):
                if i < window_size:
                    momentum_consistency.append(0)
                else:
                    recent_dirs = df['momentum_direction_1h'].iloc[i-window_size:i]
                    if len(recent_dirs) > 0:
                        # Count how many match the most recent direction
                        last_dir = recent_dirs.iloc[-1]
                        consistency = (recent_dirs == last_dir).sum() / len(recent_dirs)
                        momentum_consistency.append(consistency)
                    else:
                        momentum_consistency.append(0)
            df['momentum_consistency'] = momentum_consistency
        
        return df
    
    def _add_spike_features(self, df):
        """
        Add spike detection features to identify unusual price movements.
        
        Spike features help identify anomalies and unusual market conditions:
        - Spike indicators (price > quantile threshold)
        - Spike magnitude (how extreme is the spike)
        - Time since last spike
        - Spike frequency
        
        Args:
            df (pd.DataFrame): DataFrame with 'gas' column.
            
        Returns:
            pd.DataFrame: DataFrame with added spike features.
        """
        # Detect sample rate
        sample_rate = self._detect_sample_rate(df)
        
        if sample_rate >= 50:
            spike_windows = [60, 180, 360, 720, 1440]  # 1h, 3h, 6h, 12h, 24h
            spike_hours = [1, 3, 6, 12, 24]
        else:
            spike_windows = [12, 36, 72, 144, 288]  # 1h, 3h, 6h, 12h, 24h
            spike_hours = [1, 3, 6, 12, 24]
        
        for window, hours in zip(spike_windows, spike_hours):
            rolling_median = df['gas'].rolling(window).median()
            rolling_q95 = df['gas'].rolling(window).quantile(0.95)
            rolling_q90 = df['gas'].rolling(window).quantile(0.90)
            rolling_q75 = df['gas'].rolling(window).quantile(0.75)
            
            # Spike indicator: is current price above 95th percentile?
            df[f'is_spike_{hours}h'] = (df['gas'] > rolling_q95).astype(int)
            
            # Spike indicator at different thresholds
            df[f'is_spike_90pct_{hours}h'] = (df['gas'] > rolling_q90).astype(int)
            df[f'is_spike_75pct_{hours}h'] = (df['gas'] > rolling_q75).astype(int)
            
            # Spike magnitude: how many times median/mean is the current price?
            df[f'spike_magnitude_{hours}h'] = df['gas'] / (rolling_median + 1e-8)
            
            # Spike magnitude relative to mean
            rolling_mean = df['gas'].rolling(window).mean()
            df[f'spike_magnitude_mean_{hours}h'] = df['gas'] / (rolling_mean + 1e-8)
            
            # Distance from median (standardized)
            rolling_std = df['gas'].rolling(window).std()
            df[f'spike_zscore_{hours}h'] = (df['gas'] - rolling_median) / (rolling_std + 1e-8)
        
        # Time since last spike (countdown feature)
        # Count periods since last spike across different windows
        if 'is_spike_24h' in df.columns:
            time_since_spike = []
            counter = 0
            for spike in df['is_spike_24h'].fillna(0):
                if spike > 0:
                    counter = 0
                else:
                    counter += 1
                time_since_spike.append(counter)
            df['time_since_spike'] = time_since_spike
            
            # Normalize by max value to get 0-1 range
            if len(time_since_spike) > 0:
                max_time = max(time_since_spike) if max(time_since_spike) > 0 else 1
                df['time_since_spike_norm'] = [t / max_time for t in time_since_spike]
        
        # Spike frequency: how often do spikes occur in a rolling window?
        if 'is_spike_24h' in df.columns:
            if sample_rate >= 50:
                freq_window = 1440  # 24 hours
            else:
                freq_window = 288  # 24 hours
            
            df['spike_frequency'] = df['is_spike_24h'].rolling(freq_window).sum() / freq_window
        
        # Spike clustering: are we in a period with multiple spikes?
        if 'is_spike_24h' in df.columns:
            if sample_rate >= 50:
                cluster_window = 360  # 6 hours
            else:
                cluster_window = 72  # 6 hours
            
            df['spike_cluster_indicator'] = df['is_spike_24h'].rolling(cluster_window).sum()
            df['is_in_spike_period'] = (df['spike_cluster_indicator'] >= 2).astype(int)
        
        return df
    
    def _add_interaction_features(self, df):
        """
        Add interaction features that combine existing features for non-linear relationships.
        
        Interaction features capture how different factors interact:
        - volatility * momentum
        - spike_magnitude * congestion_level
        - lag_1h * lag_24h
        - hour * congestion_level
        
        Args:
            df (pd.DataFrame): DataFrame with feature columns.
            
        Returns:
            pd.DataFrame: DataFrame with added interaction features.
        """
        # Volatility * Momentum interactions
        if 'gas_rolling_std_1h' in df.columns and 'momentum_1h' in df.columns:
            df['volatility_momentum_1h'] = df['gas_rolling_std_1h'] * df['momentum_1h'].abs()
        
        if 'gas_rolling_std_3h' in df.columns and 'momentum_3h' in df.columns:
            df['volatility_momentum_3h'] = df['gas_rolling_std_3h'] * df['momentum_3h'].abs()
        
        # Spike magnitude * Congestion level
        if 'spike_magnitude_24h' in df.columns and 'congestion_level' in df.columns:
            df['spike_congestion'] = df['spike_magnitude_24h'] * df['congestion_level']
        
        # Spike magnitude * is_highly_congested
        if 'spike_magnitude_24h' in df.columns and 'is_highly_congested' in df.columns:
            df['spike_during_congestion'] = df['spike_magnitude_24h'] * df['is_highly_congested']
        
        # Lag interactions - short vs long term relationship
        if 'gas_lag_1h' in df.columns and 'gas_lag_24h' in df.columns:
            df['lag_1h_24h_ratio'] = df['gas_lag_1h'] / (df['gas_lag_24h'] + 1e-8)
            df['lag_1h_24h_diff'] = df['gas_lag_1h'] - df['gas_lag_24h']
        
        # Time-of-day * congestion patterns
        if 'hour' in df.columns and 'congestion_level' in df.columns:
            df['hour_congestion'] = df['hour'] * df['congestion_level']
        
        if 'hour_sin' in df.columns and 'congestion_level' in df.columns:
            df['hour_sin_congestion'] = df['hour_sin'] * df['congestion_level']
        
        # Volatility * Congestion
        if 'gas_cv_6h' in df.columns and 'congestion_level' in df.columns:
            df['volatility_congestion'] = df['gas_cv_6h'] * df['congestion_level']
        
        # Momentum * Spike
        if 'momentum_1h' in df.columns and 'is_spike_24h' in df.columns:
            df['momentum_spike'] = df['momentum_1h'] * df['is_spike_24h']
        
        # Rolling mean * Lag ratio (trend consistency)
        if 'gas_rolling_mean_6h' in df.columns and 'gas_lag_6h' in df.columns:
            df['trend_consistency'] = df['gas_rolling_mean_6h'] / (df['gas_lag_6h'] + 1e-8)
        
        # Volatility percentile * Momentum direction
        if 'gas_vol_percentile_6h' in df.columns and 'momentum_direction_1h' in df.columns:
            df['volatility_percentile_momentum'] = df['gas_vol_percentile_6h'] * df['momentum_direction_1h']
        
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
            from models.feature_pipeline import build_feature_matrix

            features, _, _ = build_feature_matrix(recent_data, include_external_features=True)
            return features.iloc[[-1]]
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
