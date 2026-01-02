import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats


def create_advanced_features(df):
    """
    Create comprehensive feature set for gas price prediction
    
    Args:
        df: DataFrame with columns ['timestamp', 'gas_price'] or ['timestamp', 'gas']
    
    Returns:
        X (features DataFrame), y (target Series)
    """
    
    df = df.copy()
    # Handle mixed timestamp formats (some with microseconds, some without)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle different column names
    if 'gas_price' not in df.columns and 'gas' in df.columns:
        df['gas_price'] = df['gas']
    elif 'gas_price' not in df.columns and 'current_gas' in df.columns:
        df['gas_price'] = df['current_gas']
    
    # ===================================================================
    # 1. TIME-BASED FEATURES (Cyclical patterns)
    # ===================================================================
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
    
    # Cyclical encoding (IMPORTANT: Preserves circular nature of time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ===================================================================
    # 2. LAG FEATURES (Past values matter!)
    # ===================================================================
    
    # Short-term lags (last few observations)
    for lag in [1, 2, 3, 6, 12]:  # 5min, 10min, 15min, 30min, 1hr
        df[f'lag_{lag}'] = df['gas_price'].shift(lag)
    
    # Medium-term lags (hours)
    for lag in [24, 48, 72]:  # 2hr, 4hr, 6hr (at 5min intervals)
        df[f'lag_{lag}'] = df['gas_price'].shift(lag)
    
    # Long-term lags (days)
    for lag in [288, 576, 2016]:  # 1 day, 2 days, 1 week
        df[f'lag_{lag}'] = df['gas_price'].shift(lag)
    
    # ===================================================================
    # 3. ROLLING STATISTICS (Trends and volatility)
    # ===================================================================
    
    # Moving averages (different windows)
    for window in [12, 24, 48, 72, 144, 288]:  # 1hr to 1 day
        df[f'ma_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).mean()
        df[f'std_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).std()
        df[f'min_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).min()
        df[f'max_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).max()
    
    # Exponential moving averages (more weight on recent data)
    for span in [12, 24, 72, 288]:
        df[f'ema_{span}'] = df['gas_price'].ewm(span=span, adjust=False).mean()
    
    # ===================================================================
    # 4. TREND FEATURES (Direction and momentum)
    # ===================================================================
    
    # Price differences (rate of change)
    for period in [1, 6, 12, 24, 72, 288]:
        df[f'diff_{period}'] = df['gas_price'].diff(period)
        df[f'pct_change_{period}'] = df['gas_price'].pct_change(period)
    
    # Trend strength (how consistently price is moving in one direction)
    for window in [12, 24, 72]:
        def calc_trend(x):
            if len(x) < 2:
                return 0
            try:
                return np.polyfit(np.arange(len(x)), x, 1)[0]
            except:
                return 0
        
        df[f'trend_strength_{window}'] = df['gas_price'].rolling(window).apply(calc_trend, raw=False)
    
    # ===================================================================
    # 5. VOLATILITY FEATURES (Price stability/instability)
    # ===================================================================
    
    # Coefficient of variation (volatility relative to mean)
    for window in [12, 24, 72]:
        mean = df['gas_price'].rolling(window, min_periods=1).mean()
        std = df['gas_price'].rolling(window, min_periods=1).std()
        df[f'cv_{window}'] = std / mean.replace(0, np.nan)
    
    # Range (max - min)
    for window in [12, 24, 72]:
        df[f'range_{window}'] = (
            df['gas_price'].rolling(window, min_periods=1).max() - 
            df['gas_price'].rolling(window, min_periods=1).min()
        )
    
    # ===================================================================
    # 6. MOMENTUM INDICATORS (Trading-inspired features)
    # ===================================================================
    
    # RSI (Relative Strength Index) - measures overbought/oversold
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    df['rsi_12'] = calculate_rsi(df['gas_price'], 12)
    df['rsi_24'] = calculate_rsi(df['gas_price'], 24)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['gas_price'].ewm(span=12, adjust=False).mean()
    ema_26 = df['gas_price'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    for window in [24, 72]:
        ma = df['gas_price'].rolling(window, min_periods=1).mean()
        std = df['gas_price'].rolling(window, min_periods=1).std()
        df[f'bb_upper_{window}'] = ma + (2 * std)
        df[f'bb_lower_{window}'] = ma - (2 * std)
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / ma.replace(0, np.nan)
        df[f'bb_position_{window}'] = (df['gas_price'] - df[f'bb_lower_{window}']) / (
            (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']).replace(0, np.nan)
        )
    
    # ===================================================================
    # 7. INTERACTION FEATURES (Combinations that capture complex patterns)
    # ===================================================================
    
    # Time of day × Day of week interactions
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 18)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Price × Time interactions
    df['price_x_hour'] = df['gas_price'] * df['hour']
    df['price_x_weekend'] = df['gas_price'] * df['is_weekend']
    
    # ===================================================================
    # 8. STATISTICAL FEATURES (Distribution properties)
    # ===================================================================
    
    # Skewness and kurtosis (shape of distribution)
    for window in [24, 72]:
        df[f'skew_{window}'] = df['gas_price'].rolling(window, min_periods=1).skew()
        df[f'kurt_{window}'] = df['gas_price'].rolling(window, min_periods=1).kurt()
    
    # Percentiles
    for window in [24, 72]:
        df[f'q25_{window}'] = df['gas_price'].rolling(window, min_periods=1).quantile(0.25)
        df[f'q50_{window}'] = df['gas_price'].rolling(window, min_periods=1).quantile(0.50)
        df[f'q75_{window}'] = df['gas_price'].rolling(window, min_periods=1).quantile(0.75)
    
    # Distance from moving average (how far from "normal")
    for window in [24, 72]:
        ma = df['gas_price'].rolling(window, min_periods=1).mean()
        df[f'dist_from_ma_{window}'] = df['gas_price'] - ma
        df[f'dist_from_ma_pct_{window}'] = (df['gas_price'] - ma) / ma.replace(0, np.nan)
    
    # ===================================================================
    # 9. AUTOCORRELATION FEATURES (How much current price predicts future)
    # ===================================================================
    
    # Autocorrelation at different lags
    for lag in [1, 6, 12, 24]:
        def calc_autocorr(x, lag_val):
            if len(x) < lag_val + 1:
                return 0
            try:
                return pd.Series(x).autocorr(lag=lag_val) if not pd.isna(pd.Series(x).autocorr(lag=lag_val)) else 0
            except:
                return 0
        
        df[f'autocorr_{lag}'] = df['gas_price'].rolling(window=50, min_periods=lag+1).apply(
            lambda x: calc_autocorr(x, lag), raw=False
        )
    
    # ===================================================================
    # CLEAN UP
    # ===================================================================
    
    # Get feature columns (exclude timestamp and target)
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'gas_price', 'gas', 'current_gas']]
    
    # Fill NaN values with 0 (for features that couldn't be calculated)
    # But preserve NaN in gas_price for target variable alignment
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Return features and target separately
    X = df[feature_columns]
    
    # Return gas_price as target if it exists and has values
    if 'gas_price' in df.columns:
        y = df['gas_price']
        return X, y
    else:
        return X, None


def prepare_training_data(historical_data):
    """
    Prepare data with advanced features for model training
    
    Args:
        historical_data: List of dicts with 'timestamp' and 'gas_price' or 'current_gas'
    
    Returns:
        X (features DataFrame), y (target Series)
    """
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Ensure we have the right column names
    if 'gas_price' not in df.columns:
        if 'current_gas' in df.columns:
            df['gas_price'] = df['current_gas']
        elif 'gwei' in df.columns:
            df['gas_price'] = df['gwei']
        else:
            raise ValueError("No gas price column found in data")
    
    # Create features
    X, y = create_advanced_features(df)
    
    print(f"Created {X.shape[1]} features from {len(df)} data points")
    print(f"Final dataset: {X.shape[0]} samples")
    
    return X, y

