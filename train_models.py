#!/usr/bin/env python3
"""
Gweizy Model Training Script

Standalone version of train_models_colab.ipynb
Trains prediction models and spike detectors for gas price prediction.
Refactored for Hybrid Strategy (4h Regressor -> 1h Classifier).

Usage:
    python3 train_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import time
import warnings
import json
warnings.filterwarnings('ignore')

# Database path configuration - prefer root gas_data.db (fresher data from worker)
DB_PATH = 'gas_data.db'
if not os.path.exists(DB_PATH):
    DB_PATH = 'backend/gas_data.db'
if not os.path.exists(DB_PATH):
    print(f"❌ Error: Database file not found. Please provide gas_data.db")
    print(f"   Expected locations: ./gas_data.db or backend/gas_data.db")
    sys.exit(1)
print(f"✅ Using database: {DB_PATH}")


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
def log(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {msg}")

# Connect to database
engine = create_engine(f'sqlite:///{DB_PATH}')

# Load data (note: database uses 'current_gas' column, we'll rename to 'gas_price')
query = """
SELECT timestamp, current_gas, block_number, base_fee, priority_fee,
       gas_used, gas_limit, utilization
FROM gas_prices
ORDER BY timestamp DESC
"""

df = pd.read_sql(query, engine)
# Rename current_gas to gas_price for consistency with feature engineering code
df.rename(columns={'current_gas': 'gas_price'}, inplace=True)

log(f"📊 Loaded {len(df):,} raw records (≈2s cadence)")
log(f"📅 Raw date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
log(f"⛽ Raw gas price range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei")

# Load mempool_stats data and join with gas_prices
log("\n📡 Loading mempool_stats data...")
try:
    mempool_query = """
    SELECT timestamp, tx_count
    FROM mempool_stats
    ORDER BY timestamp ASC
    """
    df_mempool = pd.read_sql(mempool_query, engine)
    
    if len(df_mempool) > 0:
        df_mempool['timestamp'] = pd.to_datetime(df_mempool['timestamp'])
        log(f"   ✅ Loaded {len(df_mempool):,} mempool_stats records")
        log(f"   📅 Mempool date range: {df_mempool['timestamp'].min()} to {df_mempool['timestamp'].max()}")
        
        # Convert gas_prices timestamp to datetime for joining
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Left join mempool_stats with gas_prices on timestamp
        # Use merge_asof for nearest timestamp matching (since timestamps may not align exactly)
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_mempool.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=5),  # Match within 5 seconds
            suffixes=('', '_mempool')
        )
        
        log(f"   ✅ Joined mempool data: {df['tx_count'].notna().sum():,} rows have mempool data")
        log(f"   📊 Mempool coverage: {df['tx_count'].notna().sum() / len(df) * 100:.1f}%")
        
        # Fill missing mempool data with 0 (for rows without mempool data)
        df['tx_count'] = df['tx_count'].fillna(0)
    else:
        log(f"   ⚠️  No mempool_stats data found - mempool features will be zero")
        df['tx_count'] = 0
except Exception as e:
    log(f"   ⚠️  Error loading mempool_stats: {e} - mempool features will be zero")
    df['tx_count'] = 0

# -------------------------------------------
# Load Mempool Stats (Leading Indicators)
# -------------------------------------------
log("\n📡 Loading mempool statistics (leading indicators)...")
try:
    mempool_query = """
    SELECT timestamp, tx_count
    FROM mempool_stats
    ORDER BY timestamp ASC
    """
    df_mempool = pd.read_sql(mempool_query, engine)
    
    if len(df_mempool) > 0:
        df_mempool['timestamp'] = pd.to_datetime(df_mempool['timestamp'])
        log(f"   ✅ Loaded {len(df_mempool):,} mempool records")
        log(f"   📅 Mempool date range: {df_mempool['timestamp'].min()} to {df_mempool['timestamp'].max()}")
        
        # Merge with gas_prices using nearest timestamp (forward fill)
        # Convert timestamps to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge using merge_asof for nearest timestamp matching (forward fill)
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_mempool.sort_values('timestamp'),
            on='timestamp',
            direction='forward',  # Forward fill - use next available mempool data
            suffixes=('', '_mempool')
        )
        
        log(f"   ✅ Merged mempool data: {df['tx_count'].notna().sum():,} records have mempool data")
    else:
        log(f"   ⚠️  No mempool data found (table may be empty - mempool worker may not be running)")
        df['tx_count'] = None
except Exception as e:
    log(f"   ⚠️  Failed to load mempool stats: {e}")
    log(f"   ⚠️  Continuing without mempool features (mempool_stats table may not exist yet)")
    df['tx_count'] = None

# -------------------------------------------
# "Tick Bar" Sampling: Only keep rows where market actually moved
# -------------------------------------------
log("\n⏱️  Tick Bar Sampling: Filtering for market movements (>1% change)")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
raw_len = len(df)

# Group by minute and aggregate
df['minute'] = df['timestamp'].dt.floor('1T')
agg_dict = {
    'gas_price': 'last',  # Close price (last value in the minute)
    'gas_used': 'sum',    # Total volume per minute
    'gas_limit': 'mean',
    'base_fee': 'mean',
    'priority_fee': 'mean',
    'block_number': 'max',
}
# Add optional columns if they exist
if 'utilization' in df.columns:
    agg_dict['utilization'] = 'mean'
if 'tx_count' in df.columns:
    agg_dict['tx_count'] = 'mean'
df_grouped = df.groupby('minute').agg(agg_dict).reset_index()
df_grouped.rename(columns={'minute': 'timestamp'}, inplace=True)

# CRITICAL: Drop rows where gas_price changes by less than 1% compared to previous row
# This filters out "silence" - we only want to train on actual market movements
df_grouped = df_grouped.sort_values('timestamp').reset_index(drop=True)
df_grouped['price_pct_change'] = df_grouped['gas_price'].pct_change().abs() * 100
# Keep first row (no previous to compare) and rows with >1% change
df_grouped = df_grouped[(df_grouped['price_pct_change'].isna()) | (df_grouped['price_pct_change'] >= 1.0)]
df_grouped = df_grouped.drop(columns=['price_pct_change'])

log(f"   Raw samples: {raw_len:,} → After tick bar filtering: {len(df_grouped):,} rows")
log(f"   Filtered out {raw_len - len(df_grouped):,} rows ({100*(raw_len - len(df_grouped))/raw_len:.1f}%) with <1% price change")
log(f"   Date range: {df_grouped['timestamp'].min()} to {df_grouped['timestamp'].max()}")

# Work with the filtered frame going forward
df = df_grouped

# Data quality checks
log("\n🔍 Data Quality Checks:")
# Check for missing values
missing_count = df['gas_price'].isna().sum()
if missing_count > 0:
    log(f"   ⚠️  Missing gas_price values: {missing_count} ({missing_count/len(df)*100:.1f}%)")

# Check for duplicates
duplicates = df.duplicated(subset=['timestamp']).sum()
if duplicates > 0:
    log(f"   ⚠️  Duplicate timestamps: {duplicates}")

# Check for temporal gaps
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time_diff'] = df['timestamp'].diff()

log("🔧 Starting feature engineering...")

# Sort by timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Sample if too large (use all data on Colab - we have resources!)
MAX_RECORDS = 100000  # Can handle more on Colab
if len(df) > MAX_RECORDS:
    log(f"⚠️ Sampling {MAX_RECORDS:,} from {len(df):,} records")
    recent = df.tail(MAX_RECORDS // 5)
    older = df.head(len(df) - MAX_RECORDS // 5).sample(MAX_RECORDS - len(recent), random_state=42)
    df = pd.concat([older, recent]).sort_values('timestamp').reset_index(drop=True)
    log(f"✅ Using {len(df):,} records")

# Enhanced outlier handling using Winsorization (cap extreme values instead of removing)
Q1, Q3 = df['gas_price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
outliers_before = ((df['gas_price'] < lower) | (df['gas_price'] > upper)).sum()

if outliers_before > 0:
    log(f"   🔧 Applying Winsorization (capping {outliers_before:,} outliers)")
    log(f"      Lower bound: {lower:.6f} gwei, Upper bound: {upper:.6f} gwei")
    # Cap outliers but keep them (Winsorization)
    df['gas_price'] = df['gas_price'].clip(lower, upper)
    
    # Verify
    outliers_after = ((df['gas_price'] < lower) | (df['gas_price'] > upper)).sum()
    if outliers_after == 0:
        log(f"      ✅ All outliers successfully capped")
else:
    log(f"   ✅ No outliers detected using 3*IQR method")

# Time features (only hour - removed day_of_week since we only have ~14h of data)
log("   Adding time features...")
df['hour'] = df['timestamp'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
# REMOVED: day_of_week, is_weekend, day_sin, day_cos (misleading with only 14h of data)

# Mempool features (leading indicators)
log("   Adding mempool features (leading indicators)...")
if 'tx_count' in df.columns and df['tx_count'].notna().any():
    # mempool_velocity: 1st derivative (change in tx_count per time period)
    df['mempool_velocity'] = df['tx_count'].diff().fillna(0)

    # mempool_acceleration: 2nd derivative (change of the change)
    df['mempool_acceleration'] = df['mempool_velocity'].diff().fillna(0)

    # Additional mempool features for robustness
    # Rolling statistics of mempool activity
    for window in [6, 12, 24]:
        df[f'mempool_ma_{window}'] = df['tx_count'].rolling(window=window, min_periods=1).mean()
        df[f'mempool_std_{window}'] = df['tx_count'].rolling(window=window, min_periods=1).std()
        df[f'mempool_max_{window}'] = df['tx_count'].rolling(window=window, min_periods=1).max()

    # Mempool momentum (rate of change over different windows)
    for window in [6, 12, 24]:
        df[f'mempool_momentum_{window}'] = df['tx_count'] - df['tx_count'].shift(window).fillna(df['tx_count'])

    log(f"   ✅ Added mempool features: velocity, acceleration, and rolling statistics")
else:
    log(f"   ⚠️  No tx_count data available - creating zero mempool features")
    df['tx_count'] = 0
    df['mempool_velocity'] = 0
    df['mempool_acceleration'] = 0
    for window in [6, 12, 24]:
        df[f'mempool_ma_{window}'] = 0
        df[f'mempool_std_{window}'] = 0
        df[f'mempool_max_{window}'] = 0
        df[f'mempool_momentum_{window}'] = 0

# EIP-1559 pressure features (utilization-aware) + Physics features
log("   Adding EIP-1559 pressure features with physics derivatives...")
# Utilization is stored as percent; convert to fraction
# Handle missing utilization column (for older data)
if 'utilization' in df.columns:
    df['utilization_frac'] = df['utilization'] / 100.0
    df['pressure_index'] = df['utilization_frac'] - 0.50  # Positive => upward pressure
    
    # NEW: Physics features - derivatives of utilization (pressure)
    # pressure_velocity: 1st derivative (change per minute)
    df['pressure_velocity'] = df['pressure_index'].diff().fillna(0)
    
    # pressure_acceleration: 2nd derivative (change of the change)
    df['pressure_acceleration'] = df['pressure_velocity'].diff().fillna(0)
    
    for window in [10, 30, 60]:
        df[f'pressure_cum_{window}m'] = df['pressure_index'].rolling(window=window, min_periods=1).sum()
else:
    log("   ⚠️  Utilization column not found - using base_fee as proxy")
    # Use base_fee changes as proxy for utilization
    if 'base_fee' in df.columns:
        df['base_fee_pct_change'] = df['base_fee'].pct_change().fillna(0)
        df['pressure_index'] = df['base_fee_pct_change']  # Positive change = upward pressure
        
        # NEW: Physics features from base_fee proxy
        df['pressure_velocity'] = df['pressure_index'].diff().fillna(0)
        df['pressure_acceleration'] = df['pressure_velocity'].diff().fillna(0)
        
        for window in [10, 30, 60]:
            df[f'pressure_cum_{window}m'] = df['pressure_index'].rolling(window=window, min_periods=1).sum()
    else:
        log("   ⚠️  No utilization or base_fee available - skipping pressure features")
        df['pressure_index'] = 0
        df['pressure_velocity'] = 0
        df['pressure_acceleration'] = 0
        for window in [10, 30, 60]:
            df[f'pressure_cum_{window}m'] = 0

# Lag features
log("   Adding lag features...")
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'gas_lag_{lag}'] = df['gas_price'].shift(lag)

# Rolling statistics
log("   Adding rolling statistics...")
for window in [6, 12, 24, 48]:
    df[f'gas_ma_{window}'] = df['gas_price'].rolling(window).mean()
    df[f'gas_std_{window}'] = df['gas_price'].rolling(window).std()
    df[f'gas_min_{window}'] = df['gas_price'].rolling(window).min()
    df[f'gas_max_{window}'] = df['gas_price'].rolling(window).max()

# Price change features
log("   Adding price change features...")
df['gas_pct_change_1'] = df['gas_price'].pct_change(1) * 100
df['gas_pct_change_6'] = df['gas_price'].pct_change(6) * 100
df['gas_pct_change_12'] = df['gas_price'].pct_change(12) * 100
df['gas_pct_change_24'] = df['gas_price'].pct_change(24) * 100

# Volatility
df['volatility_6h'] = df['gas_price'].rolling(6).std() / df['gas_price'].rolling(6).mean()
df['volatility_24h'] = df['gas_price'].rolling(24).std() / df['gas_price'].rolling(24).mean()

# Momentum
df['momentum_6'] = df['gas_price'] - df['gas_price'].shift(6)
df['momentum_12'] = df['gas_price'] - df['gas_price'].shift(12)
df['momentum_24'] = df['gas_price'] - df['gas_price'].shift(24)

# EMA
df['ema_6'] = df['gas_price'].ewm(span=6).mean()
df['ema_12'] = df['gas_price'].ewm(span=12).mean()
df['ema_24'] = df['gas_price'].ewm(span=24).mean()

# Enhanced: Volatility-normalized features (robust to extreme values)
for window in [6, 12, 24]:
    rolling_std = df['gas_price'].rolling(window=window, min_periods=1).std()
    rolling_mean = df['gas_price'].rolling(window=window, min_periods=1).mean()
    # Price normalized by volatility (Z-score relative to rolling window)
    df[f'price_zscore_{window}'] = (df['gas_price'] - rolling_mean) / (rolling_std + 1e-8)
    # Price normalized by mean (percentage of mean)
    df[f'price_rel_mean_{window}'] = df['gas_price'] / (rolling_mean + 1e-8)
    # Distance from mean in terms of std
    df[f'distance_std_{window}'] = (df['gas_price'] - rolling_mean).abs() / (rolling_std + 1e-8)

# Enhanced: Regime indicators (low/medium/high volatility periods)
for window in [12, 24]:
    rolling_std = df['gas_price'].rolling(window=window, min_periods=1).std()
    median_std = rolling_std.rolling(window*2, min_periods=1).median()
    df[f'regime_low_vol_{window}'] = (rolling_std < median_std * 0.5).astype(int)
    df[f'regime_high_vol_{window}'] = (rolling_std > median_std * 1.5).astype(int)
    df[f'regime_extreme_vol_{window}'] = (rolling_std > rolling_std.rolling(window*2, min_periods=1).quantile(0.9)).astype(int)

# Enhanced: Momentum features relative to volatility
for window in [6, 12, 24]:
    rolling_std = df['gas_price'].rolling(window=window, min_periods=1).std()
    momentum = df['gas_price'] - df['gas_price'].shift(window)
    df[f'momentum_vol_adj_{window}'] = momentum / (rolling_std + 1e-8)  # Momentum in std units

# Enhanced: Spike detection features
spike_threshold = df['gas_price'].quantile(0.9)
for window in [6, 12, 24]:
    max_price = df['gas_price'].rolling(window=window, min_periods=1).max()
    df[f'is_spike_{window}'] = (df['gas_price'] > spike_threshold).astype(int)
    df[f'near_spike_{window}'] = (max_price > spike_threshold).astype(int)

# Enhanced: Time-to-peak features (time since last spike)
for window in [6, 12, 24]:
    spike_mask = df['gas_price'] > spike_threshold
    # Calculate time since last spike (in periods)
    time_since_spike = np.zeros(len(df))
    last_spike_idx = -np.inf
    for i in range(len(df)):
        if spike_mask.iloc[i]:
            last_spike_idx = i
        if i - last_spike_idx < 1000:  # Cap at 1000 periods
            time_since_spike[i] = i - last_spike_idx
        else:
            time_since_spike[i] = 1000  # Cap
    df[f'time_since_spike_{window}'] = time_since_spike

# Enhanced: Volatility regime change detection
for window in [12, 24]:
    rolling_std = df['gas_price'].rolling(window=window, min_periods=1).std()
    # Detect regime changes (large changes in volatility)
    vol_change = rolling_std.pct_change().abs()
    df[f'vol_regime_change_{window}'] = (vol_change > vol_change.rolling(window*2, min_periods=1).quantile(0.75)).astype(int)
    # Volatility trend (increasing/decreasing)
    vol_trend = rolling_std.rolling(6, min_periods=1).mean() - rolling_std.rolling(12, min_periods=1).mean()
    df[f'vol_trend_{window}'] = np.sign(vol_trend)  # -1, 0, or 1

# Enhanced: Mean reversion features
for window in [12, 24, 48]:
    rolling_mean = df['gas_price'].rolling(window=window, min_periods=1).mean()
    # Distance from mean (normalized)
    df[f'distance_from_mean_{window}'] = (df['gas_price'] - rolling_mean) / (rolling_mean + 1e-8)
    # Mean reversion indicator (price above mean tends to revert down)
    df[f'mean_reversion_{window}'] = -np.sign(df[f'distance_from_mean_{window}']) * df[f'distance_from_mean_{window}'].abs()
    
    # Price relative to recent mean (short vs long term)
    short_mean = df['gas_price'].rolling(window//2, min_periods=1).mean()
    long_mean = rolling_mean
    df[f'short_long_ratio_{window}'] = short_mean / (long_mean + 1e-8)

# NEW: Supply/Demand Proxy Feature (Critical for stationarity)
# Base Fee increases when blocks are >50% full, providing a utilization proxy
# We can infer demand pressure from base_fee changes
if 'base_fee' in df.columns:
    log("   Adding supply/demand proxy features from base_fee...")
    # Base fee change rate (proxy for network utilization)
    df['base_fee_change'] = df['base_fee'].pct_change().fillna(0)
    df['base_fee_momentum'] = df['base_fee'].diff().fillna(0)
    
    # Rolling base fee changes (persistent demand)
    for window in [6, 12, 24]:
        df[f'base_fee_change_{window}'] = df['base_fee'].pct_change(window).fillna(0)
        df[f'base_fee_ma_{window}'] = df['base_fee'].rolling(window=window, min_periods=1).mean()
    
    # Pseudo-utilization: base_fee relative to its recent mean (high = high demand)
    base_fee_ma_24h = df['base_fee'].rolling(window=24, min_periods=1).mean()
    df['pseudo_utilization'] = df['base_fee'] / (base_fee_ma_24h + 1e-8)
    
    # Base fee acceleration (second derivative) - captures demand changes
    df['base_fee_acceleration'] = df['base_fee'].diff().diff().fillna(0)

# NEW: Ratio Features (More stationary than absolutes)
# Focus on ratios which are more invariant to price levels
log("   Adding ratio features (more stationary)...")
if 'base_fee' in df.columns and 'priority_fee' in df.columns:
    # Priority fee to base fee ratio (competition indicator)
    df['priority_base_ratio'] = df['priority_fee'] / (df['base_fee'] + 1e-8)
    
    # Gas price to base fee ratio
    df['gas_base_ratio'] = df['gas_price'] / (df['base_fee'] + 1e-8)

# Price-to-moving-average ratios (normalized price levels)
for window in [6, 12, 24, 48]:
    ma = df['gas_price'].rolling(window=window, min_periods=1).mean()
    df[f'price_ma_ratio_{window}'] = df['gas_price'] / (ma + 1e-8)

# Enhanced: Volatility × Momentum interactions (already exists but ensure they're prominent)
# These capture regime-specific behavior
for window in [6, 12, 24]:
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    mom = df['gas_price'] - df['gas_price'].shift(window)
    if f'vol_momentum_interact_{window}' not in df.columns:
        df[f'vol_momentum_interact_{window}'] = vol * mom.abs()

# Advanced: Fourier features for periodic patterns (daily cycles only)
# REMOVED: Weekly cycles (day_of_week) since we only have ~14h of data
from scipy.fft import fft
import math

# Daily cycles (24-hour patterns)
for period in [24, 12, 8, 6]:  # Different frequencies
    cycles_per_day = 24 / period
    df[f'hour_sin_{period}h'] = np.sin(2 * np.pi * df['hour'] / period)
    df[f'hour_cos_{period}h'] = np.cos(2 * np.pi * df['hour'] / period)

# Advanced: Autocorrelation features (lag correlations)
for lag in [1, 2, 3, 6, 12, 24]:
    df[f'autocorr_{lag}'] = df['gas_price'].shift(lag)
    # Correlation with shifted version
    rolling_corr = df['gas_price'].rolling(window=min(48, len(df)//10), min_periods=1).corr(df['gas_price'].shift(lag))
    df[f'autocorr_coef_{lag}'] = rolling_corr.fillna(0)

# Advanced: Interaction features (products/ratios of key features)
for window in [6, 12, 24]:
    # Volatility × momentum
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    mom = df['gas_price'] - df['gas_price'].shift(window)
    df[f'vol_momentum_interact_{window}'] = vol * mom.abs()
    
    # Price level × volatility
    price_level = df['gas_price'] / (df['gas_price'].rolling(window=window*2, min_periods=1).mean() + 1e-8)
    df[f'price_vol_interact_{window}'] = price_level * vol
    
    # Mean reversion × momentum (only if mean_reversion feature exists for this window)
    if f'mean_reversion_{window}' in df.columns:
        mean_rev = df[f'mean_reversion_{window}']
        df[f'mean_rev_momentum_{window}'] = mean_rev * mom

# Advanced: Rolling correlations between key features
for window in [12, 24]:
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    price = df['gas_price']
    # Correlation between price and volatility
    rolling_corr_price_vol = price.rolling(window=window*2, min_periods=1).corr(vol)
    df[f'corr_price_vol_{window}'] = rolling_corr_price_vol.fillna(0)
    
    # Correlation between price and momentum
    mom = df['gas_price'] - df['gas_price'].shift(window)
    rolling_corr_price_mom = price.rolling(window=window*2, min_periods=1).corr(mom)
    df[f'corr_price_mom_{window}'] = rolling_corr_price_mom.fillna(0)

# Advanced: Trend decomposition features
for window in [12, 24, 48]:
    price = df['gas_price']
    # Simple moving average (trend)
    trend = price.rolling(window=window, min_periods=1).mean()
    df[f'trend_{window}'] = trend
    
    # Detrended price (residual)
    df[f'detrended_{window}'] = price - trend
    
    # Trend strength (how much price follows trend)
    df[f'trend_strength_{window}'] = 1 - (df[f'detrended_{window}'].abs() / (price.abs() + 1e-8))
    
    # Trend direction (up/down/sideways)
    trend_diff = trend.diff()
    df[f'trend_direction_{window}'] = np.sign(trend_diff).fillna(0)

# Advanced: Price level indicators (categorical bands)
for window in [24, 48]:
    price = df['gas_price']
    rolling_mean = price.rolling(window=window, min_periods=1).mean()
    rolling_std = price.rolling(window=window, min_periods=1).std()
    
    # Price bands: low (below mean - 0.5*std), medium (within ±0.5*std), high (above mean + 0.5*std)
    df[f'price_band_low_{window}'] = (price < (rolling_mean - 0.5 * rolling_std)).astype(int)
    df[f'price_band_medium_{window}'] = ((price >= (rolling_mean - 0.5 * rolling_std)) & (price <= (rolling_mean + 0.5 * rolling_std))).astype(int)
    df[f'price_band_high_{window}'] = (price > (rolling_mean + 0.5 * rolling_std)).astype(int)
    
    # Z-score bands (normalized)
    z_score = (price - rolling_mean) / (rolling_std + 1e-8)
    df[f'price_z_low_{window}'] = (z_score < -1).astype(int)
    df[f'price_z_normal_{window}'] = ((z_score >= -1) & (z_score <= 1)).astype(int)
    df[f'price_z_high_{window}'] = (z_score > 1).astype(int)

# NEW: Micro-Structure Features (Short-term congestion)
log("   Adding micro-structure features (congestion & fee divergence)...")
if 'utilization' in df.columns:
    # Consecutive high utilization (approximate with rolling sum of boolean)
    high_util = (df['utilization'] > 80).astype(int)  # Assuming utilization is 0-100
    df['consecutive_high_util'] = high_util.rolling(window=6, min_periods=1).sum() # Count in last 6 blocks/minutes

if 'priority_fee' in df.columns:
    df['priority_fee_volatility'] = df['priority_fee'].rolling(window=6, min_periods=1).std().fillna(0)

if 'base_fee' in df.columns:
    df['gas_base_divergence'] = df['gas_price'] - df['base_fee']

# NEW: Mempool Features (Leading Indicators)
log("   Adding mempool features (leading indicators of congestion)...")
if 'tx_count' in df.columns and df['tx_count'].notna().any():
    # mempool_velocity: Raw tx_count (transactions per second)
    df['mempool_velocity'] = df['tx_count'].fillna(0)
    
    # mempool_acceleration: Difference in tx_count from previous second
    df['mempool_acceleration'] = df['tx_count'].diff().fillna(0)
    
    # mempool_surge: Boolean (1 if tx_count > rolling_mean + 2*std)
    if df['tx_count'].notna().sum() > 10:  # Need enough data for rolling stats
        rolling_mean = df['tx_count'].rolling(window=60, min_periods=1).mean()
        rolling_std = df['tx_count'].rolling(window=60, min_periods=1).std()
        df['mempool_surge'] = ((df['tx_count'] > (rolling_mean + 2 * rolling_std)) & (df['tx_count'].notna())).astype(int)
    else:
        df['mempool_surge'] = 0
    
    log(f"      ✅ Created mempool features: velocity, acceleration, surge")
    log(f"      📊 Mempool data coverage: {df['tx_count'].notna().sum():,} / {len(df):,} records ({100*df['tx_count'].notna().sum()/len(df):.1f}%)")
else:
    log(f"      ⚠️  Skipping mempool features (no mempool data available)")
    df['mempool_velocity'] = 0
    df['mempool_acceleration'] = 0
    df['mempool_surge'] = 0
    
# Drop NaN rows but be more selective - only drop rows where critical features are NaN
initial_len = len(df)
# Only drop rows where gas_price is NaN (critical feature)
df = df.dropna(subset=['gas_price'])
# Fill remaining NaN values with 0 for numeric columns (except gas_price which we already handled)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
log(f"✅ Features created: {len(df):,} samples, {len(df.columns)} features")
if initial_len > len(df):
    log(f"   ⚠️  Dropped {initial_len - len(df):,} rows with missing gas_price")
print(f"\n📊 Feature columns: {list(df.columns)}")

log("🎯 Creating prediction targets...")

# Estimate steps per hour from data
time_diffs = df['timestamp'].diff().dropna()
if len(time_diffs) > 0:
    median_interval_seconds = time_diffs.median().total_seconds()
    if pd.isna(median_interval_seconds) or median_interval_seconds == 0:
        # Default to 1-minute intervals (60 steps per hour)
        median_interval_seconds = 60
        steps_per_hour = 60
    else:
        median_interval_minutes = median_interval_seconds / 60
        steps_per_hour = max(1, int(3600 / median_interval_seconds))  # seconds in hour / interval
    log(f"   Detected interval: {median_interval_seconds:.1f} seconds ({median_interval_seconds/60:.2f} minutes)")
    log(f"   Steps per hour: {steps_per_hour}")
else:
    # Default to 1-minute intervals
    steps_per_hour = 60
    log(f"   ⚠️  Could not detect interval, defaulting to 60 steps/hour (1-minute intervals)")

# Future price targets - use actual time differences instead of fixed steps
# This is more robust when data intervals vary
targets = {}
total_samples = len(df)

# Horizons now based on 1-minute cadence
horizon_steps = {
    '1h': 60,    # 60 minutes ahead
    '4h': 240,   # 240 minutes ahead
    '24h': 1440  # keep 24h if data is long enough
}

for name, steps in horizon_steps.items():
    # Shift by steps (minutes) on the filtered data
    future_price = df['gas_price'].shift(-steps)
    current_price = df['gas_price'].copy()
    
    # Log-returns (stationary target)
    log_current = np.log(current_price + 1e-8)
    log_future = np.log(future_price + 1e-8)
    target_log_return = log_future - log_current
    
    # TARGET A (Classifier): spike_class
    # 0 (Stable): Log-return between -5% and +5%
    # 1 (Surge): Log-return > +5%
    # 2 (Crash): Log-return < -5%
    target_log_return_pct = target_log_return * 100  # Convert to percentage
    spike_class = pd.Series(1, index=target_log_return_pct.index)  # Default to Stable (1)
    spike_class[target_log_return_pct > 5.0] = 1  # Surge
    spike_class[target_log_return_pct < -5.0] = 2  # Crash
    spike_class[(target_log_return_pct >= -5.0) & (target_log_return_pct <= 5.0)] = 0  # Stable
    
    # TARGET B (Regressor): volatility_scaled_return
    # Normalize log-return by rolling mean and std to make small variations comparable
    rolling_window = min(60, len(df) // 10)  # Use 60 periods or 10% of data, whichever is smaller
    # Fill NaN in target_log_return before calculating rolling stats (use forward fill)
    target_log_return_filled = target_log_return.ffill().fillna(0)
    rolling_mean = target_log_return_filled.rolling(window=rolling_window, min_periods=1).mean()
    rolling_std = target_log_return_filled.rolling(window=rolling_window, min_periods=1).std()
    # Only calculate volatility_scaled where target_log_return is not NaN
    volatility_scaled_return = pd.Series(np.nan, index=target_log_return.index)
    valid_mask = ~target_log_return.isna()
    volatility_scaled_return[valid_mask] = (target_log_return[valid_mask] - rolling_mean[valid_mask]) / (rolling_std[valid_mask] + 1e-8)
    
    targets[name] = {
        'target_log_return': target_log_return,
        'spike_class': spike_class,  # Target A: Classifier
        'volatility_scaled_return': volatility_scaled_return,  # Target B: Regressor
        'target_price': future_price,
        'current_price': current_price,
        'original': future_price
    }
    valid = (~future_price.isna() & ~target_log_return.isna()).sum()
    log(f"   {name} ({steps} min): {valid:,} valid targets out of {total_samples:,} total")

# Check if we have enough data for each horizon
min_samples_needed = 100  # Minimum samples needed for training
available_horizons = []
for name, target_data in targets.items():
    valid_count = (~target_data['target_price'].isna()).sum()
    if valid_count >= min_samples_needed:
        available_horizons.append(name)
        log(f"   ✅ {name}: Sufficient data ({valid_count:,} samples)")
    else:
        log(f"   ⚠️  {name}: Insufficient data ({valid_count:,} samples, need {min_samples_needed})")

if not available_horizons:
    raise ValueError(f"Not enough data for any horizon! Minimum {min_samples_needed} samples needed.")

# Select feature columns
exclude_cols = ['timestamp', 'block_number', 'time_diff']  # Exclude time_diff to prevent dtype errors
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].copy()

# CRITICAL: Ensure X has no NaN (fill any remaining NaN with 0)
# This is essential after tick bar filtering which might create gaps
numeric_cols_X = X.select_dtypes(include=[np.number]).columns
X[numeric_cols_X] = X[numeric_cols_X].fillna(0)
# For any remaining non-numeric columns, fill with 0 or empty string
X = X.fillna(0)

log(f"📊 Feature matrix: {X.shape[0]:,} samples, {X.shape[1]} features")
log(f"   X NaN check: {X.isna().any(axis=1).sum()} rows with any NaN (should be 0)")

# Store original feature count for later reference
original_feature_count = X.shape[1]

from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import QuantileTransformer
from sklearn.inspection import permutation_importance

def select_features(X, y, n_samples, verbose=True):
    """
    Enhanced feature selection based on dataset size to prevent overfitting.
    Uses multiple methods (mutual info, correlation, F-regression, RFE, model importance).
    Dynamic feature count: min(20, samples/25) to prevent overfitting.
    """
    # Dynamic feature count based on dataset size
    target_features = min(20, max(10, int(n_samples / 25)))
    
    if X.shape[1] <= target_features:
        if verbose:
            log(f"   ✅ Already have {X.shape[1]} features (≤ target {target_features})")
        return X, list(X.columns)
    
    if verbose:
        log(f"   🔍 Feature selection: {X.shape[1]} → {target_features} features")
        log(f"   📊 Dataset size: {n_samples:,} samples → target {target_features} features")
    
    # Exclude non-numeric columns (time_diff, etc.) to prevent dtype errors
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols].copy()
    
    # If we lost columns, log it
    if X_numeric.shape[1] < X.shape[1]:
        dropped_cols = set(X.columns) - set(X_numeric.columns)
        if verbose:
            log(f"   ⚠️  Excluded {len(dropped_cols)} non-numeric columns: {', '.join(dropped_cols)}")
        X = X_numeric
    
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    y_clean = y.fillna(y.median())
    
    # Ensure we have enough samples for feature selection
    if len(X_clean) < 50:
        if verbose:
            log(f"   ⚠️  Too few samples for feature selection, using all features")
        return X, list(X.columns)
    
    scores_dict = {}
    
    # 1. Mutual Information (captures non-linear relationships)
    try:
        n_neighbors = min(3, max(1, len(X_clean) // 20))
        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42, n_neighbors=n_neighbors)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        scores_dict['MI'] = mi_scores
    except Exception as e:
        if verbose:
            log(f"   ⚠️  Mutual info failed: {e}")
        mi_scores = pd.Series(0, index=X.columns)
        scores_dict['MI'] = mi_scores
    
    # 2. Correlation (linear relationships)
    try:
        corr_scores = X_clean.corrwith(y_clean).abs().fillna(0)
        if corr_scores.max() > 0:
            corr_scores = corr_scores / corr_scores.max()
        scores_dict['Corr'] = corr_scores
    except Exception as e:
        if verbose:
            log(f"   ⚠️  Correlation failed: {e}")
        corr_scores = pd.Series(0, index=X.columns)
        scores_dict['Corr'] = corr_scores
    
    # 3. F-regression (statistical significance)
    try:
        f_selector = SelectKBest(f_regression, k=min(target_features * 2, X.shape[1]))
        f_selector.fit(X_clean, y_clean)
        f_scores = pd.Series(f_selector.scores_, index=X.columns).fillna(0)
        if f_scores.max() > 0:
            f_scores = f_scores / f_scores.max()
        scores_dict['F'] = f_scores
    except Exception as e:
        if verbose:
            log(f"   ⚠️  F-regression failed: {e}")
        f_scores = pd.Series(0, index=X.columns)
        scores_dict['F'] = f_scores
    
    # 4. RFE with Ridge (recursive feature elimination)
    try:
        if len(X_clean) >= 100:  # RFE needs more data
            rfe_estimator = Ridge(alpha=1.0)
            rfe = RFE(estimator=rfe_estimator, n_features_to_select=target_features, step=max(1, X.shape[1] // 50))
            rfe.fit(X_clean, y_clean)
            # Convert ranking to scores (lower rank = higher score)
            rfe_scores = pd.Series(1.0 / (rfe.ranking_ + 1), index=X.columns)
            if rfe_scores.max() > 0:
                rfe_scores = rfe_scores / rfe_scores.max()
            scores_dict['RFE'] = rfe_scores
    except Exception as e:
        if verbose:
            log(f"   ⚠️  RFE failed: {e}")
    
    # 5. Model-based importance (RandomForest)
    try:
        if len(X_clean) >= 50:
            rf_temp = RandomForestRegressor(
                n_estimators=min(50, len(X_clean) // 10),
                max_depth=min(10, int(np.log2(len(X_clean)))),
                min_samples_split=max(2, len(X_clean) // 50),
                random_state=42,
                n_jobs=-1
            )
            rf_temp.fit(X_clean, y_clean)
            rf_scores = pd.Series(rf_temp.feature_importances_, index=X.columns)
            if rf_scores.max() > 0:
                rf_scores = rf_scores / rf_scores.max()
            scores_dict['RF'] = rf_scores
    except Exception as e:
        if verbose:
            log(f"   ⚠️  Model importance failed: {e}")
    
    # Combine scores with weights (SHAP/MI gets highest weight)
    weights = {
        'MI': 0.30,
        'Corr': 0.25,
        'F': 0.20,
        'RFE': 0.15,
        'RF': 0.10
    }
    
    combined_scores = pd.Series(0.0, index=X.columns)
    total_weight = 0.0
    
    for method, weight in weights.items():
        if method in scores_dict:
            combined_scores += scores_dict[method] * weight
            total_weight += weight
    
    if total_weight > 0:
        combined_scores = combined_scores / total_weight
    
    # Select top features
    selected_features = combined_scores.nlargest(target_features).index.tolist()
    
    if verbose:
        log(f"   ✅ Selected {len(selected_features)} features")
        log(f"   📊 Top 10 features:")
        for i, feat in enumerate(selected_features[:10], 1):
            score = combined_scores[feat]
            log(f"      {i}. {feat}: {score:.4f}")
    
    return X[selected_features], selected_features

print("✅ Enhanced feature selection function defined")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, SGDRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_predict
from sklearn.base import clone
import joblib

# Try importing LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    log("   ⚠️  LightGBM not available, skipping LightGBM models")

# Create output directory
os.makedirs('trained_models', exist_ok=True)

results = {}

# Only train models for horizons with sufficient data
# horizons_to_train = available_horizons if 'available_horizons' in locals() else ['1h', '4h', '24h']

# ==============================================================================
# PHASE 1: Train 4h Regressor (Trend Setter)
# ==============================================================================
horizon_4h = '4h'
log(f"\n{'='*60}")
log(f"🎯 PHASE 1: Training 4h Regressor (Trend Setter)")
log(f"{'='*60}")

# Prepare 4h Targets
if horizon_4h not in targets:
    log(f"⚠️  Skipping 4h Regressor - Insufficient data for 4h horizon")
    model_4h = None
else:
    y_volatility_4h = targets[horizon_4h]['volatility_scaled_return']
    y_price_4h = targets[horizon_4h]['target_price']
    
    # Align X and y (remove NaNs)
    valid_idx_4h = ~(X.isna().any(axis=1) | y_volatility_4h.isna())
    X_4h = X[valid_idx_4h]
    y_4h = y_volatility_4h[valid_idx_4h]
    
    log(f"   Valid samples (4h): {len(X_4h):,}")
    
    # Feature Selection for 4h
    log(f"   🔍 Feature selection for 4h...")
    X_4h_selected, selected_features_4h = select_features(X_4h, y_4h, len(X_4h), verbose=True)
    
    # Train/Val/Test Split
    n_total_4h = len(X_4h_selected)
    test_start_4h = int(n_total_4h * 0.8)
    X_train_val_4h = X_4h_selected.iloc[:test_start_4h]
    X_test_4h = X_4h_selected.iloc[test_start_4h:]
    y_train_val_4h = y_4h.iloc[:test_start_4h]
    y_test_4h = y_4h.iloc[test_start_4h:]
    y_orig_test_4h = y_price_4h[valid_idx_4h].iloc[test_start_4h:]
    current_test_4h = targets[horizon_4h]['current_price'][valid_idx_4h].iloc[test_start_4h:]
    
    # Train 4h Model
    log(f"   🚀 Training LightGBM Regressor for 4h...")
    scaler_4h = RobustScaler()
    X_train_val_4h_scaled = scaler_4h.fit_transform(X_train_val_4h)
    X_test_4h_scaled = scaler_4h.transform(X_test_4h)
    
    model_4h = lgb.LGBMRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.01, num_leaves=15,
        objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
    )
    model_4h.fit(X_train_val_4h_scaled, y_train_val_4h)
    
    # Evaluate 4h
    y_pred_4h_vol = model_4h.predict(X_test_4h_scaled)
    # Reconstruct prices (simplified for quick check)
    # Note: Full reconstruction requires rolling stats, skipping for brevity in Phase 1 log
    
    # Save 4h Model
    model_data_4h = {
        'regressor': model_4h,
        'feature_names': selected_features_4h,
        'scaler': scaler_4h,
        'trained_at': datetime.now().isoformat()
    }
    joblib.dump(model_data_4h, 'trained_models/model_4h.pkl')
    log(f"   💾 Saved 4h model to trained_models/model_4h.pkl")

# ==============================================================================
# PHASE 2: Feature Injection (Model Stacking) - With Proper OOF Predictions
# ==============================================================================
log(f"\n{'='*60}")
log(f"💉 PHASE 2: Injecting 4h Trend Signal (Leak-Free OOF)")
log(f"{'='*60}")

if model_4h is not None:
    # IMPORTANT: To prevent data leakage, we use Out-of-Fold (OOF) predictions
    # for the training portion and regular predictions for the test portion.

    # 1. Prepare features for prediction
    X_for_4h_pred = X[selected_features_4h]

    # 2. Initialize trend signal array with NaN
    trend_signal_4h = np.full(len(X), np.nan)

    # 3. Get the valid indices (same as used in Phase 1)
    valid_idx_4h_bool = ~(X[selected_features_4h].isna().any(axis=1) | targets[horizon_4h]['volatility_scaled_return'].isna())
    valid_indices = np.where(valid_idx_4h_bool)[0]

    # 4. Split indices into train and test (same split as Phase 1)
    n_valid = len(valid_indices)
    train_end_idx = int(n_valid * 0.8)
    train_indices = valid_indices[:train_end_idx]
    test_indices = valid_indices[train_end_idx:]

    log(f"   📊 Train samples: {len(train_indices):,}, Test samples: {len(test_indices):,}")

    # 5. Generate OOF predictions for training data using TimeSeriesSplit
    log(f"   🔄 Generating Out-of-Fold predictions for training data...")
    X_train_for_oof = X_for_4h_pred.iloc[train_indices]
    y_train_for_oof = targets[horizon_4h]['volatility_scaled_return'].iloc[train_indices]

    # Use TimeSeriesSplit for proper temporal cross-validation
    tscv_oof = TimeSeriesSplit(n_splits=5)
    oof_predictions = np.full(len(train_indices), np.nan)

    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(tscv_oof.split(X_train_for_oof)):
        # Get fold data
        X_fold_train = X_train_for_oof.iloc[train_fold_idx]
        y_fold_train = y_train_for_oof.iloc[train_fold_idx]
        X_fold_val = X_train_for_oof.iloc[val_fold_idx]

        # Scale using only training fold data
        fold_scaler = RobustScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = fold_scaler.transform(X_fold_val)

        # Train fold model with same hyperparameters
        fold_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.01, num_leaves=15,
            objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
        )
        fold_model.fit(X_fold_train_scaled, y_fold_train)

        # Predict on validation fold
        oof_predictions[val_fold_idx] = fold_model.predict(X_fold_val_scaled)

        log(f"      Fold {fold_idx + 1}/5: Train={len(train_fold_idx)}, Val={len(val_fold_idx)}")

    # Fill training indices with OOF predictions
    for i, idx in enumerate(train_indices):
        if not np.isnan(oof_predictions[i]):
            trend_signal_4h[idx] = oof_predictions[i]

    # 6. Generate predictions for test data using the final trained model
    log(f"   🎯 Generating predictions for test data using final model...")
    X_test_for_pred = X_for_4h_pred.iloc[test_indices]
    X_test_scaled = scaler_4h.transform(X_test_for_pred)
    test_predictions = model_4h.predict(X_test_scaled)

    # Fill test indices with model predictions
    for i, idx in enumerate(test_indices):
        trend_signal_4h[idx] = test_predictions[i]

    # 7. Handle any remaining NaN values (fill with 0 for non-valid indices)
    trend_signal_4h = np.nan_to_num(trend_signal_4h, nan=0.0)

    # 8. Add to X
    X = X.copy()
    X['trend_signal_4h'] = trend_signal_4h

    # Calculate statistics for valid predictions only
    valid_preds = trend_signal_4h[trend_signal_4h != 0]
    if len(valid_preds) > 0:
        log(f"   ✅ Added 'trend_signal_4h' feature (OOF leak-free)")
        log(f"      Range: {valid_preds.min():.4f} to {valid_preds.max():.4f}")
        log(f"      Mean: {valid_preds.mean():.4f}, Std: {valid_preds.std():.4f}")
    else:
        log(f"   ✅ Added 'trend_signal_4h' feature (all zeros - no valid predictions)")
else:
    log(f"   ⚠️  Skipping Feature Injection (4h model failed)")
    X['trend_signal_4h'] = 0

# ==============================================================================
# PHASE 3: Train 1h Classifier (Action Taker)
# ==============================================================================
horizon_1h = '1h'
log(f"\n{'='*60}")
log(f"🎯 PHASE 3: Training 1h Classifier (Action Taker)")
log(f"{'='*60}")

# Define Target: action_class
# 0 (Wait): Log-return < -0.05
# 1 (Normal): -0.05 <= Log-return <= 0.05
# 2 (Urgent): Log-return > 0.05
target_log_return_1h = targets[horizon_1h]['target_log_return']
action_class = pd.Series(1, index=target_log_return_1h.index) # Default Normal
action_class[target_log_return_1h < -0.05] = 0 # Wait
action_class[target_log_return_1h > 0.05] = 2 # Urgent

# Align Data
valid_idx_1h = ~(X.isna().any(axis=1) | action_class.isna())
X_1h = X[valid_idx_1h]
y_1h = action_class[valid_idx_1h].astype(int)

log(f"   Valid samples (1h): {len(X_1h):,}")
log(f"   Class Distribution: {y_1h.value_counts().to_dict()}")

# Feature Selection for 1h (now includes trend_signal_4h)
X_1h_selected, selected_features_1h = select_features(X_1h, y_1h, len(X_1h), verbose=True)

# Train/Test Split
n_total_1h = len(X_1h_selected)
test_start_1h = int(n_total_1h * 0.8)
X_train_1h = X_1h_selected.iloc[:test_start_1h]
X_test_1h = X_1h_selected.iloc[test_start_1h:]
y_train_1h = y_1h.iloc[:test_start_1h]
y_test_1h = y_1h.iloc[test_start_1h:]

# Train Classifier
log(f"   🚀 Training LightGBM Classifier for 1h (Unbalanced weights)...")
scaler_1h = RobustScaler()
X_train_1h_scaled = scaler_1h.fit_transform(X_train_1h)
X_test_1h_scaled = scaler_1h.transform(X_test_1h)

model_1h = lgb.LGBMClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=31,
    objective='multiclass', num_class=3, metric='multi_logloss',
    random_state=42, n_jobs=-1, verbose=-1
)
model_1h.fit(X_train_1h_scaled, y_train_1h)

# Evaluate with Probability Gating
y_proba_1h = model_1h.predict_proba(X_test_1h_scaled)

def apply_gating(probs, threshold=0.7):
    # probs is (n_samples, 3) -> [Wait, Normal, Urgent] (assuming classes 0, 1, 2)
    # Default to Normal (1)
    preds = np.ones(len(probs), dtype=int) * 1
    
    # If P(Urgent) > threshold -> Urgent (2)
    # If P(Wait) > threshold -> Wait (0)
    # Urgent takes precedence if both high (rare with softmax)
    
    # Check Urgent (Class 2)
    urgent_mask = probs[:, 2] > threshold
    preds[urgent_mask] = 2
    
    # Check Wait (Class 0) - only if not already Urgent
    wait_mask = (probs[:, 0] > threshold) & (~urgent_mask)
    preds[wait_mask] = 0
    
    return preds

log(f"\n   🔍 Probability Gating Analysis (Precision vs Recall):")
best_threshold = 0.7  # Default start
best_score = 0

for threshold in [0.5, 0.6, 0.7, 0.8]:
    y_pred_gated = apply_gating(y_proba_1h, threshold)
    
    # Calculate metrics
    report = classification_report(y_test_1h, y_pred_gated, output_dict=True, zero_division=0)
    
    # Metrics for Class 2 (Urgent) and Class 0 (Wait)
    urgent_prec = report.get('2', {}).get('precision', 0)
    urgent_rec = report.get('2', {}).get('recall', 0)
    wait_prec = report.get('0', {}).get('precision', 0)
    wait_rec = report.get('0', {}).get('recall', 0)
    normal_count = (y_pred_gated == 1).sum()
    
    log(f"      Threshold {threshold:.1f}: Urgent P={urgent_prec:.2f}/R={urgent_rec:.2f} | Wait P={wait_prec:.2f}/R={wait_rec:.2f} | Normal Preds={normal_count}")
    
    # Score: Weighted average of F1s for active classes (Wait/Urgent)
    current_score = (report.get('2', {}).get('f1-score', 0) + report.get('0', {}).get('f1-score', 0)) / 2
    if current_score > best_score:
        best_score = current_score
        best_threshold = threshold

log(f"   ✨ Selected Inference Threshold: {best_threshold}")
y_pred_1h = apply_gating(y_proba_1h, best_threshold)
acc_1h = accuracy_score(y_test_1h, y_pred_1h)
log(f"   ✅ 1h Final Accuracy: {acc_1h:.4f}")

log(f"\n   📊 Classification Report (Threshold {best_threshold}):")
print(classification_report(y_test_1h, y_pred_1h, target_names=['Wait', 'Normal', 'Urgent'], zero_division=0))

log(f"\n   📊 Confusion Matrix:")
print(confusion_matrix(y_test_1h, y_pred_1h))

# Save 1h Model
model_data_1h = {
    'classifier': model_1h,
    'feature_names': selected_features_1h,
    'scaler': scaler_1h,
    'trained_at': datetime.now().isoformat(),
    'model_type': 'hybrid_classifier',
    'inference_config': {'threshold': best_threshold, 'default_class': 1}
}
joblib.dump(model_data_1h, 'trained_models/model_1h.pkl')
log(f"   💾 Saved 1h Hybrid Classifier to trained_models/model_1h.pkl")

log(f"\n{'='*60}")
log("🎉 HYBRID TRAINING COMPLETE!")
log(f"{'='*60}\n")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# Try importing XGBoost for spike detection
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    log("   ⚠️  XGBoost not available for spike detectors, using GradientBoosting")

# Spike detection thresholds (must match HybridPredictor)
NORMAL_THRESHOLD = 0.01   # < 0.01 gwei = Normal
ELEVATED_THRESHOLD = 0.05  # 0.01-0.05 gwei = Elevated
# > 0.05 gwei = Spike

def create_spike_features(df):
    """Create enhanced features for spike detection with volatility clustering, acceleration, and regime indicators"""
    df = df.copy()
    df = df.sort_values('timestamp')
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Recent volatility
    for window in [6, 12, 24, 48]:
        df[f'volatility_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).std()
        df[f'range_{window}'] = (
            df['gas_price'].rolling(window=window, min_periods=1).max() -
            df['gas_price'].rolling(window=window, min_periods=1).min()
        )
        df[f'mean_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).mean()
        df[f'is_rising_{window}'] = (df['gas_price'] > df[f'mean_{window}']).astype(int)
    
    # Rate of change
    for lag in [1, 2, 3, 6, 12]:
        df[f'pct_change_{lag}'] = df['gas_price'].pct_change(lag).fillna(0)
        df[f'diff_{lag}'] = df['gas_price'].diff(lag).fillna(0)
    
    # Enhanced: Volatility clustering indicators
    # Volatility clustering: high volatility tends to be followed by high volatility
    for window in [6, 12, 24]:
        vol = df['gas_price'].rolling(window=window, min_periods=1).std()
        vol_lag = vol.shift(1)
        df[f'volatility_clustering_{window}'] = (vol > vol.quantile(0.75)) & (vol_lag > vol_lag.quantile(0.75))
        df[f'volatility_clustering_{window}'] = df[f'volatility_clustering_{window}'].astype(int)
        # Volatility persistence: how long has volatility been high?
        vol_high = (vol > vol.rolling(window*2, min_periods=1).quantile(0.75)).astype(int)
        vol_persistence = vol_high.groupby((vol_high != vol_high.shift()).cumsum()).cumsum()
        df[f'volatility_persistence_{window}'] = vol_persistence
    
    # Enhanced: Rate of change acceleration (second derivative)
    # First derivative (velocity)
    df['velocity_1'] = df['gas_price'].diff(1)
    df['velocity_6'] = df['gas_price'].diff(6)
    df['velocity_12'] = df['gas_price'].diff(12)
    # Second derivative (acceleration)
    df['acceleration_1'] = df['velocity_1'].diff(1)
    df['acceleration_6'] = df['velocity_6'].diff(1)
    # Acceleration magnitude
    df['acceleration_magnitude'] = df['acceleration_1'].abs()
    # Is acceleration increasing?
    df['acceleration_increasing'] = (df['acceleration_1'] > df['acceleration_1'].shift(1)).astype(int)
    
    # Enhanced: Market regime indicators
    # Quiet vs active periods based on volatility
    for window in [12, 24, 48]:
        vol = df['gas_price'].rolling(window=window, min_periods=1).std()
        vol_median = vol.rolling(window*2, min_periods=1).median()
        df[f'regime_quiet_{window}'] = (vol < vol_median * 0.5).astype(int)
        df[f'regime_active_{window}'] = (vol > vol_median * 1.5).astype(int)
        df[f'regime_extreme_{window}'] = (vol > vol.rolling(window*2, min_periods=1).quantile(0.9)).astype(int)
    
    # Regime transition indicators
    df['regime_transition'] = (
        (df['regime_quiet_24'] != df['regime_quiet_24'].shift(1)) |
        (df['regime_active_24'] != df['regime_active_24'].shift(1))
    ).astype(int)
    
    # Cross-horizon features: 4h spike affects 1h predictions
    for window in [6, 12, 24]:
        spike_4h = (df['gas_price'].rolling(window=window*4, min_periods=1).max() > ELEVATED_THRESHOLD).astype(int)
        df[f'spike_4h_indicator_{window}'] = spike_4h
        spike_24h = (df['gas_price'].rolling(window=window*24, min_periods=1).max() > ELEVATED_THRESHOLD).astype(int)
        df[f'spike_24h_indicator_{window}'] = spike_24h
    
    # Recent spike indicator
    df['recent_spike'] = (
        df['gas_price'].rolling(window=24, min_periods=1).max() > ELEVATED_THRESHOLD
    ).astype(int)
    
    # Replace inf/-inf with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def classify_price(price):
    """Classify gas price into Normal(0)/Elevated(1)/Spike(2)"""
    if pd.isna(price):
        return np.nan
    if price < NORMAL_THRESHOLD:
        return 0  # Normal
    elif price < ELEVATED_THRESHOLD:
        return 1  # Elevated
    else:
        return 2  # Spike

# Create spike features
log("\n🔍 Creating spike detection features...")
df_spike = create_spike_features(df)

# Feature names for spike detection (including enhanced features)
spike_feature_names = [
    'hour', 'day_of_week', 'is_weekend', 'is_business_hours'
]
for window in [6, 12, 24, 48]:
    spike_feature_names.extend([
        f'volatility_{window}', f'range_{window}', f'mean_{window}', f'is_rising_{window}'
    ])
for lag in [1, 2, 3, 6, 12]:
    spike_feature_names.extend([f'pct_change_{lag}', f'diff_{lag}'])
# Enhanced features
for window in [6, 12, 24]:
    spike_feature_names.extend([
        f'volatility_clustering_{window}', f'volatility_persistence_{window}'
    ])
spike_feature_names.extend([
    'velocity_1', 'velocity_6', 'velocity_12',
    'acceleration_1', 'acceleration_6', 'acceleration_magnitude', 'acceleration_increasing'
])
for window in [12, 24, 48]:
    spike_feature_names.extend([
        f'regime_quiet_{window}', f'regime_active_{window}', f'regime_extreme_{window}'
    ])
spike_feature_names.extend(['regime_transition'])
for window in [6, 12, 24]:
    spike_feature_names.extend([
        f'spike_4h_indicator_{window}', f'spike_24h_indicator_{window}'
    ])
spike_feature_names.append('recent_spike')

# Ensure all features exist
spike_feature_names = [f for f in spike_feature_names if f in df_spike.columns]
X_spike = df_spike[spike_feature_names]

log(f"   Spike features: {len(spike_feature_names)}")
log(f"   Samples: {len(X_spike):,}")

# Create spike detection targets
spike_targets = {}
for horizon, hours in [('1h', 1), ('4h', 4), ('24h', 24)]:
    steps = hours * steps_per_hour
    future_price = df_spike['gas_price'].shift(-steps)
    spike_targets[horizon] = future_price.apply(classify_price)

spike_results = {}

# Train spike detectors for each horizon
for horizon in ['1h', '4h', '24h']:
    log(f"\n{'='*60}")
    log(f"🎯 Training Spike Detector for {horizon}")
    log(f"{'='*60}")
    
    y_spike = spike_targets[horizon]
    
    # Remove NaN targets
    valid_idx = ~y_spike.isna()
    X_clean = X_spike[valid_idx]
    y_clean = y_spike[valid_idx].astype(int)
    
    log(f"   Valid samples: {len(X_clean):,}")
    
    if len(X_clean) < 100:
        log(f"   ⚠️  Skipping {horizon} - insufficient data ({len(X_clean)} samples, need at least 100)")
        continue
    
    # Class distribution
    class_counts = y_clean.value_counts().sort_index()
    class_names = ['Normal', 'Elevated', 'Spike']
    log(f"   Class Distribution:")
    for cls, count in class_counts.items():
        pct = count / len(y_clean) * 100
        log(f"      {class_names[cls]}: {count:,} ({pct:.1f}%)")
    
    # Check if we have at least 2 classes (required for classification)
    unique_classes = np.unique(y_clean)
    if len(unique_classes) < 2:
        log(f"   ⚠️  Skipping {horizon} - only {len(unique_classes)} class(es) found: {unique_classes}")
        log(f"      Need at least 2 classes for spike detection. All prices are classified as the same category.")
        continue
    
    # Train/val/test split (60/20/20, temporal)
    n_total = len(X_clean)
    train_end = int(n_total * 0.6)
    val_end = int(n_total * 0.8)
    
    X_train_spike = X_clean.iloc[:train_end]
    X_val_spike = X_clean.iloc[train_end:val_end]
    X_test_spike = X_clean.iloc[val_end:]
    y_train_spike = y_clean.iloc[:train_end]
    y_val_spike = y_clean.iloc[train_end:val_end]
    y_test_spike = y_clean.iloc[val_end:]
    
    log(f"   Train: {len(X_train_spike):,}, Val: {len(X_val_spike):,}, Test: {len(X_test_spike):,}")
    
    # Calculate class weights for balancing
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes_train = np.unique(y_train_spike)
    # Ensure we have all expected classes (0, 1, 2) even if some are missing
    expected_classes = np.array([0, 1, 2])
    if len(unique_classes_train) < 3:
        log(f"   ⚠️  Only {len(unique_classes_train)} classes in training data: {unique_classes_train}")
        # Use balanced weights for existing classes
        class_weights = compute_class_weight('balanced', classes=unique_classes_train, y=y_train_spike)
        class_weight_dict = dict(zip(unique_classes_train, class_weights))
        # Add missing classes with weight 1.0
        for cls in expected_classes:
            if cls not in class_weight_dict:
                class_weight_dict[cls] = 1.0
    else:
        class_weights = compute_class_weight('balanced', classes=unique_classes_train, y=y_train_spike)
        class_weight_dict = dict(zip(unique_classes_train, class_weights))
    log(f"   Class weights: {class_weight_dict}")
    
    # Remap classes to start from 0 for XGBoost compatibility
    # XGBoost requires classes to be [0, 1, 2, ...] but we might have [2] or [1, 2]
    class_mapping = {old_cls: new_cls for new_cls, old_cls in enumerate(sorted(unique_classes_train))}
    y_train_spike_remapped = y_train_spike.map(class_mapping)
    y_val_spike_remapped = y_val_spike.map(class_mapping)
    y_test_spike_remapped = y_test_spike.map(class_mapping)
    
    # Update class_weight_dict to use remapped classes
    class_weight_dict_remapped = {class_mapping[old_cls]: weight for old_cls, weight in class_weight_dict.items() if old_cls in class_mapping}
    
    # Train XGBoost or GradientBoosting classifier with class balancing
    if XGBOOST_AVAILABLE:
        log(f"   Training XGBoost classifier with class balancing...")
        model_spike = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            min_child_weight=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Will use sample_weight instead
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        # Use sample weights for class balancing (use remapped classes)
        sample_weights = np.array([class_weight_dict_remapped[y] for y in y_train_spike_remapped])
        model_spike.fit(X_train_spike, y_train_spike_remapped, sample_weight=sample_weights)
    else:
        log(f"   Training GradientBoosting classifier with class balancing...")
        model_spike = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        # Use sample weights for class balancing (use remapped classes)
        sample_weights = np.array([class_weight_dict_remapped[y] for y in y_train_spike_remapped])
        model_spike.fit(X_train_spike, y_train_spike_remapped, sample_weight=sample_weights)
    
    # Calibrate the classifier for better probability estimates
    log(f"   Calibrating classifier for better probability estimates...")
    calibrated_model = CalibratedClassifierCV(model_spike, method='isotonic', cv=3)
    calibrated_model.fit(X_train_spike, y_train_spike_remapped, sample_weight=sample_weights)
    
    # Evaluate on validation and test sets (use remapped classes)
    y_pred_val_remapped = calibrated_model.predict(X_val_spike)
    y_pred_test_remapped = calibrated_model.predict(X_test_spike)
    y_proba_val = calibrated_model.predict_proba(X_val_spike)
    y_proba_test = calibrated_model.predict_proba(X_test_spike)
    
    # Remap predictions back to original class labels
    reverse_mapping = {new_cls: old_cls for old_cls, new_cls in class_mapping.items()}
    y_pred_val = np.array([reverse_mapping.get(pred, pred) for pred in y_pred_val_remapped])
    y_pred_test = np.array([reverse_mapping.get(pred, pred) for pred in y_pred_test_remapped])
    
    # Use original (non-remapped) labels for evaluation
    accuracy_val = accuracy_score(y_val_spike, y_pred_val)
    accuracy_test = accuracy_score(y_test_spike, y_pred_test)
    f1_val = f1_score(y_val_spike, y_pred_val, average='weighted', zero_division=0)
    f1_test = f1_score(y_test_spike, y_pred_test, average='weighted', zero_division=0)
    
    log(f"\n   ✅ Model Performance:")
    log(f"      Val Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.1f}%)")
    log(f"      Test Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.1f}%)")
    log(f"      Val F1 Score (weighted): {f1_val:.4f}")
    log(f"      Test F1 Score (weighted): {f1_test:.4f}")
    
    # Classification report on test set
    report = classification_report(y_test_spike, y_pred_test, target_names=class_names, zero_division=0, output_dict=True)
    log(f"   Test Set Detailed Report:")
    for cls_name in class_names:
        if cls_name.lower() in report:
            log(f"      {cls_name}: precision={report[cls_name.lower()]['precision']:.3f}, recall={report[cls_name.lower()]['recall']:.3f}, f1={report[cls_name.lower()]['f1-score']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_spike, y_pred_test)
    log(f"   Confusion Matrix (Test):")
    log(f"      {cm}")
    
    # Class-specific metrics
    log(f"   Class-specific Performance (Test):")
    for i, cls_name in enumerate(class_names):
        if i < len(cm):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            log(f"      {cls_name}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Save spike detector with enhanced metrics
    spike_detector_data = {
        'model': calibrated_model,  # Save calibrated model
        'base_model': model_spike,  # Also save base model for reference
        'feature_names': spike_feature_names,
        'metrics': {
            'accuracy_val': accuracy_val,
            'accuracy_test': accuracy_test,
            'f1_score_val': f1_val,
            'f1_score_test': f1_test,
            'confusion_matrix': cm.tolist()
        },
        'class_distribution': class_counts.to_dict(),
        'class_weights': class_weight_dict,
        'is_calibrated': True,
        'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting',
        'trained_at': datetime.now().isoformat()
    }
    
    spike_path = f'trained_models/spike_detector_{horizon}.pkl'
    joblib.dump(spike_detector_data, spike_path)
    log(f"   💾 Saved calibrated spike detector to {spike_path}")
    
    spike_results[horizon] = {
        'accuracy_val': accuracy_val,
        'accuracy_test': accuracy_test,
        'f1_score_val': f1_val,
        'f1_score_test': f1_test
    }

log(f"\n{'='*60}")
log("✅ Spike Detector Training Complete!")
log(f"{'='*60}")

log(f"\n{'='*60}")
log("🎉 TRAINING COMPLETE!")
log(f"{'='*60}\n")

# Print training summary across all horizons
if results:
    log("📊 Training Summary:")
    for h in ['1h', '4h', '24h']:
        if h in results:
            r = results[h]
            log(f"\n   {h} Horizon:")
            log(f"      Best Model: {r.get('best_model', 'Unknown')}")
            log(f"      Test R²: {r.get('r2', 0):.4f}")
            log(f"      Test MAE: {r.get('mae', 0):.6f} gwei")
            overfitting_gap = r.get('overfitting_gap', 1)
            status = '✅' if overfitting_gap < 0.1 else ('⚠️' if overfitting_gap < 0.2 else '❌')
            log(f"      Overfitting Gap: {overfitting_gap:.4f} {status}")
    
    log(f"\n{'='*60}\n")
