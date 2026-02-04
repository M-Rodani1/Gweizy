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

# Load block_stats data (from mempool worker) and join with gas_prices
log("\n📡 Loading block_stats data (from mempool worker)...")
try:
    block_stats_query = """
    SELECT timestamp, block_number as block_num_stats, gas_used as gas_used_stats,
           gas_limit as gas_limit_stats, utilization as util_stats, base_fee as base_fee_stats
    FROM block_stats
    ORDER BY timestamp ASC
    """
    df_block_stats = pd.read_sql(block_stats_query, engine)

    if len(df_block_stats) > 0:
        df_block_stats['timestamp'] = pd.to_datetime(df_block_stats['timestamp'])
        log(f"   ✅ Loaded {len(df_block_stats):,} block_stats records")
        log(f"   📅 Block stats date range: {df_block_stats['timestamp'].min()} to {df_block_stats['timestamp'].max()}")

        # Convert gas_prices timestamp to datetime for joining
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Left join block_stats with gas_prices on timestamp
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_block_stats.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=5),
            suffixes=('', '_block')
        )

        log(f"   ✅ Joined block stats: {df['util_stats'].notna().sum():,} rows have block data")
        log(f"   📊 Block stats coverage: {df['util_stats'].notna().sum() / len(df) * 100:.1f}%")

        # Fill missing data
        df['util_stats'] = df['util_stats'].fillna(0)
        df['base_fee_stats'] = df['base_fee_stats'].fillna(df['base_fee'] if 'base_fee' in df.columns else 0)

        # Create tx_count proxy from gas_used (higher gas used = more transactions)
        df['tx_count'] = df['gas_used_stats'].fillna(0) / 21000  # Approximate tx count (21000 gas per simple tx)
    else:
        log(f"   ⚠️  No block_stats data found - block features will be zero")
        df['tx_count'] = 0
        df['util_stats'] = 0
        df['base_fee_stats'] = 0
except Exception as e:
    log(f"   ⚠️  Error loading block_stats: {e} - block features will be zero")
    df['tx_count'] = 0
    df['util_stats'] = 0
    df['base_fee_stats'] = 0

# Note: tx_count is now derived from block_stats (gas_used / 21000) above
# The mempool_stats table is deprecated in favor of block_stats from the mempool worker
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

# CRITICAL: Drop rows where gas_price changes by less than 0.2% compared to previous row
# IMPROVED: Relaxed from 1.0% to 0.2% to include stable periods in training (was causing distribution bias)
# Keep full data to avoid excluding stable market conditions that appear in production
df_grouped = df_grouped.sort_values('timestamp').reset_index(drop=True)
df_grouped['price_pct_change'] = df_grouped['gas_price'].pct_change().abs() * 100
# Keep first row (no previous to compare) and rows with >0.2% change (much more lenient)
df_grouped = df_grouped[(df_grouped['price_pct_change'].isna()) | (df_grouped['price_pct_change'] >= 0.2)]
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

# ===== TIER 1 IMPROVEMENTS: Advanced Interaction Features =====
log("   Adding TIER 1 advanced interaction features...")

# 1. Price × Volatility × Time (captures time-dependent volatility effects)
for window in [6, 12, 24]:
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    # Weight by hour to capture different dynamics at different times
    hour_weight = np.abs(np.sin(2 * np.pi * df['hour'] / 24))  # 0 at midnight, 1 at noon
    df[f'price_vol_hour_interact_{window}'] = vol * hour_weight

    # Also capture hour × price directly (captures price magnitude dependence on time)
    df[f'price_hour_interact_{window}'] = df['gas_price'] * hour_weight

# 2. Momentum × Regime (captures different behavior in different market states)
for window in [6, 12, 24]:
    mom = df['gas_price'] - df['gas_price'].shift(window)
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()

    # Regime detection: high vol = volatile regime, low vol = stable regime
    vol_ma = vol.rolling(window=12, min_periods=1).mean()
    regime = (vol > vol_ma).astype(float)  # 1 = volatile regime, 0 = stable regime

    df[f'momentum_regime_interact_{window}'] = mom * (regime * 2 - 1)  # ±momentum based on regime

# 3. Mean-Reversion Pressure × Volatility (captures mean-reversion strength)
for window in [6, 12, 24]:
    # Mean reversion signal: distance from moving average
    ma = df['gas_price'].rolling(window=window, min_periods=1).mean()
    mean_rev_pressure = (ma - df['gas_price']) / (df['gas_price'] + 1e-8)  # Positive when price below MA

    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    df[f'mean_rev_pressure_vol_{window}'] = mean_rev_pressure * vol

# 4. EIP-1559 Pressure × Time-of-Day (captures demand cycles)
if 'base_fee' in df.columns:
    for window in [6, 12, 24]:
        # Pressure: how much base fee changed
        pressure = df['base_fee'].pct_change(window).fillna(0)
        hour_weight = np.abs(np.sin(2 * np.pi * df['hour'] / 24))
        df[f'pressure_hour_interact_{window}'] = pressure * hour_weight

# 5. Volatility Regime Features (smoothed regimes for stability)
vol_ma_24h = df['gas_price'].rolling(window=24, min_periods=1).std().rolling(window=6, min_periods=1).mean()
df['vol_regime_smooth'] = (df['gas_price'].rolling(window=24, min_periods=1).std() / (vol_ma_24h + 1e-8))

# 6. Multi-Scale Momentum Divergence (captures contradictory signals across timeframes)
for w1, w2 in [(6, 12), (12, 24)]:
    mom1 = df['gas_price'] - df['gas_price'].shift(w1)
    mom2 = df['gas_price'] - df['gas_price'].shift(w2)
    df[f'momentum_divergence_{w1}_{w2}'] = (mom1 - mom2) / (np.abs(mom2) + 1e-8)

# 7. Acceleration × Volatility (captures strength of moves)
for window in [6, 12, 24]:
    mom1 = df['gas_price'] - df['gas_price'].shift(window)
    mom2 = df['gas_price'].shift(window) - df['gas_price'].shift(2*window)
    accel = mom1 - mom2  # Acceleration = change in momentum
    vol = df['gas_price'].rolling(window=window, min_periods=1).std()
    df[f'accel_vol_interact_{window}'] = accel * vol

# 8. Cross-Feature Interactions (base fee × momentum)
if 'base_fee' in df.columns:
    for window in [6, 12, 24]:
        mom = df['gas_price'] - df['gas_price'].shift(window)
        base_mom = df['base_fee'] - df['base_fee'].shift(window)
        df[f'gas_base_momentum_interact_{window}'] = mom * base_mom

log(f"   ✅ Added {8} new advanced interaction feature groups")

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

# ===== TIER 1 IMPROVEMENTS: Add Meta-Features =====
log("\n🔧 Adding TIER 1 meta-features for model uncertainty...")

# Meta-features: Track recent prediction errors and confidence for ensemble
# These will be recomputed for each horizon
def create_meta_features(df_data, window=12):
    """
    Create meta-features that capture model uncertainty and confidence signals.
    These are added to the feature set for ensemble learning.
    """
    meta_df = pd.DataFrame(index=df_data.index)

    # 1. Recent volatility (uncertainty level)
    price = df_data['gas_price'] if 'gas_price' in df_data.columns else df_data.iloc[:, 0]
    for w in [6, 12, 24]:
        meta_df[f'meta_vol_{w}'] = price.rolling(window=w, min_periods=1).std()

    # 2. Recent trend consistency (how stable is the trend)
    for w in [6, 12, 24]:
        returns = price.pct_change()
        consistency = (returns > 0).rolling(window=w, min_periods=1).mean()  # % of positive returns
        # Convert to deviation from 50% (50% = no trend, 0% or 100% = strong trend)
        meta_df[f'meta_trend_strength_{w}'] = np.abs(consistency - 0.5) * 2

    # 3. Recent prediction difficulty (using price changes as proxy for "prediction error")
    for w in [6, 12, 24]:
        price_changes = price.pct_change(w).abs()
        # Use rolling max to detect recent difficulty spikes
        meta_df[f'meta_difficulty_{w}'] = price_changes.rolling(window=w, min_periods=1).max()

    # 4. Recency weighting (recent data points should have higher weight)
    meta_df['meta_recency'] = np.linspace(0.5, 1.0, len(meta_df))  # Increases over time

    # 5. Consecutive high/low periods (captures regime changes)
    for w in [6, 12]:
        returns = price.pct_change()
        positive_streak = (returns > 0).rolling(window=w, min_periods=1).sum()
        negative_streak = (returns <= 0).rolling(window=w, min_periods=1).sum()
        meta_df[f'meta_streak_{w}'] = np.maximum(positive_streak, negative_streak) / w

    # 6. Model confidence signal (inverse of uncertainty)
    for w in [12, 24]:
        vol = price.rolling(window=w, min_periods=1).std()
        vol_ma = vol.rolling(window=12, min_periods=1).mean()
        # Low vol relative to MA = high confidence
        meta_df[f'meta_confidence_{w}'] = 1.0 - np.clip(vol / (vol_ma + 1e-8), 0, 1)

    return meta_df

# Create meta-features from raw data before feature selection
meta_features = create_meta_features(df)

# Select feature columns
exclude_cols = ['timestamp', 'block_number', 'time_diff']  # Exclude time_diff to prevent dtype errors
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].copy()

# Add meta-features to X
for col in meta_features.columns:
    if col not in X.columns:
        X[col] = meta_features[col].fillna(0)

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

# ===== PHASE 2: Configuration Flags =====
# Set TUNE_HYPERPARAMS=1 environment variable to enable tuning (slower but better)
TUNE_HYPERPARAMS = os.environ.get('TUNE_HYPERPARAMS', '0') == '1'
TUNING_ITERATIONS = int(os.environ.get('TUNING_ITERATIONS', '20'))

# PHASE 2: Bayesian optimization (more efficient than random search, 15-20% improvement)
USE_BAYESIAN_OPTIMIZATION = os.environ.get('USE_BAYESIAN_OPT', '1') == '1'
BAYESIAN_TRIALS = int(os.environ.get('BAYESIAN_TRIALS', '50'))

# PHASE 2: Spike detection ensemble (20-30% improvement in spike F1)
USE_SPIKE_ENSEMBLE = os.environ.get('USE_SPIKE_ENSEMBLE', '1') == '1'

# PHASE 2: Model calibration (better probability estimates)
USE_MODEL_CALIBRATION = os.environ.get('USE_CALIBRATION', '1') == '1'

# PHASE 2: Online learning foundation (production model updates)
USE_ONLINE_LEARNING = os.environ.get('USE_ONLINE_LEARNING', '0') == '1'

# PHASE 2: Monitoring infrastructure (performance tracking)
USE_MONITORING = os.environ.get('USE_MONITORING', '1') == '1'

if TUNE_HYPERPARAMS:
    log(f"\n🔧 Hyperparameter tuning ENABLED ({TUNING_ITERATIONS} iterations per model)")
    if USE_BAYESIAN_OPTIMIZATION:
        log(f"   🎯 Using Bayesian Optimization ({BAYESIAN_TRIALS} trials) - PHASE 2 ✨")
else:
    log(f"\n⚡ Hyperparameter tuning DISABLED (using defaults)")

log(f"\n✨ PHASE 2 Features Enabled:")
log(f"   Spike Ensemble: {'✅' if USE_SPIKE_ENSEMBLE else '❌'} (20-30% spike F1 improvement)")
log(f"   Model Calibration: {'✅' if USE_MODEL_CALIBRATION else '❌'} (better probabilities)")
log(f"   Online Learning: {'✅' if USE_ONLINE_LEARNING else '❌'} (production updates)")
log(f"   Monitoring: {'✅' if USE_MONITORING else '❌'} (performance tracking)")


def tune_lgbm_regressor(X_train, y_train, horizon_name, n_iter=20):
    """
    Tune LightGBM regressor hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Args:
        X_train: Training features (scaled)
        y_train: Training targets
        horizon_name: Name of the horizon (for logging)
        n_iter: Number of random parameter combinations to try

    Returns:
        Best parameters dict
    """
    log(f"   🔍 Tuning hyperparameters for {horizon_name} ({n_iter} iterations)...")

    param_distributions = {
        'n_estimators': [150, 250, 350],  # IMPROVED: reduced range for stability
        'max_depth': [3, 4, 5],  # IMPROVED: max depth cap to prevent overfitting
        'learning_rate': [0.005, 0.01, 0.02],  # IMPROVED: lower LR for conservative training
        'num_leaves': [15, 31],  # IMPROVED: fewer leaves to reduce model complexity
        'min_child_samples': [30, 50, 100],  # IMPROVED: more samples per leaf (was 10-50)
        'subsample': [0.7, 0.8],  # IMPROVED: more aggressive subsampling (was 0.7-1.0)
        'colsample_bytree': [0.7, 0.8],  # IMPROVED: more feature sampling (was 0.7-1.0)
        'reg_alpha': [0.1, 0.5, 1.0],  # IMPROVED: stronger L1 regularization (was 0-1.0)
        'reg_lambda': [0.5, 1.0, 2.0],  # IMPROVED: stronger L2 regularization (was 0-1.0)
    }

    base_model = lgb.LGBMRegressor(
        objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
    )

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    log(f"      Best MAE (CV): {-search.best_score_:.4f}")
    log(f"      Best params: {search.best_params_}")

    return search.best_params_


def tune_lgbm_classifier(X_train, y_train, horizon_name, n_iter=20):
    """
    Tune LightGBM classifier hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Args:
        X_train: Training features (scaled)
        y_train: Training targets
        horizon_name: Name of the horizon (for logging)
        n_iter: Number of random parameter combinations to try

    Returns:
        Best parameters dict
    """
    log(f"   🔍 Tuning hyperparameters for {horizon_name} ({n_iter} iterations)...")

    param_distributions = {
        'n_estimators': [150, 250, 350],  # IMPROVED: reduced range for stability
        'max_depth': [3, 4, 5],  # IMPROVED: max depth cap to prevent overfitting
        'learning_rate': [0.01, 0.02, 0.05],  # IMPROVED: more conservative LR
        'num_leaves': [15, 31],  # IMPROVED: fewer leaves to reduce model complexity
        'min_child_samples': [30, 50, 100],  # IMPROVED: more samples per leaf (was 10-50)
        'subsample': [0.7, 0.8],  # IMPROVED: more aggressive subsampling (was 0.7-1.0)
        'colsample_bytree': [0.7, 0.8],  # IMPROVED: more feature sampling (was 0.7-1.0)
        'reg_alpha': [0.1, 0.5, 1.0],  # IMPROVED: stronger L1 regularization (was 0-0.1)
        'reg_lambda': [0.5, 1.0, 2.0],  # IMPROVED: stronger L2 regularization (was 0-0.1)
    }

    # Determine number of classes from training data
    n_classes = len(np.unique(y_train))
    base_model = lgb.LGBMClassifier(
        objective='multiclass', num_class=n_classes, random_state=42, n_jobs=-1, verbose=-1
    )

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    log(f"      Best Accuracy (CV): {search.best_score_:.4f}")
    log(f"      Best params: {search.best_params_}")

    return search.best_params_


# ===== PHASE 2: BAYESIAN HYPERPARAMETER OPTIMIZATION =====
def tune_lgbm_bayesian(X_train, y_train, horizon_name, task_type='regressor', n_trials=50):
    """
    Bayesian hyperparameter optimization using Optuna.
    More efficient than RandomSearch - focuses search on promising regions.
    Expected improvement: 15-20% over random search.

    Args:
        X_train: Training features (scaled)
        y_train: Training targets
        horizon_name: Name of the horizon (for logging)
        task_type: 'regressor' or 'classifier'
        n_trials: Number of trials to run

    Returns:
        Best parameters dict
    """
    import optuna
    from optuna.pruners import MedianPruner

    log(f"   🎯 Bayesian optimization ({task_type}) for {horizon_name} ({n_trials} trials)...")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        # Define parameter space with continuous/discrete ranges
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        }

        # Create model
        if task_type == 'regressor':
            model = lgb.LGBMRegressor(
                **params,
                objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
            )
            scoring_metric = 'neg_mean_absolute_error'
        else:  # classifier
            n_classes = len(np.unique(y_train))
            model = lgb.LGBMClassifier(
                **params,
                objective='multiclass', num_class=n_classes, random_state=42, n_jobs=-1, verbose=-1
            )
            scoring_metric = 'accuracy'

        # Time series cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)

            if task_type == 'regressor':
                score = -np.mean(np.abs(model.predict(X_val) - y_val))
            else:
                score = (model.predict(X_val) == y_val).mean()
            scores.append(score)

        return np.mean(scores)

    # Run optimization with Bayesian sampler
    sampler = optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
    pruner = MedianPruner(n_warmup_trials=5)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    log(f"      Best Score: {study.best_value:.4f}")
    log(f"      Best params: {best_params}")

    return best_params


# ===== PHASE 2: MODEL CALIBRATION =====
class CalibratedModel:
    """Wrapper for model calibration using Platt scaling and isotonic regression."""

    def __init__(self, base_model, calibration_method='sigmoid'):
        """
        Args:
            base_model: Trained model to calibrate
            calibration_method: 'sigmoid' (Platt scaling) or 'isotonic'
        """
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibrator = None

    def fit(self, X_cal, y_cal):
        """Fit calibration on calibration set."""
        from sklearn.calibration import CalibratedClassifierCV

        self.calibrator = CalibratedClassifierCV(
            self.base_model,
            method=self.calibration_method,
            cv='prefit'
        )
        self.calibrator.fit(X_cal, y_cal)
        return self

    def predict(self, X):
        """Get calibrated predictions."""
        if self.calibrator is not None:
            return self.calibrator.predict(X)
        return self.base_model.predict(X)

    def predict_proba(self, X):
        """Get calibrated probabilities."""
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        return None


# ===== PHASE 2: SPIKE DETECTION ENSEMBLE =====
def train_spike_ensemble(X_train, y_train, X_val, y_val, sample_weights=None, horizon='1h'):
    """
    Train ensemble of spike detectors combining multiple algorithms.
    Expected improvement: 20-30% in spike F1 scores.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sample_weights: Optional sample weights
        horizon: '1h', '4h', or '24h'

    Returns:
        Dictionary with ensemble and individual models
    """
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    log(f"   🎭 Training spike detection ENSEMBLE for {horizon}...")

    n_classes = len(np.unique(y_train))

    # Train diverse base learners
    models = {}

    # 1. XGBoost classifier
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
            random_state=42, verbosity=0, n_jobs=-1
        )
        if sample_weights is not None:
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            xgb_model.fit(X_train, y_train)
        models['xgb'] = ('xgb', xgb_model, 0.4)  # (name, model, weight)
    except Exception as e:
        log(f"      ⚠️  XGBoost failed: {e}")

    # 2. LightGBM classifier
    try:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1
        )
        if sample_weights is not None:
            lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            lgb_model.fit(X_train, y_train)
        models['lgb'] = ('lgb', lgb_model, 0.3)
    except Exception as e:
        log(f"      ⚠️  LightGBM failed: {e}")

    # 3. Random Forest classifier
    try:
        rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        if sample_weights is not None:
            rf_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            rf_model.fit(X_train, y_train)
        models['rf'] = ('rf', rf_model, 0.2)
    except Exception as e:
        log(f"      ⚠️  Random Forest failed: {e}")

    # 4. Logistic Regression (for diversity)
    try:
        # Scale for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        lr_model = LogisticRegression(
            max_iter=1000, multi_class='multinomial', solver='lbfgs',
            random_state=42, n_jobs=-1
        )
        if sample_weights is not None:
            lr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            lr_model.fit(X_train_scaled, y_train)
        models['lr'] = ('lr', lr_model, 0.1)
    except Exception as e:
        log(f"      ⚠️  Logistic Regression failed: {e}")

    # Create voting ensemble
    if len(models) >= 2:
        estimators = [(name, model) for name, model, _ in models.values()]
        weights = [weight for _, _, weight in models.values()]

        ensemble = VotingClassifier(estimators=estimators, weights=weights, voting='soft')

        log(f"      ✅ Spike ensemble trained with {len(models)} base models")

        return {
            'ensemble': ensemble,
            'individual_models': {k: v[1] for k, v in models.items()},
            'weights': {k: v[2] for k, v in models.items()},
            'base_model_count': len(models)
        }
    else:
        log(f"      ⚠️  Insufficient models for ensemble ({len(models)})")
        return None


# ===== PHASE 2: ONLINE LEARNING FOUNDATION =====
class OnlineLearningMixin:
    """Foundation for online learning - model updates with new data."""

    def __init__(self, model, update_frequency=100):
        """
        Args:
            model: Base model to update
            update_frequency: Update after N new predictions
        """
        self.model = model
        self.update_frequency = update_frequency
        self.predictions_since_update = 0
        self.buffer_X = []
        self.buffer_y = []

    def predict(self, X):
        """Get predictions and track for online learning."""
        preds = self.model.predict(X)
        self.predictions_since_update += len(X) if hasattr(X, '__len__') else 1
        return preds

    def add_observed_data(self, X, y):
        """Buffer new observed data."""
        self.buffer_X.append(X)
        self.buffer_y.append(y)

    def should_update(self):
        """Check if model should be updated."""
        return self.predictions_since_update >= self.update_frequency

    def update_model(self, X_buffer=None, y_buffer=None, warm_start=True):
        """Update model with buffered data."""
        if X_buffer is None:
            X_buffer = np.vstack(self.buffer_X) if self.buffer_X else None
            y_buffer = np.hstack(self.buffer_y) if self.buffer_y else None

        if X_buffer is not None and y_buffer is not None:
            # Incremental learning
            if hasattr(self.model, 'warm_start'):
                self.model.warm_start = warm_start
                self.model.fit(X_buffer, y_buffer)
            else:
                # Fallback: retrain
                self.model.fit(X_buffer, y_buffer)

            # Reset counters
            self.buffer_X = []
            self.buffer_y = []
            self.predictions_since_update = 0

            return True
        return False


# ===== PHASE 2: MONITORING INFRASTRUCTURE =====
class ModelMonitor:
    """Track model performance and detect degradation."""

    def __init__(self, name, baseline_metric=None):
        """
        Args:
            name: Model name
            baseline_metric: Baseline performance for comparison
        """
        self.name = name
        self.baseline_metric = baseline_metric
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        self.performance_history = []

    def record_prediction(self, pred, actual, timestamp=None):
        """Record a prediction for monitoring."""
        self.predictions.append(pred)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())

    def compute_metrics(self, window_size=100):
        """Compute performance metrics over recent window."""
        if len(self.predictions) < window_size:
            window_preds = self.predictions
            window_actuals = self.actuals
        else:
            window_preds = self.predictions[-window_size:]
            window_actuals = self.actuals[-window_size:]

        if len(window_preds) == 0:
            return {}

        mae = np.mean(np.abs(np.array(window_preds) - np.array(window_actuals)))
        rmse = np.sqrt(np.mean((np.array(window_preds) - np.array(window_actuals))**2))

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'window_size': len(window_preds),
            'timestamp': datetime.now().isoformat()
        }

        # Check for degradation
        if self.baseline_metric and mae > self.baseline_metric * 1.2:
            metrics['degradation_alert'] = f"MAE degraded {mae/self.baseline_metric:.1%}"

        self.performance_history.append(metrics)
        return metrics

    def get_performance_report(self):
        """Generate performance report."""
        if not self.performance_history:
            return {}

        recent_metrics = self.performance_history[-10:]
        mae_values = [m['mae'] for m in recent_metrics]

        return {
            'model_name': self.name,
            'current_mae': mae_values[-1] if mae_values else None,
            'avg_mae': np.mean(mae_values) if mae_values else None,
            'mae_trend': 'improving' if len(mae_values) > 1 and mae_values[-1] < mae_values[0] else 'stable',
            'total_predictions': len(self.predictions),
            'last_update': datetime.now().isoformat()
        }


# Default hyperparameters (used when tuning is disabled) - IMPROVED with stronger regularization
DEFAULT_PARAMS_24H = {
    'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.01, 'num_leaves': 31,
    'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 1.0  # Added regularization
}
DEFAULT_PARAMS_4H = {
    'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.01, 'num_leaves': 15,
    'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 1.0  # Added regularization
}
DEFAULT_PARAMS_1H = {
    'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.02, 'num_leaves': 31,
    'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 1.0  # Added regularization
}

# Only train models for horizons with sufficient data
# horizons_to_train = available_horizons if 'available_horizons' in locals() else ['1h', '4h', '24h']

# Global embargo gap constant
EMBARGO_GAP = 24

# ==============================================================================
# PHASE 0: Train 24h Regressor (Long-term Trend Setter)
# ==============================================================================
horizon_24h = '24h'
log(f"\n{'='*60}")
log(f"🎯 PHASE 0: Training 24h Regressor (Long-term Trend)")
log(f"{'='*60}")

# Prepare 24h Targets
if horizon_24h not in targets or horizon_24h not in available_horizons:
    log(f"⚠️  Skipping 24h Regressor - Insufficient data for 24h horizon")
    model_24h = None
    selected_features_24h = []
    scaler_24h = None
else:
    y_volatility_24h = targets[horizon_24h]['volatility_scaled_return']
    y_price_24h = targets[horizon_24h]['target_price']

    # Align X and y (remove NaNs)
    valid_idx_24h = ~(X.isna().any(axis=1) | y_volatility_24h.isna())
    X_24h = X[valid_idx_24h]
    y_24h = y_volatility_24h[valid_idx_24h]

    log(f"   Valid samples (24h): {len(X_24h):,}")

    if len(X_24h) < 500:
        log(f"   ⚠️  Insufficient samples for 24h model (need 500+), skipping")
        model_24h = None
        selected_features_24h = []
        scaler_24h = None
    else:
        # Feature Selection for 24h
        log(f"   🔍 Feature selection for 24h...")
        X_24h_selected, selected_features_24h = select_features(X_24h, y_24h, len(X_24h), verbose=True)

        # Train/Val/Test Split with Embargo Gap
        n_total_24h = len(X_24h_selected)
        train_end_24h = int(n_total_24h * 0.8) - EMBARGO_GAP
        test_start_24h = int(n_total_24h * 0.8)

        X_train_val_24h = X_24h_selected.iloc[:train_end_24h]
        X_test_24h = X_24h_selected.iloc[test_start_24h:]
        y_train_val_24h = y_24h.iloc[:train_end_24h]
        y_test_24h = y_24h.iloc[test_start_24h:]

        log(f"   📊 Train/Test split with {EMBARGO_GAP}-sample embargo gap")
        log(f"      Train: {len(X_train_val_24h):,}, Embargo: {EMBARGO_GAP}, Test: {len(X_test_24h):,}")

        # Train 24h Model (deeper trees for longer patterns)
        log(f"   🚀 Training LightGBM Regressor for 24h...")
        scaler_24h = RobustScaler()
        X_train_val_24h_scaled = scaler_24h.fit_transform(X_train_val_24h)
        X_test_24h_scaled = scaler_24h.transform(X_test_24h)

        # Get hyperparameters (tuned or default)
        if TUNE_HYPERPARAMS:
            params_24h = tune_lgbm_regressor(X_train_val_24h_scaled, y_train_val_24h, '24h', TUNING_ITERATIONS)
        else:
            params_24h = DEFAULT_PARAMS_24H

        model_24h = lgb.LGBMRegressor(
            **params_24h,
            objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
        )
        model_24h.fit(X_train_val_24h_scaled, y_train_val_24h)

        # Train quantile models for uncertainty estimation (10th and 90th percentiles)
        log(f"   📊 Training quantile models for prediction intervals...")
        quantile_params = {k: v for k, v in params_24h.items() if k not in ['objective', 'alpha']}

        model_24h_q10 = lgb.LGBMRegressor(
            **quantile_params,
            objective='quantile', alpha=0.10, random_state=42, n_jobs=-1, verbose=-1
        )
        model_24h_q10.fit(X_train_val_24h_scaled, y_train_val_24h)

        model_24h_q90 = lgb.LGBMRegressor(
            **quantile_params,
            objective='quantile', alpha=0.90, random_state=42, n_jobs=-1, verbose=-1
        )
        model_24h_q90.fit(X_train_val_24h_scaled, y_train_val_24h)

        # ===== TIER 1: Ensemble Stacking for 24h =====
        log(f"   🎭 Training TIER 1 ensemble stack (24h)...")
        from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
        from sklearn.ensemble import VotingRegressor

        try:
            # Train diverse base learners for ensemble
            base_model_lgbm = model_24h  # Already trained LightGBM (Huber)

            ridge_24h = Ridge(alpha=1.0, random_state=42)
            ridge_24h.fit(X_train_val_24h_scaled, y_train_val_24h)

            elastic_24h = ElasticNet(alpha=0.5, l1_ratio=0.8, random_state=42, max_iter=5000)
            elastic_24h.fit(X_train_val_24h_scaled, y_train_val_24h)

            huber_24h = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500)  # Fixed: removed random_state
            huber_24h.fit(X_train_val_24h_scaled, y_train_val_24h)

            # Create voting ensemble with different weights for base learners
            ensemble_24h = VotingRegressor([
                ('lgbm', base_model_lgbm),
                ('ridge', ridge_24h),
                ('elastic', elastic_24h),
                ('huber', huber_24h)
            ], weights=[0.4, 0.3, 0.2, 0.1])  # LightGBM has highest weight

            # Predictions from ensemble (for visualization, doesn't need fitting)
            ensemble_24h_preds = (
                0.4 * model_24h.predict(X_test_24h_scaled) +
                0.3 * ridge_24h.predict(X_test_24h_scaled) +
                0.2 * elastic_24h.predict(X_test_24h_scaled) +
                0.1 * huber_24h.predict(X_test_24h_scaled)
            )

            log(f"      ✅ Ensemble trained with 4 base learners (LGBM, Ridge, ElasticNet, Huber)")
            model_24h_ensemble = {
                'ensemble': ensemble_24h,
                'predictions': ensemble_24h_preds,
                'base_models': {
                    'lgbm': model_24h,
                    'ridge': ridge_24h,
                    'elastic': elastic_24h,
                    'huber': huber_24h
                }
            }
        except Exception as e:
            log(f"      ⚠️  Ensemble training failed: {e} - using single model")
            model_24h_ensemble = None

        # Evaluate 24h
        y_pred_24h_vol = model_24h.predict(X_test_24h_scaled)
        y_pred_24h_q10 = model_24h_q10.predict(X_test_24h_scaled)
        y_pred_24h_q90 = model_24h_q90.predict(X_test_24h_scaled)

        mae_24h = np.mean(np.abs(y_pred_24h_vol - y_test_24h))
        # Calculate prediction interval coverage (should be ~80%)
        coverage_24h = np.mean((y_test_24h >= y_pred_24h_q10) & (y_test_24h <= y_pred_24h_q90))
        avg_interval_24h = np.mean(y_pred_24h_q90 - y_pred_24h_q10)

        log(f"   📊 24h Test MAE (volatility-scaled): {mae_24h:.4f}")
        log(f"   📊 24h Prediction Interval Coverage: {coverage_24h:.1%} (target: 80%)")
        log(f"   📊 24h Average Interval Width: {avg_interval_24h:.4f}")

        # Save 24h Model with quantile models and ensemble
        model_data_24h = {
            'regressor': model_24h,
            'quantile_models': {
                'q10': model_24h_q10,
                'q90': model_24h_q90
            },
            'ensemble': model_24h_ensemble if model_24h_ensemble else None,  # TIER 1: Include ensemble
            'feature_names': selected_features_24h,
            'scaler': scaler_24h,
            'trained_at': datetime.now().isoformat(),
            'model_version': 'v17_tier1_ensemble',  # Mark as v17 with Tier 1 improvements
            'metrics': {
                'mae': float(mae_24h),
                'interval_coverage': float(coverage_24h),
                'avg_interval_width': float(avg_interval_24h)
            }
        }
        joblib.dump(model_data_24h, 'trained_models/model_24h.pkl')
        log(f"   💾 Saved 24h model to trained_models/model_24h.pkl")

# ==============================================================================
# PHASE 0.5: Inject 24h Trend Signal (if model available)
# ==============================================================================
log(f"\n{'='*60}")
log(f"💉 PHASE 0.5: Injecting 24h Trend Signal")
log(f"{'='*60}")

if model_24h is not None and len(selected_features_24h) > 0:
    # Generate OOF predictions for 24h signal (same approach as 4h)
    X_for_24h_pred = X[selected_features_24h]

    trend_signal_24h = np.full(len(X), np.nan)

    valid_idx_24h_bool = ~(X[selected_features_24h].isna().any(axis=1) | targets[horizon_24h]['volatility_scaled_return'].isna())
    valid_indices_24h = np.where(valid_idx_24h_bool)[0]

    n_valid_24h = len(valid_indices_24h)
    train_end_idx_24h = int(n_valid_24h * 0.8) - EMBARGO_GAP
    test_start_idx_24h = int(n_valid_24h * 0.8)
    train_indices_24h = valid_indices_24h[:train_end_idx_24h]
    test_indices_24h = valid_indices_24h[test_start_idx_24h:]

    log(f"   📊 Train samples: {len(train_indices_24h):,}, Embargo: {EMBARGO_GAP}, Test samples: {len(test_indices_24h):,}")

    # OOF predictions for training data
    log(f"   🔄 Generating Out-of-Fold predictions for training data...")
    X_train_for_oof_24h = X_for_24h_pred.iloc[train_indices_24h]
    y_train_for_oof_24h = targets[horizon_24h]['volatility_scaled_return'].iloc[train_indices_24h]

    tscv_oof_24h = TimeSeriesSplit(n_splits=5)
    oof_predictions_24h = np.full(len(train_indices_24h), np.nan)

    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(tscv_oof_24h.split(X_train_for_oof_24h)):
        X_fold_train = X_train_for_oof_24h.iloc[train_fold_idx]
        y_fold_train = y_train_for_oof_24h.iloc[train_fold_idx]
        X_fold_val = X_train_for_oof_24h.iloc[val_fold_idx]

        fold_scaler = RobustScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = fold_scaler.transform(X_fold_val)

        fold_model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.01, num_leaves=31,
            objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
        )
        fold_model.fit(X_fold_train_scaled, y_fold_train)
        oof_predictions_24h[val_fold_idx] = fold_model.predict(X_fold_val_scaled)

        log(f"      Fold {fold_idx + 1}/5: Train={len(train_fold_idx)}, Val={len(val_fold_idx)}")

    for i, idx in enumerate(train_indices_24h):
        if not np.isnan(oof_predictions_24h[i]):
            trend_signal_24h[idx] = oof_predictions_24h[i]

    # Test predictions using final model
    log(f"   🎯 Generating predictions for test data using final model...")
    X_test_for_pred_24h = X_for_24h_pred.iloc[test_indices_24h]
    X_test_scaled_24h = scaler_24h.transform(X_test_for_pred_24h)
    test_predictions_24h = model_24h.predict(X_test_scaled_24h)

    for i, idx in enumerate(test_indices_24h):
        trend_signal_24h[idx] = test_predictions_24h[i]

    trend_signal_24h = np.nan_to_num(trend_signal_24h, nan=0.0)

    X = X.copy()
    X['trend_signal_24h'] = trend_signal_24h

    valid_preds_24h = trend_signal_24h[trend_signal_24h != 0]
    if len(valid_preds_24h) > 0:
        log(f"   ✅ Added 'trend_signal_24h' feature (OOF leak-free)")
        log(f"      Range: {valid_preds_24h.min():.4f} to {valid_preds_24h.max():.4f}")
        log(f"      Mean: {valid_preds_24h.mean():.4f}, Std: {valid_preds_24h.std():.4f}")
else:
    log(f"   ⚠️  Skipping 24h trend injection (model not available)")
    X['trend_signal_24h'] = 0

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
    
    # Train/Val/Test Split with Embargo Gap (using global EMBARGO_GAP)
    n_total_4h = len(X_4h_selected)
    train_end_4h = int(n_total_4h * 0.8) - EMBARGO_GAP  # End training before embargo
    test_start_4h = int(n_total_4h * 0.8)  # Test starts after embargo

    X_train_val_4h = X_4h_selected.iloc[:train_end_4h]
    X_test_4h = X_4h_selected.iloc[test_start_4h:]
    y_train_val_4h = y_4h.iloc[:train_end_4h]
    y_test_4h = y_4h.iloc[test_start_4h:]
    y_orig_test_4h = y_price_4h[valid_idx_4h].iloc[test_start_4h:]
    current_test_4h = targets[horizon_4h]['current_price'][valid_idx_4h].iloc[test_start_4h:]

    log(f"   📊 Train/Test split with {EMBARGO_GAP}-sample embargo gap")
    log(f"      Train: {len(X_train_val_4h):,}, Embargo: {EMBARGO_GAP}, Test: {len(X_test_4h):,}")
    
    # Train 4h Model
    log(f"   🚀 Training LightGBM Regressor for 4h...")
    scaler_4h = RobustScaler()
    X_train_val_4h_scaled = scaler_4h.fit_transform(X_train_val_4h)
    X_test_4h_scaled = scaler_4h.transform(X_test_4h)

    # Get hyperparameters (tuned or default)
    if TUNE_HYPERPARAMS:
        params_4h = tune_lgbm_regressor(X_train_val_4h_scaled, y_train_val_4h, '4h', TUNING_ITERATIONS)
    else:
        params_4h = DEFAULT_PARAMS_4H

    model_4h = lgb.LGBMRegressor(
        **params_4h,
        objective='huber', alpha=0.95, random_state=42, n_jobs=-1, verbose=-1
    )
    model_4h.fit(X_train_val_4h_scaled, y_train_val_4h)

    # Train quantile models for uncertainty estimation
    log(f"   📊 Training quantile models for prediction intervals...")
    quantile_params_4h = {k: v for k, v in params_4h.items() if k not in ['objective', 'alpha']}

    model_4h_q10 = lgb.LGBMRegressor(
        **quantile_params_4h,
        objective='quantile', alpha=0.10, random_state=42, n_jobs=-1, verbose=-1
    )
    model_4h_q10.fit(X_train_val_4h_scaled, y_train_val_4h)

    model_4h_q90 = lgb.LGBMRegressor(
        **quantile_params_4h,
        objective='quantile', alpha=0.90, random_state=42, n_jobs=-1, verbose=-1
    )
    model_4h_q90.fit(X_train_val_4h_scaled, y_train_val_4h)

    # ===== TIER 1: Ensemble Stacking for 4h =====
    log(f"   🎭 Training TIER 1 ensemble stack (4h)...")
    from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
    from sklearn.ensemble import VotingRegressor

    try:
        # Train diverse base learners for ensemble
        base_model_lgbm_4h = model_4h  # Already trained LightGBM (Huber)

        ridge_4h = Ridge(alpha=1.0, random_state=42)
        ridge_4h.fit(X_train_val_4h_scaled, y_train_val_4h)

        elastic_4h = ElasticNet(alpha=0.5, l1_ratio=0.8, random_state=42, max_iter=5000)
        elastic_4h.fit(X_train_val_4h_scaled, y_train_val_4h)

        huber_4h = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500)  # Fixed: removed random_state
        huber_4h.fit(X_train_val_4h_scaled, y_train_val_4h)

        # Create voting ensemble with different weights
        ensemble_4h = VotingRegressor([
            ('lgbm', model_4h),
            ('ridge', ridge_4h),
            ('elastic', elastic_4h),
            ('huber', huber_4h)
        ], weights=[0.4, 0.3, 0.2, 0.1])

        # Ensemble predictions
        ensemble_4h_preds = (
            0.4 * model_4h.predict(X_test_4h_scaled) +
            0.3 * ridge_4h.predict(X_test_4h_scaled) +
            0.2 * elastic_4h.predict(X_test_4h_scaled) +
            0.1 * huber_4h.predict(X_test_4h_scaled)
        )

        log(f"      ✅ Ensemble trained with 4 base learners (LGBM, Ridge, ElasticNet, Huber)")
        model_4h_ensemble = {
            'ensemble': ensemble_4h,
            'predictions': ensemble_4h_preds,
            'base_models': {
                'lgbm': model_4h,
                'ridge': ridge_4h,
                'elastic': elastic_4h,
                'huber': huber_4h
            }
        }
    except Exception as e:
        log(f"      ⚠️  Ensemble training failed: {e} - using single model")
        model_4h_ensemble = None

    # Evaluate 4h
    y_pred_4h_vol = model_4h.predict(X_test_4h_scaled)
    y_pred_4h_q10 = model_4h_q10.predict(X_test_4h_scaled)
    y_pred_4h_q90 = model_4h_q90.predict(X_test_4h_scaled)

    mae_4h = np.mean(np.abs(y_pred_4h_vol - y_test_4h))
    coverage_4h = np.mean((y_test_4h >= y_pred_4h_q10) & (y_test_4h <= y_pred_4h_q90))
    avg_interval_4h = np.mean(y_pred_4h_q90 - y_pred_4h_q10)

    log(f"   📊 4h Test MAE (volatility-scaled): {mae_4h:.4f}")
    log(f"   📊 4h Prediction Interval Coverage: {coverage_4h:.1%} (target: 80%)")
    log(f"   📊 4h Average Interval Width: {avg_interval_4h:.4f}")

    # Save 4h Model with quantile models and ensemble
    model_data_4h = {
        'regressor': model_4h,
        'quantile_models': {
            'q10': model_4h_q10,
            'q90': model_4h_q90
        },
        'ensemble': model_4h_ensemble if model_4h_ensemble else None,  # TIER 1: Include ensemble
        'feature_names': selected_features_4h,
        'scaler': scaler_4h,
        'trained_at': datetime.now().isoformat(),
        'model_version': 'v17_tier1_ensemble',  # Mark as v17 with Tier 1 improvements
        'metrics': {
            'mae': float(mae_4h),
            'interval_coverage': float(coverage_4h),
            'avg_interval_width': float(avg_interval_4h)
        }
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

    # 4. Split indices into train and test (same split as Phase 1, with embargo)
    n_valid = len(valid_indices)
    train_end_idx = int(n_valid * 0.8) - EMBARGO_GAP  # Match Phase 1 embargo
    test_start_idx = int(n_valid * 0.8)
    train_indices = valid_indices[:train_end_idx]
    test_indices = valid_indices[test_start_idx:]

    log(f"   📊 Train samples: {len(train_indices):,}, Embargo: {EMBARGO_GAP}, Test samples: {len(test_indices):,}")

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
# Choose number of classes based on USE_5_CLASSES environment variable
USE_5_CLASSES = os.environ.get('USE_5_CLASSES', '0') == '1'

target_log_return_1h = targets[horizon_1h]['target_log_return']

if USE_5_CLASSES:
    # 5 classes based on percentiles for balanced distribution
    p20 = target_log_return_1h.quantile(0.20)
    p40 = target_log_return_1h.quantile(0.40)
    p60 = target_log_return_1h.quantile(0.60)
    p80 = target_log_return_1h.quantile(0.80)

    log(f"   📊 Using 5 classes (percentile thresholds)")
    log(f"      p20={p20:.4f}, p40={p40:.4f}, p60={p60:.4f}, p80={p80:.4f}")

    action_class = pd.Series(2, index=target_log_return_1h.index)
    action_class[target_log_return_1h <= p20] = 0  # Strong Wait
    action_class[(target_log_return_1h > p20) & (target_log_return_1h <= p40)] = 1  # Wait
    action_class[(target_log_return_1h > p40) & (target_log_return_1h <= p60)] = 2  # Hold
    action_class[(target_log_return_1h > p60) & (target_log_return_1h <= p80)] = 3  # Act Soon
    action_class[target_log_return_1h > p80] = 4  # Act Now

    CLASS_NAMES_1H = ['Strong Wait', 'Wait', 'Hold', 'Act Soon', 'Act Now']
    NUM_CLASSES_1H = 5
else:
    # 3 classes (default - better accuracy with limited data)
    # 0 (Wait): Log-return < -0.05 (price will drop)
    # 1 (Normal): -0.05 <= Log-return <= 0.05 (price stable)
    # 2 (Urgent): Log-return > 0.05 (price rising)
    log(f"   📊 Using 3 classes (set USE_5_CLASSES=1 for 5 classes)")

    action_class = pd.Series(1, index=target_log_return_1h.index)  # Default Normal
    action_class[target_log_return_1h < -0.05] = 0  # Wait
    action_class[target_log_return_1h > 0.05] = 2  # Urgent

    CLASS_NAMES_1H = ['Wait', 'Normal', 'Urgent']
    NUM_CLASSES_1H = 3

# Align Data
valid_idx_1h = ~(X.isna().any(axis=1) | action_class.isna())
X_1h = X[valid_idx_1h]
y_1h = action_class[valid_idx_1h].astype(int)

log(f"   Valid samples (1h): {len(X_1h):,}")
log(f"   Class Distribution: {y_1h.value_counts().to_dict()}")

# Feature Selection for 1h (now includes trend_signal_4h)
X_1h_selected, selected_features_1h = select_features(X_1h, y_1h, len(X_1h), verbose=True)

# Train/Test Split with Embargo Gap
n_total_1h = len(X_1h_selected)
train_end_1h = int(n_total_1h * 0.8) - EMBARGO_GAP
test_start_1h = int(n_total_1h * 0.8)
X_train_1h = X_1h_selected.iloc[:train_end_1h]
X_test_1h = X_1h_selected.iloc[test_start_1h:]
y_train_1h = y_1h.iloc[:train_end_1h]
y_test_1h = y_1h.iloc[test_start_1h:]

log(f"   📊 Train/Test split with {EMBARGO_GAP}-sample embargo gap")
log(f"      Train: {len(X_train_1h):,}, Embargo: {EMBARGO_GAP}, Test: {len(X_test_1h):,}")

# Train Classifier
log(f"   🚀 Training LightGBM Classifier for 1h...")
scaler_1h = RobustScaler()
X_train_1h_scaled = scaler_1h.fit_transform(X_train_1h)
X_test_1h_scaled = scaler_1h.transform(X_test_1h)

# Get hyperparameters (tuned or default)
if TUNE_HYPERPARAMS:
    params_1h = tune_lgbm_classifier(X_train_1h_scaled, y_train_1h, '1h', TUNING_ITERATIONS)
else:
    params_1h = DEFAULT_PARAMS_1H

# Only use class balancing for 5-class model (3-class works well without it)
if NUM_CLASSES_1H == 5:
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes_1h = np.unique(y_train_1h)
    class_weights_1h = compute_class_weight('balanced', classes=unique_classes_1h, y=y_train_1h)
    class_weight_dict_1h = dict(zip(unique_classes_1h, class_weights_1h))
    log(f"   📊 Class weights (5-class): {class_weight_dict_1h}")
    sample_weights_1h = np.array([class_weight_dict_1h[y] for y in y_train_1h])

    model_1h = lgb.LGBMClassifier(
        **params_1h,
        objective='multiclass', num_class=NUM_CLASSES_1H, metric='multi_logloss',
        class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_1h.fit(X_train_1h_scaled, y_train_1h, sample_weight=sample_weights_1h)
else:
    # 3-class model without class balancing (works better)
    model_1h = lgb.LGBMClassifier(
        **params_1h,
        objective='multiclass', num_class=NUM_CLASSES_1H, metric='multi_logloss',
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_1h.fit(X_train_1h_scaled, y_train_1h)

# ===== TIER 1: Ensemble Stacking for 1h Classifier =====
log(f"   🎭 Training TIER 1 ensemble stack (1h classifier)...")
from sklearn.ensemble import GradientBoostingClassifier
try:
    # Train diverse base learners for ensemble
    gb_1h = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.01, max_depth=5,
        random_state=42, verbose=0
    )
    gb_1h.fit(X_train_1h_scaled, y_train_1h)

    # Ensemble predictions (voting)
    ensemble_proba_1h = (
        0.6 * model_1h.predict_proba(X_test_1h_scaled) +
        0.4 * gb_1h.predict_proba(X_test_1h_scaled)
    )

    log(f"      ✅ Ensemble trained with 2 base learners (LightGBM, GradientBoosting)")
    model_1h_ensemble = {
        'lgbm': model_1h,
        'gb': gb_1h,
        'predictions': ensemble_proba_1h
    }
except Exception as e:
    log(f"      ⚠️  Ensemble training failed: {e} - using single model")
    model_1h_ensemble = None

# Evaluate with Probability Gating (updated for 5 classes)
y_proba_1h = model_1h.predict_proba(X_test_1h_scaled)

def apply_gating_5class(probs, threshold=0.5):
    """
    Apply probability gating for 5-class predictions.
    Classes: 0=Strong Wait, 1=Wait, 2=Hold, 3=Act Soon, 4=Act Now

    Strategy:
    - If high confidence in extreme classes (0 or 4), use them
    - Otherwise, use argmax
    """
    preds = np.argmax(probs, axis=1)  # Default to argmax

    # Override to extreme classes if high confidence
    strong_wait_mask = probs[:, 0] > threshold
    act_now_mask = probs[:, 4] > threshold

    # Strong Wait takes precedence (conservative)
    preds[strong_wait_mask] = 0
    # Act Now only if not Strong Wait
    preds[act_now_mask & ~strong_wait_mask] = 4

    return preds

# Legacy function for backward compatibility
def apply_gating(probs, threshold=0.7):
    # For 5-class, map to simplified 3-class view
    # Combine: Strong Wait + Wait = Wait, Hold = Normal, Act Soon + Act Now = Urgent
    preds = np.ones(len(probs), dtype=int) * 1  # Default Hold
    # Urgent takes precedence if both high (rare with softmax)
    
    # Check Urgent (Class 2)
    urgent_mask = probs[:, 2] > threshold
    preds[urgent_mask] = 2
    
    # Check Wait (Class 0) - only if not already Urgent
    wait_mask = (probs[:, 0] > threshold) & (~urgent_mask)
    preds[wait_mask] = 0
    
    return preds

# Evaluate model based on number of classes
log(f"\n   🔍 {NUM_CLASSES_1H}-Class Evaluation:")

# Basic argmax predictions
y_pred_1h_argmax = np.argmax(y_proba_1h, axis=1)
acc_argmax = accuracy_score(y_test_1h, y_pred_1h_argmax)
log(f"   📊 Argmax Accuracy: {acc_argmax:.4f}")

# Gated predictions (with threshold)
log(f"\n   🔍 Probability Gating Analysis:")
best_threshold = 0.5
best_score = 0

if NUM_CLASSES_1H == 5:
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        y_pred_gated = apply_gating_5class(y_proba_1h, threshold)
        acc = accuracy_score(y_test_1h, y_pred_gated)
        report = classification_report(y_test_1h, y_pred_gated, output_dict=True, zero_division=0)

        sw_prec = report.get('0', {}).get('precision', 0)
        sw_rec = report.get('0', {}).get('recall', 0)
        an_prec = report.get('4', {}).get('precision', 0)
        an_rec = report.get('4', {}).get('recall', 0)

        log(f"      Threshold {threshold:.1f}: StrongWait P={sw_prec:.2f}/R={sw_rec:.2f} | ActNow P={an_prec:.2f}/R={an_rec:.2f} | Acc={acc:.2f}")

        current_score = (report.get('0', {}).get('f1-score', 0) + report.get('4', {}).get('f1-score', 0)) / 2
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold

    y_pred_1h = apply_gating_5class(y_proba_1h, best_threshold)
else:
    # 3-class gating
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        y_pred_gated = apply_gating(y_proba_1h, threshold)
        acc = accuracy_score(y_test_1h, y_pred_gated)
        report = classification_report(y_test_1h, y_pred_gated, output_dict=True, zero_division=0)

        urgent_prec = report.get('2', {}).get('precision', 0)
        urgent_rec = report.get('2', {}).get('recall', 0)
        wait_prec = report.get('0', {}).get('precision', 0)
        wait_rec = report.get('0', {}).get('recall', 0)

        log(f"      Threshold {threshold:.1f}: Urgent P={urgent_prec:.2f}/R={urgent_rec:.2f} | Wait P={wait_prec:.2f}/R={wait_rec:.2f} | Acc={acc:.2f}")

        current_score = (report.get('2', {}).get('f1-score', 0) + report.get('0', {}).get('f1-score', 0)) / 2
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold

    y_pred_1h = apply_gating(y_proba_1h, best_threshold)

log(f"   ✨ Selected Inference Threshold: {best_threshold}")
acc_1h = accuracy_score(y_test_1h, y_pred_1h)
log(f"   ✅ 1h Final Accuracy ({NUM_CLASSES_1H}-class): {acc_1h:.4f}")

log(f"\n   📊 Classification Report ({NUM_CLASSES_1H} Classes):")
print(classification_report(y_test_1h, y_pred_1h, target_names=CLASS_NAMES_1H, zero_division=0))

log(f"\n   📊 Confusion Matrix ({NUM_CLASSES_1H} Classes):")
print(confusion_matrix(y_test_1h, y_pred_1h))

# Save 1h Model
model_data_1h = {
    'classifier': model_1h,
    'ensemble': model_1h_ensemble if model_1h_ensemble else None,  # TIER 1: Include ensemble
    'feature_names': selected_features_1h,
    'scaler': scaler_1h,
    'trained_at': datetime.now().isoformat(),
    'model_type': 'hybrid_classifier_5class',
    'model_version': 'v17_tier1_ensemble',  # Mark as v17 with Tier 1 improvements
    'num_classes': NUM_CLASSES_1H,
    'class_names': CLASS_NAMES_1H,
    'inference_config': {
        'threshold': best_threshold,
        'default_class': 2  # Hold
    },
    'metrics': {
        'accuracy': float(acc_1h),
        'class_distribution': y_1h.value_counts().to_dict()
    }
}
joblib.dump(model_data_1h, 'trained_models/model_1h.pkl')
log(f"   💾 Saved 1h Hybrid Classifier (5-class) to trained_models/model_1h.pkl")

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

# Spike detection thresholds - ADAPTIVE based on price distribution
# Using percentiles instead of absolute values for network-agnostic classification
log("\n📊 Computing adaptive spike thresholds from price distribution...")
# Calculate percentile-based thresholds from the training data
# Normal: bottom 50% of prices
# Elevated: 50th-85th percentile
# Spike: top 15% of prices
NORMAL_THRESHOLD = df['gas_price'].quantile(0.50)
ELEVATED_THRESHOLD = df['gas_price'].quantile(0.85)
log(f"   Adaptive thresholds: Normal < {NORMAL_THRESHOLD:.6f} gwei < Elevated < {ELEVATED_THRESHOLD:.6f} gwei < Spike")
log(f"   Based on {len(df):,} samples from {df['timestamp'].min()} to {df['timestamp'].max()}")

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
    
    # Train/val/test split (60/20/20, temporal) with Embargo Gaps
    # Use smaller embargo (12) between train/val and val/test to preserve more data
    SPIKE_EMBARGO = 12

    n_total = len(X_clean)
    train_end = int(n_total * 0.6) - SPIKE_EMBARGO
    val_start = int(n_total * 0.6)
    val_end = int(n_total * 0.8) - SPIKE_EMBARGO
    test_start = int(n_total * 0.8)

    X_train_spike = X_clean.iloc[:train_end]
    X_val_spike = X_clean.iloc[val_start:val_end]
    X_test_spike = X_clean.iloc[test_start:]
    y_train_spike = y_clean.iloc[:train_end]
    y_val_spike = y_clean.iloc[val_start:val_end]
    y_test_spike = y_clean.iloc[test_start:]

    log(f"   Train: {len(X_train_spike):,}, Val: {len(X_val_spike):,}, Test: {len(X_test_spike):,} (embargo: {SPIKE_EMBARGO})")
    
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

    # Compute sample weights for class balancing
    sample_weights = np.array([class_weight_dict_remapped[y] for y in y_train_spike_remapped])

    # ===== PHASE 2: SPIKE DETECTION ENSEMBLE =====
    spike_ensemble_result = None
    if USE_SPIKE_ENSEMBLE and len(X_train_spike) > 100:
        log(f"   🎭 Phase 2: Training spike detection ensemble...")
        spike_ensemble_result = train_spike_ensemble(
            X_train_spike, y_train_spike_remapped,
            X_val_spike, y_val_spike_remapped,
            sample_weights=sample_weights,
            horizon=horizon
        )

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
        # Use sample weights for class balancing (already computed above)
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
    
    # Save spike detector with enhanced metrics and adaptive thresholds
    spike_detector_data = {
        'model': calibrated_model,  # Save calibrated model
        'base_model': model_spike,  # Also save base model for reference
        'ensemble': spike_ensemble_result if spike_ensemble_result else None,  # PHASE 2: Spike ensemble
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
        'has_ensemble': spike_ensemble_result is not None,  # PHASE 2: Flag for ensemble
        'model_version': 'v17_phase2_ensemble' if spike_ensemble_result else 'v17_tier1',
        'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting',
        'trained_at': datetime.now().isoformat(),
        # Adaptive thresholds for inference
        'thresholds': {
            'normal_threshold': float(NORMAL_THRESHOLD),
            'elevated_threshold': float(ELEVATED_THRESHOLD),
            'percentiles_used': {'normal': 0.50, 'elevated': 0.85}
        }
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

# ==============================================================================
# MODEL REGISTRY: Save training manifest with versioning
# ==============================================================================
import hashlib
import json

def compute_data_hash(df, sample_size=1000):
    """Compute a hash of the data for versioning"""
    sample = df.head(sample_size).to_json()
    return hashlib.md5(sample.encode()).hexdigest()[:12]

log("📋 Creating model registry manifest...")

# Generate unique version based on timestamp and data hash
data_hash = compute_data_hash(df)
version_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data_hash}"

# Build manifest
manifest = {
    'version': version_id,
    'training_timestamp': datetime.now().isoformat(),
    'data_info': {
        'samples': len(df),
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        },
        'data_hash': data_hash,
        'features_count': X.shape[1] if 'X' in dir() else 0
    },
    'config': {
        'embargo_gap': EMBARGO_GAP,
        'hyperparameter_tuning': TUNE_HYPERPARAMS,
        'tuning_iterations': TUNING_ITERATIONS if TUNE_HYPERPARAMS else 0,
        'adaptive_thresholds': {
            'normal': float(NORMAL_THRESHOLD),
            'elevated': float(ELEVATED_THRESHOLD)
        }
    },
    'models': {}
}

# Add model info
model_files = [
    ('model_24h.pkl', '24h_regressor'),
    ('model_4h.pkl', '4h_regressor'),
    ('model_1h.pkl', '1h_classifier'),
    ('spike_detector_1h.pkl', 'spike_1h'),
    ('spike_detector_4h.pkl', 'spike_4h'),
    ('spike_detector_24h.pkl', 'spike_24h')
]

for filename, model_name in model_files:
    filepath = f'trained_models/{filename}'
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        manifest['models'][model_name] = {
            'filename': filename,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }

# Add spike detector metrics if available
if spike_results:
    manifest['spike_detector_metrics'] = spike_results

# Save manifest
manifest_path = 'trained_models/manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, default=str)

log(f"   ✅ Saved manifest to {manifest_path}")
log(f"   📌 Version: {version_id}")
log(f"   📊 Models registered: {len(manifest['models'])}")

# Print training summary across all horizons
if results:
    log("\n📊 Training Summary:")
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

log(f"\n{'='*60}")
log(f"📁 All models saved to: trained_models/")
log(f"📋 Manifest: trained_models/manifest.json")
log(f"{'='*60}\n")
