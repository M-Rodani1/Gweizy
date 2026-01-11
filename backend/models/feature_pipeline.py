import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import logging

from models.advanced_features import create_advanced_features

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v2"  # Updated for mempool features


def normalize_gas_dataframe(records: Union[Sequence[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    """Normalize raw gas records into a DataFrame with timestamp and gas_price."""
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)

    if 'timestamp' not in df.columns:
        raise ValueError("Missing timestamp column in records")

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    if 'gas_price' not in df.columns:
        for key in ('gwei', 'current_gas', 'gas'):
            if key in df.columns:
                df['gas_price'] = df[key]
                break

    if 'gas_price' not in df.columns:
        raise ValueError("Missing gas price column in records")

    df['gas_price'] = pd.to_numeric(df['gas_price'], errors='coerce')
    df = df.dropna(subset=['gas_price'])

    return df.sort_values('timestamp').reset_index(drop=True)


def detect_sample_rate_minutes(df: pd.DataFrame) -> Optional[float]:
    """Estimate sample rate in minutes from timestamp deltas."""
    if len(df) < 2 or 'timestamp' not in df.columns:
        return None

    diffs = df['timestamp'].diff().dropna()
    if diffs.empty:
        return None

    median_seconds = diffs.dt.total_seconds().median()
    if not median_seconds or median_seconds <= 0:
        return None

    return float(median_seconds / 60)


def get_steps_per_hour(sample_rate_minutes: Optional[float], default_steps: int = 12) -> int:
    """Convert sample rate to steps per hour (defaults to 12 for 5-min data)."""
    if not sample_rate_minutes:
        return default_steps
    steps = int(round(60 / sample_rate_minutes))
    return max(1, steps)


def add_external_features(df: pd.DataFrame, steps_per_hour: int) -> pd.DataFrame:
    """Add lightweight external-style features derived from gas_price."""
    df = df.copy()

    df['estimated_block_time'] = 2.0

    window = max(1, steps_per_hour)
    rolling_mean = df['gas_price'].rolling(window=window, min_periods=1).mean()
    recent_volatility = df['gas_price'].rolling(window=window, min_periods=1).std().fillna(0)

    df['recent_volatility'] = recent_volatility
    df['congestion_score'] = (recent_volatility / rolling_mean.replace(0, np.nan)).fillna(0)
    df['gas_change'] = df['gas_price'].diff().abs().fillna(0)

    spike_threshold = df['gas_price'].quantile(0.9) if len(df) > 1 else df['gas_price'].iloc[0]
    time_since_spike = []
    counter = 0
    for price in df['gas_price'].fillna(0):
        if price > spike_threshold:
            counter = 0
        else:
            counter += 1
        time_since_spike.append(counter)
    df['time_since_spike'] = time_since_spike

    df['momentum_1h'] = df['gas_price'].pct_change(steps_per_hour).fillna(0)
    df['momentum_4h'] = df['gas_price'].pct_change(steps_per_hour * 4).fillna(0)

    return df


def add_mempool_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add mempool-derived features as leading indicators for gas price prediction.

    Mempool features provide real-time insight into network demand before
    it's reflected in gas prices.
    """
    df = df.copy()

    try:
        from data.mempool_collector import get_mempool_collector
        collector = get_mempool_collector()
        mempool_features = collector.get_current_features()

        # Add mempool features to the last row (most recent observation)
        for feature_name, value in mempool_features.items():
            df[feature_name] = 0.0  # Default for historical rows
            df.loc[df.index[-1], feature_name] = value

        # Forward-fill mempool features for consistency
        mempool_cols = [c for c in df.columns if c.startswith('mempool_')]
        df[mempool_cols] = df[mempool_cols].ffill()

        logger.debug(f"Added {len(mempool_features)} mempool features")

    except Exception as e:
        logger.debug(f"Mempool features unavailable: {e}")
        # Add placeholder columns with default values
        default_mempool_features = {
            'mempool_pending_count': 0,
            'mempool_avg_gas_price': 0,
            'mempool_median_gas_price': 0,
            'mempool_p90_gas_price': 0,
            'mempool_gas_price_spread': 0,
            'mempool_large_tx_ratio': 0,
            'mempool_is_congested': 0,
            'mempool_arrival_rate': 0,
            'mempool_count_momentum': 0,
            'mempool_gas_momentum': 0
        }
        for feature_name, value in default_mempool_features.items():
            df[feature_name] = value

    return df


def build_feature_matrix(
    records: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    include_external_features: bool = True,
    include_mempool_features: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Build the feature matrix and metadata from raw records.

    Args:
        records: Raw gas price records with timestamp and price
        include_external_features: Add derived features like volatility, momentum
        include_mempool_features: Add mempool-derived leading indicators

    Returns:
        Tuple of (feature_matrix, metadata_dict, enriched_dataframe)
    """
    df = normalize_gas_dataframe(records)

    sample_rate_minutes = detect_sample_rate_minutes(df)
    steps_per_hour = get_steps_per_hour(sample_rate_minutes)

    if include_external_features:
        df = add_external_features(df, steps_per_hour)

    if include_mempool_features:
        df = add_mempool_features(df)

    features, _ = create_advanced_features(df)

    metadata = {
        'pipeline_version': PIPELINE_VERSION,
        'feature_names': list(features.columns),
        'sample_rate_minutes': sample_rate_minutes,
        'steps_per_hour': steps_per_hour,
        'external_features': include_external_features,
        'mempool_features': include_mempool_features
    }

    return features, metadata, df


def build_horizon_targets(
    gas_series: pd.Series,
    steps_per_hour: int,
    horizons: Tuple[int, ...] = (1, 4, 24)
) -> Dict[str, pd.Series]:
    """Create shifted targets for each horizon (in hours)."""
    targets = {}
    for horizon in horizons:
        targets[f'{horizon}h'] = gas_series.shift(-(steps_per_hour * horizon))
    return targets
