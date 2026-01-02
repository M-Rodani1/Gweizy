"""
State representation for the RL environment.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GasState:
    """Represents the current state of the gas market."""
    current_price: float
    price_history: List[float]
    hour: int
    day_of_week: int
    volatility: float
    momentum: float
    urgency: float
    time_waiting: int


class StateBuilder:
    """Builds state vectors for the RL agent."""

    def __init__(self, history_length: int = 24):
        self.history_length = history_length
        self.state_dim = 1 + history_length + 4 + 3 + 2  # 34 features

    def build_state(self, gas_state: GasState, price_stats: Dict) -> np.ndarray:
        """Convert GasState to normalized numpy array with improved normalization."""
        features = []
        mean_price = price_stats.get('mean', 0.001)
        std_price = price_stats.get('std', 0.0005) or 0.0005
        min_price = price_stats.get('min', mean_price * 0.5)
        max_price = price_stats.get('max', mean_price * 2.0)

        # Current price (normalized using robust scaling)
        # Use robust scaling: (x - median) / IQR for better outlier handling
        median_price = price_stats.get('median', mean_price)
        iqr = price_stats.get('iqr', std_price * 1.5) or std_price * 1.5
        if iqr > 1e-8:
            normalized_price = (gas_state.current_price - median_price) / iqr
        else:
            normalized_price = (gas_state.current_price - mean_price) / (std_price + 1e-8)
        features.append(np.clip(normalized_price, -3, 3) / 3.0)  # Normalize to [-1, 1]

        # Price history (normalized with consistent scaling)
        history = np.array(gas_state.price_history[-self.history_length:])
        if len(history) < self.history_length:
            padding = np.full(self.history_length - len(history), gas_state.current_price)
            history = np.concatenate([padding, history])
        
        # Normalize history using same method
        if iqr > 1e-8:
            normalized_history = (history - median_price) / iqr
        else:
            normalized_history = (history - mean_price) / (std_price + 1e-8)
        normalized_history = np.clip(normalized_history, -3, 3) / 3.0  # Normalize to [-1, 1]
        features.extend(normalized_history.tolist())

        # Time features (cyclical - already in [-1, 1] range)
        features.extend([
            np.sin(2 * np.pi * gas_state.hour / 24),
            np.cos(2 * np.pi * gas_state.hour / 24),
            np.sin(2 * np.pi * gas_state.day_of_week / 7),
            np.cos(2 * np.pi * gas_state.day_of_week / 7)
        ])

        # Technical indicators (normalized to [-1, 1])
        # Volatility: normalize by typical volatility range
        typical_volatility = price_stats.get('typical_volatility', 0.1) or 0.1
        vol_normalized = np.clip(gas_state.volatility / typical_volatility, 0, 2) - 1.0  # [-1, 1]
        features.append(vol_normalized)
        
        # Momentum: already in reasonable range, just clip
        features.append(np.clip(gas_state.momentum, -1, 1))
        
        # Percentile rank: normalize to [-1, 1] (0.5 becomes 0)
        percentile = price_stats.get('percentile_rank', None)
        if percentile is None:
            percentile = (gas_state.current_price - min_price) / (max_price - min_price + 1e-8)
        percentile_normalized = 2 * np.clip(percentile, 0, 1) - 1.0  # [0, 1] -> [-1, 1]
        features.append(percentile_normalized)

        # Urgency and time waiting (normalized to [-1, 1])
        features.append(2 * gas_state.urgency - 1.0)  # [0, 1] -> [-1, 1]
        time_waiting_norm = min(gas_state.time_waiting / 100.0, 1.0)
        features.append(2 * time_waiting_norm - 1.0)  # [0, 1] -> [-1, 1]

        return np.array(features, dtype=np.float32)

    def get_state_dim(self) -> int:
        return self.state_dim
