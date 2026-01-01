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
        """Convert GasState to normalized numpy array."""
        features = []
        mean_price = price_stats.get('mean', 0.001)
        std_price = price_stats.get('std', 0.0005) or 0.0005

        # Current price (normalized)
        normalized_price = (gas_state.current_price - mean_price) / std_price
        features.append(np.clip(normalized_price, -3, 3))

        # Price history (normalized)
        history = np.array(gas_state.price_history[-self.history_length:])
        if len(history) < self.history_length:
            padding = np.full(self.history_length - len(history), gas_state.current_price)
            history = np.concatenate([padding, history])
        normalized_history = np.clip((history - mean_price) / std_price, -3, 3)
        features.extend(normalized_history.tolist())

        # Time features (cyclical)
        features.extend([
            np.sin(2 * np.pi * gas_state.hour / 24),
            np.cos(2 * np.pi * gas_state.hour / 24),
            np.sin(2 * np.pi * gas_state.day_of_week / 7),
            np.cos(2 * np.pi * gas_state.day_of_week / 7)
        ])

        # Technical indicators
        features.append(np.clip(gas_state.volatility / 0.1, 0, 1))
        features.append(np.clip(gas_state.momentum, -1, 1))
        
        min_p, max_p = price_stats.get('min', mean_price*0.8), price_stats.get('max', mean_price*1.2)
        percentile = (gas_state.current_price - min_p) / (max_p - min_p + 1e-8)
        features.append(np.clip(percentile, 0, 1))

        # Urgency and time waiting
        features.append(gas_state.urgency)
        features.append(min(gas_state.time_waiting / 100, 1.0))

        return np.array(features, dtype=np.float32)

    def get_state_dim(self) -> int:
        return self.state_dim
