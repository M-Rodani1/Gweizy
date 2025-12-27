"""
State Representation for RL Transaction Agent

Handles state construction, normalization, and feature engineering
for the observation space used by the RL agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import deque


@dataclass
class MarketState:
    """Raw market state data"""
    current_gas: float
    timestamp: datetime
    predictions: Dict[str, float] = field(default_factory=dict)
    network_congestion: float = 0.5
    pending_tx_count: int = 0
    block_utilization: float = 0.5


@dataclass
class StateConfig:
    """Configuration for state builder"""
    # History lengths
    price_history_length: int = 20
    velocity_window: int = 5
    acceleration_window: int = 3

    # Normalization parameters (learned from data)
    gas_mean: float = 0.01
    gas_std: float = 0.005
    gas_min: float = 0.001
    gas_max: float = 0.1

    # Feature toggles
    include_predictions: bool = True
    include_time_features: bool = True
    include_market_features: bool = True


class StateBuilder:
    """
    Builds normalized state vectors from raw market data

    The state representation is critical for RL performance.
    Good state design captures relevant information while
    maintaining appropriate dimensionality.
    """

    def __init__(self, config: Optional[StateConfig] = None):
        self.config = config or StateConfig()

        # Rolling history for velocity/acceleration
        self.price_history = deque(maxlen=self.config.price_history_length)
        self.timestamp_history = deque(maxlen=self.config.price_history_length)

        # Statistics for adaptive normalization
        self.running_mean = self.config.gas_mean
        self.running_std = self.config.gas_std
        self.running_min = self.config.gas_min
        self.running_max = self.config.gas_max
        self.sample_count = 0

    def update_history(self, gas_price: float, timestamp: datetime):
        """Update price history with new observation"""
        self.price_history.append(gas_price)
        self.timestamp_history.append(timestamp)

        # Update running statistics
        self._update_statistics(gas_price)

    def _update_statistics(self, gas_price: float, alpha: float = 0.01):
        """Update running statistics with exponential moving average"""
        self.sample_count += 1

        if self.sample_count == 1:
            self.running_mean = gas_price
            self.running_std = 0.001
        else:
            # EMA update
            delta = gas_price - self.running_mean
            self.running_mean += alpha * delta
            self.running_std = np.sqrt(
                (1 - alpha) * (self.running_std ** 2) + alpha * (delta ** 2)
            )

        # Update min/max
        self.running_min = min(self.running_min, gas_price)
        self.running_max = max(self.running_max, gas_price)

    def build_state(
        self,
        market_state: MarketState,
        transaction_context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Build normalized state vector

        Args:
            market_state: Current market state
            transaction_context: Transaction-specific context

        Returns:
            Normalized state vector (numpy array)
        """
        features = []

        # 1. Current gas price (normalized)
        gas_norm = self._normalize_gas(market_state.current_gas)
        features.append(gas_norm)

        # 2. Gas velocity (rate of change)
        velocity = self._calculate_velocity()
        features.append(velocity)

        # 3. Gas acceleration
        acceleration = self._calculate_acceleration()
        features.append(acceleration)

        # 4. Predictions (normalized)
        if self.config.include_predictions:
            pred_1h = market_state.predictions.get('1h', market_state.current_gas)
            pred_4h = market_state.predictions.get('4h', market_state.current_gas)
            pred_24h = market_state.predictions.get('24h', market_state.current_gas)

            features.append(self._normalize_gas(pred_1h))
            features.append(self._normalize_gas(pred_4h))
            features.append(self._normalize_gas(pred_24h))

        # 5. Time features (cyclical encoding)
        if self.config.include_time_features:
            hour = market_state.timestamp.hour
            day = market_state.timestamp.weekday()

            # Cyclical encoding for smooth transitions
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(np.sin(2 * np.pi * day / 7))
            features.append(np.cos(2 * np.pi * day / 7))

        # 6. Market features
        if self.config.include_market_features:
            features.append(np.clip(market_state.network_congestion, 0, 1))
            features.append(self._calculate_volatility())

        # 7. Transaction context features
        if transaction_context:
            steps_remaining = transaction_context.get('steps_remaining', 60)
            max_steps = transaction_context.get('max_steps', 60)
            urgency = transaction_context.get('urgency', 0.5)

            features.append(steps_remaining / max_steps)  # Normalized steps remaining
            features.append(np.clip(urgency, 0, 1))

        # 8. Relative position features
        relative_to_mean = (market_state.current_gas - self.running_mean) / (self.running_std + 1e-8)
        features.append(np.clip(relative_to_mean, -3, 3) / 3)  # Normalize to [-1, 1]

        return np.array(features, dtype=np.float32)

    def _normalize_gas(self, gas_price: float) -> float:
        """Normalize gas price to roughly [-1, 1] range"""
        # Use robust normalization
        normalized = (gas_price - self.running_mean) / (self.running_std + 1e-8)
        return np.clip(normalized, -3, 3) / 3  # Clip and scale to [-1, 1]

    def _calculate_velocity(self) -> float:
        """Calculate rate of change in gas price"""
        if len(self.price_history) < 2:
            return 0.0

        window = min(self.config.velocity_window, len(self.price_history))
        recent = list(self.price_history)[-window:]

        if len(recent) < 2:
            return 0.0

        # Simple velocity: (latest - oldest) / time
        velocity = (recent[-1] - recent[0]) / (self.running_std + 1e-8)
        return np.clip(velocity, -2, 2) / 2

    def _calculate_acceleration(self) -> float:
        """Calculate acceleration (rate of change of velocity)"""
        if len(self.price_history) < 4:
            return 0.0

        window = min(self.config.acceleration_window, len(self.price_history) // 2)

        if window < 2:
            return 0.0

        recent = list(self.price_history)

        # Calculate two velocities
        v1 = (recent[-window] - recent[-2*window]) if len(recent) >= 2*window else 0
        v2 = (recent[-1] - recent[-window])

        acceleration = (v2 - v1) / (self.running_std + 1e-8)
        return np.clip(acceleration, -2, 2) / 2

    def _calculate_volatility(self) -> float:
        """Calculate recent price volatility"""
        if len(self.price_history) < 5:
            return 0.0

        recent = list(self.price_history)[-10:]
        std = np.std(recent)
        volatility = std / (self.running_mean + 1e-8)

        return np.clip(volatility, 0, 1)

    def get_state_dim(self) -> int:
        """Get the dimensionality of the state vector"""
        dim = 3  # gas, velocity, acceleration

        if self.config.include_predictions:
            dim += 3  # 1h, 4h, 24h

        if self.config.include_time_features:
            dim += 4  # hour_sin, hour_cos, day_sin, day_cos

        if self.config.include_market_features:
            dim += 2  # congestion, volatility

        dim += 2  # steps_remaining, urgency (transaction context)
        dim += 1  # relative_to_mean

        return dim

    def reset(self):
        """Reset state builder for new episode"""
        self.price_history.clear()
        self.timestamp_history.clear()

    def get_statistics(self) -> Dict:
        """Get current running statistics"""
        return {
            'mean': self.running_mean,
            'std': self.running_std,
            'min': self.running_min,
            'max': self.running_max,
            'sample_count': self.sample_count,
            'history_length': len(self.price_history)
        }


class StateNormalizer:
    """
    Handles state normalization for stable RL training

    Uses running mean/std normalization which is standard
    practice for continuous control RL algorithms.
    """

    def __init__(self, state_dim: int, clip_range: float = 10.0):
        self.state_dim = state_dim
        self.clip_range = clip_range

        # Running statistics
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 1e-4  # Small initial count to avoid division by zero

    def update(self, states: np.ndarray):
        """Update running statistics with batch of states"""
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_count = states.shape[0]

        # Welford's online algorithm for stable variance calculation
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics"""
        normalized = (state - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        """Convert normalized state back to original scale"""
        return normalized_state * np.sqrt(self.var) + self.mean

    def save(self, path: str):
        """Save normalizer state"""
        np.savez(
            path,
            mean=self.mean,
            var=self.var,
            count=self.count
        )

    def load(self, path: str):
        """Load normalizer state"""
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count']


def create_state_builder(
    include_predictions: bool = True,
    include_time: bool = True,
    include_market: bool = True,
    **kwargs
) -> StateBuilder:
    """
    Factory function to create state builders

    Args:
        include_predictions: Include prediction features
        include_time: Include time-based features
        include_market: Include market features
        **kwargs: Override StateConfig parameters

    Returns:
        Configured StateBuilder
    """
    config = StateConfig(
        include_predictions=include_predictions,
        include_time_features=include_time,
        include_market_features=include_market,
        **kwargs
    )

    return StateBuilder(config=config)
