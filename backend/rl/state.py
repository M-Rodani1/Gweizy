"""
State representation for the RL environment.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
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
    # Phase 4A: Enhanced features
    max_wait_steps: int = 48  # Maximum steps in episode (for time remaining)


class StateBuilder:
    """Builds state vectors for the RL agent with enhanced technical indicators."""

    def __init__(self, history_length: int = 24, use_enhanced_features: bool = True):
        self.history_length = history_length
        self.use_enhanced_features = use_enhanced_features
        self.tech_indicator_count = 4  # RSI, MACD, signal, Bollinger position
        # Phase 4A: Enhanced features
        self.enhanced_feature_count = 10 if use_enhanced_features else 0  # velocity (3), mean_rev (2), support/resist (2), accel, trend, time_remaining
        self.state_version = "v3_48" if use_enhanced_features else "v2_38"
        # Base features: 1 (current_price) + history_length + 4 (time) + 3 (volatility/momentum/percentile) + 2 (urgency/time_waiting)
        # Technical indicators: +4 (RSI, MACD, signal, Bollinger position)
        # Enhanced (Phase 4A): +10 (velocity x3, mean_rev x2, support/resist x2, accel, trend, time_remaining)
        # Total: 1 + history_length + 4 + 3 + 2 + 4 + 10 = 24 + history_length = 48 features (with 24 history)
        self.state_dim = 1 + history_length + 4 + 3 + 2 + self.tech_indicator_count + self.enhanced_feature_count
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = ["current_price"]
        names.extend([f"history_{i+1}" for i in range(self.history_length)])
        names.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
        names.extend(["volatility", "momentum", "percentile_rank"])
        names.extend(["urgency", "time_waiting"])
        names.extend(["rsi", "macd", "macd_signal", "bollinger_pos"])
        # Phase 4A: Enhanced features
        if self.use_enhanced_features:
            names.extend([
                "velocity_short", "velocity_medium", "velocity_long",  # Price velocity at different timeframes
                "mean_rev_short", "mean_rev_long",  # Distance from moving averages
                "support_dist", "resist_dist",  # Distance from support/resistance
                "acceleration",  # Price acceleration (second derivative)
                "trend_strength",  # Consecutive moves in same direction
                "time_remaining"  # Normalized time remaining in episode
            ])
        return names

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
        
        # Technical Indicators
        price_array = np.array(gas_state.price_history[-self.history_length:] + [gas_state.current_price])
        if len(price_array) >= 14:  # Need enough data for indicators
            # RSI (Relative Strength Index) - 14 period
            rsi = self._calculate_rsi(price_array, period=14)
            features.append(np.clip((rsi - 50) / 50, -1, 1))  # Normalize: [0, 100] -> [-1, 1]
            
            # MACD (Moving Average Convergence Divergence)
            macd, signal = self._calculate_macd(price_array)
            features.append(np.clip(macd / (mean_price + 1e-8), -1, 1))  # Normalize by mean price
            features.append(np.clip(signal / (mean_price + 1e-8), -1, 1))
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(price_array, period=20)
            # Position relative to bands: -1 (at lower) to +1 (at upper)
            if bb_upper > bb_lower:
                bb_position = 2 * (gas_state.current_price - bb_lower) / (bb_upper - bb_lower) - 1
                features.append(np.clip(bb_position, -1, 1))
            else:
                features.append(0.0)  # Neutral if bands not available
        else:
            # Not enough data, use neutral values
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Phase 4A: Enhanced features
        if self.use_enhanced_features:
            price_array = np.array(gas_state.price_history[-self.history_length:] + [gas_state.current_price])

            # 1. Price velocity at different timeframes (rate of change)
            velocity_short = self._calculate_velocity(price_array, period=3)
            velocity_medium = self._calculate_velocity(price_array, period=8)
            velocity_long = self._calculate_velocity(price_array, period=16)
            features.append(np.clip(velocity_short, -1, 1))
            features.append(np.clip(velocity_medium, -1, 1))
            features.append(np.clip(velocity_long, -1, 1))

            # 2. Mean reversion signals (distance from moving averages)
            mean_rev_short = self._calculate_mean_reversion(price_array, gas_state.current_price, period=5, std_price=std_price)
            mean_rev_long = self._calculate_mean_reversion(price_array, gas_state.current_price, period=20, std_price=std_price)
            features.append(np.clip(mean_rev_short, -1, 1))
            features.append(np.clip(mean_rev_long, -1, 1))

            # 3. Support/Resistance distances
            support_dist, resist_dist = self._calculate_support_resistance(price_array, gas_state.current_price, std_price=std_price)
            features.append(np.clip(support_dist, -1, 1))
            features.append(np.clip(resist_dist, -1, 1))

            # 4. Price acceleration (second derivative)
            acceleration = self._calculate_acceleration(price_array, std_price=std_price)
            features.append(np.clip(acceleration, -1, 1))

            # 5. Trend strength (consecutive moves in same direction)
            trend_strength = self._calculate_trend_strength(price_array)
            features.append(trend_strength)  # Already in [-1, 1]

            # 6. Time remaining in episode (normalized)
            time_remaining = 1.0 - (gas_state.time_waiting / max(gas_state.max_wait_steps, 1))
            features.append(2 * np.clip(time_remaining, 0, 1) - 1)  # [0, 1] -> [-1, 1]

        if len(features) != self.state_dim:
            raise ValueError(
                f"State dimension mismatch in StateBuilder: expected {self.state_dim}, got {len(features)}"
            )
        return np.array(features, dtype=np.float32)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        if len(prices) < slow:
            return 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        # For simplicity, use recent MACD values
        if len(prices) >= slow + signal:
            macd_values = []
            for i in range(slow, len(prices)):
                ema_f = self._ema(prices[:i+1], fast)
                ema_s = self._ema(prices[:i+1], slow)
                macd_values.append(ema_f - ema_s)
            signal_line = self._ema(np.array(macd_values), signal) if len(macd_values) >= signal else macd_line
        else:
            signal_line = macd_line
        
        return macd_line, signal_line
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) == 0:
            return 0.0
        if len(prices) == 1:
            return float(prices[0])
        
        multiplier = 2.0 / (period + 1)
        ema = float(prices[0])
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return 0.0, 0.0, float(prices[-1])
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return upper, lower, middle

    # Phase 4A: Enhanced feature calculation methods
    def _calculate_velocity(self, prices: np.ndarray, period: int = 5) -> float:
        """Calculate price velocity (rate of change) over a period."""
        if len(prices) < period + 1:
            return 0.0
        old_price = prices[-period - 1]
        new_price = prices[-1]
        if old_price == 0:
            return 0.0
        # Normalize velocity to reasonable range
        velocity = (new_price - old_price) / old_price
        return velocity * 10  # Scale up for better signal

    def _calculate_mean_reversion(self, prices: np.ndarray, current_price: float,
                                   period: int = 10, std_price: float = 1.0) -> float:
        """Calculate distance from moving average (mean reversion signal)."""
        if len(prices) < period:
            return 0.0
        ma = np.mean(prices[-period:])
        if std_price == 0:
            std_price = 1e-8
        # Positive = above MA (potential sell), Negative = below MA (potential buy/wait)
        distance = (current_price - ma) / std_price
        return distance

    def _calculate_support_resistance(self, prices: np.ndarray, current_price: float,
                                       lookback: int = 20, std_price: float = 1.0) -> Tuple[float, float]:
        """Calculate distance from recent support (low) and resistance (high)."""
        if len(prices) < 2:
            return 0.0, 0.0
        recent_prices = prices[-lookback:] if len(prices) >= lookback else prices
        support = np.min(recent_prices)
        resistance = np.max(recent_prices)
        if std_price == 0:
            std_price = 1e-8
        # Normalize distances
        support_dist = (current_price - support) / std_price  # Positive = above support
        resist_dist = (resistance - current_price) / std_price  # Positive = below resistance
        return support_dist, resist_dist

    def _calculate_acceleration(self, prices: np.ndarray, period: int = 5, std_price: float = 1.0) -> float:
        """Calculate price acceleration (second derivative)."""
        if len(prices) < period + 2:
            return 0.0
        # First derivative (velocity) at two points
        vel_new = prices[-1] - prices[-period // 2 - 1] if period // 2 > 0 else 0
        vel_old = prices[-period // 2 - 1] - prices[-period - 1] if len(prices) > period else 0
        # Second derivative
        if std_price == 0:
            std_price = 1e-8
        acceleration = (vel_new - vel_old) / std_price
        return acceleration * 5  # Scale for better signal

    def _calculate_trend_strength(self, prices: np.ndarray, max_lookback: int = 10) -> float:
        """Calculate trend strength based on consecutive moves in same direction."""
        if len(prices) < 2:
            return 0.0
        lookback = min(max_lookback, len(prices) - 1)
        consecutive_up = 0
        consecutive_down = 0
        for i in range(1, lookback + 1):
            if prices[-i] > prices[-i - 1]:
                if consecutive_down > 0:
                    break
                consecutive_up += 1
            elif prices[-i] < prices[-i - 1]:
                if consecutive_up > 0:
                    break
                consecutive_down += 1
            else:
                break
        # Normalize to [-1, 1]
        if consecutive_up > 0:
            return min(consecutive_up / max_lookback, 1.0)
        elif consecutive_down > 0:
            return -min(consecutive_down / max_lookback, 1.0)
        return 0.0

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def get_state_version(self) -> str:
        return self.state_version
