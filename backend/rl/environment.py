"""
Gas Transaction Environment

OpenAI Gym-compatible environment for learning optimal transaction timing.
Uses real historical gas data to simulate transaction execution.

Key concepts:
- State: Current gas price, predictions, time features, urgency
- Actions: WAIT, SUBMIT_NOW, SUBMIT_LOW, SUBMIT_HIGH
- Reward: Minimize gas cost + delay penalty + failure risk
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Available actions for the agent"""
    WAIT = 0           # Wait and check again next step
    SUBMIT_NOW = 1     # Submit at current gas price
    SUBMIT_LOW = 2     # Submit at 10% below current (may fail)
    SUBMIT_HIGH = 3    # Submit at 10% above current (faster confirmation)


@dataclass
class TransactionConfig:
    """Configuration for a transaction episode"""
    max_wait_steps: int = 60        # Max steps before forced submission (60 = 1 hour at 1min steps)
    urgency_level: float = 0.5      # 0 = no rush, 1 = very urgent
    gas_limit: int = 21000          # Standard transfer gas limit
    step_duration_minutes: int = 1  # Each step = 1 minute
    failure_probability_low: float = 0.15   # Chance of tx failing with low gas
    failure_probability_high: float = 0.01  # Chance of tx failing with high gas


class GasTransactionEnv(gym.Env):
    """
    Gym environment for transaction timing optimization.

    The agent observes gas prices and predictions, then decides when
    and how to submit a transaction to minimize cost while meeting
    urgency constraints.

    State Space (continuous, 15 dimensions):
        - current_gas: Current gas price (normalized)
        - gas_velocity: Rate of change
        - gas_acceleration: Second derivative
        - prediction_1h: Predicted gas in 1 hour
        - prediction_4h: Predicted gas in 4 hours
        - prediction_24h: Predicted gas in 24 hours
        - hour_sin, hour_cos: Time of day (cyclical)
        - day_sin, day_cos: Day of week (cyclical)
        - steps_remaining: Normalized time left
        - urgency: Transaction urgency level
        - congestion: Network congestion score
        - relative_to_mean: Current gas vs rolling average
        - volatility: Recent price volatility

    Action Space (discrete, 4 actions):
        0: WAIT - Check again next step
        1: SUBMIT_NOW - Submit at current price
        2: SUBMIT_LOW - Submit at 90% of current (risky)
        3: SUBMIT_HIGH - Submit at 110% of current (safe)

    Rewards:
        - Negative gas cost (lower is better)
        - Delay penalty (increases over time based on urgency)
        - Failure penalty (if tx fails)
        - Success bonus (for completing transaction)
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(
        self,
        historical_data: np.ndarray,
        predictions_data: Optional[np.ndarray] = None,
        config: Optional[TransactionConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.

        Args:
            historical_data: Array of shape (N, features) containing gas prices
                            First column must be gas price in gwei
            predictions_data: Optional array of predictions aligned with historical
            config: Transaction configuration
            render_mode: 'human' or 'ansi' for rendering
        """
        super().__init__()

        self.predictions_data = predictions_data
        self.config = config or TransactionConfig()
        self.render_mode = render_mode

        # Handle both 1D and 2D input
        if historical_data.ndim == 1:
            self.gas_prices = historical_data
        else:
            self.gas_prices = historical_data[:, 0]

        # Validate data
        if len(self.gas_prices) < 10:
            raise ValueError("Need at least 10 data points for training")

        # Calculate statistics for normalization
        self.gas_mean = np.mean(self.gas_prices)
        self.gas_std = np.std(self.gas_prices) + 1e-8
        self.gas_min = np.min(self.gas_prices)
        self.gas_max = np.max(self.gas_prices)

        # Define spaces
        # State: 15 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        # Actions: WAIT, SUBMIT_NOW, SUBMIT_LOW, SUBMIT_HIGH
        self.action_space = spaces.Discrete(4)

        # Episode state
        self.current_step = 0
        self.episode_start_idx = 0
        self.steps_taken = 0
        self.done = False
        self.transaction_submitted = False
        self.transaction_gas_price = 0.0
        self.total_reward = 0.0

        # History for rendering
        self.action_history = []
        self.gas_history = []
        self.reward_history = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Optional dict with 'start_idx' to specify starting point

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Choose random starting point (leaving room for episode)
        max_start = len(self.gas_prices) - self.config.max_wait_steps - 10

        if options and 'start_idx' in options:
            self.episode_start_idx = min(options['start_idx'], max_start)
        else:
            self.episode_start_idx = self.np_random.integers(0, max_start)

        # Reset episode state
        self.current_step = self.episode_start_idx
        self.steps_taken = 0
        self.done = False
        self.transaction_submitted = False
        self.transaction_gas_price = 0.0
        self.total_reward = 0.0

        # Set random urgency for variety
        if options and 'urgency' in options:
            self.config.urgency_level = options['urgency']
        else:
            self.config.urgency_level = self.np_random.uniform(0.2, 0.9)

        # Clear history
        self.action_history = []
        self.gas_history = []
        self.reward_history = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-3)

        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode ended normally
            truncated: Whether episode was cut short
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Record current gas
        current_gas = self.gas_prices[self.current_step]
        self.gas_history.append(current_gas)
        self.action_history.append(action)

        # Process action
        reward = 0.0
        terminated = False
        truncated = False

        if action == Action.WAIT:
            # Apply small waiting penalty based on urgency
            reward = self._calculate_wait_penalty()
            self.steps_taken += 1
            self.current_step += 1

            # Check if we've waited too long
            if self.steps_taken >= self.config.max_wait_steps:
                # Forced submission at current price
                reward += self._submit_transaction(current_gas, forced=True)
                terminated = True

        elif action == Action.SUBMIT_NOW:
            reward = self._submit_transaction(current_gas)
            terminated = True

        elif action == Action.SUBMIT_LOW:
            # Submit at 90% of current - risky but cheap
            low_gas = current_gas * 0.9
            reward = self._submit_transaction(low_gas, risky=True)
            terminated = True

        elif action == Action.SUBMIT_HIGH:
            # Submit at 110% of current - safe but expensive
            high_gas = current_gas * 1.1
            reward = self._submit_transaction(high_gas, safe=True)
            terminated = True

        self.done = terminated or truncated
        self.reward_history.append(reward)
        self.total_reward += reward

        observation = self._get_observation() if not self.done else np.zeros(15, dtype=np.float32)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state."""
        idx = self.current_step
        prices = self.gas_prices

        # Current gas (normalized)
        current_gas = prices[idx]
        gas_normalized = (current_gas - self.gas_mean) / self.gas_std

        # Velocity and acceleration
        if idx >= 2:
            velocity = (prices[idx] - prices[idx-1]) / self.gas_std
            prev_velocity = (prices[idx-1] - prices[idx-2]) / self.gas_std
            acceleration = velocity - prev_velocity
        else:
            velocity = 0.0
            acceleration = 0.0

        # Predictions (use actual future if no predictions provided)
        if self.predictions_data is not None and idx < len(self.predictions_data):
            pred_1h = (self.predictions_data[idx, 0] - self.gas_mean) / self.gas_std
            pred_4h = (self.predictions_data[idx, 1] - self.gas_mean) / self.gas_std if self.predictions_data.shape[1] > 1 else pred_1h
            pred_24h = (self.predictions_data[idx, 2] - self.gas_mean) / self.gas_std if self.predictions_data.shape[1] > 2 else pred_1h
        else:
            # Use actual future as "perfect prediction" for training
            pred_1h = (prices[min(idx+12, len(prices)-1)] - self.gas_mean) / self.gas_std
            pred_4h = (prices[min(idx+48, len(prices)-1)] - self.gas_mean) / self.gas_std
            pred_24h = (prices[min(idx+288, len(prices)-1)] - self.gas_mean) / self.gas_std

        # Time features (cyclical encoding)
        # Assuming 5-minute intervals, calculate hour and day
        minutes_from_start = idx * 5  # Each step is ~5 minutes
        hour = (minutes_from_start // 60) % 24
        day = (minutes_from_start // (60 * 24)) % 7

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        # Steps remaining (normalized)
        steps_remaining = (self.config.max_wait_steps - self.steps_taken) / self.config.max_wait_steps

        # Urgency
        urgency = self.config.urgency_level

        # Congestion (estimate from price since we only have 1D data now)
        congestion = (current_gas - self.gas_min) / (self.gas_max - self.gas_min + 1e-8)

        # Relative to rolling mean
        window = min(idx, 12)  # ~1 hour window
        if window > 0:
            rolling_mean = np.mean(prices[idx-window:idx+1])
            relative_to_mean = (current_gas - rolling_mean) / (rolling_mean + 1e-8)
        else:
            relative_to_mean = 0.0

        # Volatility (std of recent prices)
        if window > 2:
            volatility = np.std(prices[idx-window:idx+1]) / self.gas_std
        else:
            volatility = 0.0

        observation = np.array([
            gas_normalized,      # 0: Current gas (normalized)
            velocity,            # 1: Gas velocity
            acceleration,        # 2: Gas acceleration
            pred_1h,             # 3: 1h prediction
            pred_4h,             # 4: 4h prediction
            pred_24h,            # 5: 24h prediction
            hour_sin,            # 6: Hour (sin)
            hour_cos,            # 7: Hour (cos)
            day_sin,             # 8: Day (sin)
            day_cos,             # 9: Day (cos)
            steps_remaining,     # 10: Time remaining
            urgency,             # 11: Urgency level
            congestion,          # 12: Network congestion
            relative_to_mean,    # 13: Relative to recent mean
            volatility,          # 14: Recent volatility
        ], dtype=np.float32)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get additional episode information."""
        current_gas = self.gas_prices[self.current_step] if self.current_step < len(self.gas_prices) else 0

        return {
            'current_gas': current_gas,
            'steps_taken': self.steps_taken,
            'steps_remaining': self.config.max_wait_steps - self.steps_taken,
            'urgency': self.config.urgency_level,
            'transaction_submitted': self.transaction_submitted,
            'transaction_gas_price': self.transaction_gas_price,
            'total_reward': self.total_reward,
            'gas_mean': self.gas_mean,
            'gas_std': self.gas_std,
        }

    def _calculate_wait_penalty(self) -> float:
        """Calculate penalty for waiting one step."""
        # Higher urgency = higher penalty for waiting
        base_penalty = -0.001  # Small base penalty
        urgency_multiplier = 1 + (self.config.urgency_level * 2)  # 1x to 3x
        time_multiplier = 1 + (self.steps_taken / self.config.max_wait_steps)  # Increases over time

        return base_penalty * urgency_multiplier * time_multiplier

    def _submit_transaction(
        self,
        gas_price: float,
        risky: bool = False,
        safe: bool = False,
        forced: bool = False
    ) -> float:
        """
        Submit transaction and calculate reward.

        Args:
            gas_price: Gas price for transaction
            risky: Whether this is a low-gas risky submission
            safe: Whether this is a high-gas safe submission
            forced: Whether submission was forced due to timeout

        Returns:
            reward: Total reward for submission
        """
        self.transaction_submitted = True
        self.transaction_gas_price = gas_price

        # Check for failure
        if risky:
            fail_prob = self.config.failure_probability_low
        elif safe:
            fail_prob = self.config.failure_probability_high
        else:
            fail_prob = 0.05  # Normal submission

        failed = self.np_random.random() < fail_prob

        if failed:
            # Transaction failed - big penalty
            failure_penalty = -1.0
            # Still pay some gas for failed tx
            gas_cost = -(gas_price / self.gas_mean) * 0.3
            return failure_penalty + gas_cost

        # Transaction succeeded
        # Reward components:

        # 1. Gas cost (negative, lower is better)
        # Normalize by mean gas price
        gas_cost = -(gas_price / self.gas_mean)

        # 2. Timing bonus: reward if we got better than average price
        window_start = max(0, self.episode_start_idx)
        window_end = min(len(self.gas_prices), self.episode_start_idx + self.config.max_wait_steps)
        window_prices = self.gas_prices[window_start:window_end]
        window_mean = np.mean(window_prices)
        window_min = np.min(window_prices)

        # Bonus for beating average
        if gas_price < window_mean:
            timing_bonus = (window_mean - gas_price) / (window_mean + 1e-8)
        else:
            timing_bonus = 0.0

        # Extra bonus for getting close to minimum
        if gas_price <= window_min * 1.05:  # Within 5% of minimum
            optimal_bonus = 0.2
        else:
            optimal_bonus = 0.0

        # 3. Speed bonus (if urgent and submitted quickly)
        speed_factor = 1.0 - (self.steps_taken / self.config.max_wait_steps)
        speed_bonus = speed_factor * self.config.urgency_level * 0.1

        # 4. Forced submission penalty
        forced_penalty = -0.3 if forced else 0.0

        # 5. Success bonus
        success_bonus = 0.5

        total_reward = gas_cost + timing_bonus + optimal_bonus + speed_bonus + forced_penalty + success_bonus

        return total_reward

    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
        return None

    def _render_human(self):
        """Print current state to console."""
        print(self._render_ansi())

    def _render_ansi(self) -> str:
        """Generate ASCII representation of state."""
        current_gas = self.gas_prices[self.current_step] if self.current_step < len(self.gas_prices) else 0

        lines = [
            f"Step: {self.steps_taken}/{self.config.max_wait_steps}",
            f"Current Gas: {current_gas:.6f} gwei",
            f"Urgency: {self.config.urgency_level:.2f}",
            f"Total Reward: {self.total_reward:.4f}",
        ]

        if self.transaction_submitted:
            lines.append(f"Submitted at: {self.transaction_gas_price:.6f} gwei")

        if self.action_history:
            last_action = Action(self.action_history[-1]).name
            lines.append(f"Last Action: {last_action}")

        return "\n".join(lines)

    def close(self):
        """Clean up resources."""
        pass


def create_env_from_database(db_manager, hours: int = 168) -> GasTransactionEnv:
    """
    Create environment using data from database.

    Args:
        db_manager: DatabaseManager instance
        hours: Hours of historical data to load

    Returns:
        GasTransactionEnv instance
    """
    # Fetch historical data
    data = db_manager.get_historical_data(hours=hours)

    if not data:
        raise ValueError("No historical data available")

    # Convert to numpy array
    # Expecting data to have 'gwei' or 'gas_price' and optionally 'congestion_score'
    gas_prices = []
    congestion_scores = []

    for record in data:
        gas = record.get('gwei') or record.get('gas_price') or record.get('current_gas', 0)
        gas_prices.append(float(gas))
        congestion_scores.append(float(record.get('congestion_score', 5)))

    historical_data = np.column_stack([gas_prices, congestion_scores])

    return GasTransactionEnv(historical_data=historical_data)
