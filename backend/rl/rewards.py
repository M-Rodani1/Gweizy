"""
Reward functions for gas optimization RL agent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    wait_penalty: float = 0.015  # Per-step waiting cost (balanced to allow price observation)
    max_loss: float = -0.30  # Cap catastrophic loss at -30%
    urgency_multiplier: float = 2.0  # Optional extra wait penalty for urgency
    reward_scale: float = 1.0  # Optional scaling if enabled
    # Phase 3: Action timing
    min_wait_steps: int = 3  # Minimum steps before execution is optimal
    early_execution_penalty: float = 0.1  # Penalty for executing before min_wait_steps
    observation_bonus: float = 0.02  # Small bonus for observing during volatile periods
    # Phase 4B-2: Risk-adjusted metrics
    use_risk_adjustment: bool = False  # Enable risk-adjusted rewards
    risk_penalty_weight: float = 0.5  # Weight for variance penalty (higher = more conservative)
    target_savings: float = 0.10  # Target savings rate for risk calculation
    # Phase 4B-3: Extended observation window
    progressive_obs_bonus: bool = True  # Bonus increases with observation time
    max_obs_bonus: float = 0.15  # Maximum observation bonus at peak
    obs_bonus_peak_step: int = 8  # Step at which observation bonus peaks
    patience_bonus: float = 0.03  # Extra bonus for waiting through high volatility


class RewardCalculator:
    """Calculates rewards for RL agent actions with reward scaling and risk adjustment."""

    def __init__(self, config: Optional[RewardConfig] = None, scale_rewards: bool = False):
        self.config = config or RewardConfig()
        self.best_price_seen = None
        self.initial_price = None
        self.scale_rewards = scale_rewards
        # Phase 4B-2: Running statistics for risk adjustment
        self._savings_history = []
        self._savings_mean = 0.0
        self._savings_var = 0.0
        self._savings_count = 0

    def reset(self, initial_price: float):
        self.initial_price = initial_price
        self.best_price_seen = initial_price

    def reset_episode_stats(self):
        """Reset per-training-session statistics (call at start of training)."""
        self._savings_history = []
        self._savings_mean = 0.0
        self._savings_var = 0.0
        self._savings_count = 0

    def _update_savings_stats(self, savings: float):
        """Update running mean and variance of savings using Welford's algorithm."""
        self._savings_count += 1
        self._savings_history.append(savings)

        # Welford's online algorithm for running mean/variance
        delta = savings - self._savings_mean
        self._savings_mean += delta / self._savings_count
        delta2 = savings - self._savings_mean
        self._savings_var += delta * delta2

    def get_savings_stats(self) -> dict:
        """Get current savings statistics."""
        if self._savings_count < 2:
            return {'mean': 0.0, 'std': 0.0, 'sharpe': 0.0, 'count': self._savings_count}

        variance = self._savings_var / (self._savings_count - 1)
        std = np.sqrt(max(variance, 0))

        # Sharpe-like ratio (savings / volatility)
        sharpe = self._savings_mean / (std + 1e-8)

        return {
            'mean': self._savings_mean,
            'std': std,
            'sharpe': sharpe,
            'count': self._savings_count
        }

    def get_risk_adjusted_reward(self, base_reward: float, savings: float) -> float:
        """
        Apply risk adjustment to reward based on historical variance.

        The adjustment penalizes rewards that deviate too far from the target,
        encouraging more consistent performance.
        """
        if not self.config.use_risk_adjustment or self._savings_count < 10:
            return base_reward

        stats = self.get_savings_stats()
        std = stats['std']

        # Penalty for high variance (encourages consistency)
        variance_penalty = self.config.risk_penalty_weight * std

        # Penalty for being far from target savings
        target_deviation = abs(savings - self.config.target_savings)
        target_penalty = self.config.risk_penalty_weight * target_deviation * 0.5

        # Apply penalties
        adjusted_reward = base_reward - variance_penalty - target_penalty

        return adjusted_reward

    def _scale_reward(self, reward: float) -> float:
        """Scale reward to [-1, 1] range for stable learning."""
        if not self.scale_rewards:
            return reward

        return float(np.tanh(reward / self.config.reward_scale))

    def calculate_reward(self, action: int, current_price: float,
                        urgency: float, time_waiting: int, done: bool = False,
                        volatility: float = 0.0) -> float:
        """
        Calculate reward using RELATIVE SAVINGS approach with action timing incentives.
        Reward = Benchmark_Price - Execution_Price
        Where Benchmark_Price = initial price at episode start (Step 0).

        Phase 3 additions:
        - Early execution penalty: discourage executing before min_wait_steps
        - Observation bonus: reward waiting during volatile periods

        Args:
            volatility: Current market volatility (0-1 scale) for reward shaping
        """
        if self.initial_price is None:
            self.initial_price = current_price
            self.best_price_seen = current_price

        self.best_price_seen = min(self.best_price_seen, current_price)

        if action == 1:  # Execute
            # Direct savings-based reward (aligned with objective)
            if self.initial_price > 0:
                relative_savings = (self.initial_price - current_price) / self.initial_price
                reward = relative_savings - (time_waiting * self.config.wait_penalty)
            else:
                reward = 0.0

            # Phase 3: Early execution penalty
            # Penalize executing before observing enough price data
            if time_waiting < self.config.min_wait_steps:
                steps_short = self.config.min_wait_steps - time_waiting
                early_penalty = self.config.early_execution_penalty * steps_short
                reward -= early_penalty

            # Cap catastrophic losses
            if reward < self.config.max_loss:
                reward = self.config.max_loss

            # Phase 4B-2: Track savings and apply risk adjustment
            if self.initial_price > 0:
                savings = (self.initial_price - current_price) / self.initial_price
                self._update_savings_stats(savings)

                if self.config.use_risk_adjustment:
                    reward = self.get_risk_adjusted_reward(reward, savings)
        else:  # Wait
            # Small per-step wait penalty (optionally scaled by urgency)
            reward = -self.config.wait_penalty * (1 + urgency * self.config.urgency_multiplier)

            # Phase 3: Observation bonus during volatile periods
            # Encourage waiting when there's useful price movement to observe
            if volatility > 0.05:  # Only give bonus when volatility is meaningful
                observation_bonus = self.config.observation_bonus * min(volatility, 0.5)
                reward += observation_bonus

                # Phase 4B-3: Extra patience bonus for waiting through high volatility
                if volatility > 0.15:
                    reward += self.config.patience_bonus

            # Phase 4B-3: Progressive observation bonus
            # Encourages agent to wait longer by providing increasing bonus up to peak
            if self.config.progressive_obs_bonus and time_waiting < self.config.obs_bonus_peak_step:
                # Bonus increases linearly until peak step, then decays
                progress = time_waiting / self.config.obs_bonus_peak_step
                progressive_bonus = self.config.max_obs_bonus * progress * (1 - progress * 0.5)
                reward += progressive_bonus

        # Scale reward to [-1, 1] range for stable learning
        return self._scale_reward(reward)

    def get_recommendation(self, current_price: float, predicted_prices: list, urgency: float) -> dict:
        if not predicted_prices:
            return {'action': 'execute', 'confidence': 0.5, 'reason': 'No predictions'}
        
        min_pred = min(predicted_prices)
        potential_savings = (current_price - min_pred) / current_price if current_price > 0 else 0

        if urgency > 0.8:
            return {'action': 'execute', 'confidence': 0.9, 'reason': 'High urgency'}
        if potential_savings > 0.1:
            return {'action': 'wait', 'confidence': min(0.9, 0.5 + potential_savings), 
                   'reason': f'{potential_savings*100:.1f}% savings possible'}
        if current_price <= min_pred * 1.02:
            return {'action': 'execute', 'confidence': 0.8, 'reason': 'Near predicted minimum'}
        return {'action': 'execute', 'confidence': 0.6, 'reason': 'No significant savings expected'}
