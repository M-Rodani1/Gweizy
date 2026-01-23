"""
Reward functions for gas optimization RL agent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    wait_penalty: float = 0.001  # Per-step waiting cost
    max_loss: float = -0.30  # Cap catastrophic loss at -30%
    urgency_multiplier: float = 0.0  # Optional extra wait penalty for urgency
    reward_scale: float = 1.0  # Optional scaling if enabled


class RewardCalculator:
    """Calculates rewards for RL agent actions with reward scaling."""

    def __init__(self, config: Optional[RewardConfig] = None, scale_rewards: bool = False):
        self.config = config or RewardConfig()
        self.best_price_seen = None
        self.initial_price = None
        self.scale_rewards = scale_rewards

    def reset(self, initial_price: float):
        self.initial_price = initial_price
        self.best_price_seen = initial_price

    def _scale_reward(self, reward: float) -> float:
        """Scale reward to [-1, 1] range for stable learning."""
        if not self.scale_rewards:
            return reward

        return float(np.tanh(reward / self.config.reward_scale))

    def calculate_reward(self, action: int, current_price: float, 
                        urgency: float, time_waiting: int, done: bool = False,
                        volatility: float = 0.0) -> float:
        """
        Calculate reward using RELATIVE SAVINGS approach with volatility-scaled waiting penalty.
        Reward = Benchmark_Price - Execution_Price
        Where Benchmark_Price = initial price at episode start (Step 0).
        
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

            # Cap catastrophic losses
            if reward < self.config.max_loss:
                reward = self.config.max_loss
        else:  # Wait
            # Small per-step wait penalty (optionally scaled by urgency)
            reward = -self.config.wait_penalty * (1 + urgency * self.config.urgency_multiplier)
        
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
