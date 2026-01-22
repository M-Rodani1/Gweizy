"""
Reward functions for gas optimization RL agent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    savings_weight: float = 10.0  # Scale relative savings to a stable reward range
    max_relative_savings: float = 0.5  # Cap +/- savings at 50%
    time_penalty: float = 0.005
    urgency_multiplier: float = 2.0
    missed_opportunity_penalty: float = 0.6
    execution_bonus: float = 0.02
    execution_cost: float = 0.02  # Penalize instant execution slightly
    max_time_penalty: float = 1.0
    reward_scale: float = 5.0  # Tanh scaling factor


class RewardCalculator:
    """Calculates rewards for RL agent actions with reward scaling."""

    def __init__(self, config: Optional[RewardConfig] = None, scale_rewards: bool = True):
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
            # RELATIVE SAVINGS: (Benchmark - Execution) / Benchmark
            # Positive if execution price < benchmark (saved money)
            # Negative if execution price > benchmark (paid more)
            if self.initial_price > 0:
                relative_savings = (self.initial_price - current_price) / self.initial_price
                relative_savings = np.clip(
                    relative_savings,
                    -self.config.max_relative_savings,
                    self.config.max_relative_savings
                )
                reward = relative_savings * self.config.savings_weight
            else:
                reward = 0.0
            
            # Small bonus for executing (encourages action)
            reward += self.config.execution_bonus
            # Slight execution cost to avoid always-execute behavior
            reward -= self.config.execution_cost
            
            # Time penalty for urgent transactions (if waited too long on urgent tx)
            if urgency > 0.5:
                urgency_penalty = time_waiting * self.config.time_penalty * urgency * self.config.urgency_multiplier
                reward -= min(urgency_penalty, self.config.max_time_penalty)
        else:  # Wait
            # Volatility-scaled waiting penalty: more penalty during calm periods
            # This discourages passive behavior when market is stable
            base_wait_penalty = self.config.time_penalty * (1 + urgency * self.config.urgency_multiplier)
            # During low volatility (calm market), increase penalty to encourage action
            # During high volatility, reduce penalty to allow strategic waiting
            volatility_factor = 1.2 - min(volatility, 1.0)  # Scale: 0.2 to 1.2
            reward = -min(base_wait_penalty * volatility_factor, self.config.max_time_penalty)
            
            # If episode ends without execution, apply missed opportunity penalty
            if done:
                # Penalty for missing the best price seen
                if self.initial_price > 0:
                    missed_opportunity = (current_price - self.best_price_seen) / self.initial_price
                    missed_opportunity = np.clip(missed_opportunity, 0.0, self.config.max_relative_savings)
                    reward -= missed_opportunity * self.config.savings_weight * self.config.missed_opportunity_penalty
        
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
