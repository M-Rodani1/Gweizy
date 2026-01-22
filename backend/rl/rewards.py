"""
Reward functions for gas optimization RL agent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    savings_weight: float = 1.0
    time_penalty: float = 0.01
    urgency_multiplier: float = 2.0
    missed_opportunity_penalty: float = 0.5
    execution_bonus: float = 0.1


class RewardCalculator:
    """Calculates rewards for RL agent actions with reward scaling."""

    def __init__(self, config: Optional[RewardConfig] = None, scale_rewards: bool = True):
        self.config = config or RewardConfig()
        self.best_price_seen = None
        self.initial_price = None
        self.scale_rewards = scale_rewards
        
        # Track reward statistics for normalization
        self.reward_min = -10.0  # Estimated min reward
        self.reward_max = 10.0   # Estimated max reward
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []

    def reset(self, initial_price: float):
        self.initial_price = initial_price
        self.best_price_seen = initial_price

    def _scale_reward(self, reward: float) -> float:
        """Scale reward to [-1, 1] range for stable learning."""
        if not self.scale_rewards:
            return reward
        
        # Update running statistics
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8
            self.reward_min = min(self.reward_history)
            self.reward_max = max(self.reward_history)
        
        # Method 1: Tanh scaling (bounded)
        if abs(self.reward_max - self.reward_min) > 1e-8:
            normalized = 2 * (reward - self.reward_min) / (self.reward_max - self.reward_min) - 1
            return np.tanh(normalized)  # Ensures [-1, 1]
        else:
            # Method 2: Z-score normalization with clipping
            normalized = (reward - self.reward_mean) / self.reward_std
            return np.clip(np.tanh(normalized), -1.0, 1.0)

    def calculate_reward(self, action: int, current_price: float, 
                        urgency: float, time_waiting: int, done: bool = False) -> float:
        """
        Calculate reward using RELATIVE SAVINGS approach.
        Reward = Benchmark_Price - Execution_Price
        Where Benchmark_Price = initial price at episode start (Step 0).
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
                # Scale to meaningful reward range (multiply by 100 for percentage points)
                reward = relative_savings * 100.0 * self.config.savings_weight
            else:
                reward = 0.0
            
            # Small bonus for executing (encourages action)
            reward += self.config.execution_bonus
            
            # Time penalty for urgent transactions (if waited too long on urgent tx)
            if urgency > 0.5:
                reward -= time_waiting * self.config.time_penalty * urgency * self.config.urgency_multiplier
        else:  # Wait
            # Small negative reward for waiting (encourages eventual execution)
            reward = -self.config.time_penalty * (1 + urgency * self.config.urgency_multiplier)
            
            # If episode ends without execution, apply missed opportunity penalty
            if done:
                # Penalty for missing the best price seen
                if self.initial_price > 0:
                    missed_opportunity = (current_price - self.best_price_seen) / self.initial_price
                    reward -= missed_opportunity * self.config.missed_opportunity_penalty * 50
        
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
