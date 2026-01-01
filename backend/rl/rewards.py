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
    """Calculates rewards for RL agent actions."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.best_price_seen = None
        self.initial_price = None

    def reset(self, initial_price: float):
        self.initial_price = initial_price
        self.best_price_seen = initial_price

    def calculate_reward(self, action: int, current_price: float, 
                        urgency: float, time_waiting: int, done: bool = False) -> float:
        if self.initial_price is None:
            self.initial_price = current_price
            self.best_price_seen = current_price

        self.best_price_seen = min(self.best_price_seen, current_price)

        if action == 1:  # Execute
            reward = 0.0
            if self.initial_price > 0:
                savings_pct = (self.initial_price - current_price) / self.initial_price
                reward += savings_pct * self.config.savings_weight * 10
            
            if self.best_price_seen > 0:
                price_quality = 1.0 - (current_price - self.best_price_seen) / self.best_price_seen
                reward += max(0, min(1, price_quality)) * 0.5
            
            reward += self.config.execution_bonus
            
            if urgency > 0.5:
                reward -= time_waiting * self.config.time_penalty * urgency * self.config.urgency_multiplier
            return reward
        else:  # Wait
            reward = -self.config.time_penalty * (1 + urgency * self.config.urgency_multiplier)
            if done:
                missed = (current_price - self.best_price_seen) / self.initial_price
                reward -= missed * self.config.missed_opportunity_penalty * 10
            return reward

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
