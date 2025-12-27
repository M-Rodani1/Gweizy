"""
Reward Function Designs for RL Transaction Agent

Multiple reward shaping strategies for different optimization objectives:
- Cost minimization (lowest gas price)
- Speed optimization (fastest confirmation)
- Multi-objective (balance of cost, speed, reliability)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np


class RewardType(Enum):
    """Different reward function strategies"""
    COST_FOCUSED = "cost_focused"           # Minimize gas cost
    SPEED_FOCUSED = "speed_focused"         # Minimize wait time
    BALANCED = "balanced"                    # Balance cost and speed
    MULTI_OBJECTIVE = "multi_objective"     # Configurable weights
    ADAPTIVE = "adaptive"                    # Adjusts based on market


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Weight parameters (should sum to 1.0 for multi-objective)
    cost_weight: float = 0.4
    speed_weight: float = 0.3
    reliability_weight: float = 0.3

    # Penalty parameters
    failure_penalty: float = -10.0
    timeout_penalty: float = -5.0

    # Bonus parameters
    optimal_timing_bonus: float = 2.0
    below_average_bonus: float = 1.0

    # Normalization
    gas_price_baseline: float = 0.01  # Typical gas price in gwei
    max_wait_steps: int = 60

    # Adaptive reward parameters
    volatility_threshold: float = 0.05


class RewardCalculator:
    """
    Calculates rewards for transaction decisions

    The reward function is crucial for RL - it defines what behavior
    we want the agent to learn. Poor reward design leads to poor policies.
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.BALANCED,
        config: Optional[RewardConfig] = None
    ):
        self.reward_type = reward_type
        self.config = config or RewardConfig()

        # Historical tracking for adaptive rewards
        self.gas_history = []
        self.reward_history = []

    def calculate(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for a transaction decision

        Args:
            action: Action taken (0=wait, 1=submit_now, 2=submit_low, 3=submit_high)
            gas_price_paid: Actual gas price paid (or attempted)
            wait_steps: Number of steps waited before submission
            success: Whether transaction succeeded
            context: Additional context (predictions, market state, etc.)

        Returns:
            (total_reward, reward_breakdown)
        """
        if self.reward_type == RewardType.COST_FOCUSED:
            return self._cost_focused_reward(
                action, gas_price_paid, wait_steps, success, context
            )
        elif self.reward_type == RewardType.SPEED_FOCUSED:
            return self._speed_focused_reward(
                action, gas_price_paid, wait_steps, success, context
            )
        elif self.reward_type == RewardType.BALANCED:
            return self._balanced_reward(
                action, gas_price_paid, wait_steps, success, context
            )
        elif self.reward_type == RewardType.MULTI_OBJECTIVE:
            return self._multi_objective_reward(
                action, gas_price_paid, wait_steps, success, context
            )
        elif self.reward_type == RewardType.ADAPTIVE:
            return self._adaptive_reward(
                action, gas_price_paid, wait_steps, success, context
            )
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _cost_focused_reward(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Reward focused purely on minimizing gas cost

        Best for: Non-urgent transactions, DEX trades, batch operations
        """
        breakdown = {}

        if not success:
            breakdown['failure'] = self.config.failure_penalty
            return self.config.failure_penalty, breakdown

        # Calculate savings relative to baseline
        baseline = context.get('average_gas', self.config.gas_price_baseline)
        savings_ratio = (baseline - gas_price_paid) / (baseline + 1e-8)

        # Reward for paying below average
        cost_reward = savings_ratio * 5.0  # Scale to reasonable range
        breakdown['cost_savings'] = cost_reward

        # Small penalty for waiting too long (opportunity cost)
        wait_penalty = -0.01 * wait_steps
        breakdown['wait_penalty'] = wait_penalty

        # Bonus for hitting near-optimal price
        min_price = context.get('min_gas_seen', gas_price_paid)
        if gas_price_paid <= min_price * 1.05:  # Within 5% of best seen
            breakdown['optimal_bonus'] = self.config.optimal_timing_bonus
        else:
            breakdown['optimal_bonus'] = 0.0

        total = sum(breakdown.values())
        return total, breakdown

    def _speed_focused_reward(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Reward focused on fast transaction confirmation

        Best for: Urgent trades, time-sensitive operations, NFT mints
        """
        breakdown = {}

        if not success:
            breakdown['failure'] = self.config.failure_penalty
            return self.config.failure_penalty, breakdown

        # Strong reward for quick submission
        max_steps = self.config.max_wait_steps
        speed_reward = 3.0 * (1 - wait_steps / max_steps)
        breakdown['speed_reward'] = speed_reward

        # Bonus for immediate submission
        if wait_steps <= 3:
            breakdown['quick_bonus'] = 1.5
        else:
            breakdown['quick_bonus'] = 0.0

        # Small consideration for cost (don't overpay massively)
        baseline = context.get('average_gas', self.config.gas_price_baseline)
        if gas_price_paid > baseline * 1.5:  # More than 50% above average
            breakdown['overpay_penalty'] = -1.0
        else:
            breakdown['overpay_penalty'] = 0.0

        total = sum(breakdown.values())
        return total, breakdown

    def _balanced_reward(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Balanced reward considering cost, speed, and reliability

        Best for: General-purpose transactions, smart defaults
        """
        breakdown = {}

        if not success:
            breakdown['failure'] = self.config.failure_penalty
            return self.config.failure_penalty, breakdown

        # Cost component
        baseline = context.get('average_gas', self.config.gas_price_baseline)
        savings_ratio = (baseline - gas_price_paid) / (baseline + 1e-8)
        cost_reward = np.clip(savings_ratio * 3.0, -2.0, 2.0)
        breakdown['cost'] = cost_reward

        # Speed component
        max_steps = self.config.max_wait_steps
        speed_ratio = 1 - (wait_steps / max_steps)
        speed_reward = speed_ratio * 2.0
        breakdown['speed'] = speed_reward

        # Timing bonus - reward for good market timing
        prediction_1h = context.get('prediction_1h', gas_price_paid)
        if gas_price_paid < prediction_1h * 0.95:  # Paid less than predicted
            breakdown['timing_bonus'] = 1.0
        else:
            breakdown['timing_bonus'] = 0.0

        # Reliability bonus - consistent good decisions
        breakdown['reliability'] = 0.5  # Base reliability for success

        total = sum(breakdown.values())
        return total, breakdown

    def _multi_objective_reward(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Configurable multi-objective reward with explicit weights

        Best for: Custom optimization goals, A/B testing strategies
        """
        breakdown = {}

        if not success:
            breakdown['failure'] = self.config.failure_penalty
            return self.config.failure_penalty, breakdown

        # Cost objective (normalized to [0, 1])
        baseline = context.get('average_gas', self.config.gas_price_baseline)
        cost_score = np.clip(1 - (gas_price_paid / baseline), 0, 1)

        # Speed objective (normalized to [0, 1])
        speed_score = 1 - (wait_steps / self.config.max_wait_steps)

        # Reliability objective (binary for now, could be probabilistic)
        reliability_score = 1.0 if success else 0.0

        # Weighted combination
        weighted_cost = self.config.cost_weight * cost_score * 5.0
        weighted_speed = self.config.speed_weight * speed_score * 5.0
        weighted_reliability = self.config.reliability_weight * reliability_score * 5.0

        breakdown['weighted_cost'] = weighted_cost
        breakdown['weighted_speed'] = weighted_speed
        breakdown['weighted_reliability'] = weighted_reliability

        total = sum(breakdown.values())
        return total, breakdown

    def _adaptive_reward(
        self,
        action: int,
        gas_price_paid: float,
        wait_steps: int,
        success: bool,
        context: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Adaptive reward that adjusts based on market conditions

        In high volatility: Rewards quick action
        In low volatility: Rewards patience for better prices

        Best for: Dynamic market conditions, production use
        """
        breakdown = {}

        if not success:
            breakdown['failure'] = self.config.failure_penalty
            return self.config.failure_penalty, breakdown

        # Get market volatility
        volatility = context.get('volatility', 0.02)
        is_volatile = volatility > self.config.volatility_threshold

        if is_volatile:
            # High volatility - favor speed and reliability
            speed_weight = 0.5
            cost_weight = 0.2
            reliability_weight = 0.3
        else:
            # Low volatility - favor cost optimization
            speed_weight = 0.2
            cost_weight = 0.5
            reliability_weight = 0.3

        # Calculate components
        baseline = context.get('average_gas', self.config.gas_price_baseline)
        cost_score = np.clip(1 - (gas_price_paid / baseline), -0.5, 1)
        speed_score = 1 - (wait_steps / self.config.max_wait_steps)
        reliability_score = 1.0

        # Apply adaptive weights
        breakdown['cost'] = cost_weight * cost_score * 5.0
        breakdown['speed'] = speed_weight * speed_score * 5.0
        breakdown['reliability'] = reliability_weight * reliability_score * 5.0
        breakdown['volatility_mode'] = 0.5 if is_volatile else 0.0

        # Track for future adaptation
        self.gas_history.append(gas_price_paid)
        if len(self.gas_history) > 1000:
            self.gas_history = self.gas_history[-500:]

        total = sum(breakdown.values())
        self.reward_history.append(total)

        return total, breakdown

    def get_reward_stats(self) -> Dict:
        """Get statistics about recent rewards"""
        if not self.reward_history:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        recent = self.reward_history[-100:]
        return {
            'mean': np.mean(recent),
            'std': np.std(recent),
            'min': np.min(recent),
            'max': np.max(recent),
            'count': len(self.reward_history)
        }


def create_reward_calculator(
    objective: str = "balanced",
    **kwargs
) -> RewardCalculator:
    """
    Factory function to create reward calculators

    Args:
        objective: One of "cost", "speed", "balanced", "multi", "adaptive"
        **kwargs: Override RewardConfig parameters

    Returns:
        Configured RewardCalculator
    """
    type_map = {
        "cost": RewardType.COST_FOCUSED,
        "speed": RewardType.SPEED_FOCUSED,
        "balanced": RewardType.BALANCED,
        "multi": RewardType.MULTI_OBJECTIVE,
        "adaptive": RewardType.ADAPTIVE
    }

    reward_type = type_map.get(objective, RewardType.BALANCED)
    config = RewardConfig(**kwargs)

    return RewardCalculator(reward_type=reward_type, config=config)
