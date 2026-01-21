"""
RL Agent Service

Manages the trained DQN agent for real-time transaction timing recommendations.
Provides inference, state building, and recommendation generation.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from utils.logger import logger


@dataclass
class AgentRecommendation:
    """Recommendation from the RL agent"""
    action: str                    # WAIT, SUBMIT_NOW, SUBMIT_LOW, SUBMIT_HIGH
    confidence: float              # 0-1 confidence score
    q_values: Dict[str, float]     # Q-values for all actions
    reasoning: str                 # Human-readable explanation
    recommended_gas: float         # Recommended gas price
    expected_savings: float        # Expected savings vs immediate submission
    urgency_factor: float          # How urgent the recommendation considers this


class AgentService:
    """
    Service for managing the RL transaction timing agent

    Handles:
    - Model loading and caching
    - State construction from current market data
    - Inference and recommendation generation
    - Confidence estimation
    """

    ACTION_NAMES = {
        0: 'WAIT',
        1: 'SUBMIT_NOW'
    }

    def __init__(self, models_dir: str = 'models/rl_agents'):
        self.models_dir = models_dir
        self.agent = None
        self.is_loaded = False
        self.load_error = None

        # Statistics for normalization
        self.gas_mean = 0.01  # Default values
        self.gas_std = 0.005
        self.gas_min = 0.001
        self.gas_max = 0.1

        # Price history for velocity/acceleration
        self.price_history = []
        self.max_history = 20

        # Try to load agent on init
        self._try_load_agent()

    def _try_load_agent(self):
        """Attempt to load the trained agent"""
        try:
            from rl.agents.dqn import DQNAgent
            from rl.state import StateBuilder

            # Try different model paths (using .pkl format, not .pt)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_paths = [
                os.path.join(base_dir, self.models_dir, 'dqn_best.pkl'),
                os.path.join(base_dir, self.models_dir, 'dqn_final.pkl'),
                os.path.join(base_dir, 'models', 'rl_agents', 'chain_8453', 'dqn_final.pkl'),
                os.path.join(base_dir, 'models', 'rl_agents', 'chain_8453', 'dqn_best.pkl'),
                os.path.join(base_dir, 'models', 'rl_agents', 'dqn_final.pkl'),
                os.path.join(base_dir, 'models', 'rl_agents', 'dqn_best.pkl'),
                os.path.join(self.models_dir, 'dqn_best.pkl'),
                os.path.join(self.models_dir, 'dqn_final.pkl'),
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                self.load_error = "No trained agent found. Train using notebooks/train_models_colab.ipynb"
                logger.warning(f"RL Agent not loaded: {self.load_error}")
                return False

            # Create state builder to get state dimension
            state_builder = StateBuilder(history_length=24)
            state_dim = state_builder.get_state_dim()

            # Create agent and load weights
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=2  # WAIT or EXECUTE
            )
            self.agent.load(model_path)
            self.agent.epsilon = 0  # Disable exploration for inference

            self.is_loaded = True
            logger.info(f"RL Agent loaded from {model_path}")
            return True

        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load RL agent: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def update_statistics(self, gas_prices: list):
        """Update normalization statistics from recent data"""
        if gas_prices:
            self.gas_mean = np.mean(gas_prices)
            self.gas_std = np.std(gas_prices) + 1e-8
            self.gas_min = np.min(gas_prices)
            self.gas_max = np.max(gas_prices)

    def add_price(self, gas_price: float):
        """Add price to history for velocity calculation"""
        self.price_history.append(gas_price)
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def build_state(
        self,
        current_gas: float,
        predictions: Dict[str, float],
        urgency: float = 0.5,
        steps_remaining: int = 60,
        max_steps: int = 60
    ) -> np.ndarray:
        """
        Build state vector for agent inference

        Args:
            current_gas: Current gas price in gwei
            predictions: Dict with '1h', '4h', '24h' predictions
            urgency: Transaction urgency (0-1)
            steps_remaining: Steps remaining in episode
            max_steps: Maximum steps allowed

        Returns:
            15-dimensional state vector
        """
        # Add to history
        self.add_price(current_gas)

        # Normalize current gas
        gas_norm = (current_gas - self.gas_mean) / self.gas_std
        gas_norm = np.clip(gas_norm, -3, 3) / 3

        # Calculate velocity
        if len(self.price_history) >= 2:
            velocity = (self.price_history[-1] - self.price_history[-2]) / self.gas_std
            velocity = np.clip(velocity, -2, 2) / 2
        else:
            velocity = 0.0

        # Calculate acceleration
        if len(self.price_history) >= 3:
            v1 = self.price_history[-2] - self.price_history[-3]
            v2 = self.price_history[-1] - self.price_history[-2]
            acceleration = (v2 - v1) / self.gas_std
            acceleration = np.clip(acceleration, -2, 2) / 2
        else:
            acceleration = 0.0

        # Normalize predictions
        pred_1h = (predictions.get('1h', current_gas) - self.gas_mean) / self.gas_std
        pred_4h = (predictions.get('4h', current_gas) - self.gas_mean) / self.gas_std
        pred_24h = (predictions.get('24h', current_gas) - self.gas_mean) / self.gas_std

        pred_1h = np.clip(pred_1h, -3, 3) / 3
        pred_4h = np.clip(pred_4h, -3, 3) / 3
        pred_24h = np.clip(pred_24h, -3, 3) / 3

        # Time features
        now = datetime.now()
        hour = now.hour
        day = now.weekday()

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        # Normalized steps remaining
        steps_norm = steps_remaining / max_steps

        # Congestion estimate
        congestion = (current_gas - self.gas_min) / (self.gas_max - self.gas_min + 1e-8)
        congestion = np.clip(congestion, 0, 1)

        # Relative to rolling mean
        if len(self.price_history) >= 5:
            rolling_mean = np.mean(self.price_history[-5:])
            relative = (current_gas - rolling_mean) / (rolling_mean + 1e-8)
            relative = np.clip(relative, -1, 1)
        else:
            relative = 0.0

        # Volatility
        if len(self.price_history) >= 5:
            volatility = np.std(self.price_history[-5:]) / self.gas_std
            volatility = np.clip(volatility, 0, 1)
        else:
            volatility = 0.0

        state = np.array([
            gas_norm,           # 0: Current gas
            velocity,           # 1: Velocity
            acceleration,       # 2: Acceleration
            pred_1h,            # 3: 1h prediction
            pred_4h,            # 4: 4h prediction
            pred_24h,           # 5: 24h prediction
            hour_sin,           # 6: Hour sin
            hour_cos,           # 7: Hour cos
            day_sin,            # 8: Day sin
            day_cos,            # 9: Day cos
            steps_norm,         # 10: Steps remaining
            urgency,            # 11: Urgency
            congestion,         # 12: Congestion
            relative,           # 13: Relative to mean
            volatility,         # 14: Volatility
        ], dtype=np.float32)

        return state

    def get_recommendation(
        self,
        current_gas: float,
        predictions: Dict[str, float],
        urgency: float = 0.5
    ) -> AgentRecommendation:
        """
        Get agent's recommendation for transaction timing

        Args:
            current_gas: Current gas price in gwei
            predictions: Dict with prediction horizons
            urgency: Transaction urgency (0 = no rush, 1 = very urgent)

        Returns:
            AgentRecommendation with action, confidence, and reasoning
        """
        # If agent not loaded, return heuristic recommendation
        if not self.is_loaded or self.agent is None:
            return self._heuristic_recommendation(current_gas, predictions, urgency)

        # Build state
        state = self.build_state(current_gas, predictions, urgency)

        # Get Q-values from agent
        q_values = self.agent.get_q_values(state)

        # Select best action
        action_idx = int(np.argmax(q_values))
        action_name = self.ACTION_NAMES.get(action_idx, 'WAIT')

        # Calculate confidence from Q-value spread
        q_range = float(np.max(q_values) - np.min(q_values))
        confidence = float(min(1.0, q_range / 2.0))  # Normalize to 0-1

        # Build Q-values dict
        q_dict = {self.ACTION_NAMES[i]: float(q_values[i]) for i in range(len(q_values))}

        # Calculate recommended gas price
        recommended_gas = float(current_gas)

        # Estimate savings
        expected_savings = float(self._estimate_savings(action_name, current_gas, predictions))

        # Generate reasoning
        reasoning = self._generate_reasoning(
            action_name, q_values, current_gas, predictions, urgency
        )

        return AgentRecommendation(
            action=action_name,
            confidence=confidence,
            q_values=q_dict,
            reasoning=reasoning,
            recommended_gas=recommended_gas,
            expected_savings=expected_savings,
            urgency_factor=urgency
        )

    def _heuristic_recommendation(
        self,
        current_gas: float,
        predictions: Dict[str, float],
        urgency: float
    ) -> AgentRecommendation:
        """Fallback heuristic when agent not available"""
        pred_1h = predictions.get('1h', current_gas)
        pred_4h = predictions.get('4h', current_gas)

        # Simple heuristic
        if urgency > 0.7:
            action = 'SUBMIT_NOW'
            reasoning = "High urgency - submit immediately"
        elif pred_1h < current_gas * 0.95:
            action = 'WAIT'
            reasoning = f"Price expected to drop {((current_gas - pred_1h) / current_gas * 100):.1f}% in 1h"
        elif current_gas < self.gas_mean * 0.9:
            action = 'SUBMIT_NOW'
            reasoning = "Current price is below average - good time to submit"
        else:
            action = 'WAIT'
            reasoning = "Current price is above average - consider waiting"

        return AgentRecommendation(
            action=action,
            confidence=0.5,  # Low confidence for heuristic
            q_values={'WAIT': 0, 'SUBMIT_NOW': 0},
            reasoning=f"[Heuristic] {reasoning}",
            recommended_gas=current_gas,
            expected_savings=0.0,
            urgency_factor=urgency
        )

    def _estimate_savings(
        self,
        action: str,
        current_gas: float,
        predictions: Dict[str, float]
    ) -> float:
        """Estimate potential savings from waiting"""
        pred_1h = predictions.get('1h', current_gas)

        if action == 'WAIT':
            # Potential savings if predictions are lower
            savings = max(0, current_gas - pred_1h)
            return savings
        else:
            return 0.0

    def _generate_reasoning(
        self,
        action: str,
        q_values: np.ndarray,
        current_gas: float,
        predictions: Dict[str, float],
        urgency: float
    ) -> str:
        """Generate human-readable reasoning for the recommendation"""
        pred_1h = predictions.get('1h', current_gas)
        pred_4h = predictions.get('4h', current_gas)

        if action == 'WAIT':
            if pred_1h < current_gas:
                return f"Gas expected to drop to {pred_1h:.6f} gwei in 1h. Wait for better price."
            else:
                return "Current conditions suggest waiting may yield better prices."

        elif action == 'SUBMIT_NOW':
            if urgency > 0.7:
                return "High urgency transaction - submit at current price."
            elif current_gas < self.gas_mean:
                return f"Current price ({current_gas:.6f}) is below average ({self.gas_mean:.6f}). Good time to submit."
            else:
                return "Optimal submission window detected."


        return "Agent recommendation based on current market conditions."

    def get_status(self) -> Dict:
        """Get agent status and metrics"""
        status = {
            'loaded': self.is_loaded,
            'error': self.load_error,
            'statistics': {
                'gas_mean': self.gas_mean,
                'gas_std': self.gas_std,
                'price_history_length': len(self.price_history)
            }
        }

        if self.is_loaded and self.agent:
            status['agent_metrics'] = {
                'training_steps': getattr(self.agent, 'training_steps', 0),
                'epsilon': getattr(self.agent, 'epsilon', 0),
                'episode_rewards_count': len(getattr(self.agent, 'episode_rewards', []))
            }

        return status


# Global agent service instance
_agent_service = None


def get_agent_service() -> AgentService:
    """Get or create the global agent service"""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
