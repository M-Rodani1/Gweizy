"""
OpenAI Gym-style environment for gas price optimization.
"""
import numpy as np
from typing import Tuple, Optional, Dict, List
from .state import StateBuilder, GasState
from .rewards import RewardCalculator, RewardConfig
from .data_loader import GasDataLoader


class GasOptimizationEnv:
    """
    RL Environment for learning optimal transaction timing.
    
    Actions:
        0: Wait (don't execute transaction yet)
        1: Execute (submit transaction now)
    
    State: Vector containing price history, time features, urgency, etc.
    
    Reward: Based on gas savings, time penalties, and execution quality.
    """

    def __init__(
        self,
        data_loader: Optional[GasDataLoader] = None,
        episode_length: int = 48,
        urgency_range: Tuple[float, float] = (0.1, 0.9),
        max_wait_steps: int = 100
    ):
        self.data_loader = data_loader or GasDataLoader()
        self.episode_length = episode_length
        self.urgency_range = urgency_range
        self.max_wait_steps = max_wait_steps
        
        self.state_builder = StateBuilder(history_length=24)
        self.reward_calculator = RewardCalculator()
        
        self.action_space_n = 2  # Wait or Execute
        self.observation_space_shape = (self.state_builder.get_state_dim(),)
        
        self._price_data = None
        self._price_stats = None
        self._current_step = 0
        self._urgency = 0.5
        self._time_waiting = 0
        self._done = False
        self._episode_data = None

    def reset(self, urgency: Optional[float] = None) -> np.ndarray:
        """Reset environment for new episode. REQUIRES REAL DATA."""
        # Load episode data (will raise if no real data available)
        episodes = self.data_loader.get_episodes(self.episode_length, num_episodes=1)
        if not episodes:
            raise ValueError("No real data available. Cannot create episode.")
        self._episode_data = episodes[0]
        
        self._price_stats = self.data_loader.get_statistics()
        self._current_step = 0
        self._time_waiting = 0
        self._done = False
        
        # Random or specified urgency
        self._urgency = urgency if urgency is not None else np.random.uniform(*self.urgency_range)
        
        # Initialize reward calculator
        initial_price = self._episode_data[0]['gas_price']
        self.reward_calculator.reset(initial_price)
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.
        
        Returns:
            observation: New state
            reward: Reward for action
            done: Whether episode ended
            info: Additional info
        """
        if self._done:
            raise RuntimeError("Episode already done. Call reset().")
        
        current_data = self._episode_data[self._current_step]
        current_price = current_data['gas_price']
        max_steps = len(self._episode_data) - 1
        
        # Check if we're at the deadline and haven't executed yet
        if self._current_step >= max_steps and action == 0:
            # FORCED EXECUTION: Agent must execute at deadline
            action = 1  # Force execution
            deadline_penalty = -100.0  # Large penalty for missing deadline
            info = {
                'price': current_price,
                'step': self._current_step,
                'urgency': self._urgency,
                'time_waiting': self._time_waiting,
                'action_taken': 'forced_execute',
                'forced_execution': True,
                'deadline_penalty': deadline_penalty
            }
        else:
            info = {
                'price': current_price,
                'step': self._current_step,
                'urgency': self._urgency,
                'time_waiting': self._time_waiting,
                'action_taken': 'execute' if action == 1 else 'wait',
                'forced_execution': False
            }
            deadline_penalty = 0.0
        
        # Get current volatility for reward shaping
        current_volatility = 0.0
        if self._current_step > 0:
            price_history = [d['gas_price'] for d in self._episode_data[max(0, self._current_step-6):self._current_step+1]]
            if len(price_history) >= 2:
                current_volatility = np.std(price_history) / (np.mean(price_history) + 1e-8)
                current_volatility = min(current_volatility, 1.0)  # Cap at 1.0
        
        # Calculate reward using relative savings (benchmark_price - execution_price)
        reward = self.reward_calculator.calculate_reward(
            action=action,
            current_price=current_price,
            urgency=self._urgency,
            time_waiting=self._time_waiting,
            done=False,
            volatility=current_volatility
        )
        
        # Apply deadline penalty if forced execution
        reward += deadline_penalty
        
        # Check termination conditions
        if action == 1:  # Executed (either by choice or forced)
            self._done = True
            info['execution_price'] = current_price
            # Relative savings: benchmark (initial) - execution price
            info['savings_vs_initial'] = (self.reward_calculator.initial_price - current_price) / self.reward_calculator.initial_price
        else:  # Waited
            self._time_waiting += 1
            self._current_step += 1
            
            # Check if we've run out of time (shouldn't happen due to forced execution above, but keep as safety)
            if self._current_step > max_steps:
                self._done = True
                # Force execution with penalty
                final_price = self._episode_data[-1]['gas_price']
                final_volatility = current_volatility  # Use current volatility
                reward += self.reward_calculator.calculate_reward(1, final_price, self._urgency, self._time_waiting, done=True, volatility=final_volatility)
                reward += -100.0  # Deadline penalty
                info['forced_execution'] = True
                info['execution_price'] = final_price
                info['savings_vs_initial'] = (self.reward_calculator.initial_price - final_price) / self.reward_calculator.initial_price
        
        obs = self._get_observation() if not self._done else np.zeros(self.observation_space_shape)
        
        return obs, reward, self._done, info

    def _get_observation(self) -> np.ndarray:
        """Build current observation."""
        if self._current_step >= len(self._episode_data):
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        
        current = self._episode_data[self._current_step]
        
        # Get price history
        history_start = max(0, self._current_step - 24)
        price_history = [d['gas_price'] for d in self._episode_data[history_start:self._current_step]]
        
        # Calculate volatility and momentum
        if len(price_history) >= 2:
            volatility = np.std(price_history) / (np.mean(price_history) + 1e-8)
            momentum = (price_history[-1] - price_history[0]) / (price_history[0] + 1e-8)
        else:
            volatility = 0.0
            momentum = 0.0
        
        gas_state = GasState(
            current_price=current['gas_price'],
            price_history=price_history,
            hour=current['hour'],
            day_of_week=current['day_of_week'],
            volatility=volatility,
            momentum=momentum,
            urgency=self._urgency,
            time_waiting=self._time_waiting
        )
        
        return self.state_builder.build_state(gas_state, self._price_stats)

    def render(self):
        """Print current state."""
        if self._current_step < len(self._episode_data):
            current = self._episode_data[self._current_step]
            print(f"Step {self._current_step}: Price={current['gas_price']:.6f} gwei, "
                  f"Urgency={self._urgency:.2f}, Waiting={self._time_waiting}")
