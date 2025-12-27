"""
Reinforcement Learning Transaction Agent

This module implements an RL agent that learns optimal transaction timing
by using gas price predictions and historical data to make decisions.

Components:
- GasTransactionEnv: OpenAI Gym environment for transaction simulation
- Rewards: Various reward function designs
- Agents: DQN, PPO implementations
- Training: Training loops and utilities
"""

from .environment import GasTransactionEnv, Action, TransactionConfig
from .rewards import RewardCalculator, RewardType, create_reward_calculator
from .state import StateBuilder, StateNormalizer, create_state_builder
from .data_loader import RLDataLoader, ReplayBuffer, create_training_data

__all__ = [
    # Environment
    'GasTransactionEnv',
    'Action',
    'TransactionConfig',
    # Rewards
    'RewardCalculator',
    'RewardType',
    'create_reward_calculator',
    # State
    'StateBuilder',
    'StateNormalizer',
    'create_state_builder',
    # Data
    'RLDataLoader',
    'ReplayBuffer',
    'create_training_data',
]
__version__ = '0.1.0'
