"""
Deep Q-Network (DQN) Agent for Transaction Timing

Implements:
- Double DQN for reduced overestimation
- Dueling architecture for better value estimation
- Prioritized experience replay (optional)
- Target network for stability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN agent will not work.")


@dataclass
class DQNConfig:
    """Configuration for DQN agent"""
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128, 64])
    dueling: bool = True  # Use dueling architecture

    # Training parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64
    buffer_size: int = 100000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Target network
    target_update_freq: int = 100  # Steps between target updates
    tau: float = 0.005  # Soft update coefficient

    # Training settings
    min_buffer_size: int = 1000  # Min samples before training
    gradient_clip: float = 1.0


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:

    class DuelingQNetwork(nn.Module):
        """
        Dueling Q-Network architecture

        Separates value and advantage streams for better
        learning when actions have similar values.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_sizes: List[int],
            dueling: bool = True
        ):
            super().__init__()

            self.dueling = dueling

            # Shared feature layers
            layers = []
            prev_size = state_dim
            for size in hidden_sizes[:-1]:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(size))
                prev_size = size

            self.features = nn.Sequential(*layers)

            if dueling:
                # Value stream
                self.value_stream = nn.Sequential(
                    nn.Linear(prev_size, hidden_sizes[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[-1], 1)
                )

                # Advantage stream
                self.advantage_stream = nn.Sequential(
                    nn.Linear(prev_size, hidden_sizes[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[-1], action_dim)
                )
            else:
                # Simple Q-network
                self.q_layer = nn.Sequential(
                    nn.Linear(prev_size, hidden_sizes[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[-1], action_dim)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)

            if self.dueling:
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                # Q = V + (A - mean(A)) for stability
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                q_values = self.q_layer(features)

            return q_values


    class DQNAgent:
        """
        Deep Q-Network Agent for transaction timing

        Features:
        - Double DQN: Uses target network for action selection
        - Dueling architecture: Separate value and advantage streams
        - Soft target updates: Smoother learning
        - Gradient clipping: Training stability
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            config: Optional[DQNConfig] = None,
            device: str = "auto"
        ):
            self.config = config or DQNConfig()
            self.state_dim = state_dim
            self.action_dim = action_dim

            # Device setup
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            print(f"DQN Agent initialized on {self.device}")

            # Networks
            self.policy_net = DuelingQNetwork(
                state_dim,
                action_dim,
                self.config.hidden_sizes,
                self.config.dueling
            ).to(self.device)

            self.target_net = DuelingQNetwork(
                state_dim,
                action_dim,
                self.config.hidden_sizes,
                self.config.dueling
            ).to(self.device)

            # Initialize target network
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            # Optimizer
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.learning_rate
            )

            # Replay buffer
            self.replay_buffer = ReplayBuffer(self.config.buffer_size)

            # Training state
            self.epsilon = self.config.epsilon_start
            self.training_steps = 0
            self.episode_count = 0

            # Metrics tracking
            self.losses = []
            self.rewards_history = []
            self.epsilon_history = []

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """
            Select action using epsilon-greedy policy

            Args:
                state: Current state
                training: Whether in training mode (affects exploration)

            Returns:
                Selected action index
            """
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_dim)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

        def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ):
            """Store transition in replay buffer"""
            self.replay_buffer.push(state, action, reward, next_state, done)

        def train_step(self) -> Optional[float]:
            """
            Perform one training step

            Returns:
                Loss value or None if not enough samples
            """
            if len(self.replay_buffer) < self.config.min_buffer_size:
                return None

            # Sample batch
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config.batch_size)

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Double DQN: Use policy net to select action, target net to evaluate
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * next_q

            # Compute loss
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.gradient_clip
            )

            self.optimizer.step()

            # Update training state
            self.training_steps += 1
            self.losses.append(loss.item())

            # Soft update target network
            if self.training_steps % self.config.target_update_freq == 0:
                self._soft_update_target()

            # Decay epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            self.epsilon_history.append(self.epsilon)

            return loss.item()

        def _soft_update_target(self):
            """Soft update target network parameters"""
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data +
                    (1 - self.config.tau) * target_param.data
                )

        def end_episode(self, total_reward: float):
            """Called at the end of each episode"""
            self.episode_count += 1
            self.rewards_history.append(total_reward)

        def get_metrics(self) -> Dict:
            """Get training metrics"""
            metrics = {
                'training_steps': self.training_steps,
                'episode_count': self.episode_count,
                'epsilon': self.epsilon,
                'buffer_size': len(self.replay_buffer)
            }

            if self.losses:
                metrics['avg_loss'] = np.mean(self.losses[-100:])

            if self.rewards_history:
                metrics['avg_reward'] = np.mean(self.rewards_history[-100:])
                metrics['best_reward'] = max(self.rewards_history)

            return metrics

        def save(self, path: str):
            """Save agent state"""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'episode_count': self.episode_count,
                'config': self.config
            }, path)
            print(f"Agent saved to {path}")

        def load(self, path: str):
            """Load agent state"""
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            self.episode_count = checkpoint['episode_count']
            print(f"Agent loaded from {path}")

        def get_q_values(self, state: np.ndarray) -> np.ndarray:
            """Get Q-values for all actions (for debugging/visualization)"""
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.cpu().numpy()[0]

else:
    # Fallback when PyTorch is not available
    class DQNAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN agent. Install with: pip install torch")


def create_dqn_agent(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> 'DQNAgent':
    """
    Factory function to create DQN agent

    Args:
        state_dim: Dimension of state space
        action_dim: Number of discrete actions
        **kwargs: Override DQNConfig parameters

    Returns:
        Configured DQNAgent
    """
    config = DQNConfig(**kwargs)
    return DQNAgent(state_dim, action_dim, config)
