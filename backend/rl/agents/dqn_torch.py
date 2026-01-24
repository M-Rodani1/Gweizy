"""
PyTorch-based Deep Q-Network (DQN) agent for gas price optimization.
Features: Prioritized Experience Replay, Double DQN, Dueling Architecture, LayerNorm.
"""
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Optional
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SumTree:
    """SumTree for efficient prioritized experience replay sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: Tuple):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, Tuple, float]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.data[data_idx], self.tree[idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.min_priority = 1e-6

    def push(self, state, action, reward, next_state, done, td_error: float = None):
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + self.min_priority) ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, data, priority = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, np.array(indices), is_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


class ReplayBuffer:
    """Standard experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done, td_error: float = None):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch, np.array([]), np.ones(len(batch))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        pass

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    """N-step return buffer for computing multi-step TD targets."""

    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)

    def push(self, state, action, reward, next_state, done) -> Optional[Tuple]:
        """Add transition and return n-step transition if ready."""
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) < self.n_steps:
            return None

        # Compute n-step return
        n_step_reward = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                break

        # Get first state/action and last next_state/done
        first_state, first_action, _, _, _ = self.buffer[0]
        _, _, _, last_next_state, last_done = self.buffer[-1]

        return (first_state, first_action, n_step_reward, last_next_state, last_done)

    def flush(self) -> List[Tuple]:
        """Flush remaining transitions at episode end."""
        transitions = []
        while len(self.buffer) > 0:
            # Compute partial n-step return
            n_step_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break

            first_state, first_action, _, _, _ = self.buffer[0]
            _, _, _, last_next_state, last_done = self.buffer[-1]

            transitions.append((first_state, first_action, n_step_reward, last_next_state, last_done))
            self.buffer.popleft()

        return transitions

    def reset(self):
        """Reset buffer for new episode."""
        self.buffer.clear()


class RewardNormalizer:
    """Running reward normalization for stable training."""

    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-8):
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self.count += 1

        # Welford's online algorithm for running mean/variance
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_var += delta * delta2

        # Compute standard deviation
        std = np.sqrt(self.running_var / max(1, self.count - 1)) + self.epsilon

        # Normalize and clip
        normalized = (reward - self.running_mean) / std
        return float(np.clip(normalized, -self.clip_range, self.clip_range))

    def reset_stats(self):
        """Reset running statistics."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0


if TORCH_AVAILABLE:
    class NoisyLinear(nn.Module):
        """Noisy linear layer for exploration (Fortunato et al., 2017)."""

        def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            # Learnable parameters
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))

            # Factorized noise buffers
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))

            self._init_parameters()
            self.reset_noise()

        def _init_parameters(self):
            mu_range = 1 / np.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

        def _scale_noise(self, size: int) -> torch.Tensor:
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign() * x.abs().sqrt()

        def reset_noise(self):
            """Sample new noise."""
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)
    class DuelingQNetwork(nn.Module):
        """Dueling DQN architecture with LayerNorm for stability."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dims = hidden_dims

            # Shared feature layers
            layers = []
            in_dim = state_dim
            for hidden_dim in hidden_dims[:-1]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.shared = nn.Sequential(*layers) if layers else nn.Identity()

            # Get output dim of shared layers
            shared_out_dim = hidden_dims[-2] if len(hidden_dims) > 1 else state_dim

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(shared_out_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(shared_out_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            shared = self.shared(x)
            value = self.value_stream(shared)
            advantage = self.advantage_stream(shared)

            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
            return q_values


    class StandardQNetwork(nn.Module):
        """Standard DQN architecture with LayerNorm."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dims = hidden_dims

            layers = []
            in_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, action_dim))

            self.network = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.network(x)


    class NoisyDuelingQNetwork(nn.Module):
        """Dueling DQN with Noisy layers for exploration."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dims = hidden_dims

            # Shared feature layers (standard linear + LayerNorm)
            layers = []
            in_dim = state_dim
            for hidden_dim in hidden_dims[:-1]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.shared = nn.Sequential(*layers) if layers else nn.Identity()

            shared_out_dim = hidden_dims[-2] if len(hidden_dims) > 1 else state_dim

            # Value stream with noisy output
            self.value_hidden = nn.Sequential(
                NoisyLinear(shared_out_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU()
            )
            self.value_out = NoisyLinear(hidden_dims[-1], 1)

            # Advantage stream with noisy output
            self.advantage_hidden = nn.Sequential(
                NoisyLinear(shared_out_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU()
            )
            self.advantage_out = NoisyLinear(hidden_dims[-1], action_dim)

            self._init_shared_weights()

        def _init_shared_weights(self):
            for m in self.shared.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        def reset_noise(self):
            """Reset noise for all noisy layers."""
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)

            shared = self.shared(x)
            value = self.value_out(self.value_hidden(shared))
            advantage = self.advantage_out(self.advantage_hidden(shared))

            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
            return q_values


class DQNAgent:
    """
    PyTorch-based DQN Agent for gas price optimization.

    Features:
    - Prioritized Experience Replay (PER)
    - Double DQN (DDQN)
    - Dueling Network Architecture
    - LayerNorm for training stability
    - Proper gradient clipping
    - N-step returns for better credit assignment
    - Reward normalization for stable training
    - Noisy networks for exploration (optional)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        epsilon_decay_episodes: Optional[int] = None,
        epsilon_decay_steps: Optional[int] = None,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        lr_decay: float = 0.9999,
        lr_min: float = 0.00001,
        gradient_clip: float = 1.0,
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        target_update_tau: float = 0.005,
        use_soft_target: bool = True,
        device: str = None,
        # Phase 2 features
        n_steps: int = 3,  # N-step returns (1 = standard TD)
        use_reward_norm: bool = True,  # Reward normalization
        use_noisy_nets: bool = False  # Noisy networks for exploration
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent. Install with: pip install torch")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.target_update_tau = target_update_tau
        self.use_soft_target = use_soft_target

        # Phase 2 features
        self.n_steps = n_steps
        self.use_reward_norm = use_reward_norm
        self.use_noisy_nets = use_noisy_nets

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # State normalization
        self.state_mean = None
        self.state_std = None

        # Networks - use noisy version if enabled
        if use_noisy_nets and use_dueling:
            self.q_network = NoisyDuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = NoisyDuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        elif use_dueling:
            self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            self.q_network = StandardQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = StandardQNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)

        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta=per_beta
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

        # N-step buffer for computing multi-step returns
        self.n_step_buffer = NStepBuffer(n_steps=n_steps, gamma=gamma) if n_steps > 1 else None

        # Reward normalizer
        self.reward_normalizer = RewardNormalizer() if use_reward_norm else None

        # Training stats
        self.training_steps = 0
        self.episode_rewards = []
        self.last_td_errors = None

        # For compatibility with numpy version
        self.dueling = use_dueling

        if self.epsilon_decay_steps is not None and self.epsilon_decay_episodes is not None:
            logger.warning("Both epsilon_decay_steps and epsilon_decay_episodes set; using step-based decay.")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy (or noisy nets if enabled)."""
        # Noisy nets handle exploration internally, no epsilon needed
        if not self.use_noisy_nets and training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = self._align_state_features(state)
        state = self._normalize_state(state)
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax().item())

    def fit_state_normalizer(self, states: np.ndarray):
        """Fit normalization statistics from training states."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-8

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using fitted statistics."""
        if self.state_mean is None or self.state_std is None:
            return state
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return (state - self.state_mean) / self.state_std

    def decay_epsilon(self, episode: int):
        """Decay epsilon based on episode count."""
        if self.epsilon_decay_steps is not None:
            return
        if self.epsilon_decay_episodes is None:
            return
        if episode < self.epsilon_decay_episodes:
            decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes
            self.epsilon = self.epsilon_start - decay_rate * episode
        else:
            self.epsilon = self.epsilon_end
        self.epsilon = max(self.epsilon_end, min(self.epsilon_start, self.epsilon))

    def store_transition(self, state, action, reward, next_state, done, td_error: float = None):
        """Store transition in replay buffer with n-step returns and reward normalization."""
        # Normalize reward if enabled
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize(reward)

        # Use n-step buffer if enabled
        if self.n_step_buffer is not None:
            n_step_transition = self.n_step_buffer.push(state, action, reward, next_state, done)
            if n_step_transition is not None:
                self.replay_buffer.push(*n_step_transition, td_error=td_error)

            # Flush remaining transitions at episode end
            if done:
                for transition in self.n_step_buffer.flush():
                    self.replay_buffer.push(*transition, td_error=td_error)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done, td_error=td_error)

    def reset_episode(self):
        """Reset episode-specific state (call at start of each episode)."""
        if self.n_step_buffer is not None:
            self.n_step_buffer.reset()

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Reset noise for noisy networks
        if self.use_noisy_nets and hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Sample batch
        batch, indices, is_weights = self.replay_buffer.sample(self.batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        # Align and normalize
        states = self._align_state_features(states)
        next_states = self._align_state_features(next_states)
        states = self._normalize_state(states)
        next_states = self._normalize_state(next_states)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).to(self.device)

        # Current Q-values
        current_q = self.q_network(states_t)
        current_q_actions = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (use gamma^n for n-step returns)
        n_step_gamma = self.gamma ** self.n_steps
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: select action with online, evaluate with target
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q_values = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states_t).max(dim=1)[0]

            targets = rewards_t + n_step_gamma * next_q_values * (1 - dones_t)

        # TD errors for PER
        td_errors = (targets - current_q_actions).abs().detach().cpu().numpy()
        self.last_td_errors = td_errors

        # Update priorities
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(indices, td_errors)

        # Compute loss with importance sampling weights
        loss = (is_weights_t * F.smooth_l1_loss(current_q_actions, targets, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.use_soft_target:
            self._soft_update_target()
        elif self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay (step-based)
        if self.epsilon_decay_steps is not None:
            if self.training_steps < self.epsilon_decay_steps:
                decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
                self.epsilon = self.epsilon_start - decay_rate * self.training_steps
            else:
                self.epsilon = self.epsilon_end
            self.epsilon = max(self.epsilon_end, min(self.epsilon_start, self.epsilon))
        elif self.epsilon_decay_episodes is None:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Learning rate decay
        if self.optimizer.param_groups[0]['lr'] > self.lr_min:
            self.scheduler.step()
        self.learning_rate = self.optimizer.param_groups[0]['lr']

        return float(loss.item())

    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        state = self._align_state_features(state)
        state = self._normalize_state(state)
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy().flatten()

    def _align_state_features(self, state: np.ndarray) -> np.ndarray:
        """Align state features to match model's expected input dimension."""
        if state.ndim == 1:
            state = state.reshape(1, -1)

        current_dim = state.shape[1]
        expected_dim = self.state_dim

        if current_dim == expected_dim:
            return state

        if current_dim > expected_dim:
            return state[:, :expected_dim].copy()
        else:
            padding = np.zeros((state.shape[0], expected_dim - current_dim), dtype=state.dtype)
            return np.concatenate([state, padding], axis=1)

    def get_recommendation(self, state: np.ndarray, threshold: float = 0.6) -> dict:
        """Get action recommendation with confidence."""
        state = self._align_state_features(state)
        q_values = self.get_q_values(state)

        # Softmax for confidence
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        action = int(np.argmax(q_values))
        confidence = float(probs[action])

        return {
            'action': 'execute' if action == 1 else 'wait',
            'action_id': action,
            'confidence': confidence,
            'q_values': {'wait': float(q_values[0]), 'execute': float(q_values[1])},
            'should_act': action == 1 and confidence >= threshold
        }

    def save(self, path: str):
        """Save agent to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_decay_episodes': self.epsilon_decay_episodes,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'training_steps': self.training_steps,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'episode_rewards': self.episode_rewards,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'use_dueling': self.use_dueling,
            'target_update_tau': self.target_update_tau,
            'use_soft_target': self.use_soft_target,
            # Phase 2 parameters
            'n_steps': self.n_steps,
            'use_reward_norm': self.use_reward_norm,
            'use_noisy_nets': self.use_noisy_nets,
            'backend': 'pytorch'
        }

        torch.save(data, path)

    def load(self, path: str):
        """Load agent from file."""
        data = torch.load(path, map_location=self.device)

        # Check if this is a PyTorch model
        if data.get('backend') != 'pytorch':
            logger.warning("Loading numpy model into PyTorch agent - weights will be randomly initialized")
            return

        # Load network weights
        self.q_network.load_state_dict(data['q_network_state_dict'])
        self.target_network.load_state_dict(data['target_network_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])

        # Load training state
        self.epsilon = data.get('epsilon', 0.01)
        self.epsilon_start = data.get('epsilon_start', self.epsilon_start)
        self.epsilon_decay = data.get('epsilon_decay', self.epsilon_decay)
        self.epsilon_decay_episodes = data.get('epsilon_decay_episodes', self.epsilon_decay_episodes)
        self.epsilon_decay_steps = data.get('epsilon_decay_steps', self.epsilon_decay_steps)
        self.training_steps = data.get('training_steps', 0)
        self.episode_rewards = data.get('episode_rewards', [])
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.target_update_tau = data.get('target_update_tau', self.target_update_tau)
        self.use_soft_target = data.get('use_soft_target', self.use_soft_target)

        # Load normalizer
        self.state_mean = data.get('state_mean')
        self.state_std = data.get('state_std')

    def get_state_dict(self) -> dict:
        """Get agent state for ensemble saving."""
        return {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'hidden_dims': self.hidden_dims,
            'use_dueling': self.use_dueling,
            'n_steps': self.n_steps,
            'use_reward_norm': self.use_reward_norm,
            'use_noisy_nets': self.use_noisy_nets
        }

    def load_state_dict(self, state_dict: dict):
        """Load agent state from ensemble."""
        self.q_network.load_state_dict(state_dict['q_network_state_dict'])
        self.target_network.load_state_dict(state_dict['target_network_state_dict'])
        self.epsilon = state_dict.get('epsilon', 0.01)
        self.state_mean = state_dict.get('state_mean')
        self.state_std = state_dict.get('state_std')

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (for ensemble use)."""
        # Normalize state
        if self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.cpu().numpy().flatten()
