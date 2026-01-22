"""
Deep Q-Network (DQN) agent for gas price optimization.
Enhanced with Prioritized Experience Replay, Double DQN, and Dueling Architecture.
"""
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Optional
from collections import deque
import random

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
        """Update tree nodes from leaf to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node with given sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Tuple):
        """Add data with priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, Tuple, float]:
        """Get sample with given sum value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.data[data_idx], self.tree[idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (starts at beta, goes to 1.0)
            beta_increment: How much to increment beta per sample
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.min_priority = 1e-6
    
    def push(self, state, action, reward, next_state, done, td_error: float = None):
        """Add transition with priority based on TD error."""
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + self.min_priority) ** self.alpha
        
        self.tree.add(priority, (state, action, reward, next_state, done))
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        Returns: (batch, indices, importance_weights)
        """
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
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize
        
        return batch, np.array(indices), is_weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries


class ReplayBuffer:
    """Standard experience replay buffer for DQN training (fallback)."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, td_error: float = None):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        # Return dummy indices and weights for compatibility
        return batch, np.array([]), np.ones(len(batch))
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        pass  # No-op for standard buffer
    
    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """Simple neural network for Q-value approximation using numpy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64], dueling: bool = False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dueling = dueling
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
        if dueling:
            # Dueling architecture: shared layers + value stream + advantage stream
            # Shared layers
            dims_shared = [state_dim] + hidden_dims[:-1]  # All but last hidden layer
            for i in range(len(dims_shared) - 1):
                w = np.random.randn(dims_shared[i], dims_shared[i+1]) * np.sqrt(2.0 / dims_shared[i])
                b = np.zeros(dims_shared[i+1])
                self.weights.append(w)
                self.biases.append(b)
            
            # Value stream: state value V(s)
            last_shared_dim = dims_shared[-1] if dims_shared else state_dim
            self.value_weights = []
            self.value_biases = []
            value_dims = [last_shared_dim] + [hidden_dims[-1] if hidden_dims else 64] + [1]
            for i in range(len(value_dims) - 1):
                w = np.random.randn(value_dims[i], value_dims[i+1]) * np.sqrt(2.0 / value_dims[i])
                b = np.zeros(value_dims[i+1])
                self.value_weights.append(w)
                self.value_biases.append(b)
            
            # Advantage stream: action advantage A(s, a)
            self.advantage_weights = []
            self.advantage_biases = []
            advantage_dims = [last_shared_dim] + [hidden_dims[-1] if hidden_dims else 64] + [action_dim]
            for i in range(len(advantage_dims) - 1):
                w = np.random.randn(advantage_dims[i], advantage_dims[i+1]) * np.sqrt(2.0 / advantage_dims[i])
                b = np.zeros(advantage_dims[i+1])
                self.advantage_weights.append(w)
                self.advantage_biases.append(b)
        else:
            # Standard architecture
            dims = [state_dim] + hidden_dims + [action_dim]
            for i in range(len(dims) - 1):
                # Xavier initialization
                w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
                b = np.zeros(dims[i+1])
                self.weights.append(w)
                self.biases.append(b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if self.dueling:
            # Shared layers
            shared = x
            for i in range(len(self.weights)):
                shared = np.dot(shared, self.weights[i]) + self.biases[i]
                shared = np.maximum(0, shared)  # ReLU
            
            # Value stream: V(s)
            value = shared
            for i in range(len(self.value_weights) - 1):
                value = np.dot(value, self.value_weights[i]) + self.value_biases[i]
                value = np.maximum(0, value)  # ReLU
            value = np.dot(value, self.value_weights[-1]) + self.value_biases[-1]
            
            # Advantage stream: A(s, a)
            advantage = shared
            for i in range(len(self.advantage_weights) - 1):
                advantage = np.dot(advantage, self.advantage_weights[i]) + self.advantage_biases[i]
                advantage = np.maximum(0, advantage)  # ReLU
            advantage = np.dot(advantage, self.advantage_weights[-1]) + self.advantage_biases[-1]
            
            # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            # This ensures identifiability and stability
            q_values = value + (advantage - np.mean(advantage, axis=-1, keepdims=True))
            return q_values
        else:
            # Standard forward pass
            for i in range(len(self.weights) - 1):
                x = np.dot(x, self.weights[i]) + self.biases[i]
                x = np.maximum(0, x)  # ReLU
            
            # Output layer (no activation)
            x = np.dot(x, self.weights[-1]) + self.biases[-1]
            return x
    
    def copy_from(self, other: 'QNetwork'):
        """Copy weights from another network."""
        if self.dueling and other.dueling:
            # Copy shared layers
            for i in range(len(self.weights)):
                self.weights[i] = other.weights[i].copy()
                self.biases[i] = other.biases[i].copy()
            # Copy value stream
            for i in range(len(self.value_weights)):
                self.value_weights[i] = other.value_weights[i].copy()
                self.value_biases[i] = other.value_biases[i].copy()
            # Copy advantage stream
            for i in range(len(self.advantage_weights)):
                self.advantage_weights[i] = other.advantage_weights[i].copy()
                self.advantage_biases[i] = other.advantage_biases[i].copy()
        elif not self.dueling and not other.dueling:
            # Standard copy
            for i in range(len(self.weights)):
                self.weights[i] = other.weights[i].copy()
                self.biases[i] = other.biases[i].copy()
        else:
            raise ValueError("Cannot copy between dueling and non-dueling networks")


class DQNAgent:
    """
    Enhanced DQN Agent for learning optimal gas transaction timing.
    
    Features:
    - Prioritized Experience Replay (PER)
    - Double DQN (DDQN) to reduce overestimation bias
    - Dueling Network Architecture for better value estimation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        lr_decay: float = 0.9995,  # Learning rate decay factor
        lr_min: float = 0.00001,   # Minimum learning rate
        gradient_clip: float = 10.0,  # Gradient clipping threshold
        use_per: bool = True,  # Use Prioritized Experience Replay
        per_alpha: float = 0.6,  # PER priority exponent
        per_beta: float = 0.4,  # PER importance sampling exponent
        use_double_dqn: bool = True,  # Use Double DQN
        use_dueling: bool = True  # Use Dueling architecture
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store initial LR for scheduling
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip
        self.use_double_dqn = use_double_dqn
        
        # Networks with optional dueling architecture
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims, dueling=use_dueling)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims, dueling=use_dueling)
        self.target_network.copy_from(self.q_network)
        
        # Replay buffer (PER or standard)
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta=per_beta
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
        self.episode_rewards = []
        self.last_td_errors = None  # Store TD errors for PER priority updates

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done, td_error: float = None):
        """Store transition in replay buffer with optional TD error for PER."""
        self.replay_buffer.push(state, action, reward, next_state, done, td_error=td_error)

    def train_step(self) -> Optional[float]:
        """Perform one training step with Double DQN and PER support."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch (with priorities and importance weights if using PER)
        batch, indices, is_weights = self.replay_buffer.sample(self.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute current Q-values
        current_q = self.q_network.forward(states)
        
        # Double DQN: Use online network to select action, target network to evaluate
        if self.use_double_dqn:
            # Select best action using online network
            next_q_online = self.q_network.forward(next_states)
            next_actions = np.argmax(next_q_online, axis=1)
            
            # Evaluate using target network
            next_q_target = self.target_network.forward(next_states)
            next_q_values = next_q_target[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN: use target network for both selection and evaluation
            next_q_target = self.target_network.forward(next_states)
            next_q_values = np.max(next_q_target, axis=1)
        
        # Compute targets
        targets = current_q.copy()
        td_errors = []
        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * next_q_values[i]
            targets[i, actions[i]] = target
            
            # Calculate TD error for PER priority updates
            td_error = abs(target - current_q[i, actions[i]])
            td_errors.append(td_error)
        
        # Update priorities in PER buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(indices, np.array(td_errors))
        
        # Compute gradients and update (with importance sampling weights for PER)
        loss = self._update_network(states, targets, is_weights=is_weights)
        
        # Store TD errors for next transition storage
        self.last_td_errors = np.array(td_errors)
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Learning rate scheduling: decay learning rate over time
        self.learning_rate = max(self.lr_min, self.learning_rate * self.lr_decay)
        
        return loss

    def _update_network(self, states: np.ndarray, targets: np.ndarray, is_weights: np.ndarray = None) -> float:
        """Update network weights using gradient descent with optional importance sampling weights."""
        if is_weights is None:
            is_weights = np.ones(self.batch_size)
        
        # Forward pass
        activations = [states]
        x = states
        
        if self.q_network.dueling:
            # Dueling architecture forward pass
            # Shared layers
            for i in range(len(self.q_network.weights)):
                z = np.dot(x, self.q_network.weights[i]) + self.q_network.biases[i]
                x = np.maximum(0, z)  # ReLU
                activations.append(x)
            
            shared = x
            
            # Value stream
            value = shared
            value_activations = [shared]
            for i in range(len(self.q_network.value_weights) - 1):
                z = np.dot(value, self.q_network.value_weights[i]) + self.q_network.value_biases[i]
                value = np.maximum(0, z)
                value_activations.append(value)
            value = np.dot(value, self.q_network.value_weights[-1]) + self.q_network.value_biases[-1]
            
            # Advantage stream
            advantage = shared
            advantage_activations = [shared]
            for i in range(len(self.q_network.advantage_weights) - 1):
                z = np.dot(advantage, self.q_network.advantage_weights[i]) + self.q_network.advantage_biases[i]
                advantage = np.maximum(0, z)
                advantage_activations.append(advantage)
            advantage = np.dot(advantage, self.q_network.advantage_weights[-1]) + self.q_network.advantage_biases[-1]
            
            # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            advantage_mean = np.mean(advantage, axis=-1, keepdims=True)
            output = value + (advantage - advantage_mean)
            
            # Loss with importance sampling weights
            errors = (output - targets) ** 2
            loss = np.mean(errors * is_weights.reshape(-1, 1))
            
            # Backward pass for dueling architecture
            # Q = V + (A - mean(A)), so:
            # dQ/dV = 1 (for all actions)
            # dQ/dA_i = 1 - 1/n (for action i)
            # dQ/dA_j = -1/n (for other actions j != i)
            delta = 2 * (output - targets) * is_weights.reshape(-1, 1) / self.batch_size
            
            # Backprop through advantage stream
            # The gradient w.r.t. advantage needs to account for mean subtraction:
            # For each sample, delta_advantage = delta - mean(delta) across actions
            # This correctly implements: dL/dA_i = dL/dQ_i * (1 - 1/n) - sum_j(dL/dQ_j * 1/n)
            # Since we only have non-zero delta for the taken action, this simplifies to:
            # delta_advantage_i = delta_i * (1 - 1/n) for taken action
            # delta_advantage_j = delta_i * (-1/n) for other actions
            delta_advantage = delta - np.mean(delta, axis=1, keepdims=True)
            for i in range(len(self.q_network.advantage_weights) - 1, -1, -1):
                dW = np.dot(advantage_activations[i].T, delta_advantage)
                db = np.sum(delta_advantage, axis=0)
                
                # Gradient clipping
                dW_norm = np.linalg.norm(dW)
                if dW_norm > self.gradient_clip:
                    dW = dW * (self.gradient_clip / dW_norm)
                db_norm = np.linalg.norm(db)
                if db_norm > self.gradient_clip:
                    db = db * (self.gradient_clip / db_norm)
                
                self.q_network.advantage_weights[i] -= self.learning_rate * dW
                self.q_network.advantage_biases[i] -= self.learning_rate * db
                
                if i > 0:
                    delta_advantage = np.dot(delta_advantage, self.q_network.advantage_weights[i].T)
                    delta_advantage = delta_advantage * (advantage_activations[i] > 0)
            
            # Backprop through value stream
            delta_value = delta
            for i in range(len(self.q_network.value_weights) - 1, -1, -1):
                dW = np.dot(value_activations[i].T, delta_value)
                db = np.sum(delta_value, axis=0)
                
                # Gradient clipping
                dW_norm = np.linalg.norm(dW)
                if dW_norm > self.gradient_clip:
                    dW = dW * (self.gradient_clip / dW_norm)
                db_norm = np.linalg.norm(db)
                if db_norm > self.gradient_clip:
                    db = db * (self.gradient_clip / db_norm)
                
                self.q_network.value_weights[i] -= self.learning_rate * dW
                self.q_network.value_biases[i] -= self.learning_rate * db
                
                if i > 0:
                    delta_value = np.dot(delta_value, self.q_network.value_weights[i].T)
                    delta_value = delta_value * (value_activations[i] > 0)
            
            # Backprop through shared layers (combine gradients from both streams)
            delta_shared = delta_advantage + delta_value
            for i in range(len(self.q_network.weights) - 1, -1, -1):
                dW = np.dot(activations[i].T, delta_shared)
                db = np.sum(delta_shared, axis=0)
                
                # Gradient clipping
                dW_norm = np.linalg.norm(dW)
                if dW_norm > self.gradient_clip:
                    dW = dW * (self.gradient_clip / dW_norm)
                db_norm = np.linalg.norm(db)
                if db_norm > self.gradient_clip:
                    db = db * (self.gradient_clip / db_norm)
                
                self.q_network.weights[i] -= self.learning_rate * dW
                self.q_network.biases[i] -= self.learning_rate * db
                
                if i > 0:
                    delta_shared = np.dot(delta_shared, self.q_network.weights[i].T)
                    delta_shared = delta_shared * (activations[i] > 0)
        else:
            # Standard architecture
            for i in range(len(self.q_network.weights) - 1):
                z = np.dot(x, self.q_network.weights[i]) + self.q_network.biases[i]
                x = np.maximum(0, z)  # ReLU
                activations.append(x)
            
            output = np.dot(x, self.q_network.weights[-1]) + self.q_network.biases[-1]
            
            # Loss with importance sampling weights
            errors = (output - targets) ** 2
            loss = np.mean(errors * is_weights.reshape(-1, 1))
            
            # Backward pass
            delta = 2 * (output - targets) * is_weights.reshape(-1, 1) / self.batch_size
            
            for i in range(len(self.q_network.weights) - 1, -1, -1):
                # Gradient for weights and biases
                dW = np.dot(activations[i].T, delta)
                db = np.sum(delta, axis=0)
                
                # Gradient clipping: prevent exploding gradients
                dW_norm = np.linalg.norm(dW)
                if dW_norm > self.gradient_clip:
                    dW = dW * (self.gradient_clip / dW_norm)
                
                db_norm = np.linalg.norm(db)
                if db_norm > self.gradient_clip:
                    db = db * (self.gradient_clip / db_norm)
                
                # Update weights
                self.q_network.weights[i] -= self.learning_rate * dW
                self.q_network.biases[i] -= self.learning_rate * db
                
                if i > 0:
                    # Backprop through ReLU
                    delta = np.dot(delta, self.q_network.weights[i].T)
                    delta = delta * (activations[i] > 0)
                    
                    # Clip delta to prevent gradient explosion
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > self.gradient_clip:
                        delta = delta * (self.gradient_clip / delta_norm)
        
        return loss

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        # Ensure state is aligned before forward pass (safety check)
        state = self._align_state_features(state)
        return self.q_network.forward(state).flatten()

    def save(self, path: str):
        """Save agent to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'weights': self.q_network.weights,
            'biases': self.q_network.biases,
            'dueling': self.q_network.dueling,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'episode_rewards': self.episode_rewards,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate
        }
        if self.q_network.dueling:
            data['value_weights'] = self.q_network.value_weights
            data['value_biases'] = self.q_network.value_biases
            data['advantage_weights'] = self.q_network.advantage_weights
            data['advantage_biases'] = self.q_network.advantage_biases
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load agent from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        data_dueling = bool(data.get('dueling', False))
        if data_dueling and not all(k in data for k in ('value_weights', 'value_biases', 'advantage_weights', 'advantage_biases')):
            logger.warning("Dueling weights missing in saved model. Falling back to non-dueling load.")
            data_dueling = False

        # Infer dimensions from saved weights
        inferred_state_dim = self.state_dim
        inferred_action_dim = self.action_dim
        if 'weights' in data and len(data['weights']) > 0:
            inferred_state_dim = data['weights'][0].shape[0]
            if not data_dueling:
                inferred_action_dim = data['weights'][-1].shape[1]
        else:
            logger.warning("Could not determine input dimension from saved weights. Using initialized state_dim.")

        inferred_action_dim = int(data.get('action_dim', inferred_action_dim))

        # Rebuild networks if architecture differs
        if (
            self.q_network.dueling != data_dueling
            or self.state_dim != inferred_state_dim
            or self.action_dim != inferred_action_dim
        ):
            logger.warning(
                f"Rebuilding networks for loaded model "
                f"(dueling={data_dueling}, state_dim={inferred_state_dim}, action_dim={inferred_action_dim})."
            )
            hidden_dims = self.q_network.hidden_dims
            self.state_dim = inferred_state_dim
            self.action_dim = inferred_action_dim
            self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_dims, dueling=data_dueling)
            self.target_network = QNetwork(self.state_dim, self.action_dim, hidden_dims, dueling=data_dueling)

        # Load weights
        self.q_network.weights = data['weights']
        self.q_network.biases = data['biases']
        if data_dueling:
            self.q_network.value_weights = data['value_weights']
            self.q_network.value_biases = data['value_biases']
            self.q_network.advantage_weights = data['advantage_weights']
            self.q_network.advantage_biases = data['advantage_biases']

        self.target_network.copy_from(self.q_network)
        self.epsilon = data.get('epsilon', 0.01)
        self.training_steps = data.get('training_steps', 0)
        self.episode_rewards = data.get('episode_rewards', [])

    def _align_state_features(self, state: np.ndarray) -> np.ndarray:
        """Align state features to match model's expected input dimension.
        
        If state has more features than expected, truncate.
        If state has fewer features than expected, pad with zeros.
        """
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        current_dim = state.shape[1]
        expected_dim = self.state_dim
        
        if current_dim == expected_dim:
            return state
        
        if current_dim > expected_dim:
            # Truncate: take first N features (most important features are usually first)
            logger.warning(
                f"State dimension mismatch: input has {current_dim} features, "
                f"model expects {expected_dim}. Truncating to {expected_dim} features."
            )
            aligned_state = state[:, :expected_dim].copy()
            return aligned_state
        else:
            # Pad with zeros
            logger.warning(
                f"State dimension mismatch: input has {current_dim} features, "
                f"model expects {expected_dim}. Padding with zeros."
            )
            padding = np.zeros((state.shape[0], expected_dim - current_dim), dtype=state.dtype)
            return np.concatenate([state, padding], axis=1)
    
    def get_recommendation(self, state: np.ndarray, threshold: float = 0.6) -> dict:
        """Get action recommendation with confidence."""
        # Align state dimensions to match model expectations
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
