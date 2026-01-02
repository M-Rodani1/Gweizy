"""
Deep Q-Network (DQN) agent for gas price optimization.
"""
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """Simple neural network for Q-value approximation using numpy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
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
        
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)  # ReLU
        
        # Output layer (no activation)
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        return x
    
    def copy_from(self, other: 'QNetwork'):
        """Copy weights from another network."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()


class DQNAgent:
    """
    DQN Agent for learning optimal gas transaction timing.
    
    Uses experience replay and target network for stable learning.
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
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.copy_from(self.q_network)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute targets
        current_q = self.q_network.forward(states)
        next_q = self.target_network.forward(next_states)
        
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Compute gradients and update (simplified SGD)
        loss = self._update_network(states, targets)
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss

    def _update_network(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Update network weights using gradient descent."""
        # Forward pass
        activations = [states]
        x = states
        
        for i in range(len(self.q_network.weights) - 1):
            z = np.dot(x, self.q_network.weights[i]) + self.q_network.biases[i]
            x = np.maximum(0, z)  # ReLU
            activations.append(x)
        
        output = np.dot(x, self.q_network.weights[-1]) + self.q_network.biases[-1]
        
        # Loss
        loss = np.mean((output - targets) ** 2)
        
        # Backward pass
        delta = 2 * (output - targets) / self.batch_size
        
        for i in range(len(self.q_network.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0)
            
            # Update weights
            self.q_network.weights[i] -= self.learning_rate * dW
            self.q_network.biases[i] -= self.learning_rate * db
            
            if i > 0:
                # Backprop through ReLU
                delta = np.dot(delta, self.q_network.weights[i].T)
                delta = delta * (activations[i] > 0)
        
        return loss

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        return self.q_network.forward(state).flatten()

    def save(self, path: str):
        """Save agent to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'weights': self.q_network.weights,
            'biases': self.q_network.biases,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'episode_rewards': self.episode_rewards
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load agent from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_network.weights = data['weights']
        self.q_network.biases = data['biases']
        self.target_network.copy_from(self.q_network)
        self.epsilon = data.get('epsilon', 0.01)
        self.training_steps = data.get('training_steps', 0)
        self.episode_rewards = data.get('episode_rewards', [])

    def get_recommendation(self, state: np.ndarray, threshold: float = 0.6) -> dict:
        """Get action recommendation with confidence."""
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
