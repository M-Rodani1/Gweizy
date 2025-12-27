"""
RL Training Loop

Handles the full training pipeline:
- Episode generation
- Agent training
- Evaluation
- Checkpointing
- Logging
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rl.environment import GasTransactionEnv, TransactionConfig
from rl.agents.dqn import DQNAgent, DQNConfig
from rl.data_loader import RLDataLoader, DataLoaderConfig, Episode
from rl.rewards import RewardType


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Training duration
    num_episodes: int = 1000
    max_steps_per_episode: int = 60
    eval_frequency: int = 50  # Episodes between evaluations
    checkpoint_frequency: int = 100

    # Environment settings
    urgency_levels: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])

    # Early stopping
    early_stop_patience: int = 200
    early_stop_min_improvement: float = 0.01

    # Logging
    log_frequency: int = 10
    save_dir: str = "models/rl_agents"

    # Data
    data_hours: int = 720  # 30 days


class RLTrainer:
    """
    Trainer for RL transaction timing agents

    Handles the full training loop with:
    - Episode sampling from historical data
    - Agent training and optimization
    - Periodic evaluation
    - Checkpointing and logging
    """

    def __init__(
        self,
        agent: DQNAgent,
        config: Optional[TrainingConfig] = None
    ):
        self.agent = agent
        self.config = config or TrainingConfig()

        # Data loader
        self.data_loader = RLDataLoader(DataLoaderConfig())

        # Environments for different urgency levels
        self.envs = {}

        # Training state
        self.current_episode = 0
        self.best_eval_reward = float('-inf')
        self.episodes_without_improvement = 0

        # Metrics
        self.training_rewards = []
        self.eval_rewards = []
        self.training_losses = []

        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)

    def setup(self):
        """Load data and prepare environments"""
        print("="*60)
        print("Setting up RL Training")
        print("="*60)

        # Load data
        df = self.data_loader.load_from_database(hours=self.config.data_hours)
        episodes = self.data_loader.generate_episodes(df)
        self.train_episodes, self.val_episodes, self.test_episodes = \
            self.data_loader.split_episodes(episodes)

        print(f"\nData loaded:")
        print(f"  Train episodes: {len(self.train_episodes)}")
        print(f"  Val episodes: {len(self.val_episodes)}")
        print(f"  Test episodes: {len(self.test_episodes)}")

        # Create environments for different urgency levels
        for urgency in self.config.urgency_levels:
            tx_config = TransactionConfig(urgency_level=urgency)
            self.envs[urgency] = GasTransactionEnv(
                gas_prices=np.zeros(100),  # Placeholder, will be replaced
                config=tx_config
            )

        print(f"\nEnvironments created for urgencies: {self.config.urgency_levels}")

    def train(self) -> Dict:
        """
        Main training loop

        Returns:
            Training summary
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        self.setup()

        # Training metrics
        episode_rewards = []
        episode_losses = []

        # Episode generator
        train_gen = self.data_loader.episode_generator(
            self.train_episodes,
            shuffle=True,
            infinite=True
        )

        for episode_num in range(self.config.num_episodes):
            self.current_episode = episode_num

            # Get next episode data
            episode_data = next(train_gen)

            # Random urgency for diversity
            urgency = np.random.choice(self.config.urgency_levels)
            env = self.envs[urgency]

            # Reset environment with new data
            env.gas_prices = episode_data.gas_prices
            state, _ = env.reset()

            # Run episode
            episode_reward = 0
            episode_loss = 0
            steps = 0

            for step in range(self.config.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, training=True)

                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss += loss

                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            # End episode
            self.agent.end_episode(episode_reward)
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(steps, 1))

            self.training_rewards.append(episode_reward)
            self.training_losses.append(episode_loss / max(steps, 1))

            # Logging
            if episode_num % self.config.log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-self.config.log_frequency:])
                avg_loss = np.mean(episode_losses[-self.config.log_frequency:])
                metrics = self.agent.get_metrics()

                print(f"Episode {episode_num:4d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {metrics['epsilon']:.3f} | "
                      f"Buffer: {metrics['buffer_size']:5d}")

            # Evaluation
            if episode_num % self.config.eval_frequency == 0 and episode_num > 0:
                eval_reward = self._evaluate()
                self.eval_rewards.append(eval_reward)

                # Check for improvement
                if eval_reward > self.best_eval_reward + self.config.early_stop_min_improvement:
                    self.best_eval_reward = eval_reward
                    self.episodes_without_improvement = 0
                    self._save_checkpoint("best")
                else:
                    self.episodes_without_improvement += self.config.eval_frequency

                print(f"\n*** Evaluation: {eval_reward:.2f} (Best: {self.best_eval_reward:.2f}) ***\n")

            # Checkpointing
            if episode_num % self.config.checkpoint_frequency == 0 and episode_num > 0:
                self._save_checkpoint(f"episode_{episode_num}")

            # Early stopping
            if self.episodes_without_improvement >= self.config.early_stop_patience:
                print(f"\nEarly stopping at episode {episode_num}")
                break

        # Final evaluation and save
        final_eval = self._evaluate()
        self._save_checkpoint("final")

        # Summary
        summary = self._generate_summary()
        self._save_summary(summary)

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best eval reward: {self.best_eval_reward:.2f}")
        print(f"Final eval reward: {final_eval:.2f}")
        print(f"Total episodes: {self.current_episode}")

        return summary

    def _evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate agent on validation episodes

        Returns:
            Average reward across evaluation episodes
        """
        eval_rewards = []

        # Use validation episodes
        for i, episode_data in enumerate(self.val_episodes[:num_episodes]):
            # Use medium urgency for consistent evaluation
            urgency = 0.5
            env = self.envs[urgency]
            env.gas_prices = episode_data.gas_prices

            state, _ = env.reset()
            episode_reward = 0

            for _ in range(self.config.max_steps_per_episode):
                # Greedy action selection (no exploration)
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

                if done:
                    break

            eval_rewards.append(episode_reward)

        return np.mean(eval_rewards)

    def _save_checkpoint(self, name: str):
        """Save agent checkpoint"""
        path = os.path.join(self.config.save_dir, f"dqn_{name}.pt")
        self.agent.save(path)

    def _generate_summary(self) -> Dict:
        """Generate training summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': self.current_episode,
            'best_eval_reward': float(self.best_eval_reward),
            'final_avg_reward': float(np.mean(self.training_rewards[-100:])),
            'agent_config': {
                'hidden_sizes': self.agent.config.hidden_sizes,
                'learning_rate': self.agent.config.learning_rate,
                'gamma': self.agent.config.gamma,
                'epsilon_end': self.agent.config.epsilon_end
            },
            'training_config': {
                'num_episodes': self.config.num_episodes,
                'data_hours': self.config.data_hours,
                'urgency_levels': self.config.urgency_levels
            }
        }

    def _save_summary(self, summary: Dict):
        """Save training summary to JSON"""
        path = os.path.join(self.config.save_dir, "training_summary.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {path}")


def train_dqn_agent(
    num_episodes: int = 500,
    data_hours: int = 168,
    save_dir: str = "models/rl_agents"
) -> Dict:
    """
    Convenience function to train a DQN agent

    Args:
        num_episodes: Number of training episodes
        data_hours: Hours of historical data to use
        save_dir: Directory to save checkpoints

    Returns:
        Training summary
    """
    # Create agent
    state_dim = 15  # From environment
    action_dim = 4  # WAIT, SUBMIT_NOW, SUBMIT_LOW, SUBMIT_HIGH

    agent_config = DQNConfig(
        hidden_sizes=[128, 128, 64],
        learning_rate=1e-4,
        epsilon_decay=0.995
    )

    agent = DQNAgent(state_dim, action_dim, agent_config)

    # Create trainer
    train_config = TrainingConfig(
        num_episodes=num_episodes,
        data_hours=data_hours,
        save_dir=save_dir,
        log_frequency=10,
        eval_frequency=50
    )

    trainer = RLTrainer(agent, train_config)

    # Train
    summary = trainer.train()

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--data-hours", type=int, default=168, help="Hours of data")
    parser.add_argument("--save-dir", type=str, default="models/rl_agents")

    args = parser.parse_args()

    print("="*60)
    print("DQN Agent Training")
    print("="*60)

    try:
        summary = train_dqn_agent(
            num_episodes=args.episodes,
            data_hours=args.data_hours,
            save_dir=args.save_dir
        )

        print("\nTraining Summary:")
        print(json.dumps(summary, indent=2))

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
