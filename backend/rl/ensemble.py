"""
Ensemble DQN for robust decision making.

Phase 4B-4: Train multiple agents and combine their predictions
for more stable and reliable decisions.
"""
import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.data_loader import GasDataLoader
from rl.environment import GasOptimizationEnv
from rl.rewards import RewardConfig
from rl.train import get_dqn_agent


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    n_agents: int = 5
    seeds: Optional[List[int]] = None
    # Training params
    num_episodes: int = 2000
    episode_length: int = 72
    hidden_dims: List[int] = None
    # Decision params
    voting_threshold: float = 0.6  # Fraction of agents that must agree
    confidence_method: str = "agreement"  # "agreement" or "q_std"

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42 + i * 1000 for i in range(self.n_agents)]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class EnsembleDQN:
    """
    Ensemble of DQN agents for robust decision making.

    Combines predictions from multiple agents trained with different
    random seeds to reduce variance and improve reliability.
    """

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.agents = []
        self.agent_metadata = []
        self.state_dim = None
        self.action_dim = None

    def train(
        self,
        chain_id: int = 8453,
        verbose: bool = True,
        # Phase 2-4 parameters
        n_steps: int = 5,
        use_reward_norm: bool = True,
        min_wait_steps: int = 6,
        early_execution_penalty: float = 0.15,
        observation_bonus: float = 0.03,
        wait_penalty: float = 0.012,
        use_enhanced_features: bool = True
    ) -> List[Dict]:
        """
        Train ensemble of agents with different seeds.

        Returns list of training results for each agent.
        """
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING")
        print(f"{'='*60}")
        print(f"Number of agents: {self.config.n_agents}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Episodes per agent: {self.config.num_episodes}")
        print(f"{'='*60}\n")

        # Load data
        data_loader = GasDataLoader(use_database=True)
        data_loader.load_data(hours=720, min_records=500, chain_id=chain_id)

        # Split data
        train_data, eval_data = data_loader.split_data(train_ratio=0.8)
        train_loader = GasDataLoader(use_database=False)
        train_loader.set_cache(train_data)
        eval_loader = GasDataLoader(use_database=False)
        eval_loader.set_cache(eval_data)

        # Create reward config
        reward_config = RewardConfig(
            wait_penalty=wait_penalty,
            min_wait_steps=min_wait_steps,
            early_execution_penalty=early_execution_penalty,
            observation_bonus=observation_bonus
        )

        # Create environment
        env = GasOptimizationEnv(
            train_loader,
            episode_length=self.config.episode_length,
            max_wait_steps=self.config.episode_length,
            reward_config=reward_config,
            scale_rewards=True,
            use_enhanced_features=use_enhanced_features
        )

        self.state_dim = env.observation_space_shape[0]
        self.action_dim = env.action_space_n

        # Get DQN agent class
        DQNAgent = get_dqn_agent("auto")

        # Pre-generate episodes
        training_episodes = train_loader.get_diverse_episodes(
            episode_length=self.config.episode_length,
            num_episodes=min(self.config.num_episodes * 2, 500)
        )

        results = []

        for agent_idx, seed in enumerate(self.config.seeds):
            print(f"\n{'─'*60}")
            print(f"Training Agent {agent_idx + 1}/{self.config.n_agents} (seed={seed})")
            print(f"{'─'*60}")

            # Set random seed
            np.random.seed(seed)

            # Create agent
            agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.config.hidden_dims,
                learning_rate=0.0003,
                gamma=0.98,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_episodes=int(self.config.num_episodes * 0.5),
                epsilon_decay_steps=None,
                buffer_size=50000,
                batch_size=64,
                target_update_tau=0.001,
                use_soft_target=True,
                use_per=True,
                use_double_dqn=True,
                use_dueling=True,
                gradient_clip=1.0,
                n_steps=n_steps,
                use_reward_norm=use_reward_norm,
                use_noisy_nets=False
            )

            # Fit state normalizer
            sample_states = []
            for ep_data in training_episodes[:20]:
                state = env.reset(episode_data=ep_data[:self.config.episode_length])
                sample_states.append(state)
            if sample_states:
                agent.fit_state_normalizer(np.array(sample_states))

            # Training loop
            best_savings = float('-inf')
            savings_history = []

            for episode in range(self.config.num_episodes):
                agent.decay_epsilon(episode)

                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()

                # Use pre-generated episode with some randomization
                ep_idx = (episode + seed) % len(training_episodes)
                ep_data = training_episodes[ep_idx]
                if len(ep_data) > self.config.episode_length:
                    ep_data = ep_data[:self.config.episode_length]
                state = env.reset(episode_data=ep_data)

                total_reward = 0
                last_td_error = None

                while True:
                    action = agent.select_action(state, training=True)
                    next_state, reward, done, info = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done, td_error=last_td_error)

                    if len(agent.replay_buffer) >= agent.batch_size:
                        loss = agent.train_step()
                        if agent.last_td_errors is not None and len(agent.last_td_errors) > 0:
                            last_td_error = float(np.mean(agent.last_td_errors))

                    total_reward += reward
                    state = next_state

                    if done:
                        break

                savings = info.get('savings_vs_initial', 0)
                savings_history.append(savings)

                if len(savings_history) >= 100:
                    avg_savings = np.mean(savings_history[-100:])
                    if avg_savings > best_savings:
                        best_savings = avg_savings

                # Progress update
                if verbose and (episode + 1) % 500 == 0:
                    recent_savings = np.mean(savings_history[-100:]) if len(savings_history) >= 100 else np.mean(savings_history)
                    print(f"  Episode {episode+1}/{self.config.num_episodes}: Recent savings = {recent_savings*100:.2f}%")

            # Store agent
            self.agents.append(agent)
            self.agent_metadata.append({
                'seed': seed,
                'best_savings': best_savings,
                'final_savings': np.mean(savings_history[-100:]) if len(savings_history) >= 100 else np.mean(savings_history)
            })

            result = {
                'agent_idx': agent_idx,
                'seed': seed,
                'best_savings': best_savings,
                'final_savings': self.agent_metadata[-1]['final_savings']
            }
            results.append(result)

            print(f"  Agent {agent_idx + 1} complete: Best={best_savings*100:.2f}%, Final={result['final_savings']*100:.2f}%")

        # Print ensemble summary
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        best_savings_all = [r['best_savings'] for r in results]
        print(f"Best savings across agents: {max(best_savings_all)*100:.2f}%")
        print(f"Mean best savings: {np.mean(best_savings_all)*100:.2f}%")
        print(f"Std best savings: {np.std(best_savings_all)*100:.2f}%")
        print(f"{'='*60}\n")

        return results

    def get_q_values(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Q-values from all agents for a state.

        Returns:
            mean_q: Mean Q-values across ensemble
            std_q: Standard deviation of Q-values (uncertainty)
        """
        if not self.agents:
            raise ValueError("No agents trained. Call train() first.")

        all_q_values = []
        for agent in self.agents:
            q_values = agent.get_q_values(state)
            all_q_values.append(q_values)

        all_q_values = np.array(all_q_values)  # Shape: (n_agents, n_actions)
        mean_q = np.mean(all_q_values, axis=0)
        std_q = np.std(all_q_values, axis=0)

        return mean_q, std_q

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, float, Dict]:
        """
        Select action using ensemble voting/averaging.

        Returns:
            action: Selected action
            confidence: Confidence in the decision (0-1)
            info: Additional info about the decision
        """
        if not self.agents:
            raise ValueError("No agents trained. Call train() first.")

        # Get individual agent actions and Q-values
        agent_actions = []
        agent_q_values = []

        for agent in self.agents:
            if training:
                action = agent.select_action(state, training=True)
            else:
                q_values = agent.get_q_values(state)
                action = int(np.argmax(q_values))
                agent_q_values.append(q_values)
            agent_actions.append(action)

        agent_actions = np.array(agent_actions)

        # Voting: count actions
        action_counts = np.bincount(agent_actions, minlength=self.action_dim)
        vote_fractions = action_counts / len(self.agents)

        # Get ensemble action (majority vote)
        ensemble_action = int(np.argmax(action_counts))

        # Calculate confidence
        if self.config.confidence_method == "agreement":
            # Confidence = fraction of agents that agree
            confidence = vote_fractions[ensemble_action]
        else:  # q_std
            # Confidence = inverse of Q-value standard deviation
            if agent_q_values:
                all_q = np.array(agent_q_values)
                q_std = np.mean(np.std(all_q, axis=0))
                confidence = 1.0 / (1.0 + q_std)
            else:
                confidence = vote_fractions[ensemble_action]

        # Check if confidence meets threshold
        meets_threshold = confidence >= self.config.voting_threshold

        info = {
            'agent_actions': agent_actions.tolist(),
            'vote_fractions': vote_fractions.tolist(),
            'confidence': confidence,
            'meets_threshold': meets_threshold,
            'ensemble_action': ensemble_action
        }

        # If confidence is too low and we're not training, default to wait (action 0)
        if not training and not meets_threshold:
            # Low confidence = wait for more information
            return 0, confidence, info

        return ensemble_action, confidence, info

    def get_recommendation(self, state: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        Get action recommendation with confidence - compatible with gate interface.

        Args:
            state: Current state observation
            threshold: Confidence threshold for recommendation

        Returns:
            Dict with action, confidence, q_values, etc.
        """
        action, confidence, info = self.select_action(state, training=False)

        # Get average Q-values across agents
        q_values_list = []
        for agent in self.agents:
            q_values = agent.get_q_values(state)
            q_values_list.append(q_values)

        avg_q_values = np.mean(q_values_list, axis=0)

        return {
            'action': 'execute' if action == 1 else 'wait',
            'action_id': action,
            'confidence': confidence,
            'q_values': {
                'wait': float(avg_q_values[0]),
                'execute': float(avg_q_values[1])
            },
            'vote_fractions': info['vote_fractions'],
            'agent_actions': info['agent_actions'],
            'should_act': action == 1 and confidence >= threshold
        }

    def evaluate(
        self,
        data_loader: GasDataLoader = None,
        num_episodes: int = 100,
        episode_length: int = 72,
        verbose: bool = True
    ) -> Dict:
        """Evaluate ensemble on held-out data."""
        if not self.agents:
            raise ValueError("No agents trained. Call train() first.")

        if data_loader is None:
            # Load fresh data
            data_loader = GasDataLoader(use_database=True)
            data_loader.load_data(hours=720, min_records=500, chain_id=8453)
            _, eval_data = data_loader.split_data(train_ratio=0.8)
            data_loader = GasDataLoader(use_database=False)
            data_loader.set_cache(eval_data)

        env = GasOptimizationEnv(
            data_loader,
            episode_length=episode_length,
            max_wait_steps=episode_length
        )

        total_rewards = []
        savings = []
        wait_times = []
        confidences = []
        threshold_decisions = 0
        action_distribution = {'wait': 0, 'execute': 0}

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_confidences = []

            while True:
                action, confidence, info = self.select_action(state, training=False)
                episode_confidences.append(confidence)

                if info['meets_threshold']:
                    threshold_decisions += 1

                next_state, reward, done, step_info = env.step(action)
                total_reward += reward

                if action == 0:
                    action_distribution['wait'] += 1
                else:
                    action_distribution['execute'] += 1

                state = next_state

                if done:
                    break

            total_rewards.append(total_reward)
            savings.append(step_info.get('savings_vs_initial', 0))
            wait_times.append(step_info.get('time_waiting', 0))
            confidences.append(np.mean(episode_confidences))

        results = {
            'avg_reward': float(np.mean(total_rewards)),
            'avg_savings': float(np.mean(savings)),
            'std_savings': float(np.std(savings)),
            'median_savings': float(np.median(savings)),
            'positive_savings_rate': float(np.mean([1 if s > 0 else 0 for s in savings]) * 100),
            'avg_wait_time': float(np.mean(wait_times)),
            'avg_confidence': float(np.mean(confidences)),
            'threshold_decision_rate': threshold_decisions / sum(action_distribution.values()) * 100,
            'action_distribution': action_distribution,
            'n_episodes': num_episodes
        }

        if verbose:
            print(f"\nEnsemble Evaluation Results:")
            print(f"  Avg savings: {results['avg_savings']*100:.2f}% (+/- {results['std_savings']*100:.2f}%)")
            print(f"  Median savings: {results['median_savings']*100:.2f}%")
            print(f"  Positive rate: {results['positive_savings_rate']:.1f}%")
            print(f"  Avg confidence: {results['avg_confidence']:.2f}")
            print(f"  Avg wait time: {results['avg_wait_time']:.1f} steps")

        return results

    def save(self, filepath: str):
        """Save ensemble to file."""
        data = {
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
            'agent_metadata': self.agent_metadata,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'agents': [agent.get_state_dict() for agent in self.agents]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Ensemble saved to {filepath}")

    def load(self, filepath: str):
        """Load ensemble from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.config = EnsembleConfig(**data['config'])
        self.agent_metadata = data['agent_metadata']
        self.state_dim = data['state_dim']
        self.action_dim = data['action_dim']

        # Recreate agents
        DQNAgent = get_dqn_agent("auto")
        self.agents = []
        for agent_state in data['agents']:
            agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.config.hidden_dims
            )
            agent.load_state_dict(agent_state)
            self.agents.append(agent)

        print(f"Ensemble loaded from {filepath} ({len(self.agents)} agents)")


def main():
    """Train and evaluate ensemble."""
    import argparse

    parser = argparse.ArgumentParser(description='Ensemble DQN training')
    parser.add_argument('--n-agents', type=int, default=5, help='Number of agents in ensemble')
    parser.add_argument('--episodes', type=int, default=2000, help='Training episodes per agent')
    parser.add_argument('--eval-episodes', type=int, default=100, help='Evaluation episodes')
    parser.add_argument('--voting-threshold', type=float, default=0.6, help='Voting agreement threshold')
    parser.add_argument('--output', type=str, default='ensemble_model.pkl', help='Output model file')
    parser.add_argument('--chain-id', type=int, default=8453, help='Chain ID')

    args = parser.parse_args()

    # Create ensemble config
    config = EnsembleConfig(
        n_agents=args.n_agents,
        num_episodes=args.episodes,
        voting_threshold=args.voting_threshold
    )

    # Create and train ensemble
    ensemble = EnsembleDQN(config)
    results = ensemble.train(chain_id=args.chain_id)

    # Evaluate
    print("\nEvaluating ensemble...")
    eval_results = ensemble.evaluate(num_episodes=args.eval_episodes)

    # Save
    ensemble.save(args.output)

    # Save results
    results_file = args.output.replace('.pkl', '_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'training_results': results,
            'eval_results': eval_results,
            'config': asdict(config),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
