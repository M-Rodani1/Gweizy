"""
Hyperparameter tuning for DQN agent using Optuna.

Phase 3: Bayesian optimization with TPE sampler for efficient hyperparameter search.
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from rl.environment import GasOptimizationEnv
from rl.data_loader import GasDataLoader
from rl.rewards import RewardConfig
from rl.train import evaluate_agent, get_dqn_agent


class OptunaHyperparameterTuner:
    """Optuna-based hyperparameter tuning for DQN agent."""

    def __init__(
        self,
        chain_id: int = 8453,
        study_name: str = "dqn_optimization",
        storage: Optional[str] = None
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")

        self.chain_id = chain_id
        self.study_name = study_name
        self.storage = storage

        # Load data once for all trials
        print("Loading data for hyperparameter tuning...")
        self.data_loader = GasDataLoader(use_database=True)
        self.data_loader.load_data(hours=720, min_records=500, chain_id=chain_id)

        # Split into train/eval
        self.train_data, self.eval_data = self.data_loader.split_data(train_ratio=0.8)
        self.train_loader = GasDataLoader(use_database=False)
        self.train_loader.set_cache(self.train_data)
        self.eval_loader = GasDataLoader(use_database=False)
        self.eval_loader.set_cache(self.eval_data)

        print(f"Data loaded: {len(self.train_data)} train, {len(self.eval_data)} eval records")

    def create_objective(
        self,
        num_episodes: int = 1000,
        eval_episodes: int = 50,
        episode_length: int = 48
    ):
        """Create Optuna objective function."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {
                # Network architecture
                'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [64, 128, 256]),
                'hidden_dim_2': trial.suggest_categorical('hidden_dim_2', [32, 64, 128]),

                # Learning parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'gamma': trial.suggest_float('gamma', 0.95, 0.995),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),

                # Exploration
                'epsilon_decay_ratio': trial.suggest_float('epsilon_decay_ratio', 0.3, 0.8),

                # Target network
                'target_update_tau': trial.suggest_float('target_update_tau', 0.001, 0.01),

                # Phase 2 features
                'n_steps': trial.suggest_int('n_steps', 1, 5),
                'use_reward_norm': trial.suggest_categorical('use_reward_norm', [True, False]),

                # Phase 3: Action timing
                'min_wait_steps': trial.suggest_int('min_wait_steps', 1, 8),
                'early_execution_penalty': trial.suggest_float('early_execution_penalty', 0.0, 0.3),
                'observation_bonus': trial.suggest_float('observation_bonus', 0.0, 0.1),
                'wait_penalty': trial.suggest_float('wait_penalty', 0.005, 0.05),
            }

            # Build reward config with Phase 3 parameters
            reward_config = RewardConfig(
                wait_penalty=params['wait_penalty'],
                min_wait_steps=params['min_wait_steps'],
                early_execution_penalty=params['early_execution_penalty'],
                observation_bonus=params['observation_bonus']
            )

            # Create environment
            env = GasOptimizationEnv(
                self.train_loader,
                episode_length=episode_length,
                max_wait_steps=episode_length,
                reward_config=reward_config,
                scale_rewards=True
            )

            # Get DQN agent class
            DQNAgent = get_dqn_agent("auto")

            # Create agent
            hidden_dims = [params['hidden_dim_1'], params['hidden_dim_2']]
            epsilon_decay_episodes = int(num_episodes * params['epsilon_decay_ratio'])

            agent = DQNAgent(
                state_dim=env.observation_space_shape[0],
                action_dim=env.action_space_n,
                hidden_dims=hidden_dims,
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_episodes=epsilon_decay_episodes,
                epsilon_decay_steps=None,
                buffer_size=50000,
                batch_size=params['batch_size'],
                target_update_tau=params['target_update_tau'],
                use_soft_target=True,
                use_per=True,
                use_double_dqn=True,
                use_dueling=True,
                gradient_clip=1.0,
                # Phase 2
                n_steps=params['n_steps'],
                use_reward_norm=params['use_reward_norm'],
                use_noisy_nets=False
            )

            # Pre-generate episodes
            training_episodes = self.train_loader.get_diverse_episodes(
                episode_length=episode_length,
                num_episodes=min(num_episodes * 2, 300)
            )

            # Fit state normalizer
            sample_states = []
            for ep_data in training_episodes[:20]:
                state = env.reset(episode_data=ep_data[:episode_length])
                sample_states.append(state)
            if sample_states:
                agent.fit_state_normalizer(np.array(sample_states))

            # Training loop with intermediate evaluation for pruning
            episode_rewards = []
            avg_savings_history = []

            for episode in range(num_episodes):
                agent.decay_epsilon(episode)

                if hasattr(agent, 'reset_episode'):
                    agent.reset_episode()

                # Use pre-generated episode
                if episode < len(training_episodes):
                    ep_data = training_episodes[episode]
                    if len(ep_data) > episode_length:
                        ep_data = ep_data[:episode_length]
                    state = env.reset(episode_data=ep_data)
                else:
                    state = env.reset()

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

                episode_rewards.append(total_reward)
                avg_savings_history.append(info.get('savings_vs_initial', 0))

                # Intermediate evaluation for pruning (every 200 episodes)
                if (episode + 1) % 200 == 0:
                    recent_savings = np.mean(avg_savings_history[-100:]) if len(avg_savings_history) >= 100 else np.mean(avg_savings_history)

                    # Report to Optuna for pruning
                    trial.report(recent_savings, episode)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # Final evaluation
            eval_env = GasOptimizationEnv(
                self.eval_loader,
                episode_length=episode_length,
                max_wait_steps=episode_length,
                reward_config=reward_config
            )

            eval_results = evaluate_agent(
                agent,
                num_episodes=eval_episodes,
                verbose=False,
                data_loader=self.eval_loader,
                episode_length=episode_length,
                max_wait_steps=episode_length
            )

            # Primary metric: eval savings
            eval_savings = eval_results['avg_savings']

            # Log additional metrics
            trial.set_user_attr('eval_reward', eval_results['avg_reward'])
            trial.set_user_attr('positive_savings_rate', eval_results['positive_savings_rate'])
            trial.set_user_attr('avg_wait_time', eval_results['avg_wait_time'])
            trial.set_user_attr('train_savings_final', np.mean(avg_savings_history[-100:]))

            return eval_savings

        return objective

    def optimize(
        self,
        n_trials: int = 50,
        num_episodes: int = 1000,
        eval_episodes: int = 50,
        episode_length: int = 48,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> optuna.Study:
        """
        Run Optuna optimization.

        Args:
            n_trials: Number of trials to run
            num_episodes: Training episodes per trial
            eval_episodes: Evaluation episodes per trial
            episode_length: Steps per episode
            timeout: Maximum time in seconds (optional)
            n_jobs: Number of parallel jobs (1 = sequential)

        Returns:
            Optuna study object
        """
        # Create study with TPE sampler and median pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=400)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",  # Maximize savings
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        # Create objective
        objective = self.create_objective(
            num_episodes=num_episodes,
            eval_episodes=eval_episodes,
            episode_length=episode_length
        )

        # Optimize
        print(f"\nStarting Optuna optimization: {n_trials} trials")
        print(f"Training episodes per trial: {num_episodes}")
        print(f"Evaluation episodes: {eval_episodes}")
        print("=" * 60)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        return study

    def save_results(self, study: optuna.Study, filepath: str):
        """Save optimization results to JSON."""
        results = {
            'chain_id': self.chain_id,
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_trial.value,
                'params': study.best_trial.params,
                'user_attrs': study.best_trial.user_attrs
            },
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state),
                    'user_attrs': t.user_attrs
                }
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filepath}")
        print(f"\nBest trial #{study.best_trial.number}:")
        print(f"  Eval savings: {study.best_trial.value * 100:.2f}%")
        print(f"  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    def get_best_config(self, study: optuna.Study) -> Dict:
        """Extract best configuration from study."""
        params = study.best_trial.params

        return {
            'hidden_dims': [params['hidden_dim_1'], params['hidden_dim_2']],
            'learning_rate': params['learning_rate'],
            'gamma': params['gamma'],
            'batch_size': params['batch_size'],
            'epsilon_decay_ratio': params['epsilon_decay_ratio'],
            'target_update_tau': params['target_update_tau'],
            'n_steps': params['n_steps'],
            'use_reward_norm': params['use_reward_norm'],
            'min_wait_steps': params['min_wait_steps'],
            'early_execution_penalty': params['early_execution_penalty'],
            'observation_bonus': params['observation_bonus'],
            'wait_penalty': params['wait_penalty']
        }


def main():
    """Run hyperparameter tuning."""
    import argparse

    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for DQN agent')
    parser.add_argument('--chain-id', type=int, default=8453, help='Chain ID')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes per trial')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Evaluation episodes')
    parser.add_argument('--episode-length', type=int, default=48, help='Episode length')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--output', type=str, default='optuna_results.json', help='Output file')
    parser.add_argument('--study-name', type=str, default='dqn_phase3', help='Optuna study name')

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required. Install with: pip install optuna")
        sys.exit(1)

    # Create tuner
    tuner = OptunaHyperparameterTuner(
        chain_id=args.chain_id,
        study_name=args.study_name
    )

    # Run optimization
    study = tuner.optimize(
        n_trials=args.trials,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        timeout=args.timeout
    )

    # Save results
    tuner.save_results(study, args.output)

    # Print best config
    best_config = tuner.get_best_config(study)
    print("\nBest configuration for training:")
    print(json.dumps(best_config, indent=2))


if __name__ == '__main__':
    main()
