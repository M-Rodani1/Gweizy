"""
Walk-Forward Validation for Time-Series RL Training.

Phase 4B-1: Proper time-series cross-validation to detect overfitting
and get honest performance estimates.

Walk-forward validation trains on past data and evaluates on future data,
using rolling windows to simulate real-world deployment scenarios.
"""
import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.data_loader import GasDataLoader
from rl.environment import GasOptimizationEnv
from rl.rewards import RewardConfig
from rl.train import evaluate_agent, get_dqn_agent


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    fold_number: int
    train_start_idx: int
    train_end_idx: int
    eval_start_idx: int
    eval_end_idx: int
    train_records: int
    eval_records: int
    # Training metrics
    best_train_savings: float
    final_train_savings: float
    # Evaluation metrics
    eval_avg_reward: float
    eval_avg_savings: float
    eval_median_savings: float
    eval_positive_rate: float
    eval_avg_wait_time: float
    # Timestamps
    train_start_time: str
    train_end_time: str
    eval_start_time: str
    eval_end_time: str


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    n_folds: int
    total_train_records: int
    total_eval_records: int
    # Aggregated metrics (mean across folds)
    mean_eval_savings: float
    std_eval_savings: float
    median_eval_savings: float
    mean_positive_rate: float
    mean_wait_time: float
    # Best/worst fold performance
    best_fold_savings: float
    worst_fold_savings: float
    # Individual fold results
    folds: List[WalkForwardFold]
    # Metadata
    timestamp: str
    config: Dict


class WalkForwardValidator:
    """
    Walk-forward validation for DQN agent.

    Splits data into rolling windows:
    [===Train 1===][=Eval 1=]
         [===Train 2===][=Eval 2=]
              [===Train 3===][=Eval 3=]

    This simulates real-world deployment where we train on past
    and deploy on future (unseen) data.
    """

    def __init__(
        self,
        chain_id: int = 8453,
        n_folds: int = 5,
        train_ratio: float = 0.7,  # 70% train, 30% eval per fold
        overlap_ratio: float = 0.5,  # 50% overlap between folds
        min_train_records: int = 1000,
        min_eval_records: int = 200
    ):
        self.chain_id = chain_id
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.overlap_ratio = overlap_ratio
        self.min_train_records = min_train_records
        self.min_eval_records = min_eval_records

        # Load all data
        print(f"Loading data for walk-forward validation (chain {chain_id})...")
        self.data_loader = GasDataLoader(use_database=True)
        self.all_data = self.data_loader.load_data(hours=720, min_records=500, chain_id=chain_id)
        print(f"Loaded {len(self.all_data)} total records")

        # Calculate fold boundaries
        self.fold_boundaries = self._calculate_fold_boundaries()

    def _calculate_fold_boundaries(self) -> List[Tuple[int, int, int, int]]:
        """
        Calculate train/eval boundaries for each fold.

        Returns list of (train_start, train_end, eval_start, eval_end) tuples.
        """
        total_records = len(self.all_data)

        # Calculate window size based on n_folds and overlap
        # With overlap, we need: start + n_folds * step + window_size = total_records
        # where step = window_size * (1 - overlap_ratio)

        # Size of each fold's data (train + eval)
        fold_size = int(total_records // (1 + (self.n_folds - 1) * (1 - self.overlap_ratio)))
        fold_size = max(fold_size, self.min_train_records + self.min_eval_records)

        train_size = int(fold_size * self.train_ratio)
        eval_size = int(fold_size - train_size)

        # Ensure minimum sizes
        train_size = max(train_size, self.min_train_records)
        eval_size = max(eval_size, self.min_eval_records)

        step_size = int(fold_size * (1 - self.overlap_ratio))

        boundaries = []
        for i in range(self.n_folds):
            train_start = int(i * step_size)
            train_end = int(train_start + train_size)
            eval_start = int(train_end)
            eval_end = int(eval_start + eval_size)

            # Don't exceed data bounds
            if eval_end > total_records:
                eval_end = total_records
                eval_start = max(train_end, eval_end - eval_size)
                if eval_start >= eval_end:
                    break

            boundaries.append((train_start, train_end, eval_start, eval_end))

        return boundaries

    def run_validation(
        self,
        num_episodes: int = 1000,
        eval_episodes: int = 50,
        episode_length: int = 48,
        hidden_dims: Optional[List[int]] = None,
        verbose: bool = True,
        # Phase 2-4 parameters
        n_steps: int = 3,
        use_reward_norm: bool = True,
        min_wait_steps: int = 3,
        early_execution_penalty: float = 0.1,
        observation_bonus: float = 0.02,
        wait_penalty: float = 0.015,
        use_enhanced_features: bool = True,
        # Phase 4B-2: Risk adjustment
        use_risk_adjustment: bool = False,
        risk_penalty_weight: float = 0.5,
        target_savings: float = 0.10
    ) -> WalkForwardResults:
        """
        Run walk-forward validation across all folds.
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]

        fold_results = []

        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION")
        print(f"{'='*60}")
        print(f"Total records: {len(self.all_data)}")
        print(f"Number of folds: {len(self.fold_boundaries)}")
        print(f"Train/Eval ratio: {self.train_ratio:.0%}/{1-self.train_ratio:.0%}")
        print(f"Episodes per fold: {num_episodes}")
        print(f"{'='*60}\n")

        for fold_idx, (train_start, train_end, eval_start, eval_end) in enumerate(self.fold_boundaries):
            print(f"\n{'─'*60}")
            print(f"FOLD {fold_idx + 1}/{len(self.fold_boundaries)}")
            print(f"{'─'*60}")
            print(f"Train: records {train_start} to {train_end} ({train_end - train_start} records)")
            print(f"Eval:  records {eval_start} to {eval_end} ({eval_end - eval_start} records)")

            # Get train/eval data slices
            train_data = self.all_data[train_start:train_end]
            eval_data = self.all_data[eval_start:eval_end]

            # Get timestamps for reporting
            train_start_time = train_data[0]['timestamp'].isoformat() if train_data else ""
            train_end_time = train_data[-1]['timestamp'].isoformat() if train_data else ""
            eval_start_time = eval_data[0]['timestamp'].isoformat() if eval_data else ""
            eval_end_time = eval_data[-1]['timestamp'].isoformat() if eval_data else ""

            print(f"Train period: {train_start_time[:10]} to {train_end_time[:10]}")
            print(f"Eval period:  {eval_start_time[:10]} to {eval_end_time[:10]}")

            # Create data loaders for this fold
            train_loader = GasDataLoader(use_database=False)
            train_loader.set_cache(train_data)

            eval_loader = GasDataLoader(use_database=False)
            eval_loader.set_cache(eval_data)

            # Train agent on this fold
            fold_result = self._train_and_evaluate_fold(
                fold_idx=fold_idx,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_episodes=num_episodes,
                eval_episodes=eval_episodes,
                episode_length=episode_length,
                hidden_dims=hidden_dims,
                n_steps=n_steps,
                use_reward_norm=use_reward_norm,
                min_wait_steps=min_wait_steps,
                early_execution_penalty=early_execution_penalty,
                observation_bonus=observation_bonus,
                wait_penalty=wait_penalty,
                use_enhanced_features=use_enhanced_features,
                use_risk_adjustment=use_risk_adjustment,
                risk_penalty_weight=risk_penalty_weight,
                target_savings=target_savings,
                verbose=verbose
            )

            # Add metadata
            fold_result.train_start_idx = train_start
            fold_result.train_end_idx = train_end
            fold_result.eval_start_idx = eval_start
            fold_result.eval_end_idx = eval_end
            fold_result.train_records = train_end - train_start
            fold_result.eval_records = eval_end - eval_start
            fold_result.train_start_time = train_start_time
            fold_result.train_end_time = train_end_time
            fold_result.eval_start_time = eval_start_time
            fold_result.eval_end_time = eval_end_time

            fold_results.append(fold_result)

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Train best savings: {fold_result.best_train_savings*100:.2f}%")
            print(f"  Eval avg savings:   {fold_result.eval_avg_savings*100:.2f}%")
            print(f"  Eval positive rate: {fold_result.eval_positive_rate:.1f}%")

        # Aggregate results
        eval_savings = [f.eval_avg_savings for f in fold_results]
        positive_rates = [f.eval_positive_rate for f in fold_results]
        wait_times = [f.eval_avg_wait_time for f in fold_results]

        results = WalkForwardResults(
            n_folds=len(fold_results),
            total_train_records=sum(f.train_records for f in fold_results),
            total_eval_records=sum(f.eval_records for f in fold_results),
            mean_eval_savings=float(np.mean(eval_savings)),
            std_eval_savings=float(np.std(eval_savings)),
            median_eval_savings=float(np.median(eval_savings)),
            mean_positive_rate=float(np.mean(positive_rates)),
            mean_wait_time=float(np.mean(wait_times)),
            best_fold_savings=float(max(eval_savings)),
            worst_fold_savings=float(min(eval_savings)),
            folds=fold_results,
            timestamp=datetime.now().isoformat(),
            config={
                'n_folds': self.n_folds,
                'train_ratio': self.train_ratio,
                'overlap_ratio': self.overlap_ratio,
                'num_episodes': num_episodes,
                'eval_episodes': eval_episodes,
                'episode_length': episode_length,
                'n_steps': n_steps,
                'min_wait_steps': min_wait_steps,
                'use_risk_adjustment': use_risk_adjustment,
                'risk_penalty_weight': risk_penalty_weight,
                'target_savings': target_savings
            }
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Folds completed: {results.n_folds}")
        print(f"Mean eval savings: {results.mean_eval_savings*100:.2f}% (+/- {results.std_eval_savings*100:.2f}%)")
        print(f"Median eval savings: {results.median_eval_savings*100:.2f}%")
        print(f"Best fold: {results.best_fold_savings*100:.2f}%")
        print(f"Worst fold: {results.worst_fold_savings*100:.2f}%")
        print(f"Mean positive rate: {results.mean_positive_rate:.1f}%")
        print(f"Mean wait time: {results.mean_wait_time:.1f} steps")
        print(f"{'='*60}\n")

        return results

    def _train_and_evaluate_fold(
        self,
        fold_idx: int,
        train_loader: GasDataLoader,
        eval_loader: GasDataLoader,
        num_episodes: int,
        eval_episodes: int,
        episode_length: int,
        hidden_dims: List[int],
        n_steps: int,
        use_reward_norm: bool,
        min_wait_steps: int,
        early_execution_penalty: float,
        observation_bonus: float,
        wait_penalty: float,
        use_enhanced_features: bool,
        use_risk_adjustment: bool,
        risk_penalty_weight: float,
        target_savings: float,
        verbose: bool
    ) -> WalkForwardFold:
        """Train and evaluate a single fold."""

        # Create reward config
        reward_config = RewardConfig(
            wait_penalty=wait_penalty,
            min_wait_steps=min_wait_steps,
            early_execution_penalty=early_execution_penalty,
            observation_bonus=observation_bonus,
            use_risk_adjustment=use_risk_adjustment,
            risk_penalty_weight=risk_penalty_weight,
            target_savings=target_savings
        )

        # Create environment
        env = GasOptimizationEnv(
            train_loader,
            episode_length=episode_length,
            max_wait_steps=episode_length,
            reward_config=reward_config,
            scale_rewards=True,
            use_enhanced_features=use_enhanced_features
        )

        # Get DQN agent class
        DQNAgent = get_dqn_agent("auto")

        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space_shape[0],
            action_dim=env.action_space_n,
            hidden_dims=hidden_dims,
            learning_rate=0.0003,
            gamma=0.98,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=int(num_episodes * 0.5),
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

        # Pre-generate episodes
        training_episodes = train_loader.get_diverse_episodes(
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

        # Training loop
        best_savings = float('-inf')
        savings_history = []

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

            savings = info.get('savings_vs_initial', 0)
            savings_history.append(savings)

            if len(savings_history) >= 100:
                avg_savings = np.mean(savings_history[-100:])
                if avg_savings > best_savings:
                    best_savings = avg_savings

            # Progress update
            if verbose and (episode + 1) % 200 == 0:
                recent_savings = np.mean(savings_history[-100:]) if len(savings_history) >= 100 else np.mean(savings_history)
                print(f"  Episode {episode+1}/{num_episodes}: Recent savings = {recent_savings*100:.2f}%")

        # Final training savings
        final_savings = np.mean(savings_history[-100:]) if len(savings_history) >= 100 else np.mean(savings_history)

        # Evaluate on held-out data
        eval_results = evaluate_agent(
            agent,
            num_episodes=eval_episodes,
            verbose=False,
            data_loader=eval_loader,
            episode_length=episode_length,
            max_wait_steps=episode_length
        )

        return WalkForwardFold(
            fold_number=fold_idx + 1,
            train_start_idx=0,  # Will be set by caller
            train_end_idx=0,
            eval_start_idx=0,
            eval_end_idx=0,
            train_records=0,
            eval_records=0,
            best_train_savings=best_savings,
            final_train_savings=final_savings,
            eval_avg_reward=eval_results['avg_reward'],
            eval_avg_savings=eval_results['avg_savings'],
            eval_median_savings=eval_results['median_savings'],
            eval_positive_rate=eval_results['positive_savings_rate'],
            eval_avg_wait_time=eval_results['avg_wait_time'],
            train_start_time="",
            train_end_time="",
            eval_start_time="",
            eval_end_time=""
        )

    def save_results(self, results: WalkForwardResults, filepath: str):
        """Save results to JSON file."""
        # Convert to dict
        results_dict = {
            'n_folds': results.n_folds,
            'total_train_records': results.total_train_records,
            'total_eval_records': results.total_eval_records,
            'mean_eval_savings': results.mean_eval_savings,
            'std_eval_savings': results.std_eval_savings,
            'median_eval_savings': results.median_eval_savings,
            'mean_positive_rate': results.mean_positive_rate,
            'mean_wait_time': results.mean_wait_time,
            'best_fold_savings': results.best_fold_savings,
            'worst_fold_savings': results.worst_fold_savings,
            'folds': [asdict(f) for f in results.folds],
            'timestamp': results.timestamp,
            'config': results.config
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filepath}")


def main():
    """Run walk-forward validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Walk-forward validation for DQN agent')
    parser.add_argument('--chain-id', type=int, default=8453, help='Chain ID')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes per fold')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Evaluation episodes per fold')
    parser.add_argument('--episode-length', type=int, default=48, help='Episode length')
    parser.add_argument('--output', type=str, default='walk_forward_results.json', help='Output file')

    args = parser.parse_args()

    validator = WalkForwardValidator(
        chain_id=args.chain_id,
        n_folds=args.folds
    )

    results = validator.run_validation(
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length
    )

    validator.save_results(results, args.output)


if __name__ == '__main__':
    main()
