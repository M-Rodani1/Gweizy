"""
Hyperparameter tuning utility for DQN agent.

Provides grid search and random search capabilities for finding optimal hyperparameters.
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import itertools

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.agents.dqn import DQNAgent
from rl.environment import GasOptimizationEnv
from rl.data_loader import GasDataLoader
from rl.train import evaluate_agent


class HyperparameterTuner:
    """Hyperparameter tuning for DQN agent."""
    
    def __init__(self, chain_id: int = 8453):
        self.chain_id = chain_id
        self.results = []
        
    def grid_search(
        self,
        param_grid: Dict[str, List],
        num_episodes: int = 200,
        eval_episodes: int = 50,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Perform grid search over hyperparameter space.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
            num_episodes: Training episodes per configuration
            eval_episodes: Evaluation episodes per configuration
            verbose: Print progress
        
        Returns:
            List of results dictionaries sorted by performance
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        if verbose:
            print(f"Grid search: {len(combinations)} configurations to test")
            print(f"Parameters: {param_names}")
            print("=" * 60)
        
        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            if verbose:
                print(f"\n[{idx+1}/{len(combinations)}] Testing: {params}")
            
            result = self._test_configuration(
                params, num_episodes, eval_episodes, verbose
            )
            self.results.append(result)
            
            if verbose:
                print(f"  Result: Avg Reward = {result['avg_reward']:.3f}, "
                      f"Avg Savings = {result['avg_savings']*100:.2f}%")
        
        # Sort by performance
        self.results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        return self.results
    
    def random_search(
        self,
        param_ranges: Dict[str, Tuple],
        n_trials: int = 20,
        num_episodes: int = 200,
        eval_episodes: int = 50,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Perform random search over hyperparameter space.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            n_trials: Number of random configurations to test
            num_episodes: Training episodes per configuration
            eval_episodes: Evaluation episodes per configuration
            verbose: Print progress
        
        Returns:
            List of results dictionaries sorted by performance
        """
        if verbose:
            print(f"Random search: {n_trials} random configurations")
            print(f"Parameter ranges: {param_ranges}")
            print("=" * 60)
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            if verbose:
                print(f"\n[Trial {trial+1}/{n_trials}] Testing: {params}")
            
            result = self._test_configuration(
                params, num_episodes, eval_episodes, verbose
            )
            self.results.append(result)
            
            if verbose:
                print(f"  Result: Avg Reward = {result['avg_reward']:.3f}, "
                      f"Avg Savings = {result['avg_savings']*100:.2f}%")
        
        # Sort by performance
        self.results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        return self.results
    
    def _test_configuration(
        self,
        params: Dict,
        num_episodes: int,
        eval_episodes: int,
        verbose: bool
    ) -> Dict:
        """Test a single hyperparameter configuration."""
        # Setup environment
        data_loader = GasDataLoader()
        data_loader.load_data(hours=720, chain_id=self.chain_id)
        env = GasOptimizationEnv(data_loader, episode_length=24, max_wait_steps=24)
        
        # Extract parameters with defaults
        hidden_dims = params.get('hidden_dims', [128, 128, 64])
        learning_rate = params.get('learning_rate', 0.0003)
        gamma = params.get('gamma', 0.98)
        epsilon_end = params.get('epsilon_end', 0.02)
        epsilon_decay = params.get('epsilon_decay', 0.998)
        buffer_size = params.get('buffer_size', 50000)
        batch_size = params.get('batch_size', 128)
        target_update_freq = params.get('target_update_freq', 100)
        lr_decay = params.get('lr_decay', 0.9995)
        gradient_clip = params.get('gradient_clip', 10.0)
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space_shape[0],
            action_dim=env.action_space_n,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            lr_decay=lr_decay,
            lr_min=0.00001,
            gradient_clip=gradient_clip
        )
        
        # Train
        episode_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.train_step()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
        
        # Evaluate
        eval_results = evaluate_agent(agent, num_episodes=eval_episodes, verbose=False)
        
        # Store results
        result = {
            'params': params,
            'avg_reward': eval_results['avg_reward'],
            'avg_savings': eval_results['avg_savings'],
            'median_savings': eval_results['median_savings'],
            'positive_savings_rate': eval_results['positive_savings_rate'],
            'avg_wait_time': eval_results['avg_wait_time'],
            'training_rewards': episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_results(self, filepath: str):
        """Save tuning results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'chain_id': self.chain_id,
                'results': self.results,
                'best_config': self.results[0] if self.results else None
            }, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
        if self.results:
            print(f"Best configuration:")
            print(f"  Params: {self.results[0]['params']}")
            print(f"  Avg Reward: {self.results[0]['avg_reward']:.3f}")
            print(f"  Avg Savings: {self.results[0]['avg_savings']*100:.2f}%")


def main():
    """Example hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for DQN agent')
    parser.add_argument('--chain-id', type=int, default=8453, help='Chain ID')
    parser.add_argument('--method', choices=['grid', 'random'], default='random',
                       help='Search method')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials (random search)')
    parser.add_argument('--episodes', type=int, default=200, help='Training episodes per config')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Evaluation episodes')
    parser.add_argument('--output', type=str, default='hyperparameter_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(chain_id=args.chain_id)
    
    if args.method == 'grid':
        # Grid search example
        param_grid = {
            'learning_rate': [0.0001, 0.0003, 0.0005],
            'gamma': [0.95, 0.98, 0.99],
            'epsilon_decay': [0.995, 0.998, 0.999],
            'batch_size': [64, 128, 256]
        }
        tuner.grid_search(param_grid, num_episodes=args.episodes, 
                         eval_episodes=args.eval_episodes)
    else:
        # Random search example
        param_ranges = {
            'learning_rate': (0.0001, 0.001),
            'gamma': (0.95, 0.99),
            'epsilon_end': (0.01, 0.05),
            'epsilon_decay': (0.995, 0.999),
            'batch_size': (64, 256),
            'target_update_freq': (50, 200),
            'lr_decay': (0.999, 0.9999),
            'gradient_clip': (5.0, 20.0)
        }
        tuner.random_search(param_ranges, n_trials=args.trials,
                           num_episodes=args.episodes,
                           eval_episodes=args.eval_episodes)
    
    tuner.save_results(args.output)


if __name__ == '__main__':
    main()
