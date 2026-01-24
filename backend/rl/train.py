"""
Training script for the DQN gas optimization agent.
Enhanced with better hyperparameters, evaluation, and checkpointing.
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
from typing import Optional, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import GasOptimizationEnv
from rl.data_loader import GasDataLoader
from rl.rewards import RewardConfig

# Try to import PyTorch agent first, fall back to numpy
try:
    from rl.agents.dqn_torch import DQNAgent as DQNAgentTorch, PrioritizedReplayBuffer, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    DQNAgentTorch = None

from rl.agents.dqn import DQNAgent as DQNAgentNumpy

def get_dqn_agent(backend: str = "auto"):
    """Get appropriate DQN agent based on backend preference."""
    if backend == "auto":
        if TORCH_AVAILABLE:
            print("Using PyTorch backend (recommended)")
            return DQNAgentTorch
        else:
            print("PyTorch not available, using numpy backend")
            return DQNAgentNumpy
    elif backend == "pytorch":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch requested but not available. Install with: pip install torch")
        return DQNAgentTorch
    elif backend == "numpy":
        return DQNAgentNumpy
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'pytorch', or 'numpy'")


def train_dqn(
    num_episodes: int = 10000,
    episode_length: int = 48,  # Increased from 24 to allow more price observation
    max_wait_steps: Optional[int] = None,
    save_path: str = None,
    checkpoint_dir: str = None,
    checkpoint_freq: int = 100,
    verbose: bool = True,
    use_diverse_episodes: bool = True,
    chain_id: int = 8453,
    use_dueling: bool = True,
    hidden_dims: Optional[List[int]] = None,
    eval_every: int = 200,
    eval_episodes: int = 30,
    backend: str = "auto",  # "auto", "pytorch", or "numpy"
    # Phase 2 features (PyTorch only)
    n_steps: int = 3,  # N-step returns (1 = standard TD)
    use_reward_norm: bool = True,  # Reward normalization
    use_noisy_nets: bool = False,  # Noisy networks for exploration
    # Phase 3: Action timing parameters
    min_wait_steps: int = 3,  # Minimum steps before optimal execution
    early_execution_penalty: float = 0.1,  # Penalty for executing too early
    observation_bonus: float = 0.02,  # Bonus for waiting during volatile periods
    wait_penalty: float = 0.015,  # Per-step waiting cost
    # Phase 4A: Enhanced state features
    use_enhanced_features: bool = True  # Enable enhanced state representation
):
    """
    Train DQN agent on historical gas data for a specific chain.
    
    Args:
        num_episodes: Number of training episodes
        episode_length: Steps per episode
        max_wait_steps: Maximum wait steps before forced execution
        save_path: Where to save final trained model
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Save checkpoint every N episodes
        verbose: Print training progress
        use_diverse_episodes: Use diverse episode sampling for better coverage
        chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
        use_dueling: Whether to use dueling architecture
        hidden_dims: Hidden layer sizes
        eval_every: Evaluate on holdout every N episodes
        eval_episodes: Number of holdout episodes to evaluate
    """
    # Create chain-specific save paths
    # Use persistent storage on Railway, fallback to local
    if save_path is None:
        if os.path.exists('/data'):
            save_path = os.path.join('/data', 'models', 'rl_agents', f'chain_{chain_id}', 'dqn_final.pkl')
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(
                base_dir,
                'models', 'rl_agents', f'chain_{chain_id}', 'dqn_final.pkl'
            )
    
    if checkpoint_dir is None:
        if os.path.exists('/data'):
            checkpoint_dir = os.path.join('/data', 'models', 'rl_agents', f'chain_{chain_id}')
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_dir = os.path.join(
                base_dir,
                'models', 'rl_agents', f'chain_{chain_id}'
            )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize with chain-specific REAL data (no synthetic fallback)
    print("Loading REAL data from database (no synthetic fallback)...")
    data_loader = GasDataLoader(use_database=True)
    try:
        data_loader.load_data(hours=720, min_records=500, chain_id=chain_id)  # Require at least 500 records
        print(f"✅ Successfully loaded real data for training")
    except ValueError as e:
        print(f"❌ {e}")
        print("   Please collect real data before training DQN agent.")
        raise
    
    # Split data to avoid leakage: train on early segment, evaluate on holdout
    train_data, eval_data = data_loader.split_data(train_ratio=0.8)
    train_loader = GasDataLoader(use_database=False)
    train_loader.set_cache(train_data)
    eval_loader = GasDataLoader(use_database=False)
    eval_loader.set_cache(eval_data)

    if max_wait_steps is None:
        max_wait_steps = episode_length

    # Phase 3: Configure reward function with action timing parameters
    reward_config = RewardConfig(
        wait_penalty=wait_penalty,
        min_wait_steps=min_wait_steps,
        early_execution_penalty=early_execution_penalty,
        observation_bonus=observation_bonus
    )

    env = GasOptimizationEnv(
        train_loader,
        episode_length=episode_length,
        max_wait_steps=max_wait_steps,
        reward_config=reward_config,
        scale_rewards=True,  # Scale rewards to [-1, 1] for stable Q-learning
        use_enhanced_features=use_enhanced_features  # Phase 4A: Enhanced state representation
    )

    if hidden_dims is None:
        hidden_dims = [128, 64]

    # Select DQN backend
    DQNAgent = get_dqn_agent(backend)
    is_pytorch = (DQNAgent == DQNAgentTorch) if TORCH_AVAILABLE else False

    # Hyperparameters differ between backends
    if is_pytorch:
        # PyTorch: can handle higher learning rates due to better gradient handling
        learning_rate = 0.0003
        lr_min = 0.00005
        gradient_clip = 1.0
        batch_size = 64
    else:
        # Numpy: needs conservative hyperparameters
        learning_rate = 0.00005
        lr_min = 0.00005
        gradient_clip = 1.0
        batch_size = 32

    # Build agent kwargs
    agent_kwargs = dict(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=0.98,  # Higher discount for longer-term planning
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,  # Slower decay (used as fallback)
        epsilon_decay_episodes=num_episodes,  # Episode-based decay over full training
        epsilon_decay_steps=None,  # Disable step-based decay (was causing slow exploration)
        buffer_size=50000,  # Larger replay buffer
        batch_size=batch_size,
        target_update_freq=200,  # Update target network less frequently
        lr_decay=0.9999,  # Gradual LR decay
        lr_min=lr_min,
        gradient_clip=gradient_clip,
        use_per=True,  # Enable Prioritized Experience Replay
        per_alpha=0.6,
        per_beta=0.4,
        use_double_dqn=True,  # Enable Double DQN
        use_dueling=use_dueling,
        target_update_tau=0.001,  # Slower target updates (reduced from 0.005)
        use_soft_target=True
    )

    # Add Phase 2 features for PyTorch backend
    if is_pytorch:
        agent_kwargs.update(
            n_steps=n_steps,
            use_reward_norm=use_reward_norm,
            use_noisy_nets=use_noisy_nets
        )

    agent = DQNAgent(**agent_kwargs)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_savings = []
    losses = []
    best_avg_reward = float('-inf')
    best_avg_savings = float('-inf')
    best_eval_savings = float('-inf')
    last_eval_metrics = None
    
    # Pre-generate diverse episodes for better training
    if use_diverse_episodes:
        print("Generating diverse training episodes...")
        training_episodes = train_loader.get_diverse_episodes(
            episode_length=episode_length,
            num_episodes=min(num_episodes * 2, 500)  # Generate more episodes than needed
        )
        print(f"Generated {len(training_episodes)} diverse episodes")
    else:
        training_episodes = None

    # Fit state normalizer from a sample of training episodes
    if training_episodes:
        sample_states = []
        max_states = 2000
        for episode_data in training_episodes[:min(50, len(training_episodes))]:
            if len(sample_states) >= max_states:
                break
            state = env.reset(episode_data=episode_data[:episode_length])
            sample_states.append(state)
            done = False
            while not done and len(sample_states) < max_states:
                action = np.random.randint(0, env.action_space_n)
                next_state, _, done, _ = env.step(action)
                if not done:
                    sample_states.append(next_state)
        if sample_states:
            agent.fit_state_normalizer(np.array(sample_states))
    
    # Get chain name for display
    from data.multichain_collector import CHAINS
    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
    
    backend_name = "PyTorch" if is_pytorch else "NumPy"
    print(f"Starting Enhanced DQN training for {chain_name} (Chain ID: {chain_id})")
    print(f"Backend: {backend_name}")
    print(f"Episodes: {num_episodes}")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Network: {hidden_dims} (Dueling: {use_dueling})")
    print(f"Learning rate: {agent.learning_rate}, Gamma: {agent.gamma}")
    print(f"Features: PER={isinstance(agent.replay_buffer, PrioritizedReplayBuffer)}, "
          f"Double DQN={agent.use_double_dqn}, Dueling={use_dueling}")
    if is_pytorch:
        print(f"Phase 2: N-step={n_steps}, RewardNorm={use_reward_norm}, NoisyNets={use_noisy_nets}")
    print(f"Phase 3: MinWait={min_wait_steps}, EarlyPenalty={early_execution_penalty}, ObsBonus={observation_bonus}")
    print(f"Phase 4A: EnhancedFeatures={use_enhanced_features} (state_dim={env.observation_space_shape[0]})")
    print(f"Curriculum Learning: Enabled (episode length increases over time)")
    print("-" * 50)
    
    start_time = datetime.now()
    
    # Curriculum Learning: Start with shorter episodes, gradually increase
    # Minimum 12 steps to ensure meaningful exploration before forced execution
    min_episode_length = 12
    curriculum_episode_lengths = [
        (0, int(num_episodes * 0.2), max(min_episode_length, episode_length // 2)),  # First 20%: half length (min 12)
        (int(num_episodes * 0.2), int(num_episodes * 0.5), max(min_episode_length, int(episode_length * 0.75))),  # Next 30%: 75% length
        (int(num_episodes * 0.5), num_episodes, episode_length)  # Last 50%: full length
    ]
    
    def get_curriculum_length(episode_num: int) -> int:
        """Get episode length based on curriculum learning schedule."""
        for start, end, length in curriculum_episode_lengths:
            if start <= episode_num < end:
                return length
        return episode_length

    # Batch size schedule (ramp up as replay buffer grows)
    batch_schedule = [
        (0.0, 32),
        (0.35, 64),
        (0.7, 96)
    ]
    next_batch_idx = 1
    loss_spike_threshold = 5.0
    early_stop_loss_threshold = 1000.0  # Stop training if loss exceeds this
    early_stopped = False
    
    for episode in range(num_episodes):
        agent.decay_epsilon(episode)

        # Reset episode-specific state (n-step buffer, etc.)
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()

        # Curriculum learning: adjust episode length
        current_episode_length = get_curriculum_length(episode)
        if current_episode_length != episode_length:
            # Temporarily adjust environment episode length
            original_length = env.episode_length
            env.episode_length = current_episode_length
        
        # Use pre-generated episode if available
        if training_episodes and episode < len(training_episodes):
            episode_data = training_episodes[episode]
            if len(episode_data) > current_episode_length:
                episode_data = episode_data[:current_episode_length]
            state = env.reset(episode_data=episode_data)
        else:
            state = env.reset()
        
        total_reward = 0
        step = 0
        episode_losses = []
        last_td_error = None  # Track TD error for PER
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition with TD error for PER (if available from previous step)
            agent.store_transition(state, action, reward, next_state, done, td_error=last_td_error)
            
            # Train (only if buffer has enough samples)
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    # Store TD error for next transition (PER)
                    if agent.last_td_errors is not None and len(agent.last_td_errors) > 0:
                        # Use mean TD error as estimate for this transition
                        last_td_error = float(np.mean(agent.last_td_errors))
            
            total_reward += reward
            step += 1
            state = next_state
            
            if done:
                break
        
        # Restore original episode length
        if current_episode_length != episode_length:
            env.episode_length = original_length
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        
        savings = info.get('savings_vs_initial', 0)
        avg_savings.append(savings)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))

        # Gradually increase batch size as buffer grows
        progress = (episode + 1) / max(1, num_episodes)
        if next_batch_idx < len(batch_schedule) and progress >= batch_schedule[next_batch_idx][0]:
            agent.batch_size = batch_schedule[next_batch_idx][1]
            next_batch_idx += 1
        
        # Checkpointing
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'dqn_checkpoint_ep{episode+1}.pkl'
            )
            agent.save(checkpoint_path)
            
            # Save training metrics
            metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')
            metrics = {
                'episode': episode + 1,
                'avg_reward_last_50': float(np.mean(episode_rewards[-50:])),
                'avg_savings_last_50': float(np.mean(avg_savings[-50:])),
                'avg_loss_last_50': float(np.mean(losses[-50:]) if losses else 0),
                'epsilon': float(agent.epsilon),
                'training_steps': agent.training_steps,
                'timestamp': datetime.now().isoformat()
            }
            if last_eval_metrics:
                metrics['eval_metrics'] = last_eval_metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Track best performance (by savings)
        if len(episode_rewards) >= 100:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_savings = np.mean(avg_savings[-100:])

            if recent_avg_savings > best_avg_savings:
                best_avg_savings = recent_avg_savings
                best_avg_reward = recent_avg_reward
                best_path = os.path.join(checkpoint_dir, 'dqn_best.pkl')
                agent.save(best_path)
                if verbose:
                    print(
                        f"✓ New best model saved for {chain_name} "
                        f"(avg savings: {best_avg_savings*100:.2f}%)"
                    )

        # Periodic evaluation on holdout set
        if eval_every and (episode + 1) % eval_every == 0:
            last_eval_metrics = evaluate_agent(
                agent,
                num_episodes=eval_episodes,
                verbose=False,
                data_loader=eval_loader,
                episode_length=episode_length,
                max_wait_steps=max_wait_steps
            )
            if last_eval_metrics['avg_savings'] > best_eval_savings:
                best_eval_savings = last_eval_metrics['avg_savings']
                best_path = os.path.join(checkpoint_dir, 'dqn_best.pkl')
                agent.save(best_path)
                if verbose:
                    print(
                        f"✓ New best model saved for {chain_name} "
                        f"(eval avg savings: {best_eval_savings*100:.2f}%)"
                    )
        
        # Logging
        if verbose:
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_len = np.mean(episode_lengths[-50:])
                avg_save = np.mean(avg_savings[-50:]) * 100
                avg_loss = np.mean(losses[-50:]) if losses else 0
                
                elapsed = (datetime.now() - start_time).total_seconds()
                episodes_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (num_episodes - episode - 1) / episodes_per_sec if episodes_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"\nEpisode {episode + 1}/{num_episodes} ({episode + 1}/{num_episodes * 100:.1f}%)")
                print(f"  Avg Reward (last 50): {avg_reward:.3f}")
                print(f"  Avg Length: {avg_len:.1f}")
                print(f"  Avg Savings: {avg_save:.2f}%")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                buffer_capacity = getattr(agent.replay_buffer, "capacity", None)
                if buffer_capacity is None and hasattr(agent.replay_buffer, "tree"):
                    buffer_capacity = getattr(agent.replay_buffer.tree, "capacity", None)
                buffer_capacity = buffer_capacity if buffer_capacity is not None else "?"
                print(f"  Buffer Size: {len(agent.replay_buffer)}/{buffer_capacity}")
                print(f"  ETA: {eta_minutes:.1f} minutes")

                # Reduce learning rate if loss spikes
                if avg_loss > loss_spike_threshold and agent.learning_rate > agent.lr_min:
                    agent.learning_rate = max(agent.lr_min, agent.learning_rate * 0.5)
                    print(f"  ↓ LR reduced to {agent.learning_rate:.6f} due to loss spike")

                # Early stopping if loss diverges
                if avg_loss > early_stop_loss_threshold:
                    print(f"\n⚠️  EARLY STOPPING: Loss ({avg_loss:.2f}) exceeded threshold ({early_stop_loss_threshold})")
                    print(f"   Stopping to preserve best model. Training completed {episode + 1}/{num_episodes} episodes.")
                    early_stopped = True

                if len(episode_rewards) >= 100:
                    print(f"  Best Avg Reward (last 100): {best_avg_reward:.3f}")
                    print(f"  Best Avg Savings (last 100): {best_avg_savings*100:.2f}%")

        # Break out of training loop if early stopped
        if early_stopped:
            break

    # Save final trained agent
    agent.episode_rewards = episode_rewards
    agent.save(save_path)
    
    # Save final training summary
    summary_path = os.path.join(checkpoint_dir, 'training_summary.json')
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    actual_episodes = len(episode_rewards)
    summary = {
        'chain_id': chain_id,
        'chain_name': chain_name,
        'total_episodes': num_episodes,
        'actual_episodes': actual_episodes,
        'early_stopped': early_stopped,
        'training_time_minutes': training_time,
        'final_avg_reward_last_100': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards)),
        'final_avg_savings_last_100': float(np.mean(avg_savings[-100:])) if len(avg_savings) >= 100 else float(np.mean(avg_savings)),
        'best_avg_reward': float(best_avg_reward),
        'best_avg_savings': float(best_avg_savings),
        'best_eval_savings': float(best_eval_savings),
        'eval_metrics': last_eval_metrics,
        'total_training_steps': agent.training_steps,
        'final_epsilon': float(agent.epsilon),
        'final_buffer_size': len(agent.replay_buffer),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nModel saved to {save_path}")
    print(f"Training summary saved to {summary_path}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Total Episodes: {num_episodes}")
    print(f"Training Time: {training_time:.1f} minutes")
    print(f"Total Training Steps: {agent.training_steps}")
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")
    print(f"  Avg Savings: {np.mean(avg_savings[-100:])*100:.2f}%")
    print(f"\nBest Performance (last 100 episodes):")
    print(f"  Best Avg Reward: {best_avg_reward:.3f}")
    print(f"  Best Avg Savings: {best_avg_savings*100:.2f}%")
    print(f"\nFinal Epsilon: {agent.epsilon:.4f}")
    print(f"Final Buffer Size: {len(agent.replay_buffer)}")
    
    return agent


def evaluate_agent(
    agent,  # DQNAgent (PyTorch or Numpy)
    num_episodes: int = 100,
    verbose: bool = True,
    data_loader: Optional[GasDataLoader] = None,
    episode_length: int = 24,
    max_wait_steps: Optional[int] = None
):
    """
    Evaluate trained agent with comprehensive metrics.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of evaluation episodes
        verbose: Print detailed results
        episode_length: Steps per episode
        max_wait_steps: Maximum wait steps before forced execution
    """
    data_loader = data_loader or GasDataLoader()
    if max_wait_steps is None:
        max_wait_steps = episode_length
    env = GasOptimizationEnv(
        data_loader,
        episode_length=episode_length,
        max_wait_steps=max_wait_steps
    )
    
    total_rewards = []
    savings_list = []
    wait_times = []
    execution_prices = []
    initial_prices = []
    action_distribution = {'wait': 0, 'execute': 0}
    confidence_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        initial_price = env._episode_data[0]['gas_price']
        initial_prices.append(initial_price)
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            
            # Get recommendation for confidence
            rec = agent.get_recommendation(state, threshold=0.5)
            confidence_scores.append(rec['confidence'])
            
            if action == 0:
                action_distribution['wait'] += 1
            else:
                action_distribution['execute'] += 1
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                savings_list.append(info.get('savings_vs_initial', 0))
                wait_times.append(info.get('time_waiting', 0))
                execution_prices.append(info.get('execution_price', initial_price))
                break
        
        total_rewards.append(total_reward)
    
    # Calculate metrics
    avg_reward = np.mean(total_rewards)
    avg_savings = np.mean(savings_list)
    avg_wait = np.mean(wait_times)
    positive_savings_pct = sum(1 for s in savings_list if s > 0) / len(savings_list) * 100
    avg_confidence = np.mean(confidence_scores)
    total_actions = action_distribution['wait'] + action_distribution['execute']
    
    # Calculate savings statistics
    savings_array = np.array(savings_list)
    median_savings = np.median(savings_array) * 100
    std_savings = np.std(savings_array) * 100
    
    # Calculate price improvement
    price_improvements = []
    for i, (init, exec_price) in enumerate(zip(initial_prices, execution_prices)):
        if init > 0:
            improvement = (init - exec_price) / init
            price_improvements.append(improvement)
    
    avg_price_improvement = np.mean(price_improvements) * 100 if price_improvements else 0
    
    if verbose:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Episodes Evaluated: {num_episodes}")
        print(f"\nPerformance Metrics:")
        print(f"  Avg Reward: {avg_reward:.3f} ± {np.std(total_rewards):.3f}")
        print(f"  Avg Savings: {avg_savings*100:.2f}% ± {std_savings:.2f}%")
        print(f"  Median Savings: {median_savings:.2f}%")
        print(f"  Avg Price Improvement: {avg_price_improvement:.2f}%")
        print(f"  Positive Savings Rate: {positive_savings_pct:.1f}%")
        print(f"\nBehavior Metrics:")
        print(f"  Avg Wait Time: {avg_wait:.1f} steps")
        print(f"  Action Distribution:")
        print(f"    Wait: {action_distribution['wait']} ({(action_distribution['wait']/max(1, total_actions))*100:.1f}%)")
        print(f"    Execute: {action_distribution['execute']} ({(action_distribution['execute']/max(1, total_actions))*100:.1f}%)")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        print(f"\nSavings Distribution:")
        print(f"  Min: {np.min(savings_array)*100:.2f}%")
        print(f"  25th percentile: {np.percentile(savings_array, 25)*100:.2f}%")
        print(f"  75th percentile: {np.percentile(savings_array, 75)*100:.2f}%")
        print(f"  Max: {np.max(savings_array)*100:.2f}%")
    
    return {
        'avg_reward': float(avg_reward),
        'avg_savings': float(avg_savings),
        'median_savings': float(median_savings),
        'positive_savings_rate': float(positive_savings_pct),
        'avg_wait_time': float(avg_wait),
        'action_distribution': action_distribution,
        'avg_confidence': float(avg_confidence),
        'price_improvement': float(avg_price_improvement),
        'net_savings': float(avg_savings)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN gas optimization agent')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes (default: 10000)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--dueling', action='store_true', help='Force enable dueling architecture')
    parser.add_argument('--no-dueling', action='store_true', help='Disable dueling architecture')
    parser.add_argument('--hidden-dims', type=str, default=None, help='Comma-separated hidden dims (e.g., "128,64")')
    parser.add_argument('--eval-every', type=int, default=200, help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=30, help='Number of evaluation episodes')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'pytorch', 'numpy'],
                       help='Backend for DQN (auto=prefer pytorch, pytorch, numpy)')
    # Phase 2 features (PyTorch only)
    parser.add_argument('--n-steps', type=int, default=3, help='N-step returns (default: 3, use 1 for standard TD)')
    parser.add_argument('--no-reward-norm', action='store_true', help='Disable reward normalization')
    parser.add_argument('--noisy-nets', action='store_true', help='Use noisy networks for exploration (replaces epsilon-greedy)')
    # Phase 3: Action timing
    parser.add_argument('--min-wait-steps', type=int, default=3, help='Minimum steps before optimal execution (default: 3)')
    parser.add_argument('--early-penalty', type=float, default=0.1, help='Penalty for executing before min-wait-steps (default: 0.1)')
    parser.add_argument('--obs-bonus', type=float, default=0.02, help='Bonus for waiting during volatile periods (default: 0.02)')
    parser.add_argument('--wait-penalty', type=float, default=0.015, help='Per-step waiting cost (default: 0.015)')
    # Phase 4A: Enhanced state features
    parser.add_argument('--no-enhanced-features', action='store_true', help='Disable Phase 4A enhanced state features')
    args = parser.parse_args()

    use_dueling = True
    if args.no_dueling:
        use_dueling = False
    elif args.dueling:
        use_dueling = True

    hidden_dims = None
    if args.hidden_dims:
        hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',') if x.strip()]

    agent = train_dqn(
        num_episodes=args.episodes,
        use_dueling=use_dueling,
        hidden_dims=hidden_dims,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        backend=args.backend,
        # Phase 2 features
        n_steps=args.n_steps,
        use_reward_norm=not args.no_reward_norm,
        use_noisy_nets=args.noisy_nets,
        # Phase 3: Action timing
        min_wait_steps=args.min_wait_steps,
        early_execution_penalty=args.early_penalty,
        observation_bonus=args.obs_bonus,
        wait_penalty=args.wait_penalty,
        # Phase 4A: Enhanced state features
        use_enhanced_features=not args.no_enhanced_features
    )
    
    if args.evaluate:
        evaluate_agent(agent)
