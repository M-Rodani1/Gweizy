"""
Training script for the DQN gas optimization agent.
Enhanced with better hyperparameters, evaluation, and checkpointing.
"""
import os
import sys
import numpy as np
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import GasOptimizationEnv
from rl.agents.dqn import DQNAgent
from rl.data_loader import GasDataLoader


def train_dqn(
    num_episodes: int = 10000,
    episode_length: int = 48,
    save_path: str = None,
    checkpoint_dir: str = None,
    checkpoint_freq: int = 100,
    verbose: bool = True,
    use_diverse_episodes: bool = True,
    chain_id: int = 8453
):
    """
    Train DQN agent on historical gas data for a specific chain.
    
    Args:
        num_episodes: Number of training episodes
        episode_length: Steps per episode
        save_path: Where to save final trained model
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Save checkpoint every N episodes
        verbose: Print training progress
        use_diverse_episodes: Use diverse episode sampling for better coverage
        chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
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
    
    env = GasOptimizationEnv(data_loader, episode_length=episode_length)
    
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        hidden_dims=[128, 128, 64],  # Deeper network
        learning_rate=0.0003,  # Lower learning rate for stability
        gamma=0.98,  # Higher discount for longer-term planning
        epsilon_start=1.0,
        epsilon_end=0.02,  # Lower final epsilon
        epsilon_decay=0.998,  # Slower decay
        buffer_size=50000,  # Larger replay buffer
        batch_size=128,  # Larger batch size
        target_update_freq=100,  # Update target network less frequently
        lr_decay=0.9995,  # Learning rate decay (per training step)
        lr_min=0.00001,  # Minimum learning rate
        gradient_clip=10.0  # Gradient clipping threshold
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_savings = []
    losses = []
    best_avg_reward = float('-inf')
    best_avg_savings = float('-inf')
    
    # Pre-generate diverse episodes for better training
    if use_diverse_episodes:
        print("Generating diverse training episodes...")
        training_episodes = data_loader.get_diverse_episodes(
            episode_length=episode_length,
            num_episodes=min(num_episodes * 2, 500)  # Generate more episodes than needed
        )
        print(f"Generated {len(training_episodes)} diverse episodes")
    else:
        training_episodes = None
    
    # Get chain name for display
    from data.multichain_collector import CHAINS
    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
    
    print(f"Starting DQN training for {chain_name} (Chain ID: {chain_id})")
    print(f"Episodes: {num_episodes}")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Network: {agent.q_network.hidden_dims}")
    print(f"Learning rate: {agent.learning_rate}, Gamma: {agent.gamma}")
    print("-" * 50)
    
    start_time = datetime.now()
    
    for episode in range(num_episodes):
        # Use pre-generated episode if available
        if training_episodes and episode < len(training_episodes):
            # Inject episode data into environment
            env._episode_data = training_episodes[episode]
            state = env.reset()
        else:
            state = env.reset()
        
        total_reward = 0
        step = 0
        episode_losses = []
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train (only if buffer has enough samples)
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            total_reward += reward
            step += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        
        savings = info.get('savings_vs_initial', 0)
        avg_savings.append(savings)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
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
                'avg_reward_last_100': float(np.mean(episode_rewards[-100:])),
                'avg_savings_last_100': float(np.mean(avg_savings[-100:])),
                'epsilon': float(agent.epsilon),
                'training_steps': agent.training_steps,
                'timestamp': datetime.now().isoformat()
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Track best performance
        if len(episode_rewards) >= 100:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_savings = np.mean(avg_savings[-100:])
            
            if recent_avg_reward > best_avg_reward:
                best_avg_reward = recent_avg_reward
                best_path = os.path.join(checkpoint_dir, 'dqn_best.pkl')
                agent.save(best_path)
                if verbose:
                    print(f"✓ New best model saved for {chain_name} (avg reward: {best_avg_reward:.3f})")
            
            if recent_avg_savings > best_avg_savings:
                best_avg_savings = recent_avg_savings
        
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
                print(f"  Buffer Size: {len(agent.replay_buffer)}/{agent.replay_buffer.buffer.maxlen}")
                print(f"  ETA: {eta_minutes:.1f} minutes")
                
                if len(episode_rewards) >= 100:
                    print(f"  Best Avg Reward (last 100): {best_avg_reward:.3f}")
                    print(f"  Best Avg Savings (last 100): {best_avg_savings*100:.2f}%")
    
    # Save final trained agent
    agent.episode_rewards = episode_rewards
    agent.save(save_path)
    
    # Save final training summary
    summary_path = os.path.join(checkpoint_dir, 'training_summary.json')
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    summary = {
        'chain_id': chain_id,
        'chain_name': chain_name,
        'total_episodes': num_episodes,
        'training_time_minutes': training_time,
        'final_avg_reward_last_100': float(np.mean(episode_rewards[-100:])),
        'final_avg_savings_last_100': float(np.mean(avg_savings[-100:])),
        'best_avg_reward': float(best_avg_reward),
        'best_avg_savings': float(best_avg_savings),
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


def evaluate_agent(agent: DQNAgent, num_episodes: int = 100, verbose: bool = True):
    """
    Evaluate trained agent with comprehensive metrics.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of evaluation episodes
        verbose: Print detailed results
    """
    data_loader = GasDataLoader()
    env = GasOptimizationEnv(data_loader, episode_length=48)
    
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
        print(f"    Wait: {action_distribution['wait']} ({action_distribution['wait']/num_episodes*100:.1f}%)")
        print(f"    Execute: {action_distribution['execute']} ({action_distribution['execute']/num_episodes*100:.1f}%)")
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
        'price_improvement': float(avg_price_improvement)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN gas optimization agent')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes (default: 10000)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    args = parser.parse_args()
    
    agent = train_dqn(num_episodes=args.episodes)
    
    if args.evaluate:
        evaluate_agent(agent)
