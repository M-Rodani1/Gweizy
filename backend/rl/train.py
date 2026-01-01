"""
Training script for the DQN gas optimization agent.
"""
import os
import sys
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import GasOptimizationEnv
from rl.agents.dqn import DQNAgent
from rl.data_loader import GasDataLoader


def train_dqn(
    num_episodes: int = 500,
    episode_length: int = 48,
    save_path: str = None,
    verbose: bool = True
):
    """
    Train DQN agent on historical gas data.
    
    Args:
        num_episodes: Number of training episodes
        episode_length: Steps per episode
        save_path: Where to save trained model
        verbose: Print training progress
    """
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'dqn_agent.pkl'
        )
    
    # Initialize
    data_loader = GasDataLoader()
    env = GasOptimizationEnv(data_loader, episode_length=episode_length)
    
    agent = DQNAgent(
        state_dim=env.observation_space_shape[0],
        action_dim=env.action_space_n,
        hidden_dims=[128, 64],
        learning_rate=0.0005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=50
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_savings = []
    
    print(f"Starting DQN training: {num_episodes} episodes")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            total_reward += reward
            step += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        
        savings = info.get('savings_vs_initial', 0)
        avg_savings.append(savings)
        
        # Logging
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_len = np.mean(episode_lengths[-50:])
            avg_save = np.mean(avg_savings[-50:]) * 100
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}, Avg Length: {avg_len:.1f}")
            print(f"  Avg Savings: {avg_save:.2f}%, Epsilon: {agent.epsilon:.3f}")
    
    # Save trained agent
    agent.episode_rewards = episode_rewards
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"Final Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
    print(f"Final Avg Savings (last 100): {np.mean(avg_savings[-100:])*100:.2f}%")
    print(f"Total Training Steps: {agent.training_steps}")
    
    return agent


def evaluate_agent(agent: DQNAgent, num_episodes: int = 100):
    """Evaluate trained agent."""
    data_loader = GasDataLoader()
    env = GasOptimizationEnv(data_loader, episode_length=48)
    
    total_rewards = []
    savings_list = []
    wait_times = []
    
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                savings_list.append(info.get('savings_vs_initial', 0))
                wait_times.append(info.get('time_waiting', 0))
                break
        
        total_rewards.append(total_reward)
    
    print("\nEVALUATION RESULTS")
    print("-" * 30)
    print(f"Avg Reward: {np.mean(total_rewards):.3f}")
    print(f"Avg Savings: {np.mean(savings_list)*100:.2f}%")
    print(f"Avg Wait Time: {np.mean(wait_times):.1f} steps")
    print(f"Positive Savings: {sum(1 for s in savings_list if s > 0)/len(savings_list)*100:.1f}%")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN gas optimization agent')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    args = parser.parse_args()
    
    agent = train_dqn(num_episodes=args.episodes)
    
    if args.evaluate:
        evaluate_agent(agent)
