#!/usr/bin/env python3
"""
Train RL agent using data fetched from the production API
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime

from rl.environment import GasTransactionEnv, TransactionConfig
from rl.agents.dqn import DQNAgent, DQNConfig
from rl.data_loader import Episode


API_URL = "https://basegasfeesml-production.up.railway.app"


def fetch_historical_data(hours: int = 168) -> pd.DataFrame:
    """Fetch historical data from production API"""
    print(f"Fetching {hours} hours of historical data from API...")

    response = requests.get(f"{API_URL}/api/historical?hours={hours}", timeout=60)
    response.raise_for_status()

    data = response.json()
    records = data.get('data', [])

    if not records:
        raise ValueError("No data returned from API")

    print(f"Fetched {len(records)} records")

    # Convert to DataFrame
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['time'])
    df['gas_price'] = df['gwei']
    df = df.sort_values('timestamp')

    return df


def generate_episodes(df: pd.DataFrame, episode_length: int = 150) -> list:
    """Generate training episodes from DataFrame"""
    episodes = []
    gas_prices = df['gas_price'].values
    timestamps = df['timestamp'].values

    # Create episodes with sliding window
    step = episode_length // 2  # 50% overlap

    for i in range(0, len(gas_prices) - episode_length, step):
        episode_prices = gas_prices[i:i + episode_length]
        episode_times = timestamps[i:i + episode_length]

        episode = Episode(
            gas_prices=episode_prices.astype(np.float32),
            timestamps=episode_times,
            predictions={}
        )
        episodes.append(episode)

    return episodes


def train_agent(episodes: list, num_episodes: int = 500, save_dir: str = "models/rl_agents"):
    """Train DQN agent on episodes"""

    os.makedirs(save_dir, exist_ok=True)

    # Create agent
    state_dim = 15
    action_dim = 4

    agent_config = DQNConfig(
        hidden_sizes=[128, 128, 64],
        learning_rate=1e-4,
        epsilon_decay=0.995
    )

    agent = DQNAgent(state_dim, action_dim, agent_config)

    # Create environments for different urgency levels
    urgency_levels = [0.3, 0.5, 0.7]
    envs = {}

    for urgency in urgency_levels:
        tx_config = TransactionConfig(urgency_level=urgency)
        envs[urgency] = GasTransactionEnv(
            historical_data=np.zeros(100),
            config=tx_config
        )

    # Split episodes
    np.random.shuffle(episodes)
    n_train = int(len(episodes) * 0.8)
    n_val = int(len(episodes) * 0.1)

    train_episodes = episodes[:n_train]
    val_episodes = episodes[n_train:n_train + n_val]

    print(f"\nTraining on {len(train_episodes)} episodes")
    print(f"Validating on {len(val_episodes)} episodes")

    # Training loop
    best_eval_reward = float('-inf')
    training_rewards = []

    for ep_num in range(num_episodes):
        # Sample random episode and urgency
        episode_data = train_episodes[ep_num % len(train_episodes)]
        urgency = np.random.choice(urgency_levels)
        env = envs[urgency]

        # Reset environment with episode data
        env.gas_prices = episode_data.gas_prices
        env.gas_mean = np.mean(env.gas_prices)
        env.gas_std = np.std(env.gas_prices) + 1e-8
        env.gas_min = np.min(env.gas_prices)
        env.gas_max = np.max(env.gas_prices)
        state, _ = env.reset()

        episode_reward = 0
        episode_loss = 0
        steps = 0

        for step in range(60):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        agent.end_episode(episode_reward)
        training_rewards.append(episode_reward)

        # Logging
        if ep_num % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:]) if training_rewards else 0
            metrics = agent.get_metrics()
            print(f"Episode {ep_num:4d} | Reward: {avg_reward:7.2f} | "
                  f"Epsilon: {metrics['epsilon']:.3f} | Buffer: {metrics['buffer_size']:5d}")

        # Evaluation every 50 episodes
        if ep_num % 50 == 0 and ep_num > 0:
            eval_reward = evaluate_agent(agent, envs[0.5], val_episodes[:10])
            print(f"\n*** Eval Reward: {eval_reward:.2f} (Best: {best_eval_reward:.2f}) ***\n")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, "dqn_best.pt"))
                print("Saved new best model!")

    # Save final model
    agent.save(os.path.join(save_dir, "dqn_final.pt"))

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'best_eval_reward': float(best_eval_reward),
        'final_avg_reward': float(np.mean(training_rewards[-100:])),
        'total_training_episodes': len(train_episodes)
    }

    with open(os.path.join(save_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Models saved to: {save_dir}")

    return summary


def evaluate_agent(agent, env, episodes, num_episodes: int = 10) -> float:
    """Evaluate agent on validation episodes"""
    rewards = []

    for episode_data in episodes[:num_episodes]:
        env.gas_prices = episode_data.gas_prices
        env.gas_mean = np.mean(env.gas_prices)
        env.gas_std = np.std(env.gas_prices) + 1e-8
        env.gas_min = np.min(env.gas_prices)
        env.gas_max = np.max(env.gas_prices)
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(60):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards.append(episode_reward)

    return np.mean(rewards)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent from API data")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--data-hours", type=int, default=168, help="Hours of data to fetch")
    parser.add_argument("--save-dir", type=str, default="models/rl_agents")

    args = parser.parse_args()

    print("="*60)
    print("RL Agent Training from API Data")
    print("="*60)

    try:
        # Fetch data
        df = fetch_historical_data(hours=args.data_hours)

        # Generate episodes
        episodes = generate_episodes(df)
        print(f"Generated {len(episodes)} training episodes")

        # Train
        summary = train_agent(
            episodes=episodes,
            num_episodes=args.episodes,
            save_dir=args.save_dir
        )

        print("\nTraining Summary:")
        print(json.dumps(summary, indent=2))

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
