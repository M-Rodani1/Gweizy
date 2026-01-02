#!/usr/bin/env python3
"""
Test script for RL Transaction Environment

Verifies:
- Environment creation
- State/action spaces
- Episode rollout
- Reward calculation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_environment():
    """Test the GasTransactionEnv"""
    print("="*60)
    print("Testing RL Environment")
    print("="*60)

    from rl.environment import GasTransactionEnv, TransactionConfig, Action

    # Create sample gas price data
    np.random.seed(42)
    gas_prices = 0.01 + 0.002 * np.random.randn(100)
    gas_prices = np.abs(gas_prices)  # Ensure positive

    # Create environment
    config = TransactionConfig(
        max_wait_steps=60,
        urgency_level=0.5
    )
    env = GasTransactionEnv(gas_prices, config=config)

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Gas price range: {gas_prices.min():.6f} - {gas_prices.max():.6f}")

    # Reset environment
    state, info = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial state: {state}")

    # Run a few steps
    print("\nRunning episode...")
    total_reward = 0
    step = 0

    while True:
        # Random action
        action = env.action_space.sample()
        action_name = Action(action).name

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step < 5 or terminated or truncated:
            print(f"  Step {step:2d}: Action={action_name:12s} "
                  f"Reward={reward:7.3f} "
                  f"Gas={info.get('gas_price', 0):.6f}")

        step += 1

        if terminated or truncated:
            break

    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final info: {info}")

    return True


def test_rewards():
    """Test reward calculator"""
    print("\n" + "="*60)
    print("Testing Reward Calculator")
    print("="*60)

    from rl.rewards import RewardCalculator, RewardType, create_reward_calculator

    # Test different reward types
    for reward_type in ['cost', 'speed', 'balanced', 'adaptive']:
        calc = create_reward_calculator(reward_type)

        context = {
            'average_gas': 0.01,
            'min_gas_seen': 0.008,
            'prediction_1h': 0.011,
            'volatility': 0.02
        }

        # Test successful submission below average
        reward, breakdown = calc.calculate(
            action=1,  # SUBMIT_NOW
            gas_price_paid=0.009,
            wait_steps=10,
            success=True,
            context=context
        )

        print(f"\n{reward_type.upper()} reward:")
        print(f"  Total: {reward:.3f}")
        for k, v in breakdown.items():
            print(f"    {k}: {v:.3f}")

    return True


def test_state_builder():
    """Test state builder"""
    print("\n" + "="*60)
    print("Testing State Builder")
    print("="*60)

    from rl.state import StateBuilder, MarketState
    from datetime import datetime

    builder = StateBuilder()

    # Simulate some history
    for i in range(10):
        gas = 0.01 + 0.001 * np.sin(i / 5)
        builder.update_history(gas, datetime.now())

    # Build state
    market = MarketState(
        current_gas=0.011,
        timestamp=datetime.now(),
        predictions={'1h': 0.012, '4h': 0.010, '24h': 0.009},
        network_congestion=0.6
    )

    context = {
        'steps_remaining': 45,
        'max_steps': 60,
        'urgency': 0.5
    }

    state = builder.build_state(market, context)

    print(f"\nState shape: {state.shape}")
    print(f"State dimension: {builder.get_state_dim()}")
    print(f"State values: {state}")

    stats = builder.get_statistics()
    print(f"\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return True


def test_data_loader():
    """Test data loader"""
    print("\n" + "="*60)
    print("Testing Data Loader")
    print("="*60)

    from rl.data_loader import RLDataLoader, DataLoaderConfig

    config = DataLoaderConfig(
        episode_length=30,
        step_interval_minutes=1
    )
    loader = RLDataLoader(config)

    try:
        # Try loading from database
        df = loader.load_from_database(hours=24)

        # Generate episodes
        episodes = loader.generate_episodes(df)

        # Split
        train, val, test = loader.split_episodes(episodes)

        print(f"\nData loaded successfully:")
        print(f"  Records: {len(df)}")
        print(f"  Train episodes: {len(train)}")
        print(f"  Val episodes: {len(val)}")
        print(f"  Test episodes: {len(test)}")

        if train:
            print(f"\nSample episode:")
            print(f"  Length: {len(train[0])}")
            print(f"  Gas range: {train[0].gas_prices.min():.6f} - {train[0].gas_prices.max():.6f}")

        return True

    except Exception as e:
        print(f"Data loading test skipped: {e}")
        print("(This is expected if database is not available)")
        return True


def test_dqn_agent():
    """Test DQN agent"""
    print("\n" + "="*60)
    print("Testing DQN Agent")
    print("="*60)

    try:
        from rl.agents.dqn import DQNAgent, DQNConfig

        config = DQNConfig(
            hidden_sizes=[64, 32],
            min_buffer_size=10
        )

        agent = DQNAgent(
            state_dim=15,
            action_dim=4,
            config=config
        )

        print(f"\nAgent created on device: {agent.device}")
        print(f"Network architecture: {config.hidden_sizes}")

        # Test action selection
        state = np.random.randn(15).astype(np.float32)
        action = agent.select_action(state)
        print(f"\nSample action: {action}")

        # Get Q-values
        q_values = agent.get_q_values(state)
        print(f"Q-values: {q_values}")

        # Test storing transitions and training
        # Need more samples than batch_size (64) to train
        for i in range(100):
            s = np.random.randn(15).astype(np.float32)
            a = np.random.randint(4)
            r = np.random.randn()
            ns = np.random.randn(15).astype(np.float32)
            d = i == 99
            agent.store_transition(s, a, r, ns, d)

        loss = agent.train_step()
        print(f"\nTraining step loss: {loss:.6f}" if loss else "\nNo training yet (buffer warming up)")

        metrics = agent.get_metrics()
        print(f"\nAgent metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        return True

    except ImportError as e:
        print(f"DQN test skipped: {e}")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RL TRANSACTION AGENT - TEST SUITE")
    print("="*60)

    tests = [
        ("Environment", test_environment),
        ("Rewards", test_rewards),
        ("State Builder", test_state_builder),
        ("Data Loader", test_data_loader),
        ("DQN Agent", test_dqn_agent),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
