# DQN Agent Training Guide

This guide explains how to train and improve the DQN (Deep Q-Network) agent for gas price optimization.

## Overview

The DQN agent learns optimal transaction timing by:
- Observing gas price patterns and market conditions
- Learning when to wait vs. execute transactions
- Maximizing gas savings while considering urgency

## Quick Start

### 1. Collect Historical Data

First, ensure you have enough historical data for training:

```bash
cd backend
python scripts/collect_historical_data.py --hours 720 --verify
```

This will verify you have at least 500 records. If not, collect more:

```bash
python scripts/collect_historical_data.py --hours 720 --interval 5
```

### 2. Train the Model

Use the automated training pipeline:

```bash
python scripts/train_dqn_pipeline.py --episodes 1000
```

This will:
- Verify data quality
- Train the DQN agent for 1000 episodes
- Save checkpoints every 100 episodes
- Save the best model based on performance
- Evaluate the trained model

### 3. Evaluate a Trained Model

Evaluate an existing model:

```bash
python scripts/train_dqn_pipeline.py --evaluate-only models/rl_agents/dqn_final.pkl
```

## Training Options

### Basic Training

```bash
# Train with default settings (1000 episodes)
python scripts/train_dqn_pipeline.py

# Train with custom number of episodes
python scripts/train_dqn_pipeline.py --episodes 2000

# Collect data first, then train
python scripts/train_dqn_pipeline.py --collect-data --episodes 1000
```

### Advanced Training

```bash
# Train without evaluation (faster)
python scripts/train_dqn_pipeline.py --no-evaluate

# Quiet mode (less output)
python scripts/train_dqn_pipeline.py --quiet

# Use more historical data
python scripts/train_dqn_pipeline.py --hours 2160 --episodes 1500
```

### Direct Training Script

You can also use the training script directly:

```bash
python -m rl.train --episodes 1000 --evaluate
```

## Model Files

Trained models are saved in `backend/models/rl_agents/`:

- `dqn_final.pkl` - Final model after training
- `dqn_best.pkl` - Best performing model (auto-saved during training)
- `dqn_checkpoint_ep{N}.pkl` - Checkpoints every 100 episodes
- `training_summary.json` - Training statistics and metrics
- `training_metrics.json` - Latest training metrics

## Training Improvements

### Enhanced Data Loading

- **Database Integration**: Uses `DatabaseManager` to load real historical data
- **Data Augmentation**: Adds noise and variations to improve generalization
- **Diverse Episodes**: Samples episodes from different market conditions (high/low volatility)

### Better Hyperparameters

- **Deeper Network**: 3-layer network (128-128-64) instead of 2-layer
- **Larger Replay Buffer**: 50,000 capacity for more diverse experiences
- **Better Learning Rate**: 0.0003 for more stable training
- **Higher Discount Factor**: 0.98 for longer-term planning

### Training Features

- **Checkpointing**: Saves models every 100 episodes
- **Best Model Tracking**: Automatically saves best performing model
- **Progress Tracking**: Detailed metrics and ETA
- **Evaluation**: Comprehensive evaluation with multiple metrics

## Evaluation Metrics

After training, the model is evaluated on:

- **Average Reward**: Overall performance
- **Average Savings**: Percentage of gas saved
- **Positive Savings Rate**: % of transactions with positive savings
- **Action Distribution**: Wait vs. Execute ratio
- **Confidence Scores**: Model confidence in recommendations

## API Integration

The trained model is automatically loaded by the API:

```bash
# Check agent status
curl http://localhost:5000/api/agent/status

# Get recommendation
curl "http://localhost:5000/api/agent/recommend?urgency=0.5&gas_amount=150000"
```

## Troubleshooting

### Not Enough Data

If you see "Not enough data" errors:

1. Check database has records:
   ```bash
   python scripts/collect_historical_data.py --verify
   ```

2. Collect more data:
   ```bash
   python scripts/collect_historical_data.py --hours 720
   ```

### Model Not Loading

The API tries multiple paths:
- `models/rl_agents/dqn_final.pkl` (preferred)
- `models/rl_agents/dqn_best.pkl`
- `models/saved_models/dqn_agent.pkl`
- `models/dqn_agent.pkl`

Check logs to see which paths were tried.

### Training Takes Too Long

- Reduce episodes: `--episodes 500`
- Use `--quiet` to reduce output overhead
- Skip evaluation: `--no-evaluate`

## Next Steps

1. **More Training Data**: Collect 30+ days of historical data
2. **Hyperparameter Tuning**: Experiment with learning rates, network sizes
3. **Multi-Chain Training**: Train separate models for each chain
4. **Online Learning**: Continuously update model with new data
5. **Ensemble Methods**: Combine multiple models for better predictions

## Architecture

```
rl/
├── agents/
│   └── dqn.py          # DQN agent implementation
├── data_loader.py      # Historical data loading with augmentation
├── environment.py      # RL environment (gas optimization)
├── rewards.py          # Reward function
├── state.py            # State representation
└── train.py            # Training script with checkpointing
```

## References

- DQN Paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Experience Replay: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

