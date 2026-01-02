"""
Automated DQN training pipeline.
Collects data, trains model, evaluates, and deploys.
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.collect_historical_data import verify_data_quality, enrich_existing_data
from rl.train import train_dqn, evaluate_agent
from rl.agents.dqn import DQNAgent
from rl.state import StateBuilder
from utils.logger import logger


def run_training_pipeline(
    collect_data: bool = False,
    hours_back: int = 720,
    episodes: int = 1000,
    evaluate: bool = True,
    verbose: bool = True
):
    """
    Run complete training pipeline.
    
    Args:
        collect_data: Whether to collect additional data first
        hours_back: Hours of historical data to use
        episodes: Number of training episodes
        evaluate: Whether to evaluate after training
        verbose: Print detailed output
    """
    logger.info("=" * 60)
    logger.info("DQN Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Verify/collect data
    if collect_data:
        logger.info("\n[Step 1/4] Collecting historical data...")
        from scripts.collect_historical_data import collect_historical_data
        collect_historical_data(hours_back=hours_back, interval_minutes=5)
    else:
        logger.info("\n[Step 1/4] Verifying data quality...")
        verify_data_quality(min_records=500)
    
    # Step 2: Enrich data
    logger.info("\n[Step 2/4] Enriching data...")
    enrich_existing_data()
    
    # Step 3: Train model
    logger.info("\n[Step 3/4] Training DQN agent...")
    logger.info(f"Training for {episodes} episodes...")
    
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'rl_agents'
    )
    os.makedirs(model_dir, exist_ok=True)
    
    agent = train_dqn(
        num_episodes=episodes,
        episode_length=48,
        save_path=os.path.join(model_dir, 'dqn_final.pkl'),
        checkpoint_dir=model_dir,
        checkpoint_freq=100,
        verbose=verbose,
        use_diverse_episodes=True
    )
    
    # Step 4: Evaluate
    if evaluate:
        logger.info("\n[Step 4/4] Evaluating trained agent...")
        results = evaluate_agent(agent, num_episodes=100, verbose=verbose)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final Model: {os.path.join(model_dir, 'dqn_final.pkl')}")
        logger.info(f"Best Model: {os.path.join(model_dir, 'dqn_best.pkl')}")
        logger.info(f"\nEvaluation Results:")
        logger.info(f"  Avg Savings: {results['avg_savings']*100:.2f}%")
        logger.info(f"  Positive Savings Rate: {results['positive_savings_rate']:.1f}%")
        logger.info(f"  Avg Confidence: {results['avg_confidence']:.3f}")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {os.path.join(model_dir, 'dqn_final.pkl')}")
        logger.info("Skipping evaluation (use --evaluate to enable)")
    
    return agent


def load_and_evaluate(model_path: str, num_episodes: int = 100):
    """Load a trained model and evaluate it."""
    logger.info(f"Loading model from {model_path}...")
    
    state_builder = StateBuilder(history_length=24)
    agent = DQNAgent(
        state_dim=state_builder.get_state_dim(),
        action_dim=2
    )
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    agent.load(model_path)
    logger.info("Model loaded successfully")
    
    results = evaluate_agent(agent, num_episodes=num_episodes, verbose=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Training Pipeline')
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect additional historical data before training')
    parser.add_argument('--hours', type=int, default=720,
                       help='Hours of historical data to use (default: 720)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--no-evaluate', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--evaluate-only', type=str,
                       help='Only evaluate an existing model (provide path)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of episodes for evaluation (default: 100)')
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        # Only evaluate existing model
        load_and_evaluate(args.evaluate_only, num_episodes=args.eval_episodes)
    else:
        # Run full pipeline
        run_training_pipeline(
            collect_data=args.collect_data,
            hours_back=args.hours,
            episodes=args.episodes,
            evaluate=not args.no_evaluate,
            verbose=not args.quiet
        )

