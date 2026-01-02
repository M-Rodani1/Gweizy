"""
Automated training pipeline for all supported chains.
Trains both ML models and DQN agents for each chain.
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.multichain_collector import CHAINS
from models.model_trainer import GasModelTrainer
from models.feature_engineering import GasFeatureEngineer
from rl.train import train_dqn, evaluate_agent
from rl.agents.dqn import DQNAgent
from utils.logger import logger


def train_ml_models_for_chain(chain_id: int, hours_back: int = 720, verbose: bool = True):
    """
    Train ML models (1h, 4h, 24h) for a specific chain.
    
    Args:
        chain_id: Chain ID to train models for
        hours_back: Hours of historical data to use
        verbose: Print detailed output
    
    Returns:
        True if successful, False otherwise
    """
    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training ML Models for {chain_name} (Chain ID: {chain_id})")
    logger.info(f"{'='*60}")
    
    try:
        # Prepare training data
        engineer = GasFeatureEngineer()
        X, y_1h, y_4h, y_24h = engineer.prepare_training_data(
            hours_back=hours_back,
            chain_id=chain_id
        )
        
        if len(X) < 100:
            logger.warning(f"Not enough data for {chain_name}: {len(X)} samples")
            return False
        
        # Train models
        trainer = GasModelTrainer(chain_id=chain_id)
        trainer.train_all_models(X, y_1h, y_4h, y_24h, chain_id=chain_id)
        
        # Save models
        trainer.save_models(chain_id=chain_id)
        
        logger.info(f"✓ ML models trained and saved for {chain_name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to train ML models for {chain_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def train_dqn_for_chain(chain_id: int, episodes: int = 1000, verbose: bool = True):
    """
    Train DQN agent for a specific chain.
    
    Args:
        chain_id: Chain ID to train agent for
        episodes: Number of training episodes
        verbose: Print detailed output
    
    Returns:
        Trained agent or None if failed
    """
    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training DQN Agent for {chain_name} (Chain ID: {chain_id})")
    logger.info(f"{'='*60}")
    
    try:
        agent = train_dqn(
            num_episodes=episodes,
            chain_id=chain_id,
            verbose=verbose,
            use_diverse_episodes=True
        )
        
        logger.info(f"✓ DQN agent trained and saved for {chain_name}")
        return agent
        
    except Exception as e:
        logger.error(f"✗ Failed to train DQN agent for {chain_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def train_all_chains(
    chains: list = None,
    train_ml: bool = True,
    train_dqn: bool = True,
    ml_episodes: int = 1000,
    dqn_episodes: int = 1000,
    hours_back: int = 720,
    verbose: bool = True
):
    """
    Train models for all supported chains.
    
    Args:
        chains: List of chain IDs to train (None = all chains)
        train_ml: Whether to train ML models
        train_dqn: Whether to train DQN agents
        ml_episodes: Not used for ML (kept for consistency)
        dqn_episodes: Number of DQN training episodes
        hours_back: Hours of historical data to use
        verbose: Print detailed output
    """
    if chains is None:
        chains = list(CHAINS.keys())
    
    logger.info("=" * 60)
    logger.info("MULTI-CHAIN TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Chains: {[CHAINS[c]['name'] for c in chains]}")
    logger.info(f"Train ML: {train_ml}, Train DQN: {train_dqn}")
    logger.info(f"Hours of data: {hours_back}")
    
    results = {
        'ml_models': {},
        'dqn_agents': {},
        'start_time': datetime.now()
    }
    
    # Train ML models
    if train_ml:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Training ML Models")
        logger.info("=" * 60)
        
        for chain_id in chains:
            success = train_ml_models_for_chain(
                chain_id=chain_id,
                hours_back=hours_back,
                verbose=verbose
            )
            results['ml_models'][chain_id] = success
    
    # Train DQN agents
    if train_dqn:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Training DQN Agents")
        logger.info("=" * 60)
        
        for chain_id in chains:
            agent = train_dqn_for_chain(
                chain_id=chain_id,
                episodes=dqn_episodes,
                verbose=verbose
            )
            results['dqn_agents'][chain_id] = agent is not None
    
    # Summary
    results['end_time'] = datetime.now()
    results['duration'] = (results['end_time'] - results['start_time']).total_seconds() / 60
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total duration: {results['duration']:.1f} minutes")
    
    if train_ml:
        logger.info("\nML Models:")
        for chain_id, success in results['ml_models'].items():
            chain_name = CHAINS[chain_id]['name']
            status = "✓ Success" if success else "✗ Failed"
            logger.info(f"  {chain_name}: {status}")
    
    if train_dqn:
        logger.info("\nDQN Agents:")
        for chain_id, success in results['dqn_agents'].items():
            chain_name = CHAINS[chain_id]['name']
            status = "✓ Success" if success else "✗ Failed"
            logger.info(f"  {chain_name}: {status}")
    
    return results


def evaluate_all_chains():
    """Evaluate all trained models and agents."""
    logger.info("=" * 60)
    logger.info("EVALUATING ALL CHAINS")
    logger.info("=" * 60)
    
    for chain_id, chain_info in CHAINS.items():
        chain_name = chain_info['name']
        logger.info(f"\n{chain_name} (Chain ID: {chain_id}):")
        
        # Evaluate DQN agent
        try:
            from rl.state import StateBuilder
            state_builder = StateBuilder(history_length=24)
            agent = DQNAgent(
                state_dim=state_builder.get_state_dim(),
                action_dim=2
            )
            
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models', 'rl_agents', f'chain_{chain_id}', 'dqn_final.pkl'
            )
            
            if os.path.exists(model_path):
                agent.load(model_path)
                logger.info(f"  DQN Agent: Loaded")
                results = evaluate_agent(agent, num_episodes=50, verbose=False)
                logger.info(f"    Avg Savings: {results['avg_savings']*100:.2f}%")
                logger.info(f"    Positive Savings Rate: {results['positive_savings_rate']:.1f}%")
            else:
                logger.warning(f"  DQN Agent: Not found")
        except Exception as e:
            logger.warning(f"  DQN Agent: Error - {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models for all chains')
    parser.add_argument('--chains', type=str, help='Comma-separated chain IDs (e.g., 8453,1,42161)')
    parser.add_argument('--ml-only', action='store_true', help='Only train ML models')
    parser.add_argument('--dqn-only', action='store_true', help='Only train DQN agents')
    parser.add_argument('--episodes', type=int, default=1000, help='DQN training episodes')
    parser.add_argument('--hours', type=int, default=720, help='Hours of historical data')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained models')
    
    args = parser.parse_args()
    
    chains = None
    if args.chains:
        chains = [int(c.strip()) for c in args.chains.split(',')]
    
    train_ml = not args.dqn_only
    train_dqn = not args.ml_only
    
    if args.evaluate:
        evaluate_all_chains()
    else:
        train_all_chains(
            chains=chains,
            train_ml=train_ml,
            train_dqn=train_dqn,
            dqn_episodes=args.episodes,
            hours_back=args.hours,
            verbose=not args.quiet
        )

