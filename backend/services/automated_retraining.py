"""
Automated retraining pipeline for ML models and DQN agents.
Supports scheduled retraining and auto-retrain on accuracy drops for all chains.
"""
import os
import sys
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from data.multichain_collector import CHAINS
from models.accuracy_tracker import AccuracyTracker
from models.model_trainer import GasModelTrainer
from models.feature_engineering import GasFeatureEngineer
from rl.train import train_dqn
from utils.logger import logger


class AutomatedRetraining:
    """Automated retraining service with scheduling and accuracy monitoring."""
    
    def __init__(self):
        self.accuracy_trackers = {}  # Per-chain accuracy trackers
        self.retraining_history = []
        self.is_running = False
        self.thread = None
        
        # Configuration
        self.retrain_interval_hours = 24  # Daily retraining
        self.accuracy_drop_threshold = 0.25  # 25% increase in error triggers retrain
        self.min_accuracy_mae = 0.01  # Minimum acceptable MAE
        self.min_data_points = 500  # Minimum data points needed
    
    def check_accuracy_and_retrain(self, chain_id: int = 8453, force: bool = False) -> Dict:
        """
        Check model accuracy and retrain if needed.
        
        Args:
            chain_id: Chain ID to check/retrain
            force: Force retraining even if accuracy is good
        
        Returns:
            Dict with retraining results
        """
        chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
        logger.info(f"Checking accuracy for {chain_name} (Chain ID: {chain_id})")
        
        try:
            # Initialize accuracy tracker for chain if needed
            if chain_id not in self.accuracy_trackers:
                tracker_path = f'models/saved_models/chain_{chain_id}/accuracy_tracking.db'
                self.accuracy_trackers[chain_id] = AccuracyTracker(db_path=tracker_path)
            
            tracker = self.accuracy_trackers[chain_id]
            
            # Get recent accuracy metrics
            recent_metrics = tracker.get_recent_metrics(window_size=100)
            
            if not recent_metrics or len(recent_metrics) < 50:
                logger.info(f"Insufficient accuracy data for {chain_name}, skipping retrain check")
                return {
                    'retrained': False,
                    'reason': 'Insufficient accuracy data',
                    'chain_id': chain_id
                }
            
            # Check for accuracy degradation
            should_retrain = force
            reason = "Forced retraining" if force else None
            
            if not should_retrain:
                # Check if MAE has increased significantly
                latest_mae = recent_metrics[-1].get('mae', 0)
                baseline_mae = recent_metrics[0].get('mae', latest_mae)
                
                if latest_mae > self.min_accuracy_mae:
                    mae_increase = (latest_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0
                    
                    if mae_increase > self.accuracy_drop_threshold:
                        should_retrain = True
                        reason = f"Accuracy dropped {mae_increase*100:.1f}% (MAE: {latest_mae:.6f})"
                
                # Check if MAE is above threshold
                if latest_mae > self.min_accuracy_mae * 2:
                    should_retrain = True
                    reason = f"MAE above threshold ({latest_mae:.6f} > {self.min_accuracy_mae*2:.6f})"
            
            if should_retrain:
                logger.info(f"Retraining needed for {chain_name}: {reason}")
                return self.retrain_chain_models(chain_id, reason=reason)
            else:
                logger.info(f"No retraining needed for {chain_name}. Latest MAE: {recent_metrics[-1].get('mae', 0):.6f}")
                return {
                    'retrained': False,
                    'reason': 'Accuracy acceptable',
                    'chain_id': chain_id,
                    'latest_mae': recent_metrics[-1].get('mae', 0)
                }
                
        except Exception as e:
            logger.error(f"Error checking accuracy for {chain_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'retrained': False,
                'error': str(e),
                'chain_id': chain_id
            }
    
    def retrain_chain_models(self, chain_id: int, reason: str = "Scheduled retraining") -> Dict:
        """
        Retrain ML models for a specific chain.
        
        Args:
            chain_id: Chain ID to retrain
            reason: Reason for retraining
        
        Returns:
            Dict with retraining results
        """
        chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
        start_time = datetime.now()
        
        logger.info(f"Starting retraining for {chain_name} (Chain ID: {chain_id})")
        logger.info(f"Reason: {reason}")
        
        try:
            # Prepare training data
            engineer = GasFeatureEngineer()
            X, y_1h, y_4h, y_24h = engineer.prepare_training_data(
                hours_back=720,
                chain_id=chain_id
            )
            
            if len(X) < self.min_data_points:
                return {
                    'retrained': False,
                    'reason': f'Insufficient data: {len(X)} < {self.min_data_points}',
                    'chain_id': chain_id
                }
            
            # Train models
            trainer = GasModelTrainer(chain_id=chain_id)
            trainer.train_all_models(X, y_1h, y_4h, y_24h, chain_id=chain_id)
            
            # Save models
            trainer.save_models(chain_id=chain_id)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            result = {
                'retrained': True,
                'chain_id': chain_id,
                'chain_name': chain_name,
                'reason': reason,
                'duration_minutes': duration,
                'training_samples': len(X),
                'timestamp': datetime.now().isoformat()
            }
            
            # Record in history
            self.retraining_history.append(result)
            
            logger.info(f"✓ Successfully retrained {chain_name} models in {duration:.1f} minutes")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed to retrain {chain_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'retrained': False,
                'error': str(e),
                'chain_id': chain_id,
                'chain_name': chain_name
            }
    
    def retrain_chain_dqn(self, chain_id: int, episodes: int = 500) -> Dict:
        """
        Retrain DQN agent for a specific chain.
        
        Args:
            chain_id: Chain ID to retrain
            episodes: Number of training episodes
        
        Returns:
            Dict with retraining results
        """
        chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
        start_time = datetime.now()
        
        logger.info(f"Starting DQN retraining for {chain_name} (Chain ID: {chain_id})")
        
        try:
            agent = train_dqn(
                num_episodes=episodes,
                episode_length=24,
                max_wait_steps=24,
                chain_id=chain_id,
                verbose=False,
                use_diverse_episodes=True
            )
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            result = {
                'retrained': True,
                'chain_id': chain_id,
                'chain_name': chain_name,
                'model_type': 'dqn',
                'episodes': episodes,
                'duration_minutes': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✓ Successfully retrained {chain_name} DQN agent in {duration:.1f} minutes")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed to retrain {chain_name} DQN: {e}")
            return {
                'retrained': False,
                'error': str(e),
                'chain_id': chain_id,
                'chain_name': chain_name
            }
    
    def retrain_all_chains(self, chains: Optional[List[int]] = None, 
                          train_ml: bool = True, train_dqn: bool = False) -> Dict:
        """
        Retrain models for all chains.
        
        Args:
            chains: List of chain IDs (None = all chains)
            train_ml: Whether to train ML models
            train_dqn: Whether to train DQN agents
        
        Returns:
            Dict with results for all chains
        """
        if chains is None:
            chains = list(CHAINS.keys())
        
        results = {
            'ml_models': {},
            'dqn_agents': {},
            'start_time': datetime.now().isoformat()
        }
        
        if train_ml:
            for chain_id in chains:
                result = self.check_accuracy_and_retrain(chain_id)
                results['ml_models'][chain_id] = result
        
        if train_dqn:
            for chain_id in chains:
                result = self.retrain_chain_dqn(chain_id, episodes=500)
                results['dqn_agents'][chain_id] = result
        
        results['end_time'] = datetime.now().isoformat()
        results['duration_minutes'] = (datetime.now() - datetime.fromisoformat(results['start_time'])).total_seconds() / 60
        
        return results
    
    def start_scheduler(self):
        """Start the automated retraining scheduler."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        
        # Schedule daily retraining check for all chains
        schedule.every().day.at("02:00").do(self._daily_retraining_check)
        
        # Schedule accuracy checks every 6 hours
        schedule.every(6).hours.do(self._accuracy_check_all_chains)
        
        # Start scheduler in background thread
        def run_scheduler():
            logger.info("Automated retraining scheduler started")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.thread = threading.Thread(target=run_scheduler, daemon=True)
        self.thread.start()
        
        logger.info("✓ Automated retraining scheduler started")
    
    def stop_scheduler(self):
        """Stop the automated retraining scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("Automated retraining scheduler stopped")
    
    def _daily_retraining_check(self):
        """Daily retraining check - runs at 2 AM UTC"""
        logger.info("=" * 60)
        logger.info("DAILY RETRAINING CHECK")
        logger.info("=" * 60)
        
        results = self.retrain_all_chains(train_ml=True, train_dqn=False)
        
        logger.info(f"Retraining complete: {sum(1 for r in results['ml_models'].values() if r.get('retrained'))} chains retrained")
    
    def _accuracy_check_all_chains(self):
        """Check accuracy for all chains and retrain if needed"""
        logger.info("Checking accuracy for all chains...")
        
        for chain_id in CHAINS.keys():
            self.check_accuracy_and_retrain(chain_id, force=False)


# Global instance
_retraining_service = None

def get_retraining_service() -> AutomatedRetraining:
    """Get or create automated retraining service instance."""
    global _retraining_service
    if _retraining_service is None:
        _retraining_service = AutomatedRetraining()
    return _retraining_service
