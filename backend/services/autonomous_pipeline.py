"""
Autonomous ML Pipeline Service

A unified service that manages the entire ML lifecycle autonomously:
1. Continuous data quality monitoring
2. Automatic training when conditions are met
3. Automatic model evaluation and deployment
4. Failure handling and recovery
5. Comprehensive monitoring and alerts
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

from data.database import DatabaseManager
from models.model_registry import get_registry
from models.accuracy_tracker import AccuracyTracker
from utils.logger import logger

# Import for model reloading
try:
    from api.routes import reload_models
except ImportError:
    reload_models = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    total_records: int
    recent_records_24h: int
    recent_records_7d: int
    data_continuity_score: float  # 0-1, based on gaps
    feature_completeness: float  # 0-1, based on NaN rates
    sufficient_for_training: bool
    sufficient_for_agent_training: bool  # 500+ records for DQN agents
    days_of_data: float
    issues: List[str]


@dataclass
class TrainingDecision:
    """Decision about whether to train"""
    should_train: bool
    reason: str
    priority: str  # 'high', 'medium', 'low'
    estimated_duration_minutes: int


class AutonomousPipeline:
    """
    Autonomous ML pipeline that manages data collection, training, and deployment.
    """
    
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.last_training_time = None
        self.training_in_progress = False
        self.pipeline_history = []
        
        # Configuration
        self.check_interval_seconds = 300  # Check every 5 minutes
        self.min_data_points = 500  # Minimum samples for training
        self.optimal_data_points = 1000  # Optimal samples
        self.min_days_data = 7  # Minimum days of data
        self.optimal_days_data = 30  # Optimal days
        self.retrain_interval_hours = 24  # Retrain at least daily
        self.max_training_duration_minutes = 30  # Timeout for training
        
        # Data quality thresholds
        self.min_continuity_score = 0.7  # 70% data continuity
        self.max_feature_nan_rate = 0.1  # Max 10% NaN in features

        # Drift detection thresholds
        self.drift_threshold = 0.25  # 25% MAE increase triggers retrain
        self.drift_confidence_required = 0.8  # Need 80% confidence to act on drift

        # Alert tracking (avoid spam)
        self.last_drift_alert = {}  # {horizon: datetime}
        self.drift_alert_cooldown_hours = 1  # Don't alert more than once per hour
        
        # Model performance thresholds
        self.min_r2 = 0.1  # Minimum R² to deploy
        self.min_directional_accuracy = 0.45  # Minimum directional accuracy
        
        logger.info("AutonomousPipeline initialized")
    
    def check_data_quality(self) -> DataQualityMetrics:
        """Check data quality and availability"""
        try:
            db = DatabaseManager()
            
            # Get data counts
            all_data = db.get_historical_data(hours=720)  # 30 days
            recent_24h = db.get_historical_data(hours=24)
            recent_7d = db.get_historical_data(hours=168)
            
            total_records = len(all_data)
            recent_24h_count = len(recent_24h)
            recent_7d_count = len(recent_7d)
            
            # Calculate days of data
            if all_data:
                from datetime import datetime
                timestamps = []
                for d in all_data:
                    ts = d.get('timestamp', '')
                    if isinstance(ts, str):
                        try:
                            # Try parsing ISO format
                            if 'T' in ts or ' ' in ts:
                                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                            else:
                                timestamps.append(datetime.strptime(ts, '%Y-%m-%d'))
                        except (ValueError, TypeError):
                            try:
                                from dateutil import parser
                                timestamps.append(parser.parse(ts))
                            except (ValueError, TypeError, ImportError):
                                pass  # Skip unparseable timestamps
                    elif isinstance(ts, datetime):
                        timestamps.append(ts)
                
                if len(timestamps) > 1:
                    timestamps.sort()
                    days_of_data = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                else:
                    days_of_data = 0
            else:
                days_of_data = 0
            
            # Check data continuity (gaps)
            continuity_score = 1.0
            if len(all_data) > 1 and timestamps:
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                           for i in range(len(timestamps)-1)]
                if intervals:
                    expected_interval = 5  # 5 minutes
                    gaps = sum(1 for i in intervals if i > expected_interval * 2)
                    continuity_score = 1.0 - (gaps / len(intervals))
                    continuity_score = max(0.0, continuity_score)
            
            # Check feature completeness (would need to build features to check)
            # For now, assume good if we have data
            feature_completeness = 1.0
            
            # Determine if sufficient for training
            sufficient = (
                total_records >= self.min_data_points and
                days_of_data >= self.min_days_data and
                continuity_score >= self.min_continuity_score
            )
            
            # Determine if sufficient for agent training (lower threshold: 500 records)
            sufficient_for_agents = total_records >= 500 and days_of_data >= 3
            
            # Collect issues
            issues = []
            if total_records < self.min_data_points:
                issues.append(f"Insufficient records: {total_records} < {self.min_data_points}")
            if days_of_data < self.min_days_data:
                issues.append(f"Insufficient days: {days_of_data:.1f} < {self.min_days_data}")
            if continuity_score < self.min_continuity_score:
                issues.append(f"Poor data continuity: {continuity_score:.2%} < {self.min_continuity_score:.2%}")
            if recent_24h_count < 100:
                issues.append(f"Low recent data: {recent_24h_count} records in last 24h")
            
            return DataQualityMetrics(
                total_records=total_records,
                recent_records_24h=recent_24h_count,
                recent_records_7d=recent_7d_count,
                data_continuity_score=continuity_score,
                feature_completeness=feature_completeness,
                sufficient_for_training=sufficient,
                sufficient_for_agent_training=sufficient_for_agents,
                days_of_data=days_of_data,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return DataQualityMetrics(
                total_records=0,
                recent_records_24h=0,
                recent_records_7d=0,
                data_continuity_score=0.0,
                feature_completeness=0.0,
                sufficient_for_training=False,
                sufficient_for_agent_training=False,
                days_of_data=0.0,
                issues=[f"Error checking data: {str(e)}"]
            )
    
    def check_drift_status(self) -> Dict:
        """
        Check drift status across all horizons.
        Returns drift info and whether retraining is recommended.
        """
        try:
            tracker = AccuracyTracker()
            drift_status = {
                'horizons': {},
                'any_drifting': False,
                'should_retrain': False,
                'drifting_horizons': []
            }

            for horizon in ['1h', '4h', '24h']:
                drift = tracker.check_drift(horizon)
                drift_status['horizons'][horizon] = {
                    'is_drifting': drift.is_drifting,
                    'drift_ratio': drift.drift_ratio,
                    'mae_current': drift.mae_current,
                    'mae_baseline': drift.mae_baseline,
                    'confidence': drift.confidence,
                    'sample_size': drift.sample_size
                }

                if drift.is_drifting and drift.confidence >= self.drift_confidence_required:
                    drift_status['any_drifting'] = True
                    drift_status['drifting_horizons'].append(horizon)

                    # Log alert (with cooldown to avoid spam)
                    self._log_drift_alert(horizon, drift)

            # Recommend retrain if any horizon is drifting with high confidence
            drift_status['should_retrain'] = drift_status['any_drifting']

            return drift_status

        except Exception as e:
            logger.warning(f"Error checking drift status: {e}")
            return {
                'horizons': {},
                'any_drifting': False,
                'should_retrain': False,
                'drifting_horizons': [],
                'error': str(e)
            }

    def _log_drift_alert(self, horizon: str, drift_metrics):
        """Log drift alert with cooldown to prevent spam."""
        now = datetime.now()
        last_alert = self.last_drift_alert.get(horizon)

        if last_alert:
            hours_since = (now - last_alert).total_seconds() / 3600
            if hours_since < self.drift_alert_cooldown_hours:
                return  # Skip, within cooldown

        # Log the alert
        logger.warning(f"🚨 DRIFT ALERT [{horizon}]: MAE increased by {drift_metrics.drift_ratio*100:.1f}% "
                      f"(current: {drift_metrics.mae_current:.6f}, baseline: {drift_metrics.mae_baseline:.6f})")
        logger.warning(f"   Confidence: {drift_metrics.confidence*100:.0f}%, Samples: {drift_metrics.sample_size}")
        logger.warning(f"   → Recommend retraining models for {horizon} horizon")

        self.last_drift_alert[horizon] = now

    def should_train(self, data_quality: DataQualityMetrics) -> TrainingDecision:
        """Determine if training should be triggered (ML models and/or DQN agents)"""

        # Don't train if already training
        if self.training_in_progress:
            return TrainingDecision(
                should_train=False,
                reason="Training already in progress",
                priority="none",
                estimated_duration_minutes=0
            )

        # Check if data is sufficient
        if not data_quality.sufficient_for_training:
            return TrainingDecision(
                should_train=False,
                reason=f"Insufficient data: {', '.join(data_quality.issues)}",
                priority="none",
                estimated_duration_minutes=0
            )

        # Initialize decision variables
        should_train = False
        reason = "No training needed"
        priority = "none"

        # Check 1: Drift detection (highest priority)
        drift_status = self.check_drift_status()
        if drift_status['should_retrain']:
            should_train = True
            drifting = ', '.join(drift_status['drifting_horizons'])
            reason = f"🚨 Drift detected in {drifting} - accuracy degraded >25%"
            priority = "critical"
            logger.info(f"Auto-retrain triggered by drift: {reason}")

        # Check 2: Time-based (if not already triggered by drift)
        if not should_train:
            if self.last_training_time is None:
                should_train = True
                reason = "No previous training recorded"
                priority = "high"
            else:
                hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
                if hours_since_training >= self.retrain_interval_hours:
                    should_train = True
                    reason = f"Last training was {hours_since_training:.1f} hours ago (threshold: {self.retrain_interval_hours}h)"
                    priority = "medium"

        # Check 3: Optimal data boost (upgrade priority if lots of data)
        if should_train and priority != "critical":
            if data_quality.total_records >= self.optimal_data_points and data_quality.days_of_data >= self.optimal_days_data:
                priority = "high"
                reason = f"Optimal data available ({data_quality.total_records} records, {data_quality.days_of_data:.1f} days)"

        # Estimate training duration (based on data size)
        estimated_duration = 5  # Base 5 minutes for ML
        if data_quality.total_records > 2000:
            estimated_duration = 15
        elif data_quality.total_records > 1000:
            estimated_duration = 10

        # Add agent training time if we have enough data
        if data_quality.total_records >= 500:
            estimated_duration += 10

        return TrainingDecision(
            should_train=should_train,
            reason=reason,
            priority=priority,
            estimated_duration_minutes=estimated_duration
        )
    
    def trigger_training(self) -> Dict:
        """Trigger ML model training and optionally DQN agent training"""
        if self.training_in_progress:
            return {
                'success': False,
                'error': 'Training already in progress',
                'timestamp': datetime.now().isoformat()
            }
        
        self.training_in_progress = True
        start_time = datetime.now()
        
        try:
            logger.info("="*60)
            logger.info("🤖 AUTONOMOUS PIPELINE: Triggering Model Training")
            logger.info("="*60)
            
            # Check data quality to determine what to train
            data_quality = self.check_data_quality()
            should_train_agents = data_quality.total_records >= 500  # Minimum for agent training
            
            results = {
                'ml_training': None,
                'agent_training': None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 1: Train ML models (disabled - now done via Colab)
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(current_dir, "scripts", "retrain_models_simple.py")
            
            logger.warning("Automated training is disabled. Please use notebooks/train_models_colab.ipynb")
            raise NotImplementedError("Training must be done via Google Colab notebook: notebooks/train_models_colab.ipynb")
            
            if False:  # Disabled
                logger.info(f"Running ML training script: {script_path}")
            
            # Run ML training in subprocess
            ml_result = subprocess.run(
                [sys.executable, script_path],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=self.max_training_duration_minutes * 60
            )
            
            ml_duration = (datetime.now() - start_time).total_seconds() / 60
            
            if ml_result.returncode == 0:
                logger.info(f"✅ ML training completed successfully in {ml_duration:.1f} minutes")
                results['ml_training'] = {
                    'success': True,
                    'duration_minutes': ml_duration
                }

                # Step 1b: Train spike detector models (disabled - now in Colab)
                logger.info("🎯 Training spike detector classifiers...")
                spike_script_path = os.path.join(current_dir, "scripts", "train_spike_detectors.py")

                if False:  # Disabled - training now done via Colab
                    spike_start = datetime.now()
                    spike_result = subprocess.run(
                        [sys.executable, spike_script_path],
                        cwd=current_dir,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    spike_duration = (datetime.now() - spike_start).total_seconds() / 60

                    if spike_result.returncode == 0:
                        logger.info(f"✅ Spike detector training completed in {spike_duration:.1f} minutes")
                        results['spike_detector_training'] = {
                            'success': True,
                            'duration_minutes': spike_duration
                        }
                    else:
                        logger.warning(f"⚠️ Spike detector training failed: {spike_result.stderr[:200]}")
                        results['spike_detector_training'] = {
                            'success': False,
                            'error': spike_result.stderr[:200]
                        }
                else:
                    logger.warning(f"Spike detector script not found: {spike_script_path}")
                    results['spike_detector_training'] = {
                        'skipped': True,
                        'reason': 'Script not found'
                    }

                # Auto-reload models
                if reload_models:
                    try:
                        reload_result = reload_models()
                        if reload_result.get('success'):
                            logger.info("✅ Models auto-reloaded after training")
                        else:
                            logger.warning(f"⚠️ Model reload failed: {reload_result.get('error')}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error reloading models: {e}")

                # Evaluate new model
                evaluation = self.evaluate_new_model()
                results['evaluation'] = evaluation
            else:
                error_msg = ml_result.stderr[:500] if ml_result.stderr else "Unknown error"
                logger.error(f"❌ ML training failed: {error_msg}")
                results['ml_training'] = {
                    'success': False,
                    'error': error_msg,
                    'duration_minutes': ml_duration
                }
            
            # Step 2: Train DQN agents if we have enough data
            if should_train_agents:
                logger.info("="*60)
                logger.info("🤖 AUTONOMOUS PIPELINE: Triggering Agent Training")
                logger.info("="*60)
                logger.info(f"Data available: {data_quality.total_records} records (minimum: 500)")
                
                agent_start_time = datetime.now()
                agent_result = self._train_dqn_agents(data_quality)
                agent_duration = (datetime.now() - agent_start_time).total_seconds() / 60
                
                results['agent_training'] = {
                    'success': agent_result.get('success', False),
                    'duration_minutes': agent_duration,
                    'details': agent_result
                }
                
                if agent_result.get('success'):
                    logger.info(f"✅ Agent training completed successfully in {agent_duration:.1f} minutes")
                else:
                    logger.warning(f"⚠️ Agent training failed: {agent_result.get('error', 'Unknown error')}")
            else:
                logger.info(f"⏸️  Skipping agent training: insufficient data ({data_quality.total_records} < 500 records)")
                results['agent_training'] = {
                    'skipped': True,
                    'reason': f"Insufficient data: {data_quality.total_records} < 500 records"
                }
            
            total_duration = (datetime.now() - start_time).total_seconds() / 60
            
            # Determine overall success
            ml_success = results['ml_training'] and results['ml_training'].get('success', False)
            agent_success = results['agent_training'] and (
                results['agent_training'].get('success', False) or 
                results['agent_training'].get('skipped', False)
            )
            
            overall_success = ml_success and (agent_success or not should_train_agents)
            
            if overall_success:
                self.last_training_time = datetime.now()
                results['success'] = True
                results['duration_minutes'] = total_duration
                
                self.pipeline_history.append({
                    'action': 'training',
                    'result': results,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"✅ Autonomous training cycle completed in {total_duration:.1f} minutes")
            else:
                results['success'] = False
                results['error'] = "One or more training steps failed"
            
            return results
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Training timed out after {self.max_training_duration_minutes} minutes")
            return {
                'success': False,
                'error': f'Training timed out after {self.max_training_duration_minutes} minutes',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error triggering training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.training_in_progress = False
    
    def _train_dqn_agents(self, data_quality: DataQualityMetrics) -> Dict:
        """Train DQN agents for transaction timing recommendations"""
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # DQN training now done via Colab notebook
            logger.warning("DQN training must be done via Google Colab notebook: notebooks/train_models_colab.ipynb")
            agent_script = os.path.join(current_dir, "scripts", "train_dqn_pipeline.py")
            if False:  # Disabled - training now done via Colab
                logger.info(f"Running agent training script: {agent_script}")
            
            # Determine episodes based on data size
            episodes = 500  # Default
            if data_quality.total_records >= 5000:
                episodes = 1000  # More data = more episodes
            elif data_quality.total_records >= 2000:
                episodes = 750
            
            # Run agent training (non-blocking, but with timeout)
            result = subprocess.run(
                [sys.executable, agent_script, '--episodes', str(episodes), '--quiet'],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=20 * 60  # 20 minute timeout for agent training
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'episodes': episodes,
                    'output': result.stdout[-500:] if result.stdout else None
                }
            else:
                error_msg = result.stderr[:500] if result.stderr else result.stdout[-500] if result.stdout else "Unknown error"
                logger.warning(f"Agent training script failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'episodes': episodes
                }
                
        except subprocess.TimeoutExpired:
            logger.warning("Agent training timed out (this is acceptable, training continues in background)")
            return {
                'success': False,
                'error': 'Training timed out (may still be running)',
                'note': 'Agent training can take 20+ minutes, consider running separately'
            }
        except Exception as e:
            logger.warning(f"Error running agent training script: {e}")
            # Try direct training as fallback
            return self._train_dqn_direct(data_quality)
    
    def _train_dqn_direct(self, data_quality: DataQualityMetrics) -> Dict:
        """Train DQN agents directly using the training module"""
        try:
            from rl.train import train_dqn
            from config import Config
            import os
            
            # Determine episodes based on data size
            episodes = 500  # Default
            if data_quality.total_records >= 5000:
                episodes = 1000
            elif data_quality.total_records >= 2000:
                episodes = 750
            
            logger.info(f"Training DQN agent directly: {episodes} episodes")
            
            # Use persistent storage for saving agents
            if os.path.exists('/data'):
                save_path = os.path.join('/data', 'models', 'rl_agents', 'chain_8453', 'dqn_final.pkl')
                checkpoint_dir = os.path.join('/data', 'models', 'rl_agents', 'chain_8453')
            else:
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                save_path = os.path.join(current_dir, 'models', 'rl_agents', 'chain_8453', 'dqn_final.pkl')
                checkpoint_dir = os.path.join(current_dir, 'models', 'rl_agents', 'chain_8453')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Train agent (this may take a while, but we'll let it run)
            agent = train_dqn(
                num_episodes=episodes,
                episode_length=24,
                max_wait_steps=24,
                save_path=save_path,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=100,
                verbose=False,  # Less verbose for autonomous pipeline
                use_diverse_episodes=True,
                chain_id=8453  # Base chain
            )
            
            # Convert training_steps to native Python int
            training_steps = getattr(agent, 'training_steps', 0)
            if hasattr(training_steps, 'item'):
                training_steps = int(training_steps.item())
            elif not isinstance(training_steps, (int, float)):
                training_steps = int(training_steps)
            
            return {
                'success': True,
                'episodes': int(episodes),
                'training_steps': int(training_steps),
                'model_path': save_path
            }
            
        except Exception as e:
            logger.error(f"Error in direct agent training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_new_model(self) -> Dict:
        """Evaluate newly trained model and decide if it should be activated"""
        try:
            registry = get_registry()
            
            # Get latest model versions
            for horizon in ['1h', '4h', '24h']:
                try:
                    versions = registry.get_version_history(horizon)
                    if not versions:
                        continue
                    
                    latest = versions[0]  # Most recent
                    metrics = latest.get('metrics', {})
                    
                    r2 = metrics.get('r2', -999)
                    directional_acc = metrics.get('directional_accuracy', 0)
                    
                    # Check if model meets minimum thresholds
                    if r2 >= self.min_r2 and directional_acc >= self.min_directional_accuracy:
                        # Check if it's better than active model
                        active = registry.get_active_version(horizon)
                        if active:
                            active_metrics = active.get('metrics', {})
                            active_r2 = active_metrics.get('r2', -999)
                            
                            # Activate if new model is better
                            if r2 > active_r2:
                                try:
                                    registry.activate_version(horizon, latest['version'])
                                    logger.info(f"✅ Auto-activated better {horizon} model (R²: {r2:.4f} > {active_r2:.4f})")
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to activate {horizon} model: {e}")
                        else:
                            # No active model, activate this one
                            try:
                                registry.activate_version(horizon, latest['version'])
                                logger.info(f"✅ Auto-activated {horizon} model (no previous active model)")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to activate {horizon} model: {e}")
                    else:
                        logger.warning(f"⚠️ New {horizon} model doesn't meet thresholds (R²: {r2:.4f}, Dir Acc: {directional_acc:.2%})")
                        
                except Exception as e:
                    logger.warning(f"Error evaluating {horizon} model: {e}")
            
            return {'evaluated': True, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error evaluating new model: {e}")
            return {'evaluated': False, 'error': str(e)}
    
    def run_autonomous_cycle(self):
        """Run one cycle of the autonomous pipeline"""
        try:
            # Check data quality
            data_quality = self.check_data_quality()
            
            # Log data quality status
            if data_quality.issues:
                logger.info(f"📊 Data Quality: {len(data_quality.issues)} issues - {', '.join(data_quality.issues[:2])}")
            else:
                logger.info(f"📊 Data Quality: ✅ Good ({data_quality.total_records} records, {data_quality.days_of_data:.1f} days)")
            
            # Decide if training is needed
            decision = self.should_train(data_quality)
            
            if decision.should_train:
                logger.info(f"🤖 Training Decision: {decision.priority.upper()} priority - {decision.reason}")
                logger.info(f"   Estimated duration: {decision.estimated_duration_minutes} minutes")
                
                # Trigger training
                result = self.trigger_training()
                
                if result.get('success'):
                    logger.info("✅ Autonomous training cycle completed successfully")
                else:
                    logger.error(f"❌ Autonomous training cycle failed: {result.get('error')}")
            else:
                logger.debug(f"⏸️  No training needed: {decision.reason}")
                
        except Exception as e:
            logger.error(f"Error in autonomous cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def start(self):
        """Start the autonomous pipeline"""
        if self.is_running:
            logger.warning("Autonomous pipeline already running")
            return
        
        self.is_running = True
        
        def run_pipeline():
            logger.info("="*60)
            logger.info("🤖 AUTONOMOUS ML PIPELINE STARTED")
            logger.info("="*60)
            logger.info(f"Check interval: {self.check_interval_seconds} seconds")
            logger.info(f"Min data points: {self.min_data_points}")
            logger.info(f"Retrain interval: {self.retrain_interval_hours} hours")
            
            while self.is_running:
                try:
                    self.run_autonomous_cycle()
                except Exception as e:
                    logger.error(f"Error in pipeline cycle: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval_seconds)
        
        self.thread = threading.Thread(target=run_pipeline, daemon=True, name="AutonomousPipeline")
        self.thread.start()
        
        logger.info("✅ Autonomous pipeline started")
    
    def stop(self):
        """Stop the autonomous pipeline"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Autonomous pipeline stopped")
    
    def get_status(self) -> Dict:
        """Get current pipeline status"""
        data_quality = self.check_data_quality()
        decision = self.should_train(data_quality)
        
        return {
            'running': self.is_running,
            'training_in_progress': self.training_in_progress,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'data_quality': {
                'total_records': data_quality.total_records,
                'days_of_data': data_quality.days_of_data,
                'sufficient_for_training': data_quality.sufficient_for_training,
                'issues': data_quality.issues
            },
            'training_decision': {
                'should_train': decision.should_train,
                'reason': decision.reason,
                'priority': decision.priority
            },
            'history_count': len(self.pipeline_history)
        }


# Global instance
_autonomous_pipeline = None

def get_autonomous_pipeline() -> AutonomousPipeline:
    """Get or create autonomous pipeline instance"""
    global _autonomous_pipeline
    if _autonomous_pipeline is None:
        _autonomous_pipeline = AutonomousPipeline()
    return _autonomous_pipeline
