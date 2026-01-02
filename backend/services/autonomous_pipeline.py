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
                        except:
                            try:
                                from dateutil import parser
                                timestamps.append(parser.parse(ts))
                            except:
                                pass
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
                days_of_data=0.0,
                issues=[f"Error checking data: {str(e)}"]
            )
    
    def should_train(self, data_quality: DataQualityMetrics) -> TrainingDecision:
        """Determine if training should be triggered"""
        
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
        
        # Check if it's been too long since last training
        should_train_by_time = False
        if self.last_training_time is None:
            should_train_by_time = True
            reason = "No previous training recorded"
            priority = "high"
        else:
            hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
            if hours_since_training >= self.retrain_interval_hours:
                should_train_by_time = True
                reason = f"Last training was {hours_since_training:.1f} hours ago (threshold: {self.retrain_interval_hours}h)"
                priority = "medium"
        
        # Check if we have optimal data (high priority)
        if data_quality.total_records >= self.optimal_data_points and data_quality.days_of_data >= self.optimal_days_data:
            if should_train_by_time:
                priority = "high"
                reason = f"Optimal data available ({data_quality.total_records} records, {data_quality.days_of_data:.1f} days)"
        
        # Estimate training duration (based on data size)
        estimated_duration = 5  # Base 5 minutes
        if data_quality.total_records > 2000:
            estimated_duration = 15
        elif data_quality.total_records > 1000:
            estimated_duration = 10
        
        return TrainingDecision(
            should_train=should_train_by_time,
            reason=reason,
            priority=priority,
            estimated_duration_minutes=estimated_duration
        )
    
    def trigger_training(self) -> Dict:
        """Trigger model training"""
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
            
            # Run training script
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(current_dir, "scripts", "retrain_models_simple.py")
            
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Training script not found: {script_path}")
            
            logger.info(f"Running training script: {script_path}")
            
            # Run training in subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=self.max_training_duration_minutes * 60
            )
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            if result.returncode == 0:
                logger.info(f"✅ Training completed successfully in {duration:.1f} minutes")
                
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
                
                self.last_training_time = datetime.now()
                
                result_dict = {
                    'success': True,
                    'duration_minutes': duration,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation': evaluation
                }
                
                self.pipeline_history.append({
                    'action': 'training',
                    'result': result_dict,
                    'timestamp': datetime.now().isoformat()
                })
                
                return result_dict
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"❌ Training failed: {error_msg}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'duration_minutes': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
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

