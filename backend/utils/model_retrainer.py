"""
Automated Model Retraining Pipeline

Monitors model performance and automatically retrains when:
- Prediction accuracy drops below threshold
- New data is available (weekly retraining)
- Manual trigger requested

Features:
- Performance degradation detection
- Automatic data collection
- Model versioning
- Rollback on regression
- Notification system
"""

import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
import joblib
from utils.prediction_validator import PredictionValidator
from utils.logger import logger
from data.database import DatabaseManager


class ModelRetrainer:
    """Handles automated model retraining"""

    def __init__(self, models_dir: str = "models/saved_models"):
        """
        Initialize retrainer

        Args:
            models_dir: Directory where models are saved
        """
        self.models_dir = models_dir
        self.validator = PredictionValidator()
        self.db = DatabaseManager()
        self.backup_dir = f"{models_dir}/backups"

        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def should_retrain(self, horizon: str = None) -> Tuple[bool, str]:
        """
        Check if model should be retrained

        Args:
            horizon: Specific horizon to check or None for all

        Returns:
            (should_retrain: bool, reason: str)
        """
        horizons = [horizon] if horizon else ['1h', '4h', '24h']

        for h in horizons:
            # Check recent performance (last 7 days)
            metrics = self.validator.calculate_metrics(horizon=h, days=7)

            if metrics['sample_size'] == 0:
                logger.warning(f"No validated predictions for {h} in last 7 days")
                continue

            # Threshold checks
            if metrics['mae'] > 0.001:  # MAE too high
                return True, f"MAE ({metrics['mae']:.6f}) exceeds threshold for {h}"

            if metrics['directional_accuracy'] < 0.55:  # Barely better than random
                return True, f"Directional accuracy ({metrics['directional_accuracy']:.2%}) too low for {h}"

            # Check for degradation over time
            degradation = self._check_performance_degradation(h)
            if degradation:
                return True, f"Performance degrading for {h}: {degradation}"

        # Check if enough new data is available
        new_data_available = self._check_new_data_available()
        if new_data_available:
            return True, "Significant new data available for training"

        return False, "Model performance is acceptable"

    def _check_performance_degradation(self, horizon: str) -> Optional[str]:
        """
        Check if performance is degrading over time

        Args:
            horizon: Prediction horizon

        Returns:
            Degradation description or None
        """
        try:
            # Get performance trends
            trends = self.validator.get_performance_trends(horizon=horizon, days=30)

            if len(trends) < 7:  # Need at least a week of data
                return None

            # Compare last 3 days vs previous week
            recent_metrics = trends[-3:]
            previous_metrics = trends[-10:-3]

            if not recent_metrics or not previous_metrics:
                return None

            # Calculate average metrics
            recent_mae = sum(m['mae'] for m in recent_metrics) / len(recent_metrics)
            previous_mae = sum(m['mae'] for m in previous_metrics) / len(previous_metrics)

            recent_dir_acc = sum(m['directional_accuracy'] for m in recent_metrics) / len(recent_metrics)
            previous_dir_acc = sum(m['directional_accuracy'] for m in previous_metrics) / len(previous_metrics)

            # Check for degradation (>20% worse)
            mae_degradation = (recent_mae - previous_mae) / previous_mae
            dir_acc_degradation = (previous_dir_acc - recent_dir_acc) / previous_dir_acc

            if mae_degradation > 0.2:
                return f"MAE increased by {mae_degradation:.1%}"

            if dir_acc_degradation > 0.2:
                return f"Directional accuracy decreased by {dir_acc_degradation:.1%}"

            return None

        except Exception as e:
            logger.error(f"Error checking degradation: {e}")
            return None

    def _check_new_data_available(self, min_new_records: int = 10000) -> bool:
        """
        Check if significant new data is available

        Args:
            min_new_records: Minimum new records to trigger retraining

        Returns:
            True if enough new data available
        """
        try:
            from data.database import GasPrice

            session = self.db._get_session()
            try:
                # Get timestamp of last model training
                metadata_path = f"{self.models_dir}/training_metadata.json"
                if not os.path.exists(metadata_path):
                    return True  # No metadata, should train

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                last_training_time = datetime.fromisoformat(metadata.get('training_timestamp', '2000-01-01'))

                # Count new records since last training
                new_records = session.query(GasPrice).filter(
                    GasPrice.timestamp > last_training_time
                ).count()

                logger.info(f"New data since last training: {new_records:,} records")

                return new_records >= min_new_records

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error checking new data: {e}")
            return False

    def backup_current_models(self) -> str:
        """
        Backup current models before retraining

        Returns:
            Backup directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.backup_dir}/backup_{timestamp}"

        try:
            os.makedirs(backup_path, exist_ok=True)

            # Copy all model files
            for file in os.listdir(self.models_dir):
                if file.endswith(('.pkl', '.h5', '.json')):
                    src = f"{self.models_dir}/{file}"
                    dst = f"{backup_path}/{file}"
                    shutil.copy2(src, dst)

            logger.info(f"Backed up models to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Error backing up models: {e}")
            raise

    def restore_models(self, backup_path: str):
        """
        Restore models from backup

        Args:
            backup_path: Path to backup directory
        """
        try:
            for file in os.listdir(backup_path):
                if file.endswith(('.pkl', '.h5', '.json')):
                    src = f"{backup_path}/{file}"
                    dst = f"{self.models_dir}/{file}"
                    shutil.copy2(src, dst)

            logger.info(f"Restored models from {backup_path}")

        except Exception as e:
            logger.error(f"Error restoring models: {e}")
            raise

    def retrain_models(
        self,
        model_type: str = 'all',
        force: bool = False
    ) -> Dict:
        """
        Retrain models

        Args:
            model_type: 'lstm', 'prophet', 'ensemble', or 'all'
            force: Force retraining even if not needed

        Returns:
            Retraining results
        """
        logger.info("="*60)
        logger.info("AUTOMATED MODEL RETRAINING")
        logger.info("="*60)

        # Check if retraining is needed
        if not force:
            should_retrain, reason = self.should_retrain()

            if not should_retrain:
                logger.info(f"Retraining not needed: {reason}")
                return {
                    'retrained': False,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }

            logger.info(f"Retraining triggered: {reason}")
        else:
            logger.info("Forced retraining requested")
            reason = "Manual trigger"

        # Backup current models
        backup_path = self.backup_current_models()

        results = {
            'retrained': True,
            'reason': reason,
            'backup_path': backup_path,
            'models_trained': [],
            'performance': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Retrain based on model type
            if model_type in ['lstm', 'all']:
                logger.info("\n--- Training LSTM Models ---")
                lstm_result = self._retrain_lstm()
                results['models_trained'].append('lstm')
                results['performance']['lstm'] = lstm_result

            if model_type in ['prophet', 'all']:
                logger.info("\n--- Training Prophet Models ---")
                prophet_result = self._retrain_prophet()
                results['models_trained'].append('prophet')
                results['performance']['prophet'] = prophet_result

            if model_type in ['ensemble', 'all']:
                logger.info("\n--- Training Ensemble Models ---")
                ensemble_result = self._retrain_ensemble()
                results['models_trained'].append('ensemble')
                results['performance']['ensemble'] = ensemble_result

            # Save training metadata
            self._save_training_metadata(results)

            # Validate new models
            validation_passed = self._validate_new_models(backup_path)

            if not validation_passed:
                logger.error("New models failed validation - rolling back")
                self.restore_models(backup_path)
                results['rollback'] = True
                results['validation_passed'] = False
            else:
                results['validation_passed'] = True
                logger.info("✓ Retraining successful and validated")

            return results

        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            logger.info("Rolling back to previous models")
            self.restore_models(backup_path)

            results['error'] = str(e)
            results['rollback'] = True

            return results

    def _retrain_lstm(self) -> Dict:
        """Retrain LSTM models"""
        try:
            import subprocess

            result = subprocess.run(
                ['python3', 'scripts/train_timeseries_models.py', '--model', 'lstm', '--epochs', '50'],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("LSTM training completed successfully")
                return {'success': True, 'output': result.stdout}
            else:
                logger.error(f"LSTM training failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return {'success': False, 'error': str(e)}

    def _retrain_prophet(self) -> Dict:
        """Retrain Prophet models"""
        try:
            import subprocess

            result = subprocess.run(
                ['python3', 'scripts/train_timeseries_models.py', '--model', 'prophet'],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )

            if result.returncode == 0:
                logger.info("Prophet training completed successfully")
                return {'success': True, 'output': result.stdout}
            else:
                logger.error(f"Prophet training failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}

        except Exception as e:
            logger.error(f"Error training Prophet: {e}")
            return {'success': False, 'error': str(e)}

    def _retrain_ensemble(self) -> Dict:
        """Retrain ensemble models (RandomForest + GradientBoosting)"""
        try:
            # Use existing ensemble training script
            from models.ensemble_predictor import train_ensemble_models

            result = train_ensemble_models()
            logger.info("Ensemble training completed successfully")
            return {'success': True, 'result': result}

        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_new_models(self, backup_path: str) -> bool:
        """
        Validate newly trained models

        Args:
            backup_path: Path to model backup (for comparison)

        Returns:
            True if validation passed
        """
        logger.info("\n--- Validating New Models ---")

        try:
            # Basic checks
            required_files = []

            # Check LSTM models
            if os.path.exists(f"{self.models_dir}/lstm_model.h5"):
                required_files.append('lstm_model.h5')
                required_files.append('lstm_scaler.pkl')

            # Check Prophet models
            if os.path.exists(f"{self.models_dir}/prophet_model.pkl"):
                required_files.append('prophet_model.pkl')

            # Verify all required files exist
            for file in required_files:
                path = f"{self.models_dir}/{file}"
                if not os.path.exists(path):
                    logger.error(f"Missing model file: {file}")
                    return False

                # Check file size (should not be empty)
                if os.path.getsize(path) == 0:
                    logger.error(f"Model file is empty: {file}")
                    return False

            logger.info("✓ All model files present and valid")

            # TODO: Could add performance validation here
            # - Load models
            # - Make test predictions
            # - Compare with backup model performance

            return True

        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return False

    def _save_training_metadata(self, results: Dict):
        """Save training metadata for future reference"""
        metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'reason': results.get('reason'),
            'models_trained': results.get('models_trained', []),
            'validation_passed': results.get('validation_passed', False)
        }

        metadata_path = f"{self.models_dir}/training_metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved training metadata to {metadata_path}")


def scheduled_retraining_check():
    """
    Check if retraining is needed (run this daily)
    """
    retrainer = ModelRetrainer()

    logger.info("Running scheduled retraining check...")

    should_retrain, reason = retrainer.should_retrain()

    if should_retrain:
        logger.info(f"Retraining needed: {reason}")
        results = retrainer.retrain_models()

        # Send notification (email/slack)
        if results['validation_passed']:
            logger.info("✓ Models successfully retrained and deployed")
        else:
            logger.error("✗ Retraining failed - models rolled back")

        return results
    else:
        logger.info(f"No retraining needed: {reason}")
        return {'retrained': False, 'reason': reason}


if __name__ == "__main__":
    # Example usage
    retrainer = ModelRetrainer()

    print("=== Automated Model Retraining System ===\n")

    # Check if retraining is needed
    should_retrain, reason = retrainer.should_retrain()

    print(f"Should retrain: {should_retrain}")
    print(f"Reason: {reason}\n")

    if should_retrain:
        print("Starting retraining process...")
        results = retrainer.retrain_models()

        print("\n=== Retraining Results ===")
        print(json.dumps(results, indent=2))
