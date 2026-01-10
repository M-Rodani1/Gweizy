"""
Model Retraining API Routes

Endpoints for managing automated model retraining:
- Check if retraining is needed
- Trigger manual retraining
- View retraining history
- Rollback to previous models
"""

from flask import Blueprint, jsonify, request
from utils.model_retrainer import ModelRetrainer
from utils.logger import logger
from datetime import datetime
import os

retraining_bp = Blueprint('retraining', __name__)
retrainer = ModelRetrainer()


@retraining_bp.route('/retraining/status', methods=['GET'])
def get_retraining_status():
    """
    Check if model retraining is needed

    Returns:
        Status indicating whether retraining should be triggered
    """
    try:
        should_retrain, reason = retrainer.should_retrain()

        # Get last training info
        metadata_path = f"{retrainer.models_dir}/training_metadata.json"
        last_training = None

        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                last_training = {
                    'timestamp': metadata.get('training_timestamp'),
                    'reason': metadata.get('reason'),
                    'models_trained': metadata.get('models_trained', []),
                    'validation_passed': metadata.get('validation_passed', False)
                }

        return jsonify({
            'should_retrain': should_retrain,
            'reason': reason,
            'last_training': last_training,
            'checked_at': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error checking retraining status: {e}")
        return jsonify({'error': str(e)}), 500


@retraining_bp.route('/retraining/trigger', methods=['POST'])
def trigger_retraining():
    """
    Manually trigger model retraining

    Body:
        {
            "model_type": "lstm" | "prophet" | "ensemble" | "all",
            "force": true | false
        }

    Returns:
        Retraining results

    Note: This endpoint should be protected with authentication in production
    """
    try:
        # TODO: Add authentication check
        # if not is_admin(request):
        #     return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json() or {}
        model_type = data.get('model_type', 'all')
        force = data.get('force', False)

        # Validate model type
        valid_types = ['lstm', 'prophet', 'ensemble', 'all']
        if model_type not in valid_types:
            return jsonify({
                'error': f'Invalid model_type. Must be one of: {valid_types}'
            }), 400

        logger.info(f"Manual retraining triggered: model_type={model_type}, force={force}")

        # Run retraining in background (for production, use Celery or background worker)
        # For now, run synchronously
        results = retrainer.retrain_models(model_type=model_type, force=force)

        status_code = 200 if results.get('validation_passed', False) else 500

        return jsonify(results), status_code

    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        return jsonify({'error': str(e)}), 500


@retraining_bp.route('/retraining/history', methods=['GET'])
def get_retraining_history():
    """
    Get history of model retraining events

    Returns:
        List of past retraining events
    """
    try:
        # Get all backups (each represents a retraining event)
        backup_dir = retrainer.backup_dir
        backups = []

        if os.path.exists(backup_dir):
            for backup_folder in os.listdir(backup_dir):
                if backup_folder.startswith('backup_'):
                    backup_path = f"{backup_dir}/{backup_folder}"

                    # Extract timestamp from folder name
                    timestamp_str = backup_folder.replace('backup_', '')

                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                        # Get backup info
                        backup_info = {
                            'timestamp': timestamp.isoformat(),
                            'backup_path': backup_path,
                            'files': os.listdir(backup_path) if os.path.isdir(backup_path) else []
                        }

                        backups.append(backup_info)

                    except ValueError:
                        continue

        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'total_backups': len(backups),
            'backups': backups[:20]  # Return last 20
        }), 200

    except Exception as e:
        logger.error(f"Error getting retraining history: {e}")
        return jsonify({'error': str(e)}), 500


@retraining_bp.route('/retraining/rollback', methods=['POST'])
def rollback_models():
    """
    Rollback to a previous model backup

    Body:
        {
            "backup_path": "/path/to/backup"
        }

    Returns:
        Rollback status

    Note: This endpoint should be protected with authentication in production
    """
    try:
        # TODO: Add authentication check
        # if not is_admin(request):
        #     return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()

        if not data or 'backup_path' not in data:
            return jsonify({'error': 'backup_path is required'}), 400

        backup_path = data['backup_path']

        # Verify backup exists
        if not os.path.exists(backup_path):
            return jsonify({'error': 'Backup path does not exist'}), 404

        # Verify it's in the backups directory (security check)
        if not backup_path.startswith(retrainer.backup_dir):
            return jsonify({'error': 'Invalid backup path'}), 400

        logger.info(f"Rolling back models to {backup_path}")

        # Restore models
        retrainer.restore_models(backup_path)

        return jsonify({
            'success': True,
            'message': f'Models rolled back to {backup_path}',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error rolling back models: {e}")
        return jsonify({'error': str(e)}), 500


@retraining_bp.route('/retraining/check-data', methods=['GET'])
def check_training_data():
    """
    Check if sufficient data is available for training

    Returns:
        Data availability status
    """
    try:
        from data.database import GasPrice, DatabaseManager

        db = DatabaseManager()
        session = db._get_session()

        try:
            total_records = session.query(GasPrice).count()

            # Get date range
            oldest = session.query(GasPrice).order_by(GasPrice.timestamp.asc()).first()
            newest = session.query(GasPrice).order_by(GasPrice.timestamp.desc()).first()

            if oldest and newest:
                date_range_days = (newest.timestamp - oldest.timestamp).days
            else:
                date_range_days = 0

            # Recommended: At least 30 days of data for quality predictions
            recommended_days = 30
            sufficient = date_range_days >= recommended_days

            return jsonify({
                'total_records': total_records,
                'date_range_days': date_range_days,
                'oldest_timestamp': oldest.timestamp.isoformat() if oldest else None,
                'newest_timestamp': newest.timestamp.isoformat() if newest else None,
                'recommended_days': recommended_days,
                'sufficient_data': sufficient,
                'readiness': 'ready' if sufficient else 'collecting',
                'progress_percent': min(100, (date_range_days / recommended_days) * 100)
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error checking training data: {e}")
        return jsonify({'error': str(e)}), 500


@retraining_bp.route('/retraining/simple', methods=['POST'])
def trigger_simple_retraining():
    """
    Trigger simple model retraining using the retrain_models_simple.py script

    This runs in a background thread to avoid HTTP timeouts.
    Training typically takes 3-10 minutes.

    Returns:
        Immediate response with training status
    """
    try:
        import subprocess
        import sys
        import threading
        import os

        # Check if training is already running
        if hasattr(trigger_simple_retraining, '_training_in_progress'):
            if trigger_simple_retraining._training_in_progress:
                return jsonify({
                    'status': 'in_progress',
                    'message': 'Training already in progress. Please wait for it to complete.'
                }), 200

        logger.info("Starting simple model retraining in background thread...")

        # Get absolute path to the script
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(current_dir, "scripts", "retrain_models_simple.py")

        logger.info(f"Script path: {script_path}")
        logger.info(f"Script exists: {os.path.exists(script_path)}")

        # Mark training as in progress
        trigger_simple_retraining._training_in_progress = True

        def run_training():
            """Run training in background thread"""
            try:
                logger.info("Background training thread started")

                # Step 1: Train main RandomForest models
                logger.info("Step 1/2: Training RandomForest percentage change models...")
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=current_dir
                )

                if result.returncode != 0:
                    logger.error(f"Main model training failed: {result.stderr}")
                    logger.error(f"Output: {result.stdout[:500]}...")
                    return

                logger.info("Main model training completed successfully")

                # Step 2: Train spike detector models
                logger.info("Step 2/2: Training spike detector classifiers...")
                spike_script_path = os.path.join(current_dir, "scripts", "train_spike_detectors.py")

                if os.path.exists(spike_script_path):
                    spike_result = subprocess.run(
                        [sys.executable, spike_script_path],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout for spike detectors
                        cwd=current_dir
                    )

                    if spike_result.returncode == 0:
                        logger.info("Spike detector training completed successfully")
                    else:
                        logger.warning(f"Spike detector training failed: {spike_result.stderr}")
                        logger.warning("Continuing without spike detectors...")
                else:
                    logger.warning(f"Spike detector script not found at {spike_script_path}")

                if result.returncode == 0:
                    # Auto-reload models after successful training
                    try:
                        logger.info("🔄 Auto-reloading models after training...")
                        from api.routes import reload_models
                        reload_result = reload_models()
                        if reload_result['success']:
                            logger.info(f"✅ Models auto-reloaded: {reload_result['models_loaded']} models loaded")
                        else:
                            logger.warning(f"⚠️  Auto-reload had issues: {reload_result.get('error')}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not auto-reload models: {e}")
                        logger.info("💡 You can manually reload models via POST /api/models/reload")
                    logger.info("Retraining completed successfully")
                    logger.info(f"Output: {result.stdout[:500]}...")  # Log first 500 chars
                else:
                    logger.error(f"Retraining failed: {result.stderr}")
                    logger.error(f"Output: {result.stdout[:500]}...")
            except subprocess.TimeoutExpired:
                logger.error("Retraining timed out after 10 minutes")
            except Exception as e:
                logger.error(f"Error during training: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                # Mark training as complete
                trigger_simple_retraining._training_in_progress = False
                logger.info("Training thread completed")

        # Start training in background thread
        training_thread = threading.Thread(target=run_training, name="ModelTraining", daemon=True)
        training_thread.start()

        # Return immediately
        return jsonify({
            'status': 'started',
            'message': 'Model training started in background. This may take 3-10 minutes.',
            'timestamp': datetime.now().isoformat(),
            'note': 'Check logs or retraining status endpoint for progress'
        }), 200

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        import traceback
        trigger_simple_retraining._training_in_progress = False
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Initialize training status
trigger_simple_retraining._training_in_progress = False
