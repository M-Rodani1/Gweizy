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

            # Recommended: At least 6 months of data
            recommended_days = 180
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

    This runs synchronously and may take 3-5 minutes. For async execution,
    consider using a background task queue.

    Returns:
        Training results or error
    """
    try:
        import subprocess
        import sys

        logger.info("Starting simple model retraining...")

        # Run the retraining script
        # On Railway, working directory is /app/backend, so use relative path
        script_path = "scripts/retrain_models_simple.py"

        # Execute the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info("Retraining completed successfully")
            return jsonify({
                'status': 'success',
                'message': 'Models retrained successfully',
                'output': result.stdout,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            logger.error(f"Retraining failed: {result.stderr}")
            return jsonify({
                'status': 'error',
                'message': 'Retraining failed',
                'error': result.stderr,
                'output': result.stdout
            }), 500

    except subprocess.TimeoutExpired:
        logger.error("Retraining timed out")
        return jsonify({
            'status': 'error',
            'message': 'Retraining timed out after 10 minutes'
        }), 500
    except Exception as e:
        logger.error(f"Error during simple retraining: {e}")
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500
