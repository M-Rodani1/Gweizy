"""
Model Retraining API Routes

Endpoints for managing automated model retraining:
- Check if retraining is needed
- Trigger manual retraining
- View retraining history
- Rollback to previous models
- Track training progress in real-time
"""

from flask import Blueprint, jsonify, request
from utils.model_retrainer import ModelRetrainer
from utils.logger import logger
from api.middleware import require_admin_auth
from datetime import datetime
import time
import os
import threading

retraining_bp = Blueprint('retraining', __name__)
retrainer = ModelRetrainer()

# Training progress tracking (in-memory, thread-safe)
_training_progress = {
    'is_training': False,
    'current_step': 0,
    'total_steps': 3,
    'step_name': None,
    'step_status': None,  # 'running', 'completed', 'failed', 'skipped'
    'steps': [
        {'name': 'RandomForest Models', 'status': 'pending', 'message': None},
        {'name': 'Spike Detectors', 'status': 'pending', 'message': None},
        {'name': 'DQN Agent', 'status': 'pending', 'message': None}
    ],
    'started_at': None,
    'completed_at': None,
    'error': None
}
_progress_lock = threading.Lock()


def _update_progress(step: int = None, status: str = None, message: str = None,
                     is_training: bool = None, error: str = None, completed: bool = False):
    """Thread-safe progress update"""
    with _progress_lock:
        if is_training is not None:
            _training_progress['is_training'] = is_training
            if is_training:
                _training_progress['started_at'] = datetime.now().isoformat()
                _training_progress['completed_at'] = None
                _training_progress['error'] = None
                # Reset all steps
                for s in _training_progress['steps']:
                    s['status'] = 'pending'
                    s['message'] = None

        if step is not None:
            _training_progress['current_step'] = step
            _training_progress['step_name'] = _training_progress['steps'][step]['name'] if step < 3 else None

        if status is not None and step is not None and step < 3:
            _training_progress['steps'][step]['status'] = status
            _training_progress['step_status'] = status

        if message is not None and step is not None and step < 3:
            _training_progress['steps'][step]['message'] = message

        if error is not None:
            _training_progress['error'] = error

        if completed:
            _training_progress['is_training'] = False
            _training_progress['completed_at'] = datetime.now().isoformat()


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
@require_admin_auth
def trigger_retraining():
    """
    Manually trigger model retraining

    Body:
        {
            "model_type": "lstm" | "prophet" | "ensemble" | "all",
            "force": true | false
        }

    Requires admin authentication (X-Admin-API-Key header).

    Returns:
        Retraining results
    """
    try:
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
@require_admin_auth
def rollback_models():
    """
    Rollback to a previous model backup

    Body:
        {
            "backup_path": "/path/to/backup"
        }

    Requires admin authentication (X-Admin-API-Key header).

    Returns:
        Rollback status
    """
    try:
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

            # Recommended: At least 20 days of data for quality predictions
            recommended_days = 20
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


@retraining_bp.route('/retraining/migrate-db', methods=['POST'])
def run_database_migration():
    """
    Run database migration to add gas_used, gas_limit, and utilization columns.
    
    This endpoint runs the migration script to add the new columns to the gas_prices table.
    Safe to run multiple times - it checks if columns already exist.
    """
    try:
        from scripts.migrate_add_utilization_fields import migrate_database
        
        logger.info("Running database migration: Add utilization fields...")
        success = migrate_database()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Database migration completed successfully',
                'columns_added': ['gas_used', 'gas_limit', 'utilization']
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Database migration failed. Check logs for details.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error running database migration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to run database migration'
        }), 500


@retraining_bp.route('/retraining/simple', methods=['POST'])
def trigger_simple_retraining():
    """
    Trigger model retraining. 
    NOTE: All model training is now done via Google Colab notebook.
    See: notebooks/train_models_colab.ipynb

    This runs in a background thread to avoid HTTP timeouts.
    Training time varies by dataset size:
    - 10k records: ~3-5 minutes
    - 50k records: ~10-15 minutes  
    - 100k records: ~20-30 minutes
    
    The endpoint returns immediately. Use /retraining/training-progress to monitor.

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

        # All training is now done via Google Colab notebook
        logger.warning("Automated training is disabled. Please use Google Colab notebook for training.")
        logger.info("Training notebook: notebooks/train_models_colab.ipynb")
        
        _update_progress(is_training=False)
        
        return jsonify({
            'success': False,
            'message': 'Automated training is disabled. All model training must be done via Google Colab notebook.',
            'notebook_path': 'notebooks/train_models_colab.ipynb',
            'instructions': [
                '1. Upload your gas_data.db to Google Colab',
                '2. Run all cells in notebooks/train_models_colab.ipynb',
                '3. Download the trained models zip',
                '4. Copy models to backend/models/saved_models/',
                '5. Commit and push to deploy'
            ]
        }), 400

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


@retraining_bp.route('/retraining/training-progress', methods=['GET'])
def get_training_progress():
    """
    Get current training progress.

    Returns detailed progress information including:
    - is_training: Whether training is currently running
    - current_step: Current step index (0-2)
    - steps: Array of step statuses with names and messages
    - started_at: When training started
    - completed_at: When training completed (if done)
    - error: Any error message
    """
    with _progress_lock:
        return jsonify({
            'is_training': _training_progress['is_training'],
            'current_step': _training_progress['current_step'],
            'total_steps': _training_progress['total_steps'],
            'step_name': _training_progress['step_name'],
            'step_status': _training_progress['step_status'],
            'steps': _training_progress['steps'],
            'started_at': _training_progress['started_at'],
            'completed_at': _training_progress['completed_at'],
            'error': _training_progress['error']
        }), 200


@retraining_bp.route('/retraining/models-status', methods=['GET'])
def get_models_status():
    """
    Check which ML models are currently available.

    Returns:
        Status of all model types:
        - prediction_models: 1h, 4h, 24h prediction models
        - spike_detectors: 1h, 4h, 24h spike classification models
        - dqn_agent: Reinforcement learning agent for transaction timing
    """
    try:
        from config import Config

        models_dir = Config.MODELS_DIR
        status = {
            'prediction_models': {},
            'spike_detectors': {},
            'dqn_agent': {
                'available': False,
                'path': None
            },
            'data_status': {},
            'overall_ready': False,
            'missing_models': []
        }

        # Check prediction models (1h, 4h, 24h)
        for horizon in ['1h', '4h', '24h']:
            model_path = os.path.join(models_dir, f'model_{horizon}.pkl')
            fallback_path = f'models/saved_models/model_{horizon}.pkl'
            backend_fallback_path = f'backend/models/saved_models/model_{horizon}.pkl'

            if os.path.exists(model_path):
                status['prediction_models'][horizon] = {
                    'available': True,
                    'path': model_path
                }
            elif os.path.exists(fallback_path):
                status['prediction_models'][horizon] = {
                    'available': True,
                    'path': fallback_path
                }
            elif os.path.exists(backend_fallback_path):
                status['prediction_models'][horizon] = {
                    'available': True,
                    'path': backend_fallback_path
                }
            else:
                status['prediction_models'][horizon] = {
                    'available': False,
                    'path': None
                }
                status['missing_models'].append(f'prediction_model_{horizon}')

        # Check spike detectors (1h, 4h, 24h)
        for horizon in ['1h', '4h', '24h']:
            spike_path = os.path.join(models_dir, f'spike_detector_{horizon}.pkl')
            fallback_path = f'models/saved_models/spike_detector_{horizon}.pkl'
            backend_fallback_path = f'backend/models/saved_models/spike_detector_{horizon}.pkl'

            if os.path.exists(spike_path):
                status['spike_detectors'][horizon] = {
                    'available': True,
                    'path': spike_path
                }
            elif os.path.exists(fallback_path):
                status['spike_detectors'][horizon] = {
                    'available': True,
                    'path': fallback_path
                }
            elif os.path.exists(backend_fallback_path):
                status['spike_detectors'][horizon] = {
                    'available': True,
                    'path': backend_fallback_path
                }
            else:
                status['spike_detectors'][horizon] = {
                    'available': False,
                    'path': None
                }
                status['missing_models'].append(f'spike_detector_{horizon}')

        # Check DQN agent (supports both .pkl and .pt formats)
        dqn_paths = [
            os.path.join(models_dir, 'rl_agents', 'dqn_final.pkl'),
            os.path.join(models_dir, 'rl_agents', 'dqn_final.pt'),
            os.path.join(models_dir, 'rl_agents', 'chain_8453', 'dqn_final.pkl'),
            os.path.join(models_dir, 'rl_agents', 'chain_8453', 'dqn_final.pt'),
            'models/rl_agents/dqn_final.pkl',
            'models/rl_agents/dqn_final.pt',
            'models/rl_agents/chain_8453/dqn_final.pkl',
            'models/rl_agents/chain_8453/dqn_final.pt'
        ]

        for dqn_path in dqn_paths:
            if os.path.exists(dqn_path):
                status['dqn_agent'] = {
                    'available': True,
                    'path': dqn_path
                }
                break

        if not status['dqn_agent']['available']:
            status['missing_models'].append('dqn_agent')

        # Check data availability
        try:
            from data.database import GasPrice, DatabaseManager
            db = DatabaseManager()
            session = db._get_session()

            try:
                total_records = session.query(GasPrice).count()
                status['data_status'] = {
                    'total_records': total_records,
                    'sufficient_for_training': total_records >= 50,
                    'sufficient_for_dqn': total_records >= 500
                }
            finally:
                session.close()
        except Exception as e:
            status['data_status'] = {
                'error': str(e)
            }

        # Determine overall readiness
        prediction_ready = all(m['available'] for m in status['prediction_models'].values())
        spike_ready = all(m['available'] for m in status['spike_detectors'].values())
        dqn_ready = status['dqn_agent']['available']

        status['overall_ready'] = prediction_ready and spike_ready and dqn_ready
        status['summary'] = {
            'prediction_models_ready': prediction_ready,
            'spike_detectors_ready': spike_ready,
            'dqn_agent_ready': dqn_ready,
            'action_needed': 'POST /api/retraining/simple to train all missing models' if status['missing_models'] else None
        }

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error checking models status: {e}")
        return jsonify({'error': str(e)}), 500
