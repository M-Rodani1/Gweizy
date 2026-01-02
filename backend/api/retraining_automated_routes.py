"""
Automated retraining API routes.
Endpoints for managing automated model retraining.
"""
from flask import Blueprint, request, jsonify
from services.automated_retraining import get_retraining_service
from utils.logger import logger

automated_retraining_bp = Blueprint('automated_retraining', __name__)


@automated_retraining_bp.route('/retraining/automated/status', methods=['GET'])
def get_automated_status():
    """Get status of automated retraining scheduler."""
    try:
        service = get_retraining_service()
        
        return jsonify({
            'success': True,
            'is_running': service.is_running,
            'retrain_interval_hours': service.retrain_interval_hours,
            'accuracy_drop_threshold': service.accuracy_drop_threshold,
            'min_accuracy_mae': service.min_accuracy_mae,
            'history_count': len(service.retraining_history)
        })
    
    except Exception as e:
        logger.error(f"Error getting automated retraining status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@automated_retraining_bp.route('/retraining/automated/check/<int:chain_id>', methods=['POST'])
def check_and_retrain_chain(chain_id: int):
    """
    Check accuracy and retrain a specific chain if needed.
    
    POST body (optional):
        {
            "force": false  # Force retraining even if accuracy is good
        }
    """
    try:
        data = request.json or {}
        force = data.get('force', False)
        
        service = get_retraining_service()
        result = service.check_accuracy_and_retrain(chain_id, force=force)
        
        return jsonify({
            'success': True,
            **result
        })
    
    except Exception as e:
        logger.error(f"Error checking/retraining chain {chain_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@automated_retraining_bp.route('/retraining/automated/retrain-all', methods=['POST'])
def retrain_all_chains():
    """
    Retrain models for all chains.
    
    POST body (optional):
        {
            "chains": [8453, 1, 42161],  # Specific chains, or omit for all
            "train_ml": true,
            "train_dqn": false
        }
    """
    try:
        data = request.json or {}
        chains = data.get('chains')
        train_ml = data.get('train_ml', True)
        train_dqn = data.get('train_dqn', False)
        
        service = get_retraining_service()
        results = service.retrain_all_chains(
            chains=chains,
            train_ml=train_ml,
            train_dqn=train_dqn
        )
        
        return jsonify({
            'success': True,
            **results
        })
    
    except Exception as e:
        logger.error(f"Error retraining all chains: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@automated_retraining_bp.route('/retraining/automated/history', methods=['GET'])
def get_retraining_history():
    """Get retraining history."""
    try:
        service = get_retraining_service()
        
        limit = request.args.get('limit', 50, type=int)
        history = service.retraining_history[-limit:]
        
        return jsonify({
            'success': True,
            'history': history,
            'total_count': len(service.retraining_history)
        })
    
    except Exception as e:
        logger.error(f"Error getting retraining history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@automated_retraining_bp.route('/retraining/automated/start', methods=['POST'])
def start_scheduler():
    """Start the automated retraining scheduler."""
    try:
        service = get_retraining_service()
        service.start_scheduler()
        
        return jsonify({
            'success': True,
            'message': 'Automated retraining scheduler started'
        })
    
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@automated_retraining_bp.route('/retraining/automated/stop', methods=['POST'])
def stop_scheduler():
    """Stop the automated retraining scheduler."""
    try:
        service = get_retraining_service()
        service.stop_scheduler()
        
        return jsonify({
            'success': True,
            'message': 'Automated retraining scheduler stopped'
        })
    
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

