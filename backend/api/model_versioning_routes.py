"""
Model Versioning API Routes

Endpoints for managing model versions, rollback, and version history.
"""

from flask import Blueprint, jsonify, request
from models.model_registry import get_registry
from utils.logger import logger

versioning_bp = Blueprint('versioning', __name__)


@versioning_bp.route('/versions', methods=['GET'])
def list_versions():
    """List all model versions"""
    try:
        registry = get_registry()
        horizon = request.args.get('horizon')
        chain_id = int(request.args.get('chain_id', 8453))
        
        if horizon:
            versions = registry.get_versions(horizon, chain_id)
            return jsonify({
                'success': True,
                'horizon': horizon,
                'chain_id': chain_id,
                'versions': versions,
                'count': len(versions)
            })
        else:
            # Return summary for all models
            summary = registry.get_registry_summary()
            return jsonify({
                'success': True,
                'summary': summary
            })
    except Exception as e:
        logger.error(f"Error listing versions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@versioning_bp.route('/versions/active', methods=['GET'])
def get_active_version():
    """Get currently active version for a model"""
    try:
        registry = get_registry()
        horizon = request.args.get('horizon', '1h')
        chain_id = int(request.args.get('chain_id', 8453))
        
        active = registry.get_active_version(horizon, chain_id)
        
        if not active:
            return jsonify({
                'success': False,
                'message': f'No active version for {horizon} on chain {chain_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'active_version': active
        })
    except Exception as e:
        logger.error(f"Error getting active version: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@versioning_bp.route('/versions/activate', methods=['POST'])
def activate_version():
    """Activate a specific model version"""
    try:
        data = request.get_json()
        horizon = data.get('horizon')
        version = data.get('version')
        chain_id = data.get('chain_id', 8453)
        
        if not horizon or not version:
            return jsonify({
                'success': False,
                'error': 'horizon and version are required'
            }), 400
        
        registry = get_registry()
        registry.activate_version(horizon, version, chain_id)
        
        return jsonify({
            'success': True,
            'message': f'Activated {horizon} version {version}',
            'active_version': registry.get_active_version(horizon, chain_id)
        })
    except Exception as e:
        logger.error(f"Error activating version: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@versioning_bp.route('/versions/rollback', methods=['POST'])
def rollback_version():
    """Rollback to previous version"""
    try:
        data = request.get_json()
        horizon = data.get('horizon')
        chain_id = data.get('chain_id', 8453)
        steps = data.get('steps', 1)
        
        if not horizon:
            return jsonify({
                'success': False,
                'error': 'horizon is required'
            }), 400
        
        registry = get_registry()
        new_version = registry.rollback(horizon, chain_id, steps)
        
        return jsonify({
            'success': True,
            'message': f'Rolled back {horizon} {steps} step(s)',
            'new_version': new_version,
            'active_version': registry.get_active_version(horizon, chain_id)
        })
    except Exception as e:
        logger.error(f"Error rolling back: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@versioning_bp.route('/versions/summary', methods=['GET'])
def get_summary():
    """Get registry summary"""
    try:
        registry = get_registry()
        summary = registry.get_registry_summary()
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

