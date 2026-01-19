"""
Resilient Training API Routes

Endpoints for managing resilient model training with checkpointing and graceful shutdown.
"""

from flask import Blueprint, jsonify, request
from services.resilient_training import get_resilient_training_service
from utils.logger import logger

resilient_training_bp = Blueprint('resilient_training', __name__)

@resilient_training_bp.route('/api/training/start', methods=['POST'])
def start_training():
    """
    Start model training with resilient setup
    
    POST body (optional):
        {
            "background": true,  # Run in background (default: true)
            "force": false       # Force start even if already training (default: false)
        }
    """
    try:
        data = request.json or {}
        background = data.get('background', True)
        force = data.get('force', False)
        
        service = get_resilient_training_service()
        result = service.start_training(background=background, force=force)
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@resilient_training_bp.route('/api/training/stop', methods=['POST'])
def stop_training():
    """
    Stop running training gracefully
    
    POST body (optional):
        {
            "graceful": true  # Wait for current step to complete (default: true)
        }
    """
    try:
        data = request.json or {}
        graceful = data.get('graceful', True)
        
        service = get_resilient_training_service()
        result = service.stop_training(graceful=graceful)
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
    
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@resilient_training_bp.route('/api/training/status', methods=['GET'])
def get_training_status():
    """
    Get current training status including checkpoint info
    """
    try:
        service = get_resilient_training_service()
        status = service.get_training_status()
        
        return jsonify({
            'success': True,
            **status
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@resilient_training_bp.route('/api/training/resume', methods=['POST'])
def resume_training():
    """
    Resume training from checkpoint (if available)
    
    Note: Currently training scripts don't support resume, but this endpoint
    is provided for future implementation.
    """
    try:
        service = get_resilient_training_service()
        status = service.get_training_status()
        
        if not status.get('has_checkpoint'):
            return jsonify({
                'success': False,
                'error': 'No checkpoint available to resume from'
            }), 400
        
        # For now, just start new training
        # TODO: Implement actual resume logic in training script
        result = service.start_training(background=True, force=False)
        
        status_code = 200 if result.get('success') else 400
        return jsonify({
            **result,
            'note': 'Checkpoint exists but resume not yet implemented. Starting fresh training.'
        }), status_code
    
    except Exception as e:
        logger.error(f"Error resuming training: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
