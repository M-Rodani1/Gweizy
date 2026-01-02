"""
API routes for the autonomous ML pipeline
"""
from flask import Blueprint, jsonify
from utils.logger import logger

from services.autonomous_pipeline import get_autonomous_pipeline

autonomous_pipeline_bp = Blueprint('autonomous_pipeline', __name__)


@autonomous_pipeline_bp.route('/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get current status of the autonomous pipeline"""
    try:
        pipeline = get_autonomous_pipeline()
        status = pipeline.get_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return jsonify({'error': str(e)}), 500


@autonomous_pipeline_bp.route('/pipeline/start', methods=['POST'])
def start_pipeline():
    """Manually start the autonomous pipeline"""
    try:
        pipeline = get_autonomous_pipeline()
        if pipeline.is_running:
            return jsonify({
                'status': 'already_running',
                'message': 'Pipeline is already running'
            }), 200
        
        pipeline.start()
        return jsonify({
            'status': 'started',
            'message': 'Autonomous pipeline started'
        }), 200
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        return jsonify({'error': str(e)}), 500


@autonomous_pipeline_bp.route('/pipeline/stop', methods=['POST'])
def stop_pipeline():
    """Manually stop the autonomous pipeline"""
    try:
        pipeline = get_autonomous_pipeline()
        pipeline.stop()
        return jsonify({
            'status': 'stopped',
            'message': 'Autonomous pipeline stopped'
        }), 200
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        return jsonify({'error': str(e)}), 500


@autonomous_pipeline_bp.route('/pipeline/trigger-training', methods=['POST'])
def trigger_training():
    """Manually trigger training (bypasses normal checks)"""
    try:
        pipeline = get_autonomous_pipeline()
        result = pipeline.trigger_training()
        
        if result.get('success'):
            return jsonify({
                'status': 'success',
                'message': 'Training triggered successfully',
                'details': result
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Training failed',
                'details': result
            }), 500
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        return jsonify({'error': str(e)}), 500


@autonomous_pipeline_bp.route('/pipeline/data-quality', methods=['GET'])
def get_data_quality():
    """Get current data quality metrics"""
    try:
        pipeline = get_autonomous_pipeline()
        data_quality = pipeline.check_data_quality()
        
        return jsonify({
            'total_records': data_quality.total_records,
            'recent_records_24h': data_quality.recent_records_24h,
            'recent_records_7d': data_quality.recent_records_7d,
            'data_continuity_score': data_quality.data_continuity_score,
            'feature_completeness': data_quality.feature_completeness,
            'sufficient_for_training': data_quality.sufficient_for_training,
            'days_of_data': data_quality.days_of_data,
            'issues': data_quality.issues
        }), 200
    except Exception as e:
        logger.error(f"Error getting data quality: {e}")
        return jsonify({'error': str(e)}), 500

