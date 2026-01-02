"""
Enhanced Monitoring API Routes

Endpoints for comprehensive system monitoring.
"""

from flask import Blueprint, jsonify, request
from services.monitoring_service import get_monitoring_service
from utils.logger import logger

monitoring_bp = Blueprint('monitoring', __name__)


@monitoring_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get model performance dashboard"""
    try:
        service = get_monitoring_service()
        dashboard = service.get_model_performance_dashboard()
        return jsonify({
            'success': True,
            'dashboard': dashboard
        })
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/data-quality', methods=['GET'])
def get_data_quality():
    """Get data quality report"""
    try:
        service = get_monitoring_service()
        report = service.get_data_quality_report()
        return jsonify({
            'success': True,
            'data_quality': report
        })
    except Exception as e:
        logger.error(f"Error getting data quality: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/health', methods=['GET'])
def get_health():
    """Get system health status"""
    try:
        service = get_monitoring_service()
        health = service._get_system_health()
        return jsonify({
            'success': True,
            'health': health
        })
    except Exception as e:
        logger.error(f"Error getting health: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/summary', methods=['GET'])
def get_summary():
    """Get comprehensive monitoring summary"""
    try:
        service = get_monitoring_service()
        summary = service.get_monitoring_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

