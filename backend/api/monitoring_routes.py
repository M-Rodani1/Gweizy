"""
Enhanced Monitoring API Routes

Endpoints for comprehensive system monitoring including:
- Model performance dashboard
- Data quality reports
- API performance metrics
- System health status
"""

from flask import Blueprint, jsonify, request
from services.monitoring_service import get_monitoring_service
from api.middleware import performance_metrics, limiter, get_rate_limit
from api.cache import get_cache_stats
from utils.logger import logger, capture_exception

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
        capture_exception(e, {'endpoint': '/monitoring/dashboard'})
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
        capture_exception(e, {'endpoint': '/monitoring/data-quality'})
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
        capture_exception(e, {'endpoint': '/monitoring/health'})
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
        capture_exception(e, {'endpoint': '/monitoring/summary'})
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# API Performance Monitoring Endpoints
# =============================================================================

@monitoring_bp.route('/api-performance', methods=['GET'])
def get_api_performance():
    """
    Get API performance metrics summary.

    Returns:
        - Total requests and errors
        - Average response times (avg, p95, p99)
        - Slowest endpoints
        - Most requested endpoints
        - Recent slow requests
    """
    try:
        summary = performance_metrics.get_summary()
        return jsonify({
            'success': True,
            'performance': summary
        })
    except Exception as e:
        logger.error(f"Error getting API performance: {e}")
        capture_exception(e, {'endpoint': '/monitoring/api-performance'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/api-performance/endpoints', methods=['GET'])
def get_endpoint_performance():
    """
    Get performance metrics for all endpoints.

    Returns per-endpoint:
        - Request count
        - Error count
        - Average response time
        - P95 response time
    """
    try:
        endpoints = performance_metrics.get_all_endpoints()
        return jsonify({
            'success': True,
            'endpoints': endpoints,
            'count': len(endpoints)
        })
    except Exception as e:
        logger.error(f"Error getting endpoint performance: {e}")
        capture_exception(e, {'endpoint': '/monitoring/api-performance/endpoints'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/api-performance/endpoint/<path:endpoint_path>', methods=['GET'])
def get_single_endpoint_performance(endpoint_path):
    """
    Get detailed performance metrics for a specific endpoint.

    Args:
        endpoint_path: The endpoint path (e.g., /api/current)
    """
    try:
        # Ensure path starts with /
        if not endpoint_path.startswith('/'):
            endpoint_path = '/' + endpoint_path

        stats = performance_metrics.get_endpoint_stats(endpoint_path)
        return jsonify({
            'success': True,
            'endpoint': endpoint_path,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting endpoint stats: {e}")
        capture_exception(e, {'endpoint': '/monitoring/api-performance/endpoint'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/cache-stats', methods=['GET'])
def get_cache_statistics():
    """
    Get cache performance statistics.

    Returns:
        - Cache size and max size
        - Hit/miss counts and rates
        - Per-function cache statistics
    """
    try:
        stats = get_cache_stats()
        return jsonify({
            'success': True,
            'cache': stats
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        capture_exception(e, {'endpoint': '/monitoring/cache-stats'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/rate-limits', methods=['GET'])
def get_rate_limits():
    """
    Get configured rate limits for endpoints.

    Returns the rate limit configuration for different endpoint categories.
    """
    from api.middleware import ENDPOINT_RATE_LIMITS, default_limits

    return jsonify({
        'success': True,
        'default_limits': default_limits,
        'endpoint_limits': ENDPOINT_RATE_LIMITS
    })

