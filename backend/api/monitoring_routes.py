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


@monitoring_bp.route('/database', methods=['GET'])
def get_database_health():
    """
    Get database connection pool health and metrics.

    Returns:
        - Pool status (size, checked in/out, overflow)
        - Database health check with timing
        - Recent record counts
    """
    try:
        from data.database import DatabaseManager
        db = DatabaseManager()
        health = db.get_health_check()

        return jsonify({
            'success': True,
            'database': health
        })
    except Exception as e:
        logger.error(f"Error getting database health: {e}")
        capture_exception(e, {'endpoint': '/monitoring/database'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/circuit-breakers', methods=['GET'])
def get_circuit_breaker_status():
    """
    Get status of all circuit breakers.

    Returns:
        - Per-circuit: state (open/closed/half-open), stats, config
        - Useful for diagnosing external service issues
    """
    try:
        from utils.circuit_breaker import CircuitBreaker

        status = CircuitBreaker.get_all_status()

        return jsonify({
            'success': True,
            'circuit_breakers': status,
            'count': len(status)
        })
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        capture_exception(e, {'endpoint': '/monitoring/circuit-breakers'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/circuit-breakers/<name>/reset', methods=['POST'])
def reset_circuit_breaker(name: str):
    """
    Manually reset a circuit breaker to closed state.

    Use with caution - only reset if you've verified the underlying
    service has recovered.

    Args:
        name: Circuit breaker name (e.g., 'rpc_provider', 'owlracle_api')
    """
    try:
        from utils.circuit_breaker import CircuitBreaker

        if name not in CircuitBreaker._registry:
            return jsonify({
                'success': False,
                'error': f"Circuit breaker '{name}' not found",
                'available': list(CircuitBreaker._registry.keys())
            }), 404

        cb = CircuitBreaker._registry[name]
        cb.reset()

        return jsonify({
            'success': True,
            'message': f"Circuit breaker '{name}' has been reset to closed state",
            'status': cb.get_status()
        })
    except Exception as e:
        logger.error(f"Error resetting circuit breaker: {e}")
        capture_exception(e, {'endpoint': '/monitoring/circuit-breakers/reset'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/auto-rollback', methods=['GET'])
def get_auto_rollback_status():
    """
    Get automatic model rollback service status.

    Returns:
        - Service enabled/running state
        - Per-horizon accuracy tracking
        - Rollback thresholds
        - Recent rollback history
    """
    try:
        from services.auto_rollback_service import get_auto_rollback_service

        service = get_auto_rollback_service()
        status = service.get_status()

        return jsonify({
            'success': True,
            'auto_rollback': status
        })
    except Exception as e:
        logger.error(f"Error getting auto-rollback status: {e}")
        capture_exception(e, {'endpoint': '/monitoring/auto-rollback'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/auto-rollback/enable', methods=['POST'])
def enable_auto_rollback():
    """Enable automatic model rollback."""
    try:
        from services.auto_rollback_service import get_auto_rollback_service

        service = get_auto_rollback_service()
        service.enable()

        return jsonify({
            'success': True,
            'message': 'Auto-rollback enabled',
            'status': service.get_status()
        })
    except Exception as e:
        logger.error(f"Error enabling auto-rollback: {e}")
        capture_exception(e, {'endpoint': '/monitoring/auto-rollback/enable'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/auto-rollback/disable', methods=['POST'])
def disable_auto_rollback():
    """Disable automatic model rollback (monitoring continues)."""
    try:
        from services.auto_rollback_service import get_auto_rollback_service

        service = get_auto_rollback_service()
        service.disable()

        return jsonify({
            'success': True,
            'message': 'Auto-rollback disabled (monitoring continues)',
            'status': service.get_status()
        })
    except Exception as e:
        logger.error(f"Error disabling auto-rollback: {e}")
        capture_exception(e, {'endpoint': '/monitoring/auto-rollback/disable'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/auto-rollback/baseline/<horizon>', methods=['POST'])
def update_rollback_baseline(horizon: str):
    """
    Update accuracy baseline for a horizon.

    Use after successful retraining to set new accuracy expectations.

    Args:
        horizon: Prediction horizon ('1h', '4h', '24h')

    Body:
        accuracy: New baseline accuracy (0.0 - 1.0)
    """
    try:
        from services.auto_rollback_service import get_auto_rollback_service

        if horizon not in ['1h', '4h', '24h']:
            return jsonify({
                'success': False,
                'error': f"Invalid horizon: {horizon}"
            }), 400

        data = request.get_json() or {}
        accuracy = data.get('accuracy')

        if accuracy is None:
            return jsonify({
                'success': False,
                'error': "accuracy field required"
            }), 400

        if not 0 <= accuracy <= 1:
            return jsonify({
                'success': False,
                'error': "accuracy must be between 0 and 1"
            }), 400

        service = get_auto_rollback_service()
        service.update_baseline(horizon, accuracy)

        return jsonify({
            'success': True,
            'message': f'Baseline updated for {horizon}',
            'horizon': horizon,
            'new_baseline': accuracy
        })
    except Exception as e:
        logger.error(f"Error updating baseline: {e}")
        capture_exception(e, {'endpoint': '/monitoring/auto-rollback/baseline'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/rpc-stats', methods=['GET'])
def get_rpc_stats():
    """
    Get RPC provider statistics and health.
    
    Returns:
        - Current active RPC endpoint
        - Per-endpoint statistics (success rate, failures, rate limits)
        - Endpoint priority and health status
        - Rate limit status
    """
    try:
        from utils.rpc_manager import get_rpc_manager
        
        rpc_manager = get_rpc_manager()
        stats = rpc_manager.get_stats()
        
        return jsonify({
            'success': True,
            'rpc_stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting RPC stats: {e}")
        capture_exception(e, {'endpoint': '/monitoring/rpc-stats'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/collection-stats', methods=['GET'])
def get_collection_stats():
    """
    Get data collection service statistics.
    
    Returns:
        - Collection interval
        - Total collections and successes
        - Success rate
        - Error count
        - Collection rate (collections/hour)
        - Last collection time
    """
    try:
        from services.gas_collector_service import GasCollectorService
        
        # Try to get stats from running service
        # Note: This is a simplified version - in production you'd want
        # a singleton service instance to query
        stats = {
            'interval_seconds': 5,  # Current config
            'status': 'unknown',
            'message': 'Collection stats require access to running service instance'
        }
        
        return jsonify({
            'success': True,
            'collection_stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        capture_exception(e, {'endpoint': '/monitoring/collection-stats'})
        return jsonify({'success': False, 'error': str(e)}), 500


@monitoring_bp.route('/reliability', methods=['GET'])
def get_reliability_summary():
    """
    Get comprehensive reliability status for all systems.

    Aggregates:
        - Database health
        - Circuit breaker states
        - API performance
        - Auto-rollback status
        - Cache stats
    """
    try:
        from data.database import DatabaseManager
        from utils.circuit_breaker import CircuitBreaker
        from services.auto_rollback_service import get_auto_rollback_service

        # Database health
        db = DatabaseManager()
        db_health = db.get_health_check()

        # Circuit breakers
        circuit_status = CircuitBreaker.get_all_status()
        circuits_healthy = all(
            cb['state'] == 'closed' for cb in circuit_status.values()
        )

        # API performance
        api_summary = performance_metrics.get_summary()

        # Auto-rollback service
        try:
            rollback_service = get_auto_rollback_service()
            rollback_status = rollback_service.get_status()
            rollback_healthy = rollback_status.get('running', False)
            recent_rollbacks = rollback_status.get('total_rollbacks', 0)
        except Exception:
            rollback_healthy = False
            recent_rollbacks = 0

        # Overall status
        overall_healthy = (
            db_health.get('status') == 'healthy' and
            circuits_healthy and
            api_summary.get('error_rate_percent', 0) < 5
        )

        return jsonify({
            'success': True,
            'overall_status': 'healthy' if overall_healthy else 'degraded',
            'components': {
                'database': {
                    'status': db_health.get('status'),
                    'response_time_ms': db_health.get('response_time_ms'),
                    'data_collection_active': db_health.get('data_collection_active')
                },
                'circuit_breakers': {
                    'status': 'healthy' if circuits_healthy else 'degraded',
                    'total': len(circuit_status),
                    'open': sum(1 for cb in circuit_status.values() if cb['state'] == 'open'),
                    'half_open': sum(1 for cb in circuit_status.values() if cb['state'] == 'half_open')
                },
                'api': {
                    'total_requests': api_summary.get('total_requests', 0),
                    'error_rate_percent': api_summary.get('error_rate_percent', 0),
                    'avg_response_time_ms': api_summary.get('avg_response_time_ms', 0)
                },
                'auto_rollback': {
                    'status': 'healthy' if rollback_healthy else 'inactive',
                    'total_rollbacks': recent_rollbacks
                }
            }
        })
    except Exception as e:
        logger.error(f"Error getting reliability summary: {e}")
        capture_exception(e, {'endpoint': '/monitoring/reliability'})
        return jsonify({'success': False, 'error': str(e)}), 500

