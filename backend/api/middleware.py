"""
API Middleware Module

Provides rate limiting, performance monitoring, and request logging.
"""

from flask import request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.logger import logger, capture_exception
from config import Config
import time
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np


# =============================================================================
# Performance Metrics Storage
# =============================================================================

class PerformanceMetrics:
    """Thread-safe performance metrics tracker."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self.max_history = max_history

        # Per-endpoint metrics
        self.endpoint_times: Dict[str, List[float]] = defaultdict(list)
        self.endpoint_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_errors: Dict[str, int] = defaultdict(int)

        # Global metrics
        self.total_requests = 0
        self.total_errors = 0
        self.slow_requests: List[Dict[str, Any]] = []  # Requests > 1s

        # Time-based metrics (last hour, rolling)
        self.hourly_requests: List[datetime] = []

    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record a request's performance metrics."""
        with self._lock:
            # Update endpoint-specific metrics
            self.endpoint_times[endpoint].append(duration)
            if len(self.endpoint_times[endpoint]) > self.max_history:
                self.endpoint_times[endpoint] = self.endpoint_times[endpoint][-self.max_history:]

            self.endpoint_counts[endpoint] += 1
            self.total_requests += 1

            # Track errors
            if status_code >= 400:
                self.endpoint_errors[endpoint] += 1
                self.total_errors += 1

            # Track slow requests (> 1 second)
            if duration > 1.0:
                self.slow_requests.append({
                    'endpoint': endpoint,
                    'duration': round(duration, 3),
                    'status_code': status_code,
                    'timestamp': datetime.utcnow().isoformat()
                })
                if len(self.slow_requests) > 100:
                    self.slow_requests = self.slow_requests[-100:]

            # Track hourly requests
            now = datetime.utcnow()
            self.hourly_requests.append(now)
            cutoff = now - timedelta(hours=1)
            self.hourly_requests = [t for t in self.hourly_requests if t > cutoff]

    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint."""
        with self._lock:
            times = self.endpoint_times.get(endpoint, [])
            if not times:
                return {'available': False}

            times_arr = np.array(times)

            return {
                'available': True,
                'request_count': self.endpoint_counts.get(endpoint, 0),
                'error_count': self.endpoint_errors.get(endpoint, 0),
                'avg_response_time_ms': round(np.mean(times_arr) * 1000, 2),
                'median_response_time_ms': round(np.median(times_arr) * 1000, 2),
                'p95_response_time_ms': round(np.percentile(times_arr, 95) * 1000, 2),
                'p99_response_time_ms': round(np.percentile(times_arr, 99) * 1000, 2),
                'min_response_time_ms': round(np.min(times_arr) * 1000, 2),
                'max_response_time_ms': round(np.max(times_arr) * 1000, 2)
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._lock:
            # Calculate global averages
            all_times = []
            for times in self.endpoint_times.values():
                all_times.extend(times)

            summary = {
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'error_rate_percent': round(
                    (self.total_errors / self.total_requests * 100)
                    if self.total_requests > 0 else 0, 2
                ),
                'requests_last_hour': len(self.hourly_requests),
                'slow_request_count': len(self.slow_requests),
                'endpoints_tracked': len(self.endpoint_counts),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            if all_times:
                times_arr = np.array(all_times)
                summary['avg_response_time_ms'] = round(np.mean(times_arr) * 1000, 2)
                summary['p95_response_time_ms'] = round(np.percentile(times_arr, 95) * 1000, 2)
                summary['p99_response_time_ms'] = round(np.percentile(times_arr, 99) * 1000, 2)

            # Top 5 slowest endpoints
            endpoint_avgs = []
            for endpoint, times in self.endpoint_times.items():
                if times:
                    endpoint_avgs.append({
                        'endpoint': endpoint,
                        'avg_ms': round(np.mean(times) * 1000, 2),
                        'count': self.endpoint_counts[endpoint]
                    })

            endpoint_avgs.sort(key=lambda x: x['avg_ms'], reverse=True)
            summary['slowest_endpoints'] = endpoint_avgs[:5]

            # Most requested endpoints
            most_requested = sorted(
                self.endpoint_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            summary['most_requested'] = [
                {'endpoint': e, 'count': c} for e, c in most_requested
            ]

            # Recent slow requests
            summary['recent_slow_requests'] = self.slow_requests[-5:]

            return summary

    def get_all_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all endpoints."""
        with self._lock:
            result = {}
            for endpoint in self.endpoint_counts.keys():
                times = self.endpoint_times.get(endpoint, [])
                if times:
                    times_arr = np.array(times)
                    result[endpoint] = {
                        'request_count': self.endpoint_counts[endpoint],
                        'error_count': self.endpoint_errors.get(endpoint, 0),
                        'avg_ms': round(np.mean(times_arr) * 1000, 2),
                        'p95_ms': round(np.percentile(times_arr, 95) * 1000, 2)
                    }
            return result


# Global metrics instance
performance_metrics = PerformanceMetrics()


# =============================================================================
# Rate Limiting Configuration
# =============================================================================

# Rate limiter - more lenient in development
if Config.DEBUG:
    # Development: Allow more requests for testing
    default_limits = ["1000 per hour", "100 per minute"]
else:
    # Production: Balanced limits
    default_limits = ["500 per hour", "100 per minute"]

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=default_limits,
    storage_uri="memory://"
)

# Endpoint-specific rate limits (can be applied with @limiter.limit())
ENDPOINT_RATE_LIMITS = {
    # Analytics endpoints - moderate limits (compute-intensive)
    'analytics_volatility': '30 per minute',
    'analytics_whales': '30 per minute',
    'analytics_anomalies': '30 per minute',
    'analytics_ensemble': '30 per minute',
    'analytics_performance': '20 per minute',

    # Prediction endpoints - higher limits (frequently called)
    'current_gas': '120 per minute',
    'predictions': '60 per minute',

    # Heavy computation endpoints - stricter limits
    'patterns': '20 per minute',
    'retraining_trigger': '5 per minute',

    # Status endpoints - very lenient
    'health': '200 per minute',
    'status': '200 per minute'
}


def get_rate_limit(endpoint_key: str) -> str:
    """Get rate limit for a specific endpoint key."""
    return ENDPOINT_RATE_LIMITS.get(endpoint_key, '60 per minute')


# =============================================================================
# Request ID Middleware for Distributed Tracing
# =============================================================================

class RequestContext:
    """Thread-local storage for request context."""
    _local = threading.local()

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Get the current request ID."""
        return getattr(cls._local, 'request_id', None)

    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """Set the current request ID."""
        cls._local.request_id = request_id

    @classmethod
    def clear(cls) -> None:
        """Clear the request context."""
        cls._local.request_id = None


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"


def setup_request_id(app):
    """Add request ID tracking to all requests."""

    @app.before_request
    def assign_request_id():
        # Use incoming request ID header if present (for distributed tracing)
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            request_id = generate_request_id()

        # Store in both Flask's g object and thread-local for access in services
        g.request_id = request_id
        RequestContext.set_request_id(request_id)

    @app.after_request
    def add_request_id_header(response):
        # Add request ID to response for correlation
        request_id = getattr(g, 'request_id', None)
        if request_id:
            response.headers['X-Request-ID'] = request_id
        return response

    @app.teardown_request
    def cleanup_request_context(exception=None):
        RequestContext.clear()


# =============================================================================
# Request Logging with Performance Tracking
# =============================================================================

def log_request(app):
    """Log all incoming requests and track performance metrics."""

    @app.before_request
    def before_request():
        request.start_time = time.time()
        request_id = getattr(g, 'request_id', 'unknown')
        # Only log non-health requests to reduce noise
        if '/health' not in request.path:
            logger.debug(f"[{request_id}] {request.method} {request.path} from {request.remote_addr}")

    @app.after_request
    def after_request(response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            request_id = getattr(g, 'request_id', 'unknown')

            # Record performance metrics
            endpoint = request.path
            performance_metrics.record_request(endpoint, duration, response.status_code)

            # Log slow requests with warning
            if duration > 1.0:
                logger.warning(
                    f"[{request_id}] SLOW REQUEST: {request.method} {request.path} - "
                    f"{response.status_code} - {duration:.3f}s"
                )
            elif '/health' not in request.path:
                logger.debug(
                    f"[{request_id}] {request.method} {request.path} - "
                    f"{response.status_code} - {duration:.3f}s"
                )

            # Add performance header for debugging
            response.headers['X-Response-Time'] = f"{duration:.3f}s"

        return response


# =============================================================================
# CORS Headers
# =============================================================================

def add_cors_headers(response):
    """Add CORS headers to a response."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, PATCH'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Cache-Control, Pragma, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    response.headers['Access-Control-Allow-Credentials'] = 'false'
    return response


# =============================================================================
# Error Handlers
# =============================================================================

def error_handlers(app):
    """Register error handlers."""

    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 Not Found: {request.path}")
        response = jsonify({'error': 'Endpoint not found'})
        return add_cors_headers(response), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 Internal Error: {str(error)}")
        capture_exception(error, {'error_type': '500_internal'})
        response = jsonify({'error': 'Internal server error'})
        return add_cors_headers(response), 500

    @app.errorhandler(429)
    def ratelimit_handler(e):
        logger.warning(f"Rate limit exceeded: {request.remote_addr} on {request.path}")
        response = jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'retry_after': 60
        })
        response.headers['Retry-After'] = '60'
        return add_cors_headers(response), 429

    @app.errorhandler(503)
    def service_unavailable(error):
        logger.error(f"503 Service Unavailable: {str(error)}")
        capture_exception(error, {'error_type': '503_unavailable'})
        response = jsonify({'error': 'Service temporarily unavailable'})
        return add_cors_headers(response), 503
