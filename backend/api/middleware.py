from flask import request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.logger import logger
from config import Config
import time


# Rate limiter - more lenient in development
if Config.DEBUG:
    # Development: Allow more requests for testing
    default_limits = ["1000 per hour", "100 per minute"]
else:
    # Production: Stricter limits
    default_limits = ["200 per day", "50 per hour"]

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=default_limits,
    storage_uri="memory://"
)


def log_request(app):
    """Log all incoming requests"""
    @app.before_request
    def before_request():
        request.start_time = time.time()
        logger.info(f"{request.method} {request.path} from {request.remote_addr}")
    
    @app.after_request
    def after_request(response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
        return response


def add_cors_headers(response):
    """Add CORS headers to a response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Cache-Control, Pragma'
    return response


def error_handlers(app):
    """Register error handlers"""

    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 Not Found: {request.path}")
        response = jsonify({'error': 'Endpoint not found'})
        return add_cors_headers(response), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 Internal Error: {str(error)}")
        response = jsonify({'error': 'Internal server error'})
        return add_cors_headers(response), 500

    @app.errorhandler(429)
    def ratelimit_handler(e):
        logger.warning(f"Rate limit exceeded: {request.remote_addr}")
        response = jsonify({'error': 'Rate limit exceeded. Try again later.'})
        return add_cors_headers(response), 429

    @app.errorhandler(503)
    def service_unavailable(error):
        logger.error(f"503 Service Unavailable: {str(error)}")
        response = jsonify({'error': 'Service temporarily unavailable'})
        return add_cors_headers(response), 503

