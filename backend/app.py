"""
Main Flask Application
Base Gas Price Prediction System - ML-powered gas fee predictions
"""

from flask import Flask, jsonify
from flask_cors import CORS
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from api.routes import api_bp
from api.base_config import base_config_bp
from api.stats import stats_bp
from api.validation_routes import validation_bp
from api.onchain_routes import onchain_bp
from api.retraining_routes import retraining_bp
from api.farcaster_routes import farcaster_bp
from api.cron_routes import cron_bp
from api.analytics_routes import analytics_bp
from api.alert_routes import alert_bp
from api.agent_routes import agent_bp
from api.multichain_routes import multichain_bp
from api.accuracy_routes import accuracy_bp
from api.middleware import limiter, error_handlers, log_request
from config import Config
from utils.logger import logger

# Try to import flask-socketio, but don't fail if it's not available
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None
    logger.warning("flask-socketio not available - WebSocket features disabled")
import os
import threading
from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService

# Initialize Sentry for error tracking
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,  # 10% performance monitoring
        profiles_sample_rate=0.1,
        environment='production' if not Config.DEBUG else 'development'
    )
    logger.info("Sentry error tracking initialized")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # CORS configuration - Allow all origins for all routes
    CORS(app,
         resources={
             r"/*": {
                 "origins": "*",
                 "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
                 "allow_headers": ["Content-Type", "Authorization", "Cache-Control", "Pragma"],
                 "expose_headers": ["Content-Type", "Cache-Control"],
                 "supports_credentials": False,
                 "max_age": 3600
             }
         })
    
    # Rate limiting
    limiter.init_app(app)
    
    # Request logging
    log_request(app)
    
    # Error handlers
    error_handlers(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(stats_bp, url_prefix='/api')
    app.register_blueprint(validation_bp, url_prefix='/api')
    app.register_blueprint(onchain_bp, url_prefix='/api')
    app.register_blueprint(retraining_bp, url_prefix='/api')
    app.register_blueprint(farcaster_bp, url_prefix='/api')
    app.register_blueprint(cron_bp, url_prefix='/api')
    app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
    app.register_blueprint(alert_bp, url_prefix='/api')
    app.register_blueprint(agent_bp, url_prefix='/api')
    app.register_blueprint(multichain_bp, url_prefix='/api')
    app.register_blueprint(accuracy_bp, url_prefix='/api/accuracy')
    app.register_blueprint(base_config_bp)  # No prefix - serves at root for /config.json
    
    # Add HTTP caching and CORS headers
    @app.after_request
    def add_headers(response):
        """Add caching and CORS headers to all responses"""
        from flask import request

        # Always add CORS headers (belt and suspenders with flask-cors)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Cache-Control, Pragma'
        response.headers['Access-Control-Max-Age'] = '3600'

        # Only cache GET requests
        if request.method == 'GET':
            path = request.path

            # Long cache for static endpoints (5 minutes)
            if any(x in path for x in ['/config.json', '/manifest.json', '/api/stats']):
                response.headers['Cache-Control'] = 'public, max-age=300'

            # Medium cache for historical data (1 minute)
            elif '/historical' in path or '/analytics' in path:
                response.headers['Cache-Control'] = 'public, max-age=60'

            # Short cache for real-time data (30 seconds)
            elif any(x in path for x in ['/current', '/predictions', '/network-state']):
                response.headers['Cache-Control'] = 'public, max-age=30'

            # No cache for health checks and admin endpoints
            elif any(x in path for x in ['/health', '/validation', '/retraining']):
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'

            # Default: short cache
            else:
                response.headers['Cache-Control'] = 'public, max-age=30'

        return response

    @app.route('/')
    def index():
        return jsonify({
            'message': 'Base Gas Optimizer API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/health',
                'current': '/api/current',
                'predictions': '/api/predictions',
                'historical': '/api/historical',
                'transactions': '/api/transactions',
                'accuracy': '/api/accuracy',
                'config': '/api/config',
                'stats': '/api/stats',
                'validation': {
                    'summary': '/api/validation/summary',
                    'metrics': '/api/validation/metrics',
                    'trends': '/api/validation/trends',
                    'health': '/api/validation/health'
                },
                'onchain': {
                    'network_state': '/api/onchain/network-state',
                    'block_features': '/api/onchain/block-features/<block_number>',
                    'congestion_history': '/api/onchain/congestion-history'
                },
                'retraining': {
                    'status': '/api/retraining/status',
                    'trigger': '/api/retraining/trigger (POST)',
                    'history': '/api/retraining/history',
                    'check_data': '/api/retraining/check-data'
                },
                'analytics': {
                    'dashboard': '/api/analytics/dashboard',
                    'performance': '/api/analytics/performance',
                    'trends': '/api/analytics/trends',
                    'validation_summary': '/api/analytics/validation-summary',
                    'model_health': '/api/analytics/model-health',
                    'collection_stats': '/api/analytics/collection-stats',
                    'recent_predictions': '/api/analytics/recent-predictions'
                },
                'agent': {
                    'recommend': '/api/agent/recommend (GET/POST)',
                    'status': '/api/agent/status',
                    'actions': '/api/agent/actions',
                    'simulate': '/api/agent/simulate (POST)'
                },
                'accuracy_tracking': {
                    'metrics': '/api/accuracy/metrics',
                    'drift': '/api/accuracy/drift',
                    'summary': '/api/accuracy/summary',
                    'report': '/api/accuracy/report',
                    'features': '/api/accuracy/features',
                    'features_importance': '/api/accuracy/features/importance'
                }
            }
        })
    
    logger.info("Base Gas Optimizer API started")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Port: {Config.PORT}")

    return app


app = create_app()

# Initialize SocketIO for WebSocket support (if available)
if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    app.socketio = socketio
    logger.info("WebSocket support enabled")
else:
    socketio = None
    app.socketio = None
    logger.warning("WebSocket support disabled - flask-socketio not installed")

# Start data collection with socketio after both are initialized
use_worker_process = os.getenv('USE_WORKER_PROCESS', 'false').lower() == 'true'

if not use_worker_process:
    if not Config.DEBUG or os.getenv('ENABLE_DATA_COLLECTION', 'true').lower() == 'true':
        websocket_status = "with WebSocket support" if SOCKETIO_AVAILABLE else "without WebSocket"
        logger.info(f"Starting data collection in background threads {websocket_status}")

        # Import here to avoid circular dependency
        def start_collection_with_socketio():
            from services.gas_collector_service import GasCollectorService
            from services.onchain_collector_service import OnChainCollectorService

            logger.info("="*60)
            logger.info("STARTING BACKGROUND DATA COLLECTION")
            logger.info(f"Collection interval: {Config.COLLECTION_INTERVAL} seconds")
            logger.info("="*60)

            gas_service = GasCollectorService(Config.COLLECTION_INTERVAL, socketio=socketio if SOCKETIO_AVAILABLE else None)
            onchain_service = OnChainCollectorService(Config.COLLECTION_INTERVAL)

            gas_thread = threading.Thread(target=gas_service.start, name="GasCollector", daemon=True)
            gas_thread.start()
            logger.info("✓ Gas price collection started")

            onchain_thread = threading.Thread(target=onchain_service.start, name="OnChainCollector", daemon=True)
            onchain_thread.start()
            logger.info("✓ On-chain features collection started")

            logger.info("="*60)

        collection_thread = threading.Thread(target=start_collection_with_socketio, daemon=True)
        collection_thread.start()
else:
    logger.info("Skipping background threads - using separate worker process")

if SOCKETIO_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle client WebSocket connection"""
        logger.info('Client connected to WebSocket')
        emit('connection_established', {'message': 'Connected to gas price updates'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client WebSocket disconnection"""
        logger.info('Client disconnected from WebSocket')


if __name__ == '__main__':
    if SOCKETIO_AVAILABLE:
        socketio.run(
            app,
            debug=Config.DEBUG,
            port=Config.PORT,
            host='0.0.0.0'
        )
    else:
        app.run(
            debug=Config.DEBUG,
            port=Config.PORT,
            host='0.0.0.0'
        )
