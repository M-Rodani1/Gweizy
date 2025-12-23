"""
Main Flask Application
Base Gas Price Prediction System
"""

from flask import Flask, jsonify
from flask_cors import CORS
from api.routes import api_bp
from api.base_config import base_config_bp
from api.stats import stats_bp
from api.validation_routes import validation_bp
from api.onchain_routes import onchain_bp
from api.retraining_routes import retraining_bp
from api.farcaster_routes import farcaster_bp
from api.cron_routes import cron_bp
from api.analytics_routes import analytics_bp
from api.middleware import limiter, error_handlers, log_request
from config import Config
from utils.logger import logger
import os
import threading
from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # CORS configuration - Allow all origins with regex pattern for Cloudflare Pages
    import re

    def is_allowed_origin(origin):
        """Check if origin is allowed - matches Cloudflare Pages URLs and localhost"""
        if not origin:
            return False

        allowed_patterns = [
            r'^https://basegasfeesml\.pages\.dev$',  # Production
            r'^https://[a-f0-9]+\.basegasfeesml\.pages\.dev$',  # Preview URLs
            r'^http://localhost:\d+$',  # Local development
            r'^.*$'  # Allow all as fallback
        ]

        for pattern in allowed_patterns:
            if re.match(pattern, origin):
                return True
        return False

    CORS(app,
         resources={
             r"/api/*": {
                 "origins": is_allowed_origin,
                 "methods": ["GET", "POST", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": False
             },
             r"/config.json": {
                 "origins": is_allowed_origin,
                 "methods": ["GET", "OPTIONS"],
                 "allow_headers": ["Content-Type"],
                 "supports_credentials": False
             },
             r"/manifest.json": {
                 "origins": is_allowed_origin,
                 "methods": ["GET", "OPTIONS"],
                 "allow_headers": ["Content-Type"],
                 "supports_credentials": False
             }
         },
         supports_credentials=False)
    
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
    app.register_blueprint(base_config_bp)  # No prefix - serves at root for /config.json
    
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
                }
            }
        })
    
    logger.info("Base Gas Optimizer API started")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Port: {Config.PORT}")

    # Start background data collection
    def start_data_collection():
        """Start data collection in background threads"""
        try:
            logger.info("="*60)
            logger.info("STARTING BACKGROUND DATA COLLECTION")
            logger.info(f"Collection interval: {Config.COLLECTION_INTERVAL} seconds")
            logger.info("="*60)

            gas_service = GasCollectorService(Config.COLLECTION_INTERVAL)
            onchain_service = OnChainCollectorService(Config.COLLECTION_INTERVAL)

            # Start gas price collection
            gas_thread = threading.Thread(
                target=gas_service.start,
                name="GasCollector",
                daemon=True
            )
            gas_thread.start()
            logger.info("✓ Gas price collection started")

            # Start on-chain collection
            onchain_thread = threading.Thread(
                target=onchain_service.start,
                name="OnChainCollector",
                daemon=True
            )
            onchain_thread.start()
            logger.info("✓ On-chain features collection started")

            logger.info("="*60)
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")

    # Start collection in background threads (only if not using separate worker process)
    # On Railway: Use separate worker process (worker.py)
    # On Render/local: Use background threads
    use_worker_process = os.getenv('USE_WORKER_PROCESS', 'false').lower() == 'true'

    if not use_worker_process:
        if not Config.DEBUG or os.getenv('ENABLE_DATA_COLLECTION', 'true').lower() == 'true':
            logger.info("Starting data collection in background threads")
            collection_thread = threading.Thread(target=start_data_collection, daemon=True)
            collection_thread.start()
    else:
        logger.info("Skipping background threads - using separate worker process")

    return app


app = create_app()


if __name__ == '__main__':
    app.run(
        debug=Config.DEBUG,
        port=Config.PORT,
        host='0.0.0.0'
    )
