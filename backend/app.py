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
from api.middleware import limiter, error_handlers, log_request
from config import Config
from utils.logger import logger
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # CORS configuration
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:3001",
                "https://*.vercel.app",
                "https://*.netlify.app",
                "https://basegasfeesml.netlify.app",
                "https://basegasfeesoptimiser.netlify.app",
                os.getenv('FRONTEND_URL', '*')
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        },
        r"/config.json": {
            "origins": ["*"],  # Allow all origins for Base platform
            "methods": ["GET", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        },
        r"/manifest.json": {
            "origins": ["*"],
            "methods": ["GET", "OPTIONS"],
            "allow_headers": ["Content-Type"]
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
                }
            }
        })
    
    logger.info("Base Gas Optimizer API started")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Port: {Config.PORT}")
    
    return app


app = create_app()


if __name__ == '__main__':
    app.run(
        debug=Config.DEBUG,
        port=Config.PORT,
        host='0.0.0.0'
    )
