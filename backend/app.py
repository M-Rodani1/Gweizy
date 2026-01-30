"""
Main Flask Application
Base Gas Price Prediction System - ML-powered gas fee predictions
"""

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration
import logging
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
from api.personalization_routes import personalization_bp
from api.retraining_automated_routes import automated_retraining_bp
from api.model_versioning_routes import versioning_bp
from api.monitoring_routes import monitoring_bp
from api.mempool_routes import mempool_bp
from api.resilient_training_routes import resilient_training_bp
from api.middleware import limiter, error_handlers, log_request, setup_request_id
from config import Config
from utils.logger import logger, log_error_with_context

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
import sys
import threading
from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService
from services.hybrid_predictor import HybridPredictor

# Initialize Sentry for error tracking
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn:
    # Logging integration - capture logger.error() and above
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture INFO and above as breadcrumbs
        event_level=logging.ERROR  # Send ERROR and above to Sentry as events
    )

    def filter_sentry_events(event, hint):
        """Filter out expected exceptions that aren't real errors."""
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            # StopIteration is used by websocket libraries for control flow
            # when handling gunicorn WebSocket upgrades - not an actual error
            if exc_type is StopIteration:
                return None
        return event

    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[
            FlaskIntegration(),      # Flask request errors
            sentry_logging,          # Python logging integration
            ThreadingIntegration(    # Background thread errors
                propagate_hub=True   # Propagate Sentry context to threads
            ),
        ],
        traces_sample_rate=0.1,      # 10% performance monitoring
        profiles_sample_rate=0.1,    # 10% profiling
        environment='production' if not Config.DEBUG else 'development',

        # Filter out expected exceptions (like WebSocket StopIteration)
        before_send=filter_sentry_events,

        # Error capture settings - ALL errors should be captured
        sample_rate=1.0,             # 100% of errors (default, but explicit)
        send_default_pii=False,      # Don't send PII for security
        attach_stacktrace=True,      # Always attach stack traces
        max_breadcrumbs=50,          # Keep last 50 breadcrumbs for context

        # Enable automatic session tracking
        auto_session_tracking=True,

        # Release tracking (if version available)
        release=os.getenv('APP_VERSION', '1.0.0'),
    )
    logger.info("Sentry error tracking initialized with full coverage")
    logger.info("  - Flask errors: ✓")
    logger.info("  - Logger.error() calls: ✓")
    logger.info("  - Background thread errors: ✓")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # CORS configuration - Allow all origins for all routes
    # Explicitly allow the frontend domain and all origins
    CORS(app,
         resources={
             r"/*": {
                 "origins": ["*", "https://basegasfeesml.pages.dev", "http://localhost:3000", "http://localhost:5173"],
                 "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
                 "allow_headers": ["Content-Type", "Authorization", "Cache-Control", "Pragma", "X-Requested-With", "X-Request-ID"],
                 "expose_headers": ["Content-Type", "Cache-Control", "X-Request-ID", "X-Response-Time"],
                 "supports_credentials": False,
                 "max_age": 3600
             }
         },
         automatic_options=True,
         supports_credentials=False)
    
    # Rate limiting
    limiter.init_app(app)

    # Request ID tracking for distributed tracing (must be before log_request)
    setup_request_id(app)

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
    app.register_blueprint(agent_bp, url_prefix='/api/agent')
    app.register_blueprint(multichain_bp, url_prefix='/api')
    app.register_blueprint(accuracy_bp, url_prefix='/api/accuracy')
    app.register_blueprint(personalization_bp, url_prefix='/api')
    app.register_blueprint(versioning_bp, url_prefix='/api/versioning')
    app.register_blueprint(monitoring_bp, url_prefix='/api/monitoring')
    app.register_blueprint(mempool_bp, url_prefix='/api')
    app.register_blueprint(automated_retraining_bp, url_prefix='/api')
    app.register_blueprint(resilient_training_bp)  # Routes at /api/training/*
    
    # Register autonomous pipeline routes
    try:
        from api.autonomous_pipeline_routes import autonomous_pipeline_bp
        app.register_blueprint(autonomous_pipeline_bp, url_prefix='/api')
    except Exception as e:
        logger.warning(f"Failed to register autonomous pipeline routes: {e}")
    app.register_blueprint(base_config_bp)  # No prefix - serves at root for /config.json
    
    # Run database migrations on startup
    try:
        from scripts.migrate_add_utilization_fields import migrate_database
        logger.info("Running database migration on startup...")
        migrate_database()
        logger.info("Database migration check completed")
    except Exception as e:
        logger.warning(f"Could not run database migration on startup: {e}")
        # Don't fail startup if migration fails - it can be run manually via endpoint

    # Migrate spike detector models from pickle to joblib format (one-time migration)
    try:
        from scripts.migrate_models_to_joblib import run_migration_if_needed
        if run_migration_if_needed():
            logger.info("Model format migration check completed")
        else:
            logger.warning("Model format migration had issues - models may need manual update")
    except Exception as e:
        logger.warning(f"Could not run model format migration: {e}")
    
    # Handle OPTIONS preflight requests explicitly
    @app.before_request
    def handle_preflight():
        """Handle CORS preflight requests"""
        from flask import request
        if request.method == "OPTIONS":
            response = jsonify({})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, PATCH'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Cache-Control, Pragma, X-Requested-With'
            response.headers['Access-Control-Max-Age'] = '3600'
            return response

    # Add HTTP caching and CORS headers
    @app.after_request
    def add_headers(response):
        """Add caching and CORS headers to all responses"""
        from flask import request

        # Always add CORS headers (belt and suspenders with flask-cors)
        # CRITICAL: These must be on ALL responses, including errors
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, PATCH'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Cache-Control, Pragma, X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '3600'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
        response.headers['Access-Control-Expose-Headers'] = 'X-Request-ID, X-Response-Time'
        response.headers['Access-Control-Expose-Headers'] = 'X-Request-ID, X-Response-Time'

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
                'predictions_hybrid': '/api/predictions/hybrid',
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
                },
                'versioning': {
                    'versions': '/api/versioning/versions',
                    'active': '/api/versioning/versions/active',
                    'activate': '/api/versioning/versions/activate (POST)',
                    'rollback': '/api/versioning/versions/rollback (POST)',
                    'summary': '/api/versioning/versions/summary'
                },
                'monitoring': {
                    'dashboard': '/api/monitoring/dashboard',
                    'data-quality': '/api/monitoring/data-quality',
                    'health': '/api/monitoring/health',
                    'summary': '/api/monitoring/summary'
                },
                'mempool': {
                    'status': '/api/mempool/status',
                    'history': '/api/mempool/history',
                    'features': '/api/mempool/features'
                },
                'patterns': {
                    'analysis': '/api/patterns',
                    'description': 'Historical pattern matching for gas price prediction'
                }
            }
        })
    
    # Initialize Hybrid Predictor service (load models once on startup)
    try:
        hybrid_predictor = HybridPredictor()
        logger.info("✅ Hybrid Predictor service initialized")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize Hybrid Predictor: {e}")
        hybrid_predictor = None
    
    @app.route('/api/predictions/hybrid', methods=['GET'])
    def get_hybrid_prediction():
        """Get hybrid stacking model prediction (4h trend + 1h action)"""
        try:
            if hybrid_predictor is None:
                return jsonify({"error": "Hybrid predictor not available"}), 503
            result = hybrid_predictor.predict()
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in hybrid prediction: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    logger.info("Base Gas Optimizer API started")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Port: {Config.PORT}")

    return app


app = create_app()

# Check if we should train models on startup
TRAIN_MODELS_ON_STARTUP = os.getenv('TRAIN_MODELS', 'false').lower() == 'true'
if TRAIN_MODELS_ON_STARTUP:
    logger.info("="*60)
    logger.info("🚀 TRAIN_MODELS=true detected - Starting model training on startup")
    logger.info("="*60)
    
    def train_models_on_startup():
        """Train models in background thread on startup"""
        try:
            import subprocess
            import sys
            import time
            
            # Wait a bit for database to be ready
            time.sleep(10)
            
            logger.info("Model training is now done via Google Colab notebook.")
            logger.info("Please use notebooks/train_models_colab.ipynb for training.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "scripts", "retrain_models_simple.py")
            
            if False:  # Disabled - training now done via Colab
                logger.info(f"Running training script: {script_path}")
                logger.info("Training will take 3-10 minutes. Progress will be logged...")
                
                # Run with real-time output streaming
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=current_dir
                )
                
                # Stream output in real-time
                output_lines = []
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        logger.info(f"[TRAINING] {line}")
                        output_lines.append(line)
                
                # Wait for completion
                returncode = process.wait(timeout=600)  # 10 minute timeout
                
                if returncode == 0:
                    logger.info("✅ Model training completed successfully on startup")
                    # Log summary (last 20 lines usually contain the summary)
                    summary = "\n".join(output_lines[-20:])
                    logger.info(f"Training summary:\n{summary}")

                    # Step 2: Train spike detectors (disabled - now in Colab)
                    spike_script_path = os.path.join(current_dir, "scripts", "train_spike_detectors.py")
                    if False:  # Disabled - training now done via Colab
                        logger.info("🎯 Training spike detectors...")
                        spike_process = subprocess.Popen(
                            [sys.executable, spike_script_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            cwd=current_dir
                        )
                        for line in spike_process.stdout:
                            line = line.strip()
                            if line:
                                logger.info(f"[SPIKE] {line}")
                        spike_returncode = spike_process.wait(timeout=300)
                        if spike_returncode == 0:
                            logger.info("✅ Spike detector training completed")
                        else:
                            logger.warning(f"⚠️  Spike detector training failed (code {spike_returncode})")
                    else:
                        logger.warning(f"Spike detector script not found: {spike_script_path}")

                    # Auto-reload models after successful training
                    try:
                        logger.info("🔄 Auto-reloading models to use newly trained models...")
                        from api.routes import reload_models
                        reload_result = reload_models()
                        if reload_result['success']:
                            logger.info(f"✅ Models auto-reloaded: {reload_result['models_loaded']} models loaded")
                        else:
                            logger.warning(f"⚠️  Auto-reload had issues: {reload_result.get('error')}")
                    except Exception as e:
                        log_error_with_context(
                            e,
                            "Auto-reloading models after training",
                            context={
                                'training_completed': True,
                                'script_path': script_path,
                                'models_dir': current_dir
                            },
                            level='warning'
                        )
                        logger.info("💡 You can manually reload models via POST /api/models/reload or restart the service")
                else:
                    logger.error(f"❌ Model training failed with return code {returncode}")
                    # Log last 30 lines for debugging
                    error_output = "\n".join(output_lines[-30:])
                    logger.error(f"Last output:\n{error_output}")
                    
                    # Log detailed context for debugging
                    log_error_with_context(
                        Exception(f"Training script exited with code {returncode}"),
                        "Model training on startup",
                        context={
                            'returncode': returncode,
                            'script_path': script_path,
                            'script_exists': os.path.exists(script_path),
                            'current_dir': current_dir,
                            'output_lines_count': len(output_lines),
                            'last_output': error_output
                        }
                    )
            else:
                logger.warning(f"Training script not found: {script_path}")
                log_error_with_context(
                    FileNotFoundError(f"Training script not found: {script_path}"),
                    "Starting model training on startup",
                    context={
                        'script_path': script_path,
                        'current_dir': current_dir,
                        'scripts_dir': os.path.join(current_dir, "scripts"),
                        'scripts_dir_exists': os.path.exists(os.path.join(current_dir, "scripts"))
                    },
                    level='warning'
                )
        except Exception as e:
            log_error_with_context(
                e,
                "Startup model training thread",
                context={
                    'thread_name': 'StartupModelTraining',
                    'TRAIN_MODELS': TRAIN_MODELS_ON_STARTUP,
                    'current_dir': os.path.dirname(os.path.abspath(__file__))
                }
            )
    
    # Start training in background thread
    training_thread = threading.Thread(target=train_models_on_startup, name="StartupModelTraining", daemon=True)
    training_thread.start()
    logger.info("✓ Model training thread started (running in background)")
else:
    logger.info("TRAIN_MODELS not set or false - skipping startup training")

# Initialize SocketIO for WebSocket support (if available)
if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
    app.socketio = socketio
    logger.info("WebSocket support enabled")

    # Initialize WebSocket events module
    try:
        from services.websocket_events import init_socketio
        init_socketio(socketio)
        logger.info("WebSocket events module initialized")
    except Exception as e:
        logger.warning(f"Could not initialize websocket events: {e}")
else:
    socketio = None
    app.socketio = None
    logger.warning("WebSocket support disabled - flask-socketio not installed")

# Start data collection with socketio after both are initialized
use_worker_process = os.getenv('USE_WORKER_PROCESS', 'false').lower() == 'true'
enable_collection = os.getenv('ENABLE_DATA_COLLECTION', 'true').lower() == 'true'

logger.info(f"Data collection config: USE_WORKER_PROCESS={use_worker_process}, DEBUG={Config.DEBUG}, ENABLE_DATA_COLLECTION={enable_collection}")

if not use_worker_process:
    if not Config.DEBUG or enable_collection:
        websocket_status = "with WebSocket support" if SOCKETIO_AVAILABLE else "without WebSocket"
        logger.info(f"Starting data collection in background threads {websocket_status}")

        # Import here to avoid circular dependency
        def start_collection_with_socketio():
            try:
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

                # Start mempool collector for leading indicators
                try:
                    from data.mempool_collector import get_mempool_collector, is_collector_ready
                    mempool_collector = get_mempool_collector(timeout=5.0)
                    if mempool_collector:
                        mempool_collector.start_background_collection()
                        logger.info("✓ Mempool data collection started")
                    else:
                        logger.warning("Mempool collector initialization pending - will retry on first request")
                except Exception as e:
                    logger.warning(f"Could not start mempool collector: {e}")

                logger.info("="*60)
            except Exception as e:
                logger.error(f"CRITICAL: Failed to start data collection: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Use non-daemon thread to ensure collection survives gunicorn worker lifecycle
        collection_thread = threading.Thread(target=start_collection_with_socketio, name="CollectionStarter", daemon=False)
        collection_thread.start()
        logger.info("Collection starter thread launched")
        
        # Start automated retraining scheduler
        try:
            from services.automated_retraining import get_retraining_service
            retraining_service = get_retraining_service()
            retraining_service.start_scheduler()
            logger.info("✓ Automated retraining scheduler started")
        except Exception as e:
            logger.warning(f"Failed to start automated retraining scheduler: {e}")
        
        # Start autonomous ML pipeline (unified data collection + training)
        try:
            from services.autonomous_pipeline import get_autonomous_pipeline
            autonomous_pipeline = get_autonomous_pipeline()
            autonomous_pipeline.start()
            logger.info("✓ Autonomous ML pipeline started")
        except Exception as e:
            logger.warning(f"Failed to start autonomous pipeline: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        # Check if model training is requested via environment variable
        train_models = os.getenv('TRAIN_MODELS', 'false').lower() == 'true'
        if train_models:
            def trigger_model_training():
                """Trigger model training using resilient training service."""
                import time
                # Wait a bit for services to fully initialize
                time.sleep(10)
                
                logger.info("="*70)
                logger.info("🚀 MODEL TRAINING TRIGGERED VIA TRAIN_MODELS ENV VAR")
                logger.info("="*70)
                logger.info("   Using resilient training service with graceful shutdown")
                logger.info("="*70)
                
                try:
                    from services.resilient_training import get_resilient_training_service
                    service = get_resilient_training_service()
                    
                    # Start training in background (non-daemon thread so it can complete)
                    result = service.start_training(background=True, force=False)
                    
                    if result.get('success'):
                        logger.info(f"✓ {result.get('message', 'Training started')}")
                        logger.info("   Training runs in background - check /api/training/status for progress")
                    else:
                        logger.warning(f"⚠️  Could not start training: {result.get('error')}")
                        if result.get('status'):
                            logger.info(f"   Status: {result.get('status')}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to trigger model training: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                logger.info("="*70)
            
            training_thread = threading.Thread(target=trigger_model_training, name="ModelTrainingTrigger", daemon=True)
            training_thread.start()
            logger.info("✓ Model training trigger started (TRAIN_MODELS=true)")

        # Start automatic model rollback service
        try:
            from services.auto_rollback_service import get_auto_rollback_service
            auto_rollback = get_auto_rollback_service()
            auto_rollback.start()
            logger.info("✓ Automatic model rollback service started")
        except Exception as e:
            logger.warning(f"Failed to start auto-rollback service: {e}")

        # Check and train spike detectors if missing
        def train_spike_detectors_if_missing():
            """Train spike detectors in background if they don't exist."""
            import time
            time.sleep(30)  # Wait for other services to start

            try:
                # Check if spike detectors exist
                models_dir = os.environ.get('MODELS_DIR', '/data/models')
                horizons = ['1h', '4h', '24h']
                missing = []

                for horizon in horizons:
                    detector_path = os.path.join(models_dir, f'spike_detector_{horizon}.pkl')
                    if not os.path.exists(detector_path):
                        missing.append(horizon)

                if missing:
                    logger.info(f"⚠️ Spike detectors missing for {missing}")
                    logger.info("Please train models using notebooks/train_models_colab.ipynb")
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    spike_script = os.path.join(current_dir, "scripts", "train_spike_detectors.py")

                    if False:  # Disabled - training now done via Colab
                        import subprocess
                        process = subprocess.Popen(
                            [sys.executable, spike_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            cwd=current_dir
                        )
                        for line in process.stdout:
                            if line.strip():
                                logger.info(f"[SPIKE] {line.strip()}")
                        returncode = process.wait(timeout=300)
                        if returncode == 0:
                            logger.info("✅ Spike detector training completed")
                        else:
                            logger.warning(f"⚠️  Spike detector training failed (code {returncode})")
                    else:
                        logger.warning(f"Spike detector script not found: {spike_script}")
                else:
                    logger.info("✓ Spike detectors already exist")
            except Exception as e:
                logger.warning(f"Spike detector check/training failed: {e}")

        spike_thread = threading.Thread(target=train_spike_detectors_if_missing, name="SpikeDetectorTrainer", daemon=True)
        spike_thread.start()
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
