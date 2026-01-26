"""Gunicorn configuration for Railway deployment with graceful shutdown"""
import os
import signal
import threading
import time
from utils.logger import logger

# Track active services for graceful shutdown
_active_services = []
_shutdown_in_progress = False


def worker_int(worker):
    """
    Called when a worker receives SIGINT (Ctrl+C).
    Initiates graceful shutdown.
    """
    global _shutdown_in_progress
    _shutdown_in_progress = True
    logger.info(f"Worker {worker.pid} received SIGINT, initiating graceful shutdown...")
    _graceful_shutdown(worker)


def worker_term(worker):
    """
    Called when a worker receives SIGTERM.
    Initiates graceful shutdown with request draining.
    """
    global _shutdown_in_progress
    _shutdown_in_progress = True
    logger.info(f"Worker {worker.pid} received SIGTERM, initiating graceful shutdown...")
    _graceful_shutdown(worker)


def worker_abort(worker):
    """
    Called when a worker receives SIGABRT (timeout or critical error).
    Logs the abort for debugging.
    """
    logger.error(f"Worker {worker.pid} aborted (SIGABRT) - likely request timeout")


def _graceful_shutdown(worker):
    """
    Perform graceful shutdown:
    1. Stop accepting new connections
    2. Allow in-flight requests to complete (up to graceful_timeout)
    3. Stop background services
    4. Close database connections
    """
    logger.info(f"Worker {worker.pid}: Graceful shutdown started")

    # Stop background services
    for service_name, stop_func in _active_services:
        try:
            logger.info(f"Stopping service: {service_name}")
            stop_func()
        except Exception as e:
            logger.warning(f"Error stopping {service_name}: {e}")

    # Give in-flight requests time to complete
    logger.info(f"Worker {worker.pid}: Waiting for in-flight requests to complete...")
    time.sleep(2)  # Brief wait for request completion

    # Close database connections
    try:
        from data.database import DatabaseManager
        # The database connections will be cleaned up by SQLAlchemy's pool
        logger.info(f"Worker {worker.pid}: Database connections will be cleaned up")
    except Exception as e:
        logger.warning(f"Error during database cleanup: {e}")

    logger.info(f"Worker {worker.pid}: Graceful shutdown complete")


def register_service(name, stop_function):
    """Register a service for graceful shutdown."""
    _active_services.append((name, stop_function))
    logger.info(f"Registered service for graceful shutdown: {name}")


def post_fork(server, worker):
    """
    Called after a worker has been forked.
    This is where we start background threads since daemon threads
    don't survive the fork when using --preload.
    """
    logger.info(f"Worker {worker.pid} forked, starting data collection...")

    # Import here to avoid circular imports
    from services.gas_collector_service import GasCollectorService
    from services.onchain_collector_service import OnChainCollectorService
    from config import Config

    # Only start in the first worker to avoid duplicate collection
    if worker.age == 0:  # First worker
        def start_data_collection():
            """Start both collection services"""
            try:
                logger.info("="*60)
                logger.info("STARTING DATA COLLECTION (Background Threads)")
                logger.info("="*60)

                # Initialize services (no signal handlers in background threads)
                gas_service = GasCollectorService(register_signals=False)
                onchain_service = OnChainCollectorService(register_signals=False)

                # Register for graceful shutdown
                if hasattr(gas_service, 'stop'):
                    register_service('GasCollectorService', gas_service.stop)
                if hasattr(onchain_service, 'stop'):
                    register_service('OnChainCollectorService', onchain_service.stop)

                # Start collection loops
                gas_service.start()
                onchain_service.start()

                logger.info("Data collection services started successfully")
                logger.info("="*60)
            except Exception as e:
                logger.error(f"Failed to start data collection: {e}")

        def start_validation_scheduler():
            """Start prediction validation scheduler"""
            try:
                from services.validation_scheduler import ValidationScheduler

                logger.info("="*60)
                logger.info("STARTING PREDICTION VALIDATION SCHEDULER")
                logger.info("="*60)

                scheduler = ValidationScheduler()

                # Register for graceful shutdown
                if hasattr(scheduler, 'stop'):
                    register_service('ValidationScheduler', scheduler.stop)

                scheduler.start()
            except Exception as e:
                logger.error(f"Failed to start validation scheduler: {e}")

        # Start data collection in background thread
        collection_thread = threading.Thread(target=start_data_collection, daemon=True)
        collection_thread.start()
        logger.info(f"Data collection thread started in worker {worker.pid}")

        # Start validation scheduler in background thread
        validation_thread = threading.Thread(target=start_validation_scheduler, daemon=True)
        validation_thread.start()
        logger.info(f"Validation scheduler thread started in worker {worker.pid}")


def on_exit(server):
    """
    Called when the master process is shutting down.
    Final cleanup opportunity.
    """
    logger.info("Gunicorn master process shutting down")


def pre_request(worker, req):
    """
    Called before each request.
    Can be used to reject requests during shutdown.
    """
    if _shutdown_in_progress:
        logger.warning(f"Request received during shutdown: {req.path}")
        # Could return 503 here, but gunicorn handles this via graceful_timeout


def post_request(worker, req, environ, resp):
    """
    Called after each request completes.
    Useful for request-level cleanup or logging.
    """
    pass  # Logging handled by middleware


# =============================================================================
# Gunicorn Settings
# =============================================================================

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', '1'))
threads = int(os.getenv('GUNICORN_THREADS', '4'))
worker_class = 'geventwebsocket.gunicorn.workers.GeventWebSocketWorker'  # Required for WebSocket support

# Timeouts
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))  # Worker timeout
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', '30'))  # Shutdown grace period
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', '5'))  # Keep-alive connections

# Request handling
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', '1000'))  # Restart workers after N requests
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', '100'))  # Add randomness to prevent thundering herd

# Preload app for faster worker startup
preload_app = True

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')

# Process naming
proc_name = 'gweizy-api'
