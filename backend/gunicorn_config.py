"""Gunicorn configuration for Railway deployment"""
import os
import threading
from utils.logger import logger


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


# Gunicorn settings
bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"
workers = 1
threads = 4
timeout = 120
preload_app = True
