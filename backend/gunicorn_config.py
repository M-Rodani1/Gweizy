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

                # Initialize services
                gas_service = GasCollectorService()
                onchain_service = OnChainCollectorService()

                # Start collection loops
                gas_service.start_collection()
                onchain_service.start_collection()

                logger.info("Data collection services started successfully")
                logger.info("="*60)
            except Exception as e:
                logger.error(f"Failed to start data collection: {e}")

        # Start in background thread
        collection_thread = threading.Thread(target=start_data_collection, daemon=True)
        collection_thread.start()
        logger.info(f"Data collection thread started in worker {worker.pid}")


# Gunicorn settings
bind = f"0.0.0.0:{os.getenv('PORT', '5001')}"
workers = 1
threads = 4
timeout = 120
preload_app = True
