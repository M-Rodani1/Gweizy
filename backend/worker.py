#!/usr/bin/env python3
"""
Background Worker for Data Collection
Runs independently from the Flask API server
"""

import os
import sys
import time
import signal
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService
from config import Config
from utils.logger import logger


class DataCollectionWorker:
    """Worker process that runs data collection services"""

    def __init__(self):
        self.running = False
        self.gas_service = None
        self.onchain_service = None

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down worker...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start both collection services"""
        logger.info("="*60)
        logger.info("DATA COLLECTION WORKER STARTING")
        logger.info("="*60)
        logger.info(f"Collection interval: {Config.COLLECTION_INTERVAL} seconds")
        logger.info(f"Environment: {'Production' if not Config.DEBUG else 'Development'}")
        logger.info("="*60)

        self.running = True

        # Initialize services
        self.gas_service = GasCollectorService(Config.COLLECTION_INTERVAL)
        self.onchain_service = OnChainCollectorService(Config.COLLECTION_INTERVAL)

        logger.info("✓ Services initialized")
        logger.info("Starting collection loops...")

        # Run both services in alternating pattern
        # This is more reliable than threads in a worker process
        last_collection = 0

        try:
            while self.running:
                current_time = time.time()

                # Check if it's time to collect
                if current_time - last_collection >= Config.COLLECTION_INTERVAL:
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting collection cycle")

                    # Collect gas prices
                    try:
                        gas_data = self._collect_gas_prices()
                        if gas_data:
                            logger.info(f"  ✓ Gas: {gas_data['current_gas']:.6f} gwei")
                    except Exception as e:
                        logger.error(f"  ✗ Gas collection failed: {e}")

                    # Collect onchain features
                    try:
                        onchain_data = self._collect_onchain_features()
                        if onchain_data:
                            logger.info(f"  ✓ OnChain: Block {onchain_data['block_number']}, {onchain_data['tx_count']} txs")
                    except Exception as e:
                        logger.error(f"  ✗ OnChain collection failed: {e}")

                    last_collection = current_time

                # Sleep for a short interval to avoid busy waiting
                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.stop()

    def _collect_gas_prices(self):
        """Collect gas prices using the service"""
        return self.gas_service.collect_gas_prices()

    def _collect_onchain_features(self):
        """Collect onchain features using the service"""
        return self.onchain_service.collect_onchain_features()

    def stop(self):
        """Stop the worker"""
        logger.info("Stopping data collection worker...")
        self.running = False

        if self.gas_service:
            self.gas_service.stop()
        if self.onchain_service:
            self.onchain_service.stop()

        logger.info("Worker stopped")


def main():
    """Main entry point"""
    # Check if data collection is enabled
    if os.getenv('ENABLE_DATA_COLLECTION', 'true').lower() != 'true':
        logger.info("Data collection is disabled (ENABLE_DATA_COLLECTION=false)")
        logger.info("Worker exiting...")
        return

    logger.info("Starting data collection worker process...")

    worker = DataCollectionWorker()
    worker.start()


if __name__ == '__main__':
    main()
