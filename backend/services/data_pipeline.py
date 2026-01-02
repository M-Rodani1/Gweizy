"""
Master Data Collection Pipeline

Orchestrates both gas price and on-chain features collection.
Runs both services in parallel threads.
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Master pipeline orchestrating all data collection"""

    def __init__(self, interval_seconds=300):
        self.interval = interval_seconds
        self.gas_service = GasCollectorService(interval_seconds)
        self.onchain_service = OnChainCollectorService(interval_seconds)
        self.threads = []

    def start(self):
        """Start all collection services"""
        logger.info("="*80)
        logger.info("DATA COLLECTION PIPELINE STARTING")
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info(f"Collection interval: {self.interval} seconds ({self.interval/60:.1f} minutes)")
        logger.info("="*80)

        # Start gas price collection in separate thread
        gas_thread = threading.Thread(
            target=self.gas_service.start,
            name="GasCollector",
            daemon=True
        )
        gas_thread.start()
        self.threads.append(gas_thread)
        logger.info("✓ Gas price collection thread started")

        # Start on-chain collection in separate thread
        onchain_thread = threading.Thread(
            target=self.onchain_service.start,
            name="OnChainCollector",
            daemon=True
        )
        onchain_thread.start()
        self.threads.append(onchain_thread)
        logger.info("✓ On-chain features collection thread started")

        logger.info("="*80)
        logger.info("All services running. Press Ctrl+C to stop.")
        logger.info("="*80)

        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
                self._health_check()
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received")
            self.stop()

    def stop(self):
        """Stop all services"""
        logger.info("Stopping all collection services...")

        self.gas_service.stop()
        self.onchain_service.stop()

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("All services stopped successfully")

    def _health_check(self):
        """Periodic health check"""
        gas_health = self.gas_service.health_check()
        onchain_health = self.onchain_service.health_check()

        logger.info(
            f"Health Check - Gas: {gas_health['collections']} collections "
            f"({gas_health['errors']} errors), "
            f"OnChain: {onchain_health['collections']} collections "
            f"({onchain_health['errors']} errors)"
        )


def main():
    """Main entry point"""
    os.makedirs('logs', exist_ok=True)

    interval = int(os.getenv('COLLECTION_INTERVAL', 300))
    pipeline = DataPipeline(interval_seconds=interval)

    try:
        pipeline.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pipeline.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
