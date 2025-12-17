"""
Continuous Gas Price Collection Service

Runs in the background collecting gas prices every 5 minutes.
Designed to run as a separate process on Render or as a systemd service.
"""

import os
import sys
import time
import logging
from datetime import datetime
import signal
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import BaseGasCollector
from data.database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gas_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GasCollectorService:
    """Background service for continuous gas price collection"""

    def __init__(self, interval_seconds=300):
        """
        Args:
            interval_seconds: Collection interval (default 300 = 5 minutes)
        """
        self.interval = interval_seconds
        self.collector = BaseGasCollector()
        self.db = DatabaseManager()
        self.running = False
        self.collection_count = 0
        self.error_count = 0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def start(self):
        """Start the collection service"""
        logger.info("="*60)
        logger.info("Gas Price Collection Service Starting")
        logger.info(f"Collection interval: {self.interval} seconds ({self.interval/60:.1f} minutes)")
        logger.info("="*60)

        self.running = True

        while self.running:
            try:
                # Collect gas price
                start_time = time.time()
                data = self.collector.get_current_gas()

                if data:
                    # Save to database
                    self.db.save_gas_price(data)
                    self.collection_count += 1

                    elapsed = time.time() - start_time
                    logger.info(
                        f"✓ Collection #{self.collection_count}: "
                        f"{data['current_gas']:.6f} Gwei "
                        f"(base: {data['base_fee']:.6f}, priority: {data['priority_fee']:.6f}) "
                        f"[{elapsed:.2f}s]"
                    )

                    # Log stats every 12 collections (1 hour)
                    if self.collection_count % 12 == 0:
                        self._log_stats()
                else:
                    self.error_count += 1
                    logger.warning(f"Failed to collect gas price (error #{self.error_count})")

                # Sleep until next collection
                if self.running:
                    time.sleep(self.interval)

            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in collection loop: {e}")
                logger.error(traceback.format_exc())

                # Back off on repeated errors
                if self.error_count > 5:
                    logger.warning(f"Multiple errors ({self.error_count}), increasing backoff...")
                    time.sleep(self.interval * 2)
                else:
                    time.sleep(self.interval)

    def stop(self):
        """Stop the collection service"""
        logger.info("Stopping gas price collection service...")
        self.running = False
        self._log_stats()
        logger.info("Service stopped gracefully")

    def _log_stats(self):
        """Log collection statistics"""
        try:
            # Get recent data from database
            recent = self.db.get_historical_data(hours=24)

            if recent:
                gas_prices = [d.get('current_gas', 0) for d in recent]

                stats = {
                    'count_24h': len(gas_prices),
                    'min': min(gas_prices),
                    'max': max(gas_prices),
                    'avg': sum(gas_prices) / len(gas_prices)
                }

                logger.info("="*60)
                logger.info("24-Hour Statistics:")
                logger.info(f"  Total collections: {self.collection_count}")
                logger.info(f"  Last 24h records: {stats['count_24h']}")
                logger.info(f"  Gas price range: {stats['min']:.6f} - {stats['max']:.6f} Gwei")
                logger.info(f"  Average: {stats['avg']:.6f} Gwei")
                logger.info(f"  Error count: {self.error_count}")
                logger.info("="*60)
        except Exception as e:
            logger.warning(f"Could not generate stats: {e}")

    def health_check(self):
        """Check service health"""
        return {
            'running': self.running,
            'collections': self.collection_count,
            'errors': self.error_count,
            'error_rate': self.error_count / max(self.collection_count, 1),
            'uptime': 'running' if self.running else 'stopped'
        }


def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Get interval from environment or use default (5 minutes)
    interval = int(os.getenv('COLLECTION_INTERVAL', 300))

    # Create and start service
    service = GasCollectorService(interval_seconds=interval)

    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        service.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        service.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
