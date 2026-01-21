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

    def __init__(self, interval_seconds=300, register_signals=True, socketio=None):
        """
        Args:
            interval_seconds: Collection interval (default 300 = 5 minutes)
            register_signals: Whether to register signal handlers (False for background threads)
            socketio: Optional SocketIO instance for real-time updates
        """
        self.interval = interval_seconds
        self.collector = BaseGasCollector()
        self.db = DatabaseManager()
        self.running = False
        self.collection_count = 0
        self.error_count = 0
        self.socketio = socketio
        self.start_time = None
        self.last_collection_time = None
        self.successful_collections = 0
    
    def collect_gas_prices(self, block_number: int = None):
        """
        Collect gas prices and save to database.
        Can be called manually or by worker.
        
        Args:
            block_number: Optional specific block number to collect (for block-based polling)
        
        Returns:
            dict: Gas price data or None if failed
        """
        try:
            data = self.collector.get_current_gas(block_number=block_number)
            if data:
                self.db.save_gas_price(data)
                self.collection_count += 1
                self.successful_collections += 1
                self.last_collection_time = time.time()
            return data
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in collect_gas_prices: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        # Register signal handlers for graceful shutdown (only in main thread)
        if register_signals:
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except ValueError:
                # Signals can only be registered in the main thread
                logger.debug("Skipping signal registration (not in main thread)")

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
        self.start_time = time.time()

        while self.running:
            try:
                # Collect gas price
                start_time = time.time()
                data = self.collector.get_current_gas()

                if data:
                    # Save to database
                    self.db.save_gas_price(data)
                    self.collection_count += 1
                    self.successful_collections += 1
                    self.last_collection_time = time.time()

                    elapsed = time.time() - start_time
                    utilization = data.get('utilization', 0)
                    block_num = data.get('block_number', '?')
                    logger.info(
                        f"✓ Collection #{self.collection_count}: "
                        f"[Block {block_num}] "
                        f"Gas: {data['current_gas']:.6f} gwei | "
                        f"Util: {utilization:.1f}% | "
                        f"(base: {data['base_fee']:.6f}, priority: {data['priority_fee']:.6f}) "
                        f"[{elapsed:.2f}s]"
                    )

                    # Emit WebSocket updates with predictions and mempool
                    if self.socketio:
                        self._emit_realtime_updates(data)

                    # Log stats every 144 collections (12 minutes for 5s interval, ~1 hour for 15s)
                    # Adjusted for higher frequency collection
                    log_interval = 144 if self.interval <= 5 else 12
                    if self.collection_count % log_interval == 0:
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

    def get_collection_stats(self) -> dict:
        """Get collection service statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        success_rate = (
            (self.successful_collections / self.collection_count * 100)
            if self.collection_count > 0 else 0
        )
        
        # Calculate expected vs actual collections
        expected_collections = int(uptime / self.interval) if self.interval > 0 else 0
        collection_rate = (self.collection_count / (uptime / 3600)) if uptime > 0 else 0  # per hour
        
        return {
            'interval_seconds': self.interval,
            'running': self.running,
            'uptime_seconds': int(uptime),
            'total_collections': self.collection_count,
            'successful_collections': self.successful_collections,
            'error_count': self.error_count,
            'success_rate_percent': round(success_rate, 2),
            'expected_collections': expected_collections,
            'collection_rate_per_hour': round(collection_rate, 2),
            'last_collection_time': self.last_collection_time,
            'last_collection_ago_seconds': int(time.time() - self.last_collection_time) if self.last_collection_time else None
        }
    
    def _log_stats(self):
        """Log collection statistics"""
        try:
            # Get collection stats
            stats = self.get_collection_stats()
            logger.info(f"Collection Stats: {stats['total_collections']} total, "
                       f"{stats['success_rate_percent']:.1f}% success rate, "
                       f"{stats['collection_rate_per_hour']:.1f} collections/hour")
            
            # Get recent data from database
            recent = self.db.get_historical_data(hours=24)

            if recent:
                gas_prices = [d.get('current_gas', 0) for d in recent]

                db_stats = {
                    'count_24h': len(gas_prices),
                    'min': min(gas_prices),
                    'max': max(gas_prices),
                    'avg': sum(gas_prices) / len(gas_prices)
                }

                logger.info("="*60)
                logger.info("24-Hour Statistics:")
                logger.info(f"  Total collections: {self.collection_count}")
                logger.info(f"  Success rate: {stats['success_rate_percent']:.1f}%")
                logger.info(f"  Collection rate: {stats['collection_rate_per_hour']:.1f}/hour")
                logger.info(f"  Last 24h records: {db_stats['count_24h']}")
                logger.info(f"  Gas price range: {db_stats['min']:.6f} - {db_stats['max']:.6f} Gwei")
                logger.info(f"  Average: {db_stats['avg']:.6f} Gwei")
                logger.info(f"  Error count: {self.error_count}")
                logger.info("="*60)
        except Exception as e:
            logger.warning(f"Could not generate stats: {e}")

    def _emit_realtime_updates(self, gas_data):
        """Emit comprehensive real-time updates via WebSocket."""
        try:
            from services.websocket_events import emit_combined_update, emit_gas_update

            # Always emit gas update
            emit_gas_update(gas_data)

            # Get predictions (every collection)
            predictions = None
            mempool_status = None

            try:
                from models.ensemble_predictor import get_ensemble_predictor
                import pandas as pd

                # Get recent data for predictions
                recent = self.db.get_historical_data(hours=24)
                if recent and len(recent) >= 10:
                    df = pd.DataFrame(recent)
                    predictor = get_ensemble_predictor()
                    predictions = predictor.predict(df)
            except Exception as pred_error:
                logger.debug(f"Could not get predictions for WebSocket: {pred_error}")

            # Get mempool status
            try:
                from data.mempool_collector import get_mempool_collector
                collector = get_mempool_collector()
                features = collector.get_current_features()
                mempool_status = {
                    'pending_count': features.get('mempool_pending_count', 0),
                    'avg_gas_price': features.get('mempool_avg_gas_price', 0),
                    'is_congested': features.get('mempool_is_congested', 0) > 0,
                    'gas_momentum': features.get('mempool_gas_price_momentum', 0),
                    'count_momentum': features.get('mempool_count_momentum', 0)
                }
            except Exception as mp_error:
                logger.debug(f"Could not get mempool for WebSocket: {mp_error}")

            # Emit combined update
            emit_combined_update(gas_data, predictions, mempool_status)

            logger.debug("Real-time WebSocket updates emitted")

        except Exception as e:
            logger.warning(f"Failed to emit real-time updates: {e}")

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
