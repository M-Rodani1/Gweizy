"""
On-Chain Features Collection Service

Collects blockchain metrics that enhance gas price predictions.
Runs every 5 minutes alongside gas price collection.
"""

import os
import sys
import time
import logging
from datetime import datetime
import signal
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import BaseGasCollector
from data.database import DatabaseManager
from utils.onchain_features import OnChainFeatureExtractor
from web3 import Web3
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/onchain_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OnChainCollectorService:
    """Background service for on-chain metrics collection"""

    def __init__(self, interval_seconds=300):
        self.interval = interval_seconds
        self.w3 = Web3(Web3.HTTPProvider(Config.BASE_RPC_URL))
        self.db = DatabaseManager()
        self.feature_extractor = OnChainFeatureExtractor(self.w3)
        self.running = False
        self.collection_count = 0
        self.error_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def collect_onchain_features(self):
        """Collect on-chain metrics for current block"""
        try:
            # Get latest block
            block = self.w3.eth.get_block('latest', full_transactions=True)
            block_number = block['number']
            timestamp = datetime.fromtimestamp(block['timestamp'])

            # Transaction volume
            tx_count = len(block['transactions'])

            # Gas usage
            gas_used = block.get('gasUsed', 0)
            gas_limit = block.get('gasLimit', 1)
            gas_utilization = (gas_used / gas_limit) * 100 if gas_limit > 0 else 0

            # Base fee
            base_fee_wei = block.get('baseFeePerGas', 0)
            base_fee_gwei = base_fee_wei / 1e9

            # Analyze transactions
            contract_calls = 0
            total_gas_price = 0
            priority_fees = []
            transfer_count = 0

            for tx in block['transactions']:
                # Check if contract call (has data)
                if tx.get('input', '0x') != '0x':
                    contract_calls += 1
                else:
                    transfer_count += 1

                # Collect priority fees
                if 'maxPriorityFeePerGas' in tx:
                    priority_fees.append(tx['maxPriorityFeePerGas'])

                # Gas price
                gas_price = tx.get('gasPrice', 0)
                total_gas_price += gas_price

            # Calculate metrics
            avg_gas_price = (total_gas_price / len(block['transactions']) / 1e9) if block['transactions'] else 0
            avg_priority_fee = (sum(priority_fees) / len(priority_fees) / 1e9) if priority_fees else 0
            contract_ratio = contract_calls / tx_count if tx_count > 0 else 0

            # Network congestion score (0-100)
            congestion_score = min(100, (
                gas_utilization * 0.4 +
                (tx_count / 100) * 0.3 +
                (avg_gas_price / 0.01) * 0.3
            ))

            # Extract enhanced congestion features (Week 1 Quick Win #2)
            # These features explain 27% of gas price variance
            enhanced_features = self.feature_extractor.extract_enhanced_congestion_features(block_number)
            
            # Save to database
            features = {
                'block_number': block_number,
                'timestamp': timestamp.isoformat(),
                'tx_count': tx_count,
                'gas_used': gas_used,
                'gas_limit': gas_limit,
                'gas_utilization': gas_utilization,
                'base_fee_gwei': base_fee_gwei,
                'avg_gas_price_gwei': avg_gas_price,
                'avg_priority_fee_gwei': avg_priority_fee,
                'contract_calls': contract_calls,
                'transfers': transfer_count,
                'contract_call_ratio': contract_ratio,
                'congestion_score': congestion_score,
                'block_time': 2.0,  # Base L2 ~2 second blocks
                
                # Enhanced congestion features
                'pending_tx_count': enhanced_features.get('pending_tx_count', 0),
                'unique_senders': enhanced_features.get('unique_senders', 0),
                'unique_receivers': enhanced_features.get('unique_receivers', 0),
                'unique_addresses': enhanced_features.get('unique_addresses', 0),
                'tx_per_second': enhanced_features.get('tx_per_second', 0.0),
                'gas_utilization_ratio': enhanced_features.get('gas_utilization_ratio', 0.0),
                'avg_tx_gas': enhanced_features.get('avg_tx_gas', 0.0),
                'large_tx_ratio': enhanced_features.get('large_tx_ratio', 0.0),
                'congestion_level': enhanced_features.get('congestion_level', 0),
                'is_highly_congested': 1 if enhanced_features.get('is_highly_congested', False) else 0,
            }

            self.db.save_onchain_features(features)

            return features

        except Exception as e:
            logger.error(f"Error collecting on-chain features: {e}")
            logger.error(traceback.format_exc())
            return None

    def start(self):
        """Start the collection service"""
        logger.info("="*60)
        logger.info("On-Chain Features Collection Service Starting")
        logger.info(f"Collection interval: {self.interval} seconds")
        logger.info(f"Network: Base (Chain ID: {Config.BASE_CHAIN_ID})")
        logger.info("="*60)

        self.running = True

        while self.running:
            try:
                start_time = time.time()
                features = self.collect_onchain_features()

                if features:
                    self.collection_count += 1
                    elapsed = time.time() - start_time

                    logger.info(
                        f"✓ Collection #{self.collection_count}: "
                        f"Block {features['block_number']} - "
                        f"{features['tx_count']} txs, "
                        f"{features['gas_utilization']:.1f}% util, "
                        f"congestion: {features['congestion_score']:.1f} "
                        f"[{elapsed:.2f}s]"
                    )

                    # Log stats every hour
                    if self.collection_count % 12 == 0:
                        self._log_stats()
                else:
                    self.error_count += 1
                    logger.warning(f"Failed to collect on-chain features (error #{self.error_count})")

                if self.running:
                    time.sleep(self.interval)

            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in collection loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.interval)

    def stop(self):
        """Stop the service"""
        logger.info("Stopping on-chain collection service...")
        self.running = False
        self._log_stats()
        logger.info("Service stopped")

    def _log_stats(self):
        """Log statistics"""
        try:
            # Get recent records
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*), AVG(tx_count), AVG(gas_utilization), AVG(congestion_score)
                FROM onchain_features
                WHERE timestamp > datetime('now', '-24 hours')
            """)

            count, avg_tx, avg_util, avg_cong = cursor.fetchone()

            logger.info("="*60)
            logger.info("24-Hour On-Chain Statistics:")
            logger.info(f"  Total collections: {self.collection_count}")
            logger.info(f"  Last 24h records: {count or 0}")
            logger.info(f"  Avg transactions/block: {avg_tx or 0:.1f}")
            logger.info(f"  Avg gas utilization: {avg_util or 0:.1f}%")
            logger.info(f"  Avg congestion score: {avg_cong or 0:.1f}")
            logger.info(f"  Error count: {self.error_count}")
            logger.info("="*60)

            conn.close()
        except Exception as e:
            logger.warning(f"Could not generate stats: {e}")


def main():
    os.makedirs('logs', exist_ok=True)
    interval = int(os.getenv('COLLECTION_INTERVAL', 300))

    service = OnChainCollectorService(interval_seconds=interval)

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
