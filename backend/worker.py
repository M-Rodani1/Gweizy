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
from web3 import Web3
from web3.exceptions import BlockNotFound

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService
from config import Config
from utils.logger import logger
from utils.rpc_manager import get_rpc_manager


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
        """Start both collection services with block-based polling for Base L2 (2-second blocks)"""
        logger.info("="*60)
        logger.info("DATA COLLECTION WORKER STARTING")
        logger.info("="*60)
        logger.info("Collection mode: Block-based polling (Base L2 - ~2s blocks)")
        logger.info("Polling interval: 0.5 seconds")
        logger.info(f"Environment: {'Production' if not Config.DEBUG else 'Development'}")
        logger.info("="*60)

        self.running = True

        # Initialize services
        self.gas_service = GasCollectorService(Config.COLLECTION_INTERVAL)
        self.onchain_service = OnChainCollectorService(Config.COLLECTION_INTERVAL)

        # Initialize Web3 connection for block monitoring
        self.rpc_manager = get_rpc_manager()
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_manager.get_current_rpc()))
        
        # Track last processed block to avoid duplicates
        last_processed_block = None
        
        # Poll interval (0.5s for 2s block time)
        poll_interval = 0.5

        logger.info("✓ Services initialized")
        logger.info("Starting block-based collection loop...")

        try:
            # Initialize last_processed_block
            try:
                last_processed_block = self.w3.eth.block_number
                logger.info(f"Starting from block: {last_processed_block}")
            except Exception as e:
                logger.warning(f"Could not get initial block number: {e}, will start on first new block")

            while self.running:
                try:
                    # Poll for new block
                    current_block = self.w3.eth.block_number
                    
                    # Only process if we have a new block
                    if last_processed_block is None or current_block > last_processed_block:
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] New block detected: {current_block}")
                        
                        # Collect gas prices for this specific block
                        try:
                            gas_data = self._collect_gas_prices(block_number=current_block)
                            if gas_data:
                                utilization = gas_data.get('utilization', 0)
                                logger.info(
                                    f"  ✓ [Block {current_block}] "
                                    f"Gas: {gas_data['current_gas']:.6f} gwei | "
                                    f"Util: {utilization:.1f}%"
                                )
                        except BlockNotFound:
                            logger.debug(f"  Block {current_block} not found yet, will retry")
                        except Exception as e:
                            logger.error(f"  ✗ Gas collection failed: {e}")

                        # Collect onchain features for this block
                        try:
                            onchain_data = self._collect_onchain_features(block_number=current_block)
                            if onchain_data:
                                # Already logged in gas collection, just confirm
                                pass
                        except BlockNotFound:
                            logger.debug(f"  Block {current_block} not found yet for onchain features")
                        except Exception as e:
                            logger.error(f"  ✗ OnChain collection failed: {e}")

                        # Update last processed block
                        last_processed_block = current_block
                    else:
                        # No new block yet, just wait
                        time.sleep(poll_interval)
                        continue

                except BlockNotFound:
                    # Sometimes the node lags slightly behind the tip - this is normal
                    logger.debug("Block not found (node may be catching up), will retry")
                    time.sleep(poll_interval)
                    continue
                except Exception as e:
                    logger.error(f"Error in block polling loop: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    time.sleep(poll_interval)
                    continue

                # Small sleep between checks to avoid hammering the RPC
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.stop()

    def _collect_gas_prices(self, block_number: int = None):
        """Collect gas prices using the service for a specific block"""
        # Update collector's Web3 instance if needed
        current_rpc = self.rpc_manager.get_current_rpc()
        if self.gas_service.collector.w3.provider.endpoint_uri != current_rpc:
            self.gas_service.collector.w3 = Web3(Web3.HTTPProvider(current_rpc))
        
        # Use the service method which handles saving to database
        return self.gas_service.collect_gas_prices(block_number=block_number)

    def _collect_onchain_features(self, block_number: int = None):
        """Collect onchain features using the service for a specific block"""
        # For onchain service, we'd need to modify it similarly
        # For now, it will use latest block, but we can enhance it later
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
