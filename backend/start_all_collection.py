#!/usr/bin/env python3
"""
Start All Data Collection Services

Starts both normal data collection (gas prices + onchain features) 
and mempool data collection in parallel.

Usage:
    python start_all_collection.py

This will:
1. Start normal data collection (gas prices + onchain features)
2. Start mempool data collection
3. Run both until you press Ctrl+C
"""

import os
import sys
import time
import signal
import threading
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.gas_collector_service import GasCollectorService
from services.onchain_collector_service import OnChainCollectorService
from data.mempool_collector import get_mempool_collector
from config import Config
from utils.logger import logger

# Global service references for signal handling
gas_service = None
onchain_service = None
mempool_collector = None
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info(f"\n🛑 Received signal {signum}, shutting down all collection services...")
    running = False
    
    if mempool_collector:
        try:
            mempool_collector.stop_background_collection()
            logger.info("✓ Mempool collection stopped")
        except Exception as e:
            logger.warning(f"Error stopping mempool collector: {e}")
    
    if gas_service:
        try:
            gas_service.stop()
            logger.info("✓ Gas collection stopped")
        except Exception as e:
            logger.warning(f"Error stopping gas service: {e}")
    
    if onchain_service:
        try:
            onchain_service.stop()
            logger.info("✓ Onchain collection stopped")
        except Exception as e:
            logger.warning(f"Error stopping onchain service: {e}")
    
    logger.info("✅ All collection services stopped")
    sys.exit(0)

def start_normal_collection():
    """Start normal data collection (gas + onchain)"""
    global gas_service, onchain_service
    
    logger.info("="*80)
    logger.info("STARTING NORMAL DATA COLLECTION")
    logger.info("="*80)
    
    # Initialize services
    interval = int(os.getenv('COLLECTION_INTERVAL', 300))
    gas_service = GasCollectorService(interval_seconds=interval)
    onchain_service = OnChainCollectorService(interval_seconds=interval)
    
    # Start gas collection in separate thread
    gas_thread = threading.Thread(
        target=gas_service.start,
        name="GasCollector",
        daemon=True
    )
    gas_thread.start()
    logger.info("✓ Gas price collection thread started")
    
    # Start on-chain collection in separate thread
    onchain_thread = threading.Thread(
        target=onchain_service.start,
        name="OnChainCollector",
        daemon=True
    )
    onchain_thread.start()
    logger.info("✓ On-chain features collection thread started")
    
    logger.info("="*80)

def start_mempool_collection():
    """Start mempool data collection"""
    global mempool_collector
    
    logger.info("="*80)
    logger.info("STARTING MEMPOOL DATA COLLECTION")
    logger.info("="*80)
    
    try:
        mempool_collector = get_mempool_collector(timeout=10.0)
        if mempool_collector:
            mempool_collector.start_background_collection()
            logger.info("✓ Mempool data collection started")
            logger.info(f"   Collection interval: {mempool_collector.collection_interval} seconds")
        else:
            logger.warning("⚠️  Mempool collector initialization failed - will retry")
            return False
    except Exception as e:
        logger.error(f"❌ Could not start mempool collector: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("="*80)
    return True

def main():
    """Main entry point"""
    global running
    
    # Configure logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*80)
    logger.info("🚀 STARTING ALL DATA COLLECTION SERVICES")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Collection interval: {os.getenv('COLLECTION_INTERVAL', '300')} seconds")
    logger.info("="*80)
    logger.info("")
    
    # Start normal data collection
    start_normal_collection()
    
    # Start mempool collection
    mempool_started = start_mempool_collection()
    
    logger.info("")
    logger.info("="*80)
    logger.info("✅ ALL COLLECTION SERVICES RUNNING")
    logger.info("="*80)
    logger.info("Services active:")
    logger.info("  ✓ Gas price collection")
    logger.info("  ✓ On-chain features collection")
    if mempool_started:
        logger.info("  ✓ Mempool data collection")
    else:
        logger.info("  ⚠️  Mempool data collection (failed to start)")
    logger.info("")
    logger.info("Press Ctrl+C to stop all services")
    logger.info("="*80)
    logger.info("")
    
    # Keep main thread alive
    try:
        while running:
            time.sleep(60)
            # Periodic health check
            if gas_service and onchain_service:
                try:
                    gas_health = gas_service.health_check()
                    onchain_health = onchain_service.health_check()
                    logger.info(
                        f"Health Check - Gas: {gas_health['collections']} collections "
                        f"({gas_health['errors']} errors), "
                        f"OnChain: {onchain_health['collections']} collections "
                        f"({onchain_health['errors']} errors)"
                    )
                    if mempool_collector:
                        logger.info(
                            f"  Mempool: {len(mempool_collector.snapshot_history)} snapshots in memory"
                        )
                except Exception as e:
                    logger.debug(f"Health check error: {e}")
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        signal_handler(None, None)

if __name__ == '__main__':
    main()
