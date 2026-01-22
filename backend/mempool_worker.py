#!/usr/bin/env python3
"""
Mempool Monitoring Worker

Asynchronous background worker that monitors pending transactions via WebSocket
and aggregates them into per-second statistics for leading indicator analysis.

Usage:
    python3 backend/mempool_worker.py
"""

import asyncio
import websockets
import json
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
BASE_RPC_WSS = os.getenv('BASE_RPC_WSS', 'wss://mainnet.base.org')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///gas_data.db')

# If BASE_RPC_WSS not set, try to derive from BASE_RPC_URL
if BASE_RPC_WSS == 'wss://mainnet.base.org':
    BASE_RPC_URL = os.getenv('BASE_RPC_URL', 'https://mainnet.base.org')
    if BASE_RPC_URL.startswith('https://'):
        BASE_RPC_WSS = BASE_RPC_URL.replace('https://', 'wss://')
    elif BASE_RPC_URL.startswith('http://'):
        BASE_RPC_WSS = BASE_RPC_URL.replace('http://', 'ws://')
    logger.info(f"Derived WebSocket URL from BASE_RPC_URL: {BASE_RPC_WSS}")

# Database connection
engine = None


def init_db():
    """Initialize database schema for mempool_stats table."""
    global engine
    
    # Create engine with appropriate pool class
    if DATABASE_URL.startswith('sqlite'):
        engine = create_engine(
            DATABASE_URL,
            poolclass=NullPool,
            connect_args={'check_same_thread': False, 'timeout': 30}
        )
    else:
        # PostgreSQL or other
        engine = create_engine(DATABASE_URL, poolclass=NullPool)
    
    # Create table if it doesn't exist
    with engine.connect() as conn:
        # SQLite/PostgreSQL compatible CREATE TABLE
        if DATABASE_URL.startswith('sqlite'):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mempool_stats (
                    timestamp DATETIME PRIMARY KEY,
                    tx_count INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_stats(timestamp)
            """))
        else:
            # PostgreSQL syntax
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mempool_stats (
                    timestamp TIMESTAMP PRIMARY KEY,
                    tx_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_stats(timestamp)
            """))
        conn.commit()
    
    logger.info("✅ Database schema initialized for mempool_stats")


def save_mempool_stats(tx_count: int, timestamp: datetime):
    """Save mempool statistics to database."""
    global engine
    
    if engine is None:
        logger.error("Database engine not initialized")
        return False
    
    try:
        with engine.connect() as conn:
            if DATABASE_URL.startswith('sqlite'):
                conn.execute(text("""
                    INSERT OR REPLACE INTO mempool_stats (timestamp, tx_count, created_at)
                    VALUES (:timestamp, :tx_count, :created_at)
                """), {
                    'timestamp': timestamp,
                    'tx_count': tx_count,
                    'created_at': datetime.now()
                })
            else:
                # PostgreSQL syntax
                conn.execute(text("""
                    INSERT INTO mempool_stats (timestamp, tx_count, created_at)
                    VALUES (:timestamp, :tx_count, :created_at)
                    ON CONFLICT (timestamp) DO UPDATE SET
                        tx_count = EXCLUDED.tx_count,
                        created_at = EXCLUDED.created_at
                """), {
                    'timestamp': timestamp,
                    'tx_count': tx_count,
                    'created_at': datetime.now()
                })
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to save mempool stats: {e}")
        return False


async def mempool_monitor():
    """Main async function to monitor mempool via WebSocket."""
    pending_tx_count = 0
    last_write_time = None
    subscription_id = None
    
    while True:
        try:
            logger.info(f"Connecting to WebSocket: {BASE_RPC_WSS}")
            
            async with websockets.connect(
                BASE_RPC_WSS,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                logger.info("✅ WebSocket connected")
                
                # Subscribe to new pending transactions
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions"]
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                response = await websocket.recv()
                result = json.loads(response)
                
                if 'result' in result:
                    subscription_id = result['result']
                    logger.info(f"✅ Subscribed to pending transactions (subscription ID: {subscription_id})")
                else:
                    logger.error(f"Failed to subscribe: {response}")
                    await asyncio.sleep(5)
                    continue
                
                # Main monitoring loop
                while True:
                    try:
                        # Set timeout for receiving messages
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Check if this is a subscription notification
                        if 'params' in data and 'subscription' in data['params']:
                            if data['params']['subscription'] == subscription_id:
                                # This is a pending transaction notification
                                pending_tx_count += 1
                        
                        # Write stats every 1 second
                        now = datetime.now()
                        if last_write_time is None or (now - last_write_time).total_seconds() >= 1.0:
                            if save_mempool_stats(pending_tx_count, now):
                                logger.debug(f"Saved mempool stats: {pending_tx_count} txs at {now}")
                            pending_tx_count = 0  # Reset counter
                            last_write_time = now
                    
                    except asyncio.TimeoutError:
                        # Timeout is expected - check if we need to write stats
                        now = datetime.now()
                        if last_write_time is None or (now - last_write_time).total_seconds() >= 1.0:
                            # Write even if no transactions (0 count)
                            if save_mempool_stats(pending_tx_count, now):
                                logger.debug(f"Saved mempool stats (timeout): {pending_tx_count} txs at {now}")
                            pending_tx_count = 0
                            last_write_time = now
                        continue
                    
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break
                    
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
        
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {BASE_RPC_WSS}")
            logger.error("Please set BASE_RPC_WSS environment variable to a valid WebSocket URL")
            sys.exit(1)
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            # Reset state for reconnection
            pending_tx_count = 0
            last_write_time = None
            subscription_id = None


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Mempool Monitoring Worker")
    logger.info("=" * 60)
    logger.info(f"WebSocket URL: {BASE_RPC_WSS}")
    logger.info(f"Database URL: {DATABASE_URL}")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Start monitoring
    logger.info("Starting mempool monitoring...")
    await mempool_monitor()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down mempool worker...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
