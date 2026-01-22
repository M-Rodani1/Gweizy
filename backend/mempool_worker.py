#!/usr/bin/env python3
"""
Mempool Monitoring Worker

Asynchronous background worker that connects to Alchemy WebSocket to monitor
pending transactions as a leading indicator of gas price congestion.

Features:
- WebSocket connection to Alchemy Base mainnet
- Subscribes to alchemy_pendingTransactions
- Aggregates transaction counts in memory
- Flushes to database every 1 second
- Auto-reconnect with exponential backoff
"""

import asyncio
import websockets
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional
import os

logger = logging.getLogger(__name__)

# Alchemy WebSocket URL (hardcoded as requested)
ALCHEMY_WS_URL = "wss://base-mainnet.g.alchemy.com/v2/Rt-_YiDduM0YxHOSJJ_Yg"

# Database path - prefer root gas_data.db, fallback to backend/gas_data.db
DB_PATH = 'gas_data.db'
if not os.path.exists(DB_PATH):
    DB_PATH = 'backend/gas_data.db'
if not os.path.exists(DB_PATH):
    DB_PATH = 'gas_data.db'  # Will create if doesn't exist


def init_db():
    """
    Initialize database schema for mempool_stats table.
    Creates the table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mempool_stats (
                timestamp DATETIME PRIMARY KEY,
                tx_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mempool_stats_timestamp 
            ON mempool_stats(timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Initialized mempool_stats table in {DB_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def save_mempool_count(tx_count: int, timestamp: datetime):
    """
    Save mempool transaction count to database.
    
    Args:
        tx_count: Number of pending transactions
        timestamp: Timestamp for this measurement
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Use timestamp as primary key, so we'll use INSERT OR REPLACE
        # to handle potential duplicates
        cursor.execute("""
            INSERT OR REPLACE INTO mempool_stats (timestamp, tx_count, created_at)
            VALUES (?, ?, ?)
        """, (timestamp.isoformat(), tx_count, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.debug(f"💾 Saved mempool count: {tx_count} at {timestamp.isoformat()}")
    except Exception as e:
        logger.error(f"❌ Failed to save mempool count: {e}")


class MempoolWorker:
    """
    Asynchronous worker that monitors mempool via Alchemy WebSocket.
    """
    
    def __init__(self, ws_url: str = ALCHEMY_WS_URL, flush_interval: float = 1.0):
        """
        Initialize mempool worker.
        
        Args:
            ws_url: WebSocket URL for Alchemy
            flush_interval: Seconds between database flushes (default 1.0)
        """
        self.ws_url = ws_url
        self.flush_interval = flush_interval
        self.pending_tx_count = 0
        self.running = False
        self.ws = None  # WebSocket connection
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Connect to Alchemy WebSocket and subscribe to pending transactions."""
        try:
            logger.info(f"🔌 Connecting to Alchemy WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            logger.info("✅ Connected to Alchemy WebSocket")
            
            # Subscribe to pending transactions
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["alchemy_pendingTransactions"]
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info("📡 Subscribed to alchemy_pendingTransactions")
            
            # Wait for subscription confirmation
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            if 'result' in response_data:
                subscription_id = response_data['result']
                logger.info(f"✅ Subscription confirmed: {subscription_id}")
                return True
            else:
                logger.error(f"❌ Subscription failed: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message.
        Increments pending transaction counter.
        
        Args:
            message: JSON string message from WebSocket
        """
        try:
            data = json.loads(message)
            
            # Check if this is a pending transaction notification from Alchemy
            # Format: {"jsonrpc":"2.0","method":"eth_subscription","params":{"subscription":"...","result":{...}}}
            if data.get('method') == 'eth_subscription':
                params = data.get('params', {})
                if 'result' in params:
                    # Each notification represents a new pending transaction
                    async with self._lock:
                        self.pending_tx_count += 1
                        logger.debug(f"📨 Pending tx received (count: {self.pending_tx_count})")
            # Ignore subscription confirmation messages (they have 'result' at top level)
            elif 'result' in data and 'id' in data:
                # This is likely a subscription confirmation, ignore
                pass
            else:
                logger.debug(f"📨 Received non-transaction message: {data.get('method', 'unknown')}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
    
    async def flush_to_db(self):
        """
        Flush current transaction count to database and reset counter.
        This runs every flush_interval seconds.
        """
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                async with self._lock:
                    count = self.pending_tx_count
                    timestamp = datetime.now()
                    self.pending_tx_count = 0  # Reset counter
                
                if count > 0:
                    save_mempool_count(count, timestamp)
                    logger.info(f"💾 Flushed {count} pending transactions to DB")
                else:
                    logger.debug("💾 Flushed 0 transactions (no activity)")
                    
            except Exception as e:
                logger.error(f"❌ Error flushing to database: {e}")
    
    async def listen(self):
        """
        Listen for incoming WebSocket messages.
        """
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.ws:
                        try:
                            await self.ws.ping()
                        except:
                            pass
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️  WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"❌ Error receiving message: {e}")
                    break
        except Exception as e:
            logger.error(f"❌ Listen loop error: {e}")
    
    async def run(self):
        """
        Main run loop with auto-reconnect logic.
        """
        reconnect_delay = 5  # Start with 5 seconds
        
        while True:
            try:
                self.running = True
                
                # Connect and subscribe
                if not await self.connect():
                    logger.error("❌ Failed to connect, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                # Reset reconnect delay on successful connection
                reconnect_delay = 5
                
                # Start flush task
                flush_task = asyncio.create_task(self.flush_to_db())
                
                # Start listening
                listen_task = asyncio.create_task(self.listen())
                
                # Wait for either task to complete (they shouldn't normally)
                done, pending = await asyncio.wait(
                    [flush_task, listen_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Close WebSocket if still open
                if self.ws:
                    try:
                        await self.ws.close()
                    except:
                        pass
                    self.ws = None
                
                self.running = False
                
                # Reconnect after delay
                logger.info(f"🔄 Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                
                # Exponential backoff (cap at 60 seconds)
                reconnect_delay = min(reconnect_delay * 2, 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down mempool worker...")
                self.running = False
                if self.ws:
                    try:
                        await self.ws.close()
                    except:
                        pass
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in run loop: {e}")
                self.running = False
                if self.ws:
                    try:
                        await self.ws.close()
                    except:
                        pass
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)


async def main():
    """Main entry point for the mempool worker."""
    # Initialize database
    init_db()
    
    # Create and run worker
    worker = MempoolWorker()
    await worker.run()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the worker
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Mempool worker stopped by user")
