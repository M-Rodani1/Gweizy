#!/usr/bin/env python3
"""
Block Header Monitoring Worker

Asynchronous background worker that connects to Alchemy WebSocket to monitor
new block headers as a leading indicator of gas price congestion.

Features:
- WebSocket connection to Alchemy Base mainnet
- Subscribes to newHeads (new block headers)
- Tracks block utilization (gasUsed/gasLimit) and base fee
- Records block-level metrics to database
- Provides "heartbeat" every ~2 seconds (Base block time)
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Alchemy WebSocket URL - loaded from environment variable
ALCHEMY_WS_URL = os.getenv("BASE_RPC_WSS")

# Safety check: ensure the environment variable is set
if not ALCHEMY_WS_URL:
    raise ValueError("❌ Error: BASE_RPC_WSS is not set. Please check your .env file.")

# Database path - prefer root gas_data.db, fallback to backend/gas_data.db
DB_PATH = 'gas_data.db'
if not os.path.exists(DB_PATH):
    DB_PATH = 'backend/gas_data.db'
if not os.path.exists(DB_PATH):
    DB_PATH = 'gas_data.db'  # Will create if doesn't exist


def init_db():
    """
    Initialize database schema for block_stats table.
    Creates the table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Create block_stats table for block-level metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS block_stats (
                block_number INTEGER PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                gas_used INTEGER NOT NULL,
                gas_limit INTEGER NOT NULL,
                utilization REAL NOT NULL,
                base_fee REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_block_stats_timestamp 
            ON block_stats(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_block_stats_block_number 
            ON block_stats(block_number)
        """)
        
        # Keep old mempool_stats table for backward compatibility (if it exists)
        # But we'll use block_stats going forward
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Initialized block_stats table in {DB_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def save_block_stats(block_number: int, gas_used: int, gas_limit: int, utilization: float, base_fee: Optional[float], timestamp: datetime):
    """
    Save block statistics to database.
    
    Args:
        block_number: Block number
        gas_used: Gas used in the block
        gas_limit: Gas limit of the block
        utilization: Gas utilization percentage (gas_used / gas_limit)
        base_fee: Base fee per gas (in wei, will be converted to gwei)
        timestamp: Block timestamp
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Convert base_fee from wei to gwei if provided
        base_fee_gwei = base_fee / 1e9 if base_fee else None
        
        # Use block_number as primary key, so we'll use INSERT OR REPLACE
        # to handle potential duplicates
        cursor.execute("""
            INSERT OR REPLACE INTO block_stats 
            (block_number, timestamp, gas_used, gas_limit, utilization, base_fee, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            block_number,
            timestamp.isoformat(),
            gas_used,
            gas_limit,
            utilization,
            base_fee_gwei,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"💾 Saved block {block_number}: {utilization:.1%} utilization, base_fee={base_fee_gwei:.2f} gwei" if base_fee_gwei else f"💾 Saved block {block_number}: {utilization:.1%} utilization")
    except Exception as e:
        logger.error(f"❌ Failed to save block stats: {e}")


class BlockHeaderWorker:
    """
    Asynchronous worker that monitors new block headers via Alchemy WebSocket.
    """
    
    def __init__(self, ws_url: str = ALCHEMY_WS_URL):
        """
        Initialize block header worker.
        
        Args:
            ws_url: WebSocket URL for Alchemy
        """
        self.ws_url = ws_url
        self.running = False
        self.ws = None  # WebSocket connection
        self.subscription_id = None
        
    async def connect(self):
        """Connect to Alchemy WebSocket and subscribe to new block headers."""
        try:
            logger.info(f"🔌 Connecting to Alchemy WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            logger.info("✅ Connected to Alchemy WebSocket")
            
            # Subscribe to new block headers
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newHeads"]
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info("📡 Subscribing to newHeads (new block headers)...")
            
            # Wait for subscription confirmation
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            logger.info(f"📥 Subscription response: {json.dumps(response_data, indent=2)}")
            
            if 'result' in response_data:
                subscription_id = response_data['result']
                self.subscription_id = subscription_id
                logger.info(f"✅ Subscription confirmed: {subscription_id}")
                logger.info("🔍 Listening for new block headers...")
                logger.info("   Will track block utilization (gasUsed/gasLimit) as congestion signal")
                return True
            elif 'error' in response_data:
                error = response_data.get('error', {})
                logger.error(f"❌ Subscription failed: {error.get('message', 'Unknown error')}")
                logger.error(f"   Error code: {error.get('code', 'N/A')}")
                return False
            else:
                logger.error(f"❌ Unexpected subscription response: {response_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def hex_to_int(self, hex_str: str) -> int:
        """Convert hex string to integer."""
        if hex_str.startswith('0x'):
            return int(hex_str, 16)
        return int(hex_str, 16)
    
    async def handle_message(self, message: str):
        """
        Handle incoming WebSocket message (new block header).
        Processes block header and saves statistics.
        
        Args:
            message: JSON string message from WebSocket
        """
        try:
            data = json.loads(message)
            
            # Check if this is a new block header notification
            # Format: {"jsonrpc":"2.0","method":"eth_subscription","params":{"subscription":"...","result":{...}}}
            if data.get('method') == 'eth_subscription':
                params = data.get('params', {})
                subscription = params.get('subscription')
                
                # Verify this is for our subscription
                if subscription == self.subscription_id or subscription is None:
                    if 'result' in params:
                        block_header = params.get('result')
                        
                        if isinstance(block_header, dict):
                            # Extract block information
                            block_number_hex = block_header.get('number', '0x0')
                            block_number = self.hex_to_int(block_number_hex)
                            
                            gas_used_hex = block_header.get('gasUsed', '0x0')
                            gas_used = self.hex_to_int(gas_used_hex)
                            
                            gas_limit_hex = block_header.get('gasLimit', '0x0')
                            gas_limit = self.hex_to_int(gas_limit_hex)
                            
                            # Calculate utilization
                            utilization = gas_used / gas_limit if gas_limit > 0 else 0.0
                            
                            # Extract base fee (if available)
                            base_fee_hex = block_header.get('baseFeePerGas')
                            base_fee = self.hex_to_int(base_fee_hex) if base_fee_hex else None
                            
                            # Extract timestamp
                            timestamp_hex = block_header.get('timestamp', '0x0')
                            timestamp_int = self.hex_to_int(timestamp_hex)
                            block_timestamp = datetime.fromtimestamp(timestamp_int)
                            
                            # Log the block
                            base_fee_str = f", base_fee={base_fee/1e9:.2f} gwei" if base_fee else ""
                            logger.info(f"📦 New Block {block_number}: {utilization:.1%} full ({gas_used:,}/{gas_limit:,} gas){base_fee_str}")
                            
                            # Save to database
                            save_block_stats(
                                block_number=block_number,
                                gas_used=gas_used,
                                gas_limit=gas_limit,
                                utilization=utilization,
                                base_fee=base_fee,
                                timestamp=block_timestamp
                            )
                        else:
                            logger.warning(f"⚠️  Unexpected block header format: {type(block_header)}")
                    else:
                        logger.debug(f"📨 Subscription message without result: {params}")
                else:
                    logger.debug(f"📨 Message for different subscription: {subscription}")
            # Check for error messages
            elif 'error' in data:
                error_msg = data.get('error', {})
                logger.error(f"❌ WebSocket error: {error_msg.get('message', 'Unknown error')} (Code: {error_msg.get('code', 'N/A')})")
            # Ignore subscription confirmation messages (they have 'result' at top level)
            elif 'result' in data and 'id' in data:
                # This is likely a subscription confirmation, already handled in connect()
                pass
            else:
                logger.debug(f"📨 Received other message type: {data.get('method', 'unknown')}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  Invalid JSON message: {e}")
            logger.debug(f"   Raw message: {message[:200]}")
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}", exc_info=True)
    
    
    async def listen(self):
        """
        Listen for incoming WebSocket messages (new block headers).
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
                
                # Connect and subscribe via WebSocket
                if not await self.connect():
                    logger.error("❌ Failed to connect, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                # Reset reconnect delay on successful connection
                reconnect_delay = 5
                
                # Start listening for new block headers
                listen_task = asyncio.create_task(self.listen())
                
                # Wait for task to complete (shouldn't normally)
                await listen_task
                
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
                logger.info("🛑 Shutting down block header worker...")
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
    """Main entry point for the block header worker."""
    # Initialize database
    init_db()
    
    # Create and run worker
    worker = BlockHeaderWorker()
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
        logger.info("🛑 Block header worker stopped by user")
