"""
Mempool Data Collector for Gas Price Prediction

Collects pending transaction data and mempool statistics that serve as
leading indicators for gas price changes.

For Base (L2), we collect:
- Pending transaction count from RPC
- Gas price distribution in pending txs
- Transaction arrival rate
- Large transaction detection
- Recent block analysis for momentum

Note: Base uses a sequencer, so the "mempool" is more limited than Ethereum.
We supplement with block-level analysis for better signal.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np
from web3 import Web3
from web3.exceptions import Web3Exception
import logging

logger = logging.getLogger(__name__)


class MempoolSnapshot:
    """Represents a point-in-time snapshot of mempool state."""

    def __init__(
        self,
        timestamp: datetime,
        pending_count: int,
        gas_prices: List[float],
        total_gas: int,
        large_tx_count: int,
        avg_gas_price: float,
        median_gas_price: float,
        p90_gas_price: float,
        tx_arrival_rate: float,
        block_number: int
    ):
        self.timestamp = timestamp
        self.pending_count = pending_count
        self.gas_prices = gas_prices
        self.total_gas = total_gas
        self.large_tx_count = large_tx_count
        self.avg_gas_price = avg_gas_price
        self.median_gas_price = median_gas_price
        self.p90_gas_price = p90_gas_price
        self.tx_arrival_rate = tx_arrival_rate
        self.block_number = block_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'pending_count': self.pending_count,
            'total_gas': self.total_gas,
            'large_tx_count': self.large_tx_count,
            'avg_gas_price': self.avg_gas_price,
            'median_gas_price': self.median_gas_price,
            'p90_gas_price': self.p90_gas_price,
            'tx_arrival_rate': self.tx_arrival_rate,
            'block_number': self.block_number
        }


class MempoolCollector:
    """
    Collects and analyzes mempool/pending transaction data.

    Provides features that serve as leading indicators for gas price changes:
    - Pending transaction pressure
    - Gas price distribution in pending txs
    - Transaction arrival rate
    - Large transaction detection (whale activity)
    """

    # Thresholds
    LARGE_TX_GAS_THRESHOLD = 500_000  # Gas units for "large" transaction
    HIGH_CONGESTION_PENDING = 100  # Pending tx count indicating congestion

    def __init__(
        self,
        rpc_urls: List[str] = None,
        snapshot_interval: int = 30,  # seconds between snapshots
        history_size: int = 100,  # number of snapshots to keep
        rpc_timeout: float = 5.0  # timeout for RPC connections
    ):
        """
        Initialize mempool collector.

        Args:
            rpc_urls: List of RPC endpoints to try
            snapshot_interval: Seconds between snapshots
            history_size: Number of snapshots to keep in memory
            rpc_timeout: Timeout for RPC connections in seconds
        """
        if rpc_urls is None:
            from config import Config
            rpc_urls = [
                Config.BASE_RPC_URL,
                'https://mainnet.base.org',
                'https://base.llamarpc.com',
                'https://base-rpc.publicnode.com'
            ]

        self.rpc_urls = rpc_urls
        self.snapshot_interval = snapshot_interval
        self.history_size = history_size
        self.rpc_timeout = rpc_timeout

        # Initialize Web3 connections (try only first 2 for faster startup)
        self.w3_connections = []
        for url in rpc_urls[:2]:  # Limit to 2 RPCs for faster init
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': rpc_timeout}))
                if w3.is_connected():
                    self.w3_connections.append(w3)
                    logger.info(f"Connected to RPC: {url}")
                    break  # One connection is enough for init
            except Exception as e:
                logger.debug(f"Could not connect to {url}: {e}")

        if not self.w3_connections:
            logger.error("No RPC connections available for mempool collector")

        # Snapshot history
        self.snapshots: deque = deque(maxlen=history_size)
        self.last_snapshot_time: Optional[datetime] = None

        # For calculating arrival rate
        self.recent_tx_counts: deque = deque(maxlen=10)
        self.recent_timestamps: deque = deque(maxlen=10)

        # Background collection
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _get_web3(self) -> Optional[Web3]:
        """Get a working Web3 connection."""
        for w3 in self.w3_connections:
            try:
                if w3.is_connected():
                    return w3
            except:
                continue
        return None

    def collect_snapshot(self) -> Optional[MempoolSnapshot]:
        """
        Collect a single mempool snapshot.

        Returns:
            MempoolSnapshot or None if collection fails
        """
        w3 = self._get_web3()
        if not w3:
            logger.warning("No Web3 connection available")
            return None

        try:
            timestamp = datetime.now()

            # Get pending transactions
            pending_txs = self._get_pending_transactions(w3)

            # Get latest block for reference
            latest_block = w3.eth.get_block('latest', full_transactions=True)
            block_number = latest_block.number

            # Analyze pending transactions
            gas_prices = []
            total_gas = 0
            large_tx_count = 0

            for tx in pending_txs:
                # Get gas price (handle both legacy and EIP-1559 txs)
                gas_price = self._get_tx_gas_price(tx, latest_block)
                if gas_price:
                    gas_prices.append(gas_price)

                # Track total gas
                gas = tx.get('gas', 0) if isinstance(tx, dict) else getattr(tx, 'gas', 0)
                total_gas += gas

                # Count large transactions
                if gas > self.LARGE_TX_GAS_THRESHOLD:
                    large_tx_count += 1

            # Calculate statistics
            pending_count = len(pending_txs)
            avg_gas_price = np.mean(gas_prices) if gas_prices else 0
            median_gas_price = np.median(gas_prices) if gas_prices else 0
            p90_gas_price = np.percentile(gas_prices, 90) if gas_prices else 0

            # Calculate arrival rate
            tx_arrival_rate = self._calculate_arrival_rate(pending_count, timestamp)

            snapshot = MempoolSnapshot(
                timestamp=timestamp,
                pending_count=pending_count,
                gas_prices=gas_prices[:100],  # Keep first 100 for memory
                total_gas=total_gas,
                large_tx_count=large_tx_count,
                avg_gas_price=round(avg_gas_price, 6),
                median_gas_price=round(median_gas_price, 6),
                p90_gas_price=round(p90_gas_price, 6),
                tx_arrival_rate=round(tx_arrival_rate, 2),
                block_number=block_number
            )

            with self._lock:
                self.snapshots.append(snapshot)
                self.last_snapshot_time = timestamp

            logger.debug(f"Mempool snapshot: {pending_count} pending, avg gas: {avg_gas_price:.4f} gwei")
            return snapshot

        except Exception as e:
            logger.error(f"Error collecting mempool snapshot: {e}")
            return None

    def _get_pending_transactions(self, w3: Web3) -> List:
        """
        Get pending transactions from the mempool.

        For Base/L2s, this may be limited. We also analyze recent blocks
        for additional signal.
        """
        pending_txs = []

        try:
            # Try to get pending transactions via txpool_content (not always available)
            try:
                txpool = w3.geth.txpool.content()
                for sender_txs in txpool.get('pending', {}).values():
                    pending_txs.extend(sender_txs.values())
                for sender_txs in txpool.get('queued', {}).values():
                    pending_txs.extend(sender_txs.values())
            except Exception:
                pass  # txpool not available on many nodes

            # If no pending txs found, analyze recent blocks for momentum
            if len(pending_txs) < 5:
                # Get transactions from last 3 blocks
                latest = w3.eth.block_number
                for block_num in range(latest - 2, latest + 1):
                    try:
                        block = w3.eth.get_block(block_num, full_transactions=True)
                        pending_txs.extend(block.transactions)
                    except:
                        continue

        except Exception as e:
            logger.debug(f"Error getting pending txs: {e}")

        return pending_txs[:500]  # Limit to 500 transactions

    def _get_tx_gas_price(self, tx, latest_block) -> Optional[float]:
        """Extract gas price from transaction (gwei)."""
        try:
            # Handle dict or AttributeDict
            if isinstance(tx, dict):
                gas_price = tx.get('gasPrice')
                max_fee = tx.get('maxFeePerGas')
                max_priority = tx.get('maxPriorityFeePerGas')
            else:
                gas_price = getattr(tx, 'gasPrice', None)
                max_fee = getattr(tx, 'maxFeePerGas', None)
                max_priority = getattr(tx, 'maxPriorityFeePerGas', None)

            # EIP-1559 transaction
            if max_fee and max_priority:
                base_fee = latest_block.get('baseFeePerGas', 0) if isinstance(latest_block, dict) else getattr(latest_block, 'baseFeePerGas', 0)
                effective_price = min(max_fee, base_fee + max_priority)
                return effective_price / 1e9  # Convert to gwei

            # Legacy transaction
            if gas_price:
                return gas_price / 1e9

        except Exception:
            pass

        return None

    def _calculate_arrival_rate(self, current_count: int, timestamp: datetime) -> float:
        """Calculate transaction arrival rate (txs/second)."""
        self.recent_tx_counts.append(current_count)
        self.recent_timestamps.append(timestamp)

        if len(self.recent_tx_counts) < 2:
            return 0.0

        # Calculate rate over recent window
        time_diff = (self.recent_timestamps[-1] - self.recent_timestamps[0]).total_seconds()
        if time_diff <= 0:
            return 0.0

        # This is a simplified rate - could be improved with actual tx counting
        avg_count = np.mean(list(self.recent_tx_counts))
        return avg_count / max(1, time_diff / len(self.recent_tx_counts))

    def get_current_features(self) -> Dict[str, float]:
        """
        Get current mempool features for prediction.

        Returns dict of features that can be used in the prediction pipeline.
        """
        with self._lock:
            if not self.snapshots:
                # Collect a fresh snapshot if none exist
                self.collect_snapshot()

            if not self.snapshots:
                return self._default_features()

            latest = self.snapshots[-1]

            # Check if snapshot is stale (> 2 minutes old)
            age = (datetime.now() - latest.timestamp).total_seconds()
            if age > 120:
                new_snapshot = self.collect_snapshot()
                if new_snapshot:
                    latest = new_snapshot

            # Calculate momentum features from history
            momentum = self._calculate_momentum()

            return {
                # Current state
                'mempool_pending_count': latest.pending_count,
                'mempool_total_gas': latest.total_gas,
                'mempool_large_tx_count': latest.large_tx_count,
                'mempool_avg_gas_price': latest.avg_gas_price,
                'mempool_median_gas_price': latest.median_gas_price,
                'mempool_p90_gas_price': latest.p90_gas_price,
                'mempool_tx_arrival_rate': latest.tx_arrival_rate,

                # Derived features
                'mempool_is_congested': 1 if latest.pending_count > self.HIGH_CONGESTION_PENDING else 0,
                'mempool_gas_pressure': min(1.0, latest.total_gas / 15_000_000),  # Normalized by block gas limit

                # Momentum features
                'mempool_count_momentum': momentum['count_momentum'],
                'mempool_gas_price_momentum': momentum['gas_price_momentum'],
                'mempool_acceleration': momentum['acceleration'],

                # Meta
                'mempool_snapshot_age_seconds': age
            }

    def _calculate_momentum(self) -> Dict[str, float]:
        """Calculate momentum features from snapshot history."""
        if len(self.snapshots) < 3:
            return {
                'count_momentum': 0.0,
                'gas_price_momentum': 0.0,
                'acceleration': 0.0
            }

        snapshots = list(self.snapshots)

        # Count momentum (rate of change)
        counts = [s.pending_count for s in snapshots[-5:]]
        if len(counts) >= 2:
            count_momentum = (counts[-1] - counts[0]) / max(1, counts[0])
        else:
            count_momentum = 0.0

        # Gas price momentum
        prices = [s.avg_gas_price for s in snapshots[-5:]]
        if len(prices) >= 2 and prices[0] > 0:
            gas_price_momentum = (prices[-1] - prices[0]) / prices[0]
        else:
            gas_price_momentum = 0.0

        # Acceleration (second derivative)
        if len(prices) >= 3:
            first_half_change = prices[len(prices)//2] - prices[0]
            second_half_change = prices[-1] - prices[len(prices)//2]
            acceleration = second_half_change - first_half_change
        else:
            acceleration = 0.0

        return {
            'count_momentum': round(count_momentum, 4),
            'gas_price_momentum': round(gas_price_momentum, 4),
            'acceleration': round(acceleration, 6)
        }

    def _default_features(self) -> Dict[str, float]:
        """Return default features when no data available."""
        return {
            'mempool_pending_count': 0,
            'mempool_total_gas': 0,
            'mempool_large_tx_count': 0,
            'mempool_avg_gas_price': 0.0,
            'mempool_median_gas_price': 0.0,
            'mempool_p90_gas_price': 0.0,
            'mempool_tx_arrival_rate': 0.0,
            'mempool_is_congested': 0,
            'mempool_gas_pressure': 0.0,
            'mempool_count_momentum': 0.0,
            'mempool_gas_price_momentum': 0.0,
            'mempool_acceleration': 0.0,
            'mempool_snapshot_age_seconds': 999
        }

    def start_background_collection(self):
        """Start background thread for continuous collection."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Started mempool background collection")

    def stop_background_collection(self):
        """Stop background collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stopped mempool background collection")

    def _collection_loop(self):
        """Background collection loop."""
        save_counter = 0
        while self._running:
            try:
                snapshot = self.collect_snapshot()

                # Save to database every 2nd snapshot (every ~60s at 30s intervals)
                save_counter += 1
                if snapshot and save_counter >= 2:
                    self._save_snapshot_to_db(snapshot)
                    save_counter = 0

            except Exception as e:
                logger.error(f"Error in mempool collection loop: {e}")

            time.sleep(self.snapshot_interval)

    def _save_snapshot_to_db(self, snapshot: MempoolSnapshot):
        """Save a snapshot to the database."""
        try:
            from data.database import DatabaseManager, MempoolSnapshotRecord

            db = DatabaseManager()
            session = db._get_session()

            # Calculate momentum at save time
            momentum = self._calculate_momentum()

            record = MempoolSnapshotRecord(
                timestamp=snapshot.timestamp,
                block_number=snapshot.block_number,
                pending_count=snapshot.pending_count,
                total_gas=snapshot.total_gas,
                large_tx_count=snapshot.large_tx_count,
                avg_gas_price=snapshot.avg_gas_price,
                median_gas_price=snapshot.median_gas_price,
                p90_gas_price=snapshot.p90_gas_price,
                tx_arrival_rate=snapshot.tx_arrival_rate,
                is_congested=1 if snapshot.pending_count > self.HIGH_CONGESTION_PENDING else 0,
                count_momentum=momentum['count_momentum'],
                gas_price_momentum=momentum['gas_price_momentum']
            )

            session.add(record)
            session.commit()
            session.close()

            logger.debug(f"Saved mempool snapshot to database: {snapshot.pending_count} pending txs")

        except Exception as e:
            logger.warning(f"Could not save mempool snapshot to database: {e}")

    def get_history(self, minutes: int = 30) -> List[Dict]:
        """Get snapshot history for the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [
                s.to_dict() for s in self.snapshots
                if s.timestamp >= cutoff
            ]

    @property
    def snapshot_history(self) -> List[MempoolSnapshot]:
        """Get in-memory snapshot history."""
        with self._lock:
            return list(self.snapshots)

    def get_latest_snapshot(self) -> Optional[MempoolSnapshot]:
        """Get the most recent snapshot."""
        with self._lock:
            return self.snapshots[-1] if self.snapshots else None

    @property
    def running(self) -> bool:
        """Check if background collection is running."""
        return self._running

    @property
    def collection_interval(self) -> int:
        """Get the collection interval in seconds."""
        return self.snapshot_interval

    def get_history_from_db(self, hours: int = 24) -> List[Dict]:
        """Get historical mempool data from database."""
        try:
            from data.database import DatabaseManager, MempoolSnapshotRecord

            db = DatabaseManager()
            session = db._get_session()

            cutoff = datetime.now() - timedelta(hours=hours)
            records = session.query(MempoolSnapshotRecord).filter(
                MempoolSnapshotRecord.timestamp >= cutoff
            ).order_by(MempoolSnapshotRecord.timestamp.asc()).all()

            session.close()

            return [{
                'timestamp': r.timestamp.isoformat(),
                'block_number': r.block_number,
                'pending_count': r.pending_count,
                'total_gas': r.total_gas,
                'large_tx_count': r.large_tx_count,
                'avg_gas_price': r.avg_gas_price,
                'median_gas_price': r.median_gas_price,
                'p90_gas_price': r.p90_gas_price,
                'tx_arrival_rate': r.tx_arrival_rate,
                'is_congested': r.is_congested,
                'count_momentum': r.count_momentum,
                'gas_price_momentum': r.gas_price_momentum
            } for r in records]

        except Exception as e:
            logger.warning(f"Could not load mempool history from database: {e}")
            return []


# Global singleton instance
_mempool_collector: Optional[MempoolCollector] = None
_collector_initializing: bool = False


def get_mempool_collector(timeout: float = 5.0) -> Optional[MempoolCollector]:
    """
    Get or create the global mempool collector.

    Args:
        timeout: Max time to wait for initialization (default 5s)

    Returns:
        MempoolCollector instance or None if not ready
    """
    global _mempool_collector, _collector_initializing

    if _mempool_collector is not None:
        return _mempool_collector

    # Prevent multiple simultaneous initializations
    if _collector_initializing:
        return None

    try:
        _collector_initializing = True
        _mempool_collector = MempoolCollector(rpc_timeout=timeout)
        return _mempool_collector
    except Exception as e:
        logger.error(f"Failed to initialize mempool collector: {e}")
        return None
    finally:
        _collector_initializing = False


def is_collector_ready() -> bool:
    """Check if the mempool collector is initialized and ready."""
    return _mempool_collector is not None and len(_mempool_collector.w3_connections) > 0


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    collector = MempoolCollector()
    snapshot = collector.collect_snapshot()

    if snapshot:
        print(f"\n📊 Mempool Snapshot:")
        print(f"   Pending transactions: {snapshot.pending_count}")
        print(f"   Total gas: {snapshot.total_gas:,}")
        print(f"   Large txs: {snapshot.large_tx_count}")
        print(f"   Avg gas price: {snapshot.avg_gas_price:.4f} gwei")
        print(f"   Median gas price: {snapshot.median_gas_price:.4f} gwei")
        print(f"   P90 gas price: {snapshot.p90_gas_price:.4f} gwei")
        print(f"   TX arrival rate: {snapshot.tx_arrival_rate:.2f}/sec")

        print(f"\n📊 Features for prediction:")
        features = collector.get_current_features()
        for key, value in features.items():
            print(f"   {key}: {value}")
    else:
        print("Could not collect mempool snapshot")
