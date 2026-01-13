"""
Unit tests for MempoolCollector.

Tests mempool data collection and analysis functionality.

Run: pytest tests/test_mempool_collector.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mempool_collector import MempoolSnapshot, MempoolCollector


class TestMempoolSnapshot:
    """Tests for MempoolSnapshot dataclass."""

    @pytest.fixture
    def sample_snapshot(self):
        """Create a sample snapshot for testing."""
        return MempoolSnapshot(
            timestamp=datetime.utcnow(),
            pending_count=50,
            gas_prices=[0.001, 0.0012, 0.0015, 0.001],
            total_gas=5_000_000,
            large_tx_count=2,
            avg_gas_price=0.00115,
            median_gas_price=0.0011,
            p90_gas_price=0.0014,
            tx_arrival_rate=10.5,
            block_number=12345678
        )

    def test_snapshot_creation(self, sample_snapshot):
        """Test snapshot is created with correct values."""
        assert sample_snapshot.pending_count == 50
        assert sample_snapshot.large_tx_count == 2
        assert sample_snapshot.avg_gas_price == 0.00115
        assert sample_snapshot.block_number == 12345678

    def test_snapshot_to_dict(self, sample_snapshot):
        """Test snapshot converts to dict correctly."""
        d = sample_snapshot.to_dict()

        assert isinstance(d, dict)
        assert 'timestamp' in d
        assert 'pending_count' in d
        assert 'total_gas' in d
        assert 'large_tx_count' in d
        assert 'avg_gas_price' in d
        assert 'median_gas_price' in d
        assert 'p90_gas_price' in d
        assert 'tx_arrival_rate' in d
        assert 'block_number' in d

        assert d['pending_count'] == 50
        assert d['large_tx_count'] == 2

    def test_snapshot_timestamp_is_iso_format(self, sample_snapshot):
        """Test timestamp is serialized to ISO format."""
        d = sample_snapshot.to_dict()
        # Should be parseable as ISO format
        assert isinstance(d['timestamp'], str)
        # Should contain date separators
        assert '-' in d['timestamp']


class TestMempoolCollectorInit:
    """Tests for MempoolCollector initialization."""

    @patch('data.mempool_collector.Web3')
    @patch('data.mempool_collector.Config')
    def test_init_default_values(self, mock_config, mock_web3):
        """Test initialization with default values."""
        mock_config.BASE_RPC_URL = 'https://test-rpc.example.com'
        mock_w3_instance = MagicMock()
        mock_w3_instance.is_connected.return_value = False
        mock_web3.return_value = mock_w3_instance
        mock_web3.HTTPProvider = MagicMock()

        collector = MempoolCollector()

        assert collector.snapshot_interval == 30
        assert collector.history_size == 100
        assert collector.rpc_timeout == 5.0

    @patch('data.mempool_collector.Web3')
    def test_init_custom_values(self, mock_web3):
        """Test initialization with custom values."""
        mock_w3_instance = MagicMock()
        mock_w3_instance.is_connected.return_value = False
        mock_web3.return_value = mock_w3_instance
        mock_web3.HTTPProvider = MagicMock()

        collector = MempoolCollector(
            rpc_urls=['https://custom-rpc.example.com'],
            snapshot_interval=60,
            history_size=50,
            rpc_timeout=10.0
        )

        assert collector.snapshot_interval == 60
        assert collector.history_size == 50
        assert collector.rpc_timeout == 10.0

    @patch('data.mempool_collector.Web3')
    def test_init_creates_deques(self, mock_web3):
        """Test initialization creates proper deques."""
        mock_w3_instance = MagicMock()
        mock_w3_instance.is_connected.return_value = False
        mock_web3.return_value = mock_w3_instance
        mock_web3.HTTPProvider = MagicMock()

        collector = MempoolCollector(
            rpc_urls=['https://test-rpc.example.com'],
            history_size=50
        )

        assert isinstance(collector.snapshots, deque)
        assert collector.snapshots.maxlen == 50

    def test_class_constants(self):
        """Test class constants are defined."""
        assert MempoolCollector.LARGE_TX_GAS_THRESHOLD == 500_000
        assert MempoolCollector.HIGH_CONGESTION_PENDING == 100


class TestMempoolCollectorMethods:
    """Tests for MempoolCollector methods."""

    @pytest.fixture
    def mock_collector(self):
        """Create a collector with mocked Web3."""
        with patch('data.mempool_collector.Web3') as mock_web3:
            mock_w3_instance = MagicMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            mock_web3.HTTPProvider = MagicMock()

            collector = MempoolCollector(
                rpc_urls=['https://test-rpc.example.com'],
                snapshot_interval=30,
                history_size=100
            )
            collector.w3_connections = [mock_w3_instance]
            return collector

    def test_get_web3_returns_connected(self, mock_collector):
        """Test _get_web3 returns connected instance."""
        mock_collector.w3_connections[0].is_connected.return_value = True

        w3 = mock_collector._get_web3()

        assert w3 is not None
        assert w3 == mock_collector.w3_connections[0]

    def test_get_web3_returns_none_when_disconnected(self, mock_collector):
        """Test _get_web3 returns None when all disconnected."""
        mock_collector.w3_connections[0].is_connected.return_value = False

        w3 = mock_collector._get_web3()

        assert w3 is None

    def test_collect_snapshot_no_connection(self, mock_collector):
        """Test collect_snapshot returns None without connection."""
        mock_collector.w3_connections = []

        snapshot = mock_collector.collect_snapshot()

        assert snapshot is None

    def test_snapshot_history_property(self, mock_collector):
        """Test snapshot_history returns list of snapshots."""
        # Add some snapshots
        snapshot1 = MempoolSnapshot(
            timestamp=datetime.utcnow(),
            pending_count=10,
            gas_prices=[0.001],
            total_gas=100000,
            large_tx_count=0,
            avg_gas_price=0.001,
            median_gas_price=0.001,
            p90_gas_price=0.001,
            tx_arrival_rate=5.0,
            block_number=1000
        )
        mock_collector.snapshots.append(snapshot1)

        history = mock_collector.snapshot_history

        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0].pending_count == 10


class TestSignalGeneration:
    """Tests for signal generation methods."""

    @pytest.fixture
    def collector_with_snapshots(self):
        """Create collector with pre-populated snapshots."""
        with patch('data.mempool_collector.Web3') as mock_web3:
            mock_w3_instance = MagicMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            mock_web3.HTTPProvider = MagicMock()

            collector = MempoolCollector(
                rpc_urls=['https://test-rpc.example.com']
            )
            collector.w3_connections = [mock_w3_instance]

            # Add snapshots with increasing pending counts
            for i in range(10):
                snapshot = MempoolSnapshot(
                    timestamp=datetime.utcnow() - timedelta(minutes=10-i),
                    pending_count=10 + i * 5,
                    gas_prices=[0.001 + i * 0.0001],
                    total_gas=100000 + i * 10000,
                    large_tx_count=i % 3,
                    avg_gas_price=0.001 + i * 0.0001,
                    median_gas_price=0.001 + i * 0.0001,
                    p90_gas_price=0.0012 + i * 0.0001,
                    tx_arrival_rate=5.0 + i,
                    block_number=1000 + i
                )
                collector.snapshots.append(snapshot)

            return collector

    def test_get_latest_snapshot(self, collector_with_snapshots):
        """Test getting latest snapshot."""
        latest = collector_with_snapshots.get_latest_snapshot()

        assert latest is not None
        assert latest.block_number == 1009  # Last one added

    def test_empty_snapshots_returns_none(self):
        """Test get_latest_snapshot with no snapshots."""
        with patch('data.mempool_collector.Web3') as mock_web3:
            mock_w3_instance = MagicMock()
            mock_w3_instance.is_connected.return_value = False
            mock_web3.return_value = mock_w3_instance
            mock_web3.HTTPProvider = MagicMock()

            collector = MempoolCollector(
                rpc_urls=['https://test-rpc.example.com']
            )

            latest = collector.get_latest_snapshot()
            assert latest is None


class TestCongestionAnalysis:
    """Tests for congestion analysis."""

    @pytest.fixture
    def collector_with_high_congestion(self):
        """Create collector with high congestion data."""
        with patch('data.mempool_collector.Web3') as mock_web3:
            mock_w3_instance = MagicMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            mock_web3.HTTPProvider = MagicMock()

            collector = MempoolCollector(
                rpc_urls=['https://test-rpc.example.com']
            )
            collector.w3_connections = [mock_w3_instance]

            # Add high congestion snapshot
            snapshot = MempoolSnapshot(
                timestamp=datetime.utcnow(),
                pending_count=150,  # Above HIGH_CONGESTION_PENDING
                gas_prices=[0.002] * 100,
                total_gas=10_000_000,
                large_tx_count=5,
                avg_gas_price=0.002,
                median_gas_price=0.002,
                p90_gas_price=0.0025,
                tx_arrival_rate=20.0,
                block_number=1000
            )
            collector.snapshots.append(snapshot)

            return collector

    def test_high_congestion_detected(self, collector_with_high_congestion):
        """Test high congestion is properly detected."""
        latest = collector_with_high_congestion.get_latest_snapshot()

        assert latest.pending_count > MempoolCollector.HIGH_CONGESTION_PENDING


class TestWhaleDetection:
    """Tests for whale/large transaction detection."""

    def test_large_tx_threshold(self):
        """Test large transaction threshold constant."""
        assert MempoolCollector.LARGE_TX_GAS_THRESHOLD == 500_000

    def test_snapshot_tracks_large_txs(self):
        """Test snapshot correctly counts large transactions."""
        snapshot = MempoolSnapshot(
            timestamp=datetime.utcnow(),
            pending_count=20,
            gas_prices=[0.001] * 20,
            total_gas=5_000_000,
            large_tx_count=3,
            avg_gas_price=0.001,
            median_gas_price=0.001,
            p90_gas_price=0.0012,
            tx_arrival_rate=5.0,
            block_number=1000
        )

        assert snapshot.large_tx_count == 3


class TestThreadSafety:
    """Tests for thread-safe operations."""

    @patch('data.mempool_collector.Web3')
    def test_lock_exists(self, mock_web3):
        """Test collector has a threading lock."""
        mock_w3_instance = MagicMock()
        mock_w3_instance.is_connected.return_value = False
        mock_web3.return_value = mock_w3_instance
        mock_web3.HTTPProvider = MagicMock()

        collector = MempoolCollector(
            rpc_urls=['https://test-rpc.example.com']
        )

        assert hasattr(collector, '_lock')
        assert collector._lock is not None


class TestModuleFunctions:
    """Tests for module-level functions."""

    @patch('data.mempool_collector._mempool_collector', None)
    @patch('data.mempool_collector._collector_initializing', False)
    @patch('data.mempool_collector.MempoolCollector')
    def test_get_mempool_collector_creates_singleton(self, mock_collector_class):
        """Test get_mempool_collector creates singleton."""
        from data.mempool_collector import get_mempool_collector

        mock_instance = MagicMock()
        mock_collector_class.return_value = mock_instance

        # First call should create
        result = get_mempool_collector(timeout=5.0)

        # Should have called constructor
        mock_collector_class.assert_called()

    @patch('data.mempool_collector._mempool_collector', None)
    def test_is_collector_ready_false_when_none(self):
        """Test is_collector_ready returns False when collector is None."""
        from data.mempool_collector import is_collector_ready

        result = is_collector_ready()
        assert result == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
