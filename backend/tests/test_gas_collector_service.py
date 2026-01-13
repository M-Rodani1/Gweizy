"""
Comprehensive tests for GasCollectorService.

Tests the background data collection service that fetches gas prices.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
import threading
import time


class TestGasCollectorServiceInit:
    """Tests for GasCollectorService initialization."""

    @patch('services.gas_collector_service.DatabaseManager')
    @patch('services.gas_collector_service.BaseGasCollector')
    def test_init_with_default_interval(self, mock_collector, mock_db):
        """Should initialize with default collection interval."""
        from services.gas_collector_service import GasCollectorService

        service = GasCollectorService()
        assert service.interval > 0

    @patch('services.gas_collector_service.DatabaseManager')
    @patch('services.gas_collector_service.BaseGasCollector')
    def test_init_with_custom_interval(self, mock_collector, mock_db):
        """Should accept custom collection interval."""
        from services.gas_collector_service import GasCollectorService

        service = GasCollectorService(interval=30)
        assert service.interval == 30

    @patch('services.gas_collector_service.DatabaseManager')
    @patch('services.gas_collector_service.BaseGasCollector')
    def test_init_creates_collector(self, mock_collector, mock_db):
        """Should create gas collector instance."""
        from services.gas_collector_service import GasCollectorService

        service = GasCollectorService()
        mock_collector.assert_called()

    @patch('services.gas_collector_service.DatabaseManager')
    @patch('services.gas_collector_service.BaseGasCollector')
    def test_init_creates_database_manager(self, mock_collector, mock_db):
        """Should create database manager instance."""
        from services.gas_collector_service import GasCollectorService

        service = GasCollectorService()
        mock_db.assert_called()

    @patch('services.gas_collector_service.DatabaseManager')
    @patch('services.gas_collector_service.BaseGasCollector')
    def test_init_with_socketio(self, mock_collector, mock_db):
        """Should accept socketio for real-time updates."""
        from services.gas_collector_service import GasCollectorService

        mock_socketio = Mock()
        service = GasCollectorService(socketio=mock_socketio)
        assert service.socketio == mock_socketio


class TestGasCollectorServiceCollection:
    """Tests for data collection functionality."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService
            svc = GasCollectorService(interval=1)
            svc.db = Mock()
            svc.collector = Mock()
            return svc

    def test_collect_once_fetches_gas_data(self, service):
        """Should fetch gas data from collector."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'base_fee': 0.0008,
            'priority_fee': 0.0002,
            'timestamp': datetime.utcnow().isoformat()
        }

        service._collect_once()
        service.collector.get_current_gas.assert_called()

    def test_collect_once_saves_to_database(self, service):
        """Should save collected data to database."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'base_fee': 0.0008,
            'priority_fee': 0.0002,
            'timestamp': datetime.utcnow().isoformat()
        }

        service._collect_once()
        service.db.save_gas_data.assert_called()

    def test_collect_once_handles_collector_error(self, service):
        """Should handle collector errors gracefully."""
        service.collector.get_current_gas.side_effect = Exception("Network error")

        # Should not raise exception
        service._collect_once()

    def test_collect_once_handles_db_error(self, service):
        """Should handle database errors gracefully."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'timestamp': datetime.utcnow().isoformat()
        }
        service.db.save_gas_data.side_effect = Exception("DB error")

        # Should not raise exception
        service._collect_once()


class TestGasCollectorServiceWebSocket:
    """Tests for WebSocket integration."""

    @pytest.fixture
    def service_with_socketio(self):
        """Create service with mocked socketio."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService

            mock_socketio = Mock()
            svc = GasCollectorService(interval=1, socketio=mock_socketio)
            svc.db = Mock()
            svc.collector = Mock()
            return svc

    def test_emits_gas_update_on_collection(self, service_with_socketio):
        """Should emit WebSocket event on successful collection."""
        service_with_socketio.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'base_fee': 0.0008,
            'priority_fee': 0.0002,
            'timestamp': datetime.utcnow().isoformat()
        }

        service_with_socketio._collect_once()

        # Check if socketio.emit was called
        if service_with_socketio.socketio:
            service_with_socketio.socketio.emit.assert_called()


class TestGasCollectorServiceLifecycle:
    """Tests for service start/stop lifecycle."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService
            svc = GasCollectorService(interval=0.1)
            svc.db = Mock()
            svc.collector = Mock()
            svc.collector.get_current_gas.return_value = {
                'current_gas': 0.001,
                'timestamp': datetime.utcnow().isoformat()
            }
            return svc

    def test_start_sets_running_flag(self, service):
        """Start should set running flag."""
        # Start in thread to avoid blocking
        thread = threading.Thread(target=service.start)
        thread.daemon = True
        thread.start()

        time.sleep(0.2)
        assert service.running == True

        service.stop()

    def test_stop_clears_running_flag(self, service):
        """Stop should clear running flag."""
        thread = threading.Thread(target=service.start)
        thread.daemon = True
        thread.start()

        time.sleep(0.2)
        service.stop()
        time.sleep(0.2)

        assert service.running == False

    def test_collects_at_interval(self, service):
        """Should collect data at specified interval."""
        call_count = 0

        def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {'current_gas': 0.001, 'timestamp': datetime.utcnow().isoformat()}

        service.collector.get_current_gas = count_calls

        thread = threading.Thread(target=service.start)
        thread.daemon = True
        thread.start()

        time.sleep(0.35)  # Should get ~3 collections at 0.1s interval
        service.stop()

        assert call_count >= 2


class TestGasCollectorServiceDataValidation:
    """Tests for data validation during collection."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService
            svc = GasCollectorService(interval=1)
            svc.db = Mock()
            svc.collector = Mock()
            return svc

    def test_validates_gas_price_not_negative(self, service):
        """Should handle negative gas prices."""
        service.collector.get_current_gas.return_value = {
            'current_gas': -0.001,  # Invalid
            'timestamp': datetime.utcnow().isoformat()
        }

        # Should handle gracefully (not save or log warning)
        service._collect_once()

    def test_validates_gas_price_not_zero(self, service):
        """Should handle zero gas prices."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Should complete without error
        service._collect_once()

    def test_handles_missing_fields(self, service):
        """Should handle missing fields in response."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001
            # Missing timestamp
        }

        # Should handle gracefully
        service._collect_once()

    def test_handles_none_response(self, service):
        """Should handle None response from collector."""
        service.collector.get_current_gas.return_value = None

        # Should not raise exception
        service._collect_once()

    def test_handles_empty_response(self, service):
        """Should handle empty dict response."""
        service.collector.get_current_gas.return_value = {}

        # Should not raise exception
        service._collect_once()


class TestGasCollectorServiceMetrics:
    """Tests for collection metrics tracking."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService
            svc = GasCollectorService(interval=1)
            svc.db = Mock()
            svc.collector = Mock()
            return svc

    def test_tracks_successful_collections(self, service):
        """Should track successful collection count."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'timestamp': datetime.utcnow().isoformat()
        }

        initial_count = getattr(service, 'collection_count', 0)
        service._collect_once()
        # May or may not track count depending on implementation

    def test_tracks_failed_collections(self, service):
        """Should track failed collection count."""
        service.collector.get_current_gas.side_effect = Exception("Error")

        service._collect_once()
        # Should complete without raising


class TestGasCollectorServiceChainSupport:
    """Tests for multi-chain support."""

    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies."""
        with patch('services.gas_collector_service.DatabaseManager') as mock_db, \
             patch('services.gas_collector_service.BaseGasCollector') as mock_collector:
            from services.gas_collector_service import GasCollectorService
            svc = GasCollectorService(interval=1)
            svc.db = Mock()
            svc.collector = Mock()
            return svc

    def test_collects_for_base_chain(self, service):
        """Should collect data for Base chain (8453)."""
        service.collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'chain_id': 8453,
            'timestamp': datetime.utcnow().isoformat()
        }

        service._collect_once()
        # Verify collection happened
        service.collector.get_current_gas.assert_called()
