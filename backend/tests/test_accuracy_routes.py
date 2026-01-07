#!/usr/bin/env python3
"""
Test Accuracy API Routes.
Run: python -m pytest tests/test_accuracy_routes.py -v
"""

import sys
sys.path.append('.')

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestAccuracyMetricsEndpoint:
    """Test /accuracy/metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked tracker."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            # Import app after patching
            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_metrics_success(self, client):
        """Test successful metrics retrieval."""
        test_client, mock_tracker = client

        mock_tracker.get_current_metrics.return_value = {
            'mae': 0.0005,
            'rmse': 0.0008,
            'r2': 0.85,
            'directional_accuracy': 0.72,
            'n': 50
        }

        response = test_client.get('/accuracy/metrics')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert '1h' in data or 'success' in data

    def test_metrics_tracker_not_available(self):
        """Test response when tracker is not available."""
        with patch('api.accuracy_routes.get_tracker', return_value=None):
            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                response = client.get('/accuracy/metrics')

                assert response.status_code == 503
                data = json.loads(response.data)
                assert data['success'] == False
                assert 'not available' in data['error'].lower()


class TestAccuracyStatusEndpoint:
    """Test /accuracy/status endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked tracker."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_status_success(self, client):
        """Test successful status retrieval."""
        test_client, mock_tracker = client

        mock_tracker.get_pending_count.return_value = 10
        mock_tracker.get_validated_count.return_value = 50

        response = test_client.get('/accuracy/status')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'success' in data

    def test_status_tracker_not_available(self):
        """Test response when tracker is not available."""
        with patch('api.accuracy_routes.get_tracker', return_value=None):
            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                response = client.get('/accuracy/status')

                assert response.status_code == 503


class TestSeedTestDataEndpoint:
    """Test /accuracy/seed-test-data endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked dependencies."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_seed_data_success(self, client):
        """Test successful data seeding."""
        test_client, mock_tracker = client

        mock_tracker.record_prediction_with_actual.return_value = None
        mock_tracker.get_current_metrics.return_value = {
            'mae': 0.0005,
            'rmse': 0.0008,
            'r2': 0.85,
            'n': 20
        }

        with patch('api.accuracy_routes.BaseGasCollector') as mock_collector:
            mock_instance = MagicMock()
            mock_instance.get_current_gas.return_value = {'current_gas': 0.001}
            mock_collector.return_value = mock_instance

            response = test_client.post('/accuracy/seed-test-data')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'seeded' in data

    def test_seed_data_tracker_not_available(self):
        """Test response when tracker is not available."""
        with patch('api.accuracy_routes.get_tracker', return_value=None):
            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                response = client.post('/accuracy/seed-test-data')

                assert response.status_code == 503
                data = json.loads(response.data)
                assert data['success'] == False


class TestDiagnosticsEndpoint:
    """Test /accuracy/diagnostics endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked tracker."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.db_path = '/tmp/test.db'
            mock_tracker.predictions = {
                '1h': [MagicMock() for _ in range(10)],
                '4h': [MagicMock() for _ in range(5)],
                '24h': [MagicMock() for _ in range(3)]
            }
            mock_get_tracker.return_value = mock_tracker

            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_diagnostics_success(self, client):
        """Test successful diagnostics retrieval."""
        test_client, mock_tracker = client

        mock_tracker.get_current_metrics.return_value = {
            'mae': 0.0005,
            'rmse': 0.0008,
            'r2': 0.85,
            'n': 10
        }

        response = test_client.get('/accuracy/diagnostics')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'tracker_initialized' in data


class TestValidatePendingEndpoint:
    """Test /accuracy/validate-pending endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked tracker."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_validate_pending_success(self, client):
        """Test successful validation of pending predictions."""
        test_client, mock_tracker = client

        mock_tracker.get_pending_predictions.return_value = []
        mock_tracker.record_actual.return_value = True

        with patch('api.accuracy_routes.BaseGasCollector') as mock_collector:
            mock_instance = MagicMock()
            mock_instance.get_current_gas.return_value = {'current_gas': 0.001}
            mock_collector.return_value = mock_instance

            response = test_client.post('/accuracy/validate-pending')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True


class TestDriftEndpoint:
    """Test /accuracy/drift endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with mocked tracker."""
        with patch('api.accuracy_routes.get_tracker') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            from app import app
            app.config['TESTING'] = True
            with app.test_client() as client:
                yield client, mock_tracker

    def test_drift_check_success(self, client):
        """Test successful drift check."""
        test_client, mock_tracker = client

        mock_drift = MagicMock()
        mock_drift.mae_current = 0.0008
        mock_drift.mae_baseline = 0.0005
        mock_drift.drift_ratio = 1.6
        mock_drift.is_drifting = True
        mock_drift.confidence = 0.85
        mock_drift.sample_size = 100

        mock_tracker.check_drift.return_value = mock_drift

        response = test_client.get('/accuracy/drift')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'success' in data

    def test_drift_insufficient_data(self, client):
        """Test drift check with insufficient data."""
        test_client, mock_tracker = client

        mock_tracker.check_drift.return_value = None

        response = test_client.get('/accuracy/drift')

        assert response.status_code == 200


# Manual test runner for backwards compatibility
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.exit(result.returncode)
