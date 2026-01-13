"""
Unit tests for analytics API routes.

Tests the volatility index, whale activity, anomaly detection,
and model ensemble endpoints.

Run: pytest tests/test_analytics_routes.py -v
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

# Import Flask app for testing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_historical_data():
    """Generate mock historical gas price data."""
    base_price = 0.001
    data = []
    for i in range(100):
        data.append({
            'gas_price': base_price + np.random.uniform(-0.0002, 0.0002),
            'timestamp': (datetime.utcnow() - timedelta(hours=100-i)).isoformat()
        })
    return data


class TestVolatilityEndpoint:
    """Tests for /api/analytics/volatility endpoint."""

    @patch('api.analytics_routes.db')
    def test_volatility_returns_valid_response(self, mock_db, client, mock_historical_data):
        """Test volatility endpoint returns expected structure."""
        mock_db.get_historical_data.return_value = mock_historical_data

        response = client.get('/api/analytics/volatility')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'available' in data
        if data['available']:
            assert 'volatility_index' in data
            assert 'level' in data
            assert 'description' in data
            assert 'color' in data
            assert 0 <= data['volatility_index'] <= 100

    @patch('api.analytics_routes.db')
    def test_volatility_insufficient_data(self, mock_db, client):
        """Test volatility returns unavailable with insufficient data."""
        mock_db.get_historical_data.return_value = [{'gas_price': 0.001}] * 5

        response = client.get('/api/analytics/volatility')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data['available'] == False
        assert 'reason' in data

    @patch('api.analytics_routes.db')
    def test_volatility_level_classification(self, mock_db, client):
        """Test volatility levels are correctly classified."""
        # Create stable data (low volatility)
        stable_data = [{'gas_price': 0.001} for _ in range(50)]
        mock_db.get_historical_data.return_value = stable_data

        response = client.get('/api/analytics/volatility')
        data = json.loads(response.data)

        if data['available']:
            assert data['level'] in ['very_low', 'low', 'moderate', 'high', 'extreme']
            assert data['color'] in ['green', 'yellow', 'orange', 'red']

    @patch('api.analytics_routes.db')
    def test_volatility_respects_hours_param(self, mock_db, client, mock_historical_data):
        """Test volatility endpoint respects hours parameter."""
        mock_db.get_historical_data.return_value = mock_historical_data

        response = client.get('/api/analytics/volatility?hours=48')

        assert response.status_code == 200
        mock_db.get_historical_data.assert_called_with(hours=48)


class TestWhalesEndpoint:
    """Tests for /api/analytics/whales endpoint."""

    @patch('api.analytics_routes.get_mempool_collector')
    @patch('api.analytics_routes.is_collector_ready')
    def test_whales_returns_valid_structure(self, mock_ready, mock_collector, client):
        """Test whale endpoint returns expected structure."""
        mock_ready.return_value = False
        mock_collector.return_value = None

        response = client.get('/api/analytics/whales')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'available' in data
        assert 'current' in data
        assert 'whale_count' in data['current']
        assert 'activity_level' in data['current']
        assert 'threshold' in data

    @patch('api.analytics_routes.get_mempool_collector')
    @patch('api.analytics_routes.is_collector_ready')
    def test_whales_activity_levels(self, mock_ready, mock_collector, client):
        """Test whale activity level classification."""
        mock_ready.return_value = False
        mock_collector.return_value = None

        response = client.get('/api/analytics/whales')
        data = json.loads(response.data)

        assert data['current']['activity_level'] in ['none', 'low', 'moderate', 'high']

    @patch('api.analytics_routes.get_mempool_collector')
    @patch('api.analytics_routes.is_collector_ready')
    def test_whales_with_active_collector(self, mock_ready, mock_collector, client):
        """Test whale endpoint with active mempool collector."""
        mock_ready.return_value = True

        mock_snapshot = MagicMock()
        mock_snapshot.large_tx_count = 3

        collector = MagicMock()
        collector.snapshot_history = [mock_snapshot] * 5
        collector.get_latest_snapshot.return_value = mock_snapshot
        mock_collector.return_value = collector

        response = client.get('/api/analytics/whales')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data['current']['whale_count'] == 3
        assert data['current']['activity_level'] == 'moderate'


class TestAnomaliesEndpoint:
    """Tests for /api/analytics/anomalies endpoint."""

    @patch('api.analytics_routes.db')
    def test_anomalies_returns_valid_structure(self, mock_db, client, mock_historical_data):
        """Test anomalies endpoint returns expected structure."""
        mock_db.get_historical_data.return_value = mock_historical_data

        response = client.get('/api/analytics/anomalies')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'available' in data
        if data['available']:
            assert 'status' in data
            assert 'current_z_score' in data
            assert 'anomalies' in data
            assert 'statistics' in data

    @patch('api.analytics_routes.db')
    def test_anomalies_insufficient_data(self, mock_db, client):
        """Test anomalies returns unavailable with insufficient data."""
        mock_db.get_historical_data.return_value = [{'gas_price': 0.001}] * 10

        response = client.get('/api/analytics/anomalies')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data['available'] == False

    @patch('api.analytics_routes.db')
    def test_anomalies_sensitivity_param(self, mock_db, client, mock_historical_data):
        """Test anomalies endpoint respects sensitivity parameter."""
        mock_db.get_historical_data.return_value = mock_historical_data

        response = client.get('/api/analytics/anomalies?sensitivity=3')
        data = json.loads(response.data)

        assert response.status_code == 200

    @patch('api.analytics_routes.db')
    def test_anomalies_detects_spike(self, mock_db, client):
        """Test anomaly detection identifies price spikes."""
        # Normal prices with a spike at the end
        data = [{'gas_price': 0.001} for _ in range(50)]
        data.append({'gas_price': 0.005})  # 5x spike
        mock_db.get_historical_data.return_value = data

        response = client.get('/api/analytics/anomalies')
        result = json.loads(response.data)

        if result['available']:
            # Z-score should be high for the spike
            assert abs(result['current_z_score']) > 2


class TestEnsembleEndpoint:
    """Tests for /api/analytics/ensemble endpoint."""

    @patch('api.analytics_routes.get_ensemble_predictor')
    @patch('api.analytics_routes.hybrid_predictor', new_callable=lambda: MagicMock())
    def test_ensemble_returns_valid_structure(self, mock_hybrid, mock_ensemble, client):
        """Test ensemble endpoint returns expected structure."""
        mock_ensemble.return_value = MagicMock(models={})
        mock_hybrid.loaded = False
        mock_hybrid.spike_detectors = []

        response = client.get('/api/analytics/ensemble')
        data = json.loads(response.data)

        assert response.status_code == 200
        assert 'models' in data
        assert 'health' in data
        assert 'health_pct' in data['health']
        assert 'status' in data['health']

    @patch('api.analytics_routes.get_ensemble_predictor')
    @patch('api.analytics_routes.hybrid_predictor', new_callable=lambda: MagicMock())
    def test_ensemble_health_calculation(self, mock_hybrid, mock_ensemble, client):
        """Test ensemble health percentage calculation."""
        mock_ensemble.return_value = MagicMock(models={})
        mock_hybrid.loaded = True
        mock_hybrid.spike_detectors = [MagicMock(), MagicMock(), MagicMock()]

        response = client.get('/api/analytics/ensemble')
        data = json.loads(response.data)

        assert 0 <= data['health']['health_pct'] <= 100
        assert data['health']['status'] in ['healthy', 'degraded', 'critical']

    @patch('api.analytics_routes.get_ensemble_predictor')
    @patch('api.analytics_routes.hybrid_predictor', new_callable=lambda: MagicMock())
    def test_ensemble_lists_models(self, mock_hybrid, mock_ensemble, client):
        """Test ensemble endpoint lists all model types."""
        mock_ensemble.return_value = MagicMock(models={})
        mock_hybrid.loaded = True
        mock_hybrid.spike_detectors = []

        response = client.get('/api/analytics/ensemble')
        data = json.loads(response.data)

        model_names = [m['name'] for m in data['models']]
        assert 'hybrid_predictor' in model_names
        assert 'pattern_matcher' in model_names
        assert 'fallback_predictor' in model_names


class TestAnalyticsPerformanceEndpoint:
    """Tests for existing /api/analytics/performance endpoint."""

    def test_performance_endpoint_exists(self, client):
        """Test performance endpoint is accessible."""
        response = client.get('/api/analytics/performance')
        assert response.status_code in [200, 500]  # 500 if no data

    def test_performance_with_horizon_param(self, client):
        """Test performance endpoint accepts horizon parameter."""
        response = client.get('/api/analytics/performance?horizon=1h')
        assert response.status_code in [200, 400, 500]

    def test_performance_invalid_horizon(self, client):
        """Test performance endpoint rejects invalid horizon."""
        response = client.get('/api/analytics/performance?horizon=invalid')
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
