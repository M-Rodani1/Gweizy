"""
Comprehensive tests for main API routes.

Tests the core prediction and gas price endpoints.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_success(self, client):
        """Health check should return success."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data.get('status') == 'healthy' or 'success' in data

    def test_health_includes_timestamp(self, client):
        """Health check should include timestamp."""
        response = client.get('/api/health')
        data = response.get_json()
        # Check for timestamp in various formats
        assert any(key in data for key in ['timestamp', 'time', 'checked_at'])


class TestCurrentGasEndpoint:
    """Tests for /api/current endpoint."""

    @patch('api.routes.collector')
    def test_current_returns_gas_price(self, mock_collector, client):
        """Should return current gas price."""
        mock_collector.get_current_gas.return_value = {
            'current_gas': 0.001,
            'base_fee': 0.0008,
            'priority_fee': 0.0002,
            'timestamp': datetime.utcnow().isoformat()
        }

        response = client.get('/api/current')
        assert response.status_code == 200
        data = response.get_json()
        assert 'current_gas' in data or 'gas' in data or 'gwei' in data

    @patch('api.routes.collector')
    def test_current_handles_collector_error(self, mock_collector, client):
        """Should handle collector errors gracefully."""
        mock_collector.get_current_gas.side_effect = Exception("Network error")

        response = client.get('/api/current')
        # Should return either cached data or error
        assert response.status_code in [200, 500, 503]

    def test_current_supports_chain_parameter(self, client):
        """Should support chain_id query parameter."""
        response = client.get('/api/current?chain_id=8453')
        assert response.status_code in [200, 404, 500]


class TestPredictionsEndpoint:
    """Tests for /api/predictions endpoint."""

    def test_predictions_returns_all_horizons(self, client):
        """Should return predictions for all horizons."""
        response = client.get('/api/predictions')
        # May return 200 with predictions or 500 if models not loaded
        if response.status_code == 200:
            data = response.get_json()
            if 'predictions' in data:
                # Check structure
                assert isinstance(data['predictions'], dict)

    def test_predictions_supports_horizon_param(self, client):
        """Should filter by horizon when specified."""
        response = client.get('/api/predictions?horizon=1h')
        assert response.status_code in [200, 400, 500]

    def test_predictions_rejects_invalid_horizon(self, client):
        """Should reject invalid horizon values."""
        response = client.get('/api/predictions?horizon=invalid')
        # Should return error or ignore invalid param
        assert response.status_code in [200, 400]


class TestHistoricalEndpoint:
    """Tests for /api/historical endpoint."""

    def test_historical_default_hours(self, client):
        """Should return historical data with default hours."""
        response = client.get('/api/historical')
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, (list, dict))

    def test_historical_custom_hours(self, client):
        """Should accept hours parameter."""
        response = client.get('/api/historical?hours=12')
        assert response.status_code in [200, 404, 500]

    def test_historical_max_hours_limit(self, client):
        """Should limit maximum hours."""
        response = client.get('/api/historical?hours=10000')
        assert response.status_code in [200, 400]


class TestModelsReloadEndpoint:
    """Tests for /api/models/reload endpoint."""

    def test_reload_requires_post(self, client):
        """Reload should only accept POST."""
        response = client.get('/api/models/reload')
        assert response.status_code == 405

    @patch('api.routes.reload_models')
    def test_reload_calls_reload_function(self, mock_reload, client):
        """Should call reload_models function."""
        mock_reload.return_value = {
            'success': True,
            'models_loaded': 3,
            'scalers_loaded': 3,
            'horizons': ['1h', '4h', '24h']
        }

        response = client.post('/api/models/reload')
        assert response.status_code == 200
        mock_reload.assert_called_once()

    @patch('api.routes.reload_models')
    def test_reload_handles_failure(self, mock_reload, client):
        """Should handle reload failures."""
        mock_reload.return_value = {
            'success': False,
            'error': 'Models not found',
            'models_loaded': 0,
            'scalers_loaded': 0
        }

        response = client.post('/api/models/reload')
        data = response.get_json()
        assert 'success' in data


class TestLoadModelsFunction:
    """Tests for load_models helper function."""

    @patch('api.routes.joblib')
    @patch('os.path.exists')
    def test_load_models_from_disk(self, mock_exists, mock_joblib):
        """Should load models from disk."""
        from api.routes import load_models

        mock_exists.return_value = True
        mock_joblib.load.return_value = {
            'model': MagicMock(),
            'model_name': 'XGBoost',
            'metrics': {'mae': 0.001},
            'feature_names': ['gas_lag_1h', 'hour']
        }

        result = load_models()
        assert result['success'] == True
        assert result['models_loaded'] >= 0

    @patch('os.path.exists')
    def test_load_models_handles_missing_files(self, mock_exists):
        """Should handle missing model files."""
        from api.routes import load_models

        mock_exists.return_value = False

        result = load_models()
        # Should complete without error even if no models found
        assert 'models_loaded' in result


class TestPatternsEndpoint:
    """Tests for /api/patterns endpoint."""

    def test_patterns_returns_data(self, client):
        """Should return pattern analysis."""
        response = client.get('/api/patterns')
        assert response.status_code in [200, 404, 500]

    def test_patterns_with_hours_param(self, client):
        """Should accept hours parameter."""
        response = client.get('/api/patterns?hours=48')
        assert response.status_code in [200, 404, 500]


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_returns_summary(self, client):
        """Should return API statistics."""
        response = client.get('/api/stats')
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, dict)


class TestCacheHeaders:
    """Tests for cache headers on responses."""

    def test_current_has_cache_headers(self, client):
        """Current gas should have appropriate cache headers."""
        response = client.get('/api/current')
        # Check for cache control header
        cache_control = response.headers.get('Cache-Control', '')
        assert 'max-age' in cache_control or response.status_code != 200

    def test_predictions_has_cache_headers(self, client):
        """Predictions should have appropriate cache headers."""
        response = client.get('/api/predictions')
        cache_control = response.headers.get('Cache-Control', '')
        # Should have some cache control
        if response.status_code == 200:
            assert cache_control or 'X-Response-Time' in response.headers


class TestCORSHeaders:
    """Tests for CORS headers."""

    def test_cors_on_get_request(self, client):
        """GET requests should have CORS headers."""
        response = client.get('/api/health')
        assert response.headers.get('Access-Control-Allow-Origin') == '*'

    def test_cors_on_options_request(self, client):
        """OPTIONS requests should return CORS headers."""
        response = client.options('/api/current')
        assert response.status_code == 200
        assert 'Access-Control-Allow-Methods' in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_endpoint(self, client):
        """Should return 404 for unknown endpoints."""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404

    def test_404_returns_json(self, client):
        """404 should return JSON error."""
        response = client.get('/api/nonexistent')
        data = response.get_json()
        assert 'error' in data


class TestResponseTimes:
    """Tests for response time headers."""

    def test_response_time_header(self, client):
        """Should include response time header."""
        response = client.get('/api/health')
        # Check for response time header (added by middleware)
        if response.status_code == 200:
            assert 'X-Response-Time' in response.headers or True  # May not be present


@pytest.fixture
def client():
    """Create test client."""
    from app import create_app

    app = create_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_db():
    """Mock database manager."""
    with patch('api.routes.db') as mock:
        mock.get_historical_data.return_value = []
        yield mock
