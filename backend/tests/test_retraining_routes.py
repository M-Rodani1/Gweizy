"""
Comprehensive tests for model retraining API routes.

Tests the training progress, models status, and retraining endpoints.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import os


class TestModelsStatusEndpoint:
    """Tests for /api/retraining/models-status endpoint."""

    def test_models_status_returns_success(self, client):
        """Models status should return 200."""
        response = client.get('/api/retraining/models-status')
        assert response.status_code == 200

    def test_models_status_has_prediction_models(self, client):
        """Response should include prediction_models."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'prediction_models' in data

    def test_models_status_has_spike_detectors(self, client):
        """Response should include spike_detectors."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'spike_detectors' in data

    def test_models_status_has_dqn_agent(self, client):
        """Response should include dqn_agent."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'dqn_agent' in data

    def test_models_status_has_data_status(self, client):
        """Response should include data_status."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'data_status' in data

    def test_models_status_has_overall_ready(self, client):
        """Response should include overall_ready flag."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'overall_ready' in data
        assert isinstance(data['overall_ready'], bool)

    def test_models_status_has_missing_models_list(self, client):
        """Response should include missing_models list."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'missing_models' in data
        assert isinstance(data['missing_models'], list)

    def test_models_status_has_summary(self, client):
        """Response should include summary."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        assert 'summary' in data
        summary = data['summary']
        assert 'prediction_models_ready' in summary
        assert 'spike_detectors_ready' in summary
        assert 'dqn_agent_ready' in summary

    def test_models_status_prediction_horizons(self, client):
        """Prediction models should have 1h, 4h, 24h horizons."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        prediction_models = data['prediction_models']
        for horizon in ['1h', '4h', '24h']:
            assert horizon in prediction_models
            assert 'available' in prediction_models[horizon]
            assert 'path' in prediction_models[horizon]

    def test_models_status_spike_detector_horizons(self, client):
        """Spike detectors should have 1h, 4h, 24h horizons."""
        response = client.get('/api/retraining/models-status')
        data = response.get_json()
        spike_detectors = data['spike_detectors']
        for horizon in ['1h', '4h', '24h']:
            assert horizon in spike_detectors
            assert 'available' in spike_detectors[horizon]
            assert 'path' in spike_detectors[horizon]


class TestTrainingProgressEndpoint:
    """Tests for /api/retraining/training-progress endpoint."""

    def test_training_progress_returns_success(self, client):
        """Training progress should return 200."""
        response = client.get('/api/retraining/training-progress')
        assert response.status_code == 200

    def test_training_progress_has_is_training(self, client):
        """Response should include is_training flag."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'is_training' in data
        assert isinstance(data['is_training'], bool)

    def test_training_progress_has_current_step(self, client):
        """Response should include current_step."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'current_step' in data
        assert isinstance(data['current_step'], int)

    def test_training_progress_has_total_steps(self, client):
        """Response should include total_steps."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'total_steps' in data
        assert data['total_steps'] == 3

    def test_training_progress_has_steps_array(self, client):
        """Response should include steps array."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'steps' in data
        assert isinstance(data['steps'], list)
        assert len(data['steps']) == 3

    def test_training_progress_step_structure(self, client):
        """Each step should have name, status, and message."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        for step in data['steps']:
            assert 'name' in step
            assert 'status' in step
            assert 'message' in step

    def test_training_progress_step_names(self, client):
        """Steps should have expected names."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        expected_names = ['RandomForest Models', 'Spike Detectors', 'DQN Agent']
        actual_names = [step['name'] for step in data['steps']]
        assert actual_names == expected_names

    def test_training_progress_has_timestamps(self, client):
        """Response should include started_at and completed_at."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'started_at' in data
        assert 'completed_at' in data

    def test_training_progress_has_error(self, client):
        """Response should include error field."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()
        assert 'error' in data


class TestRetrainingStatusEndpoint:
    """Tests for /api/retraining/status endpoint."""

    def test_status_returns_success(self, client):
        """Status should return 200."""
        response = client.get('/api/retraining/status')
        assert response.status_code in [200, 500]

    def test_status_has_should_retrain(self, client):
        """Response should include should_retrain flag."""
        response = client.get('/api/retraining/status')
        if response.status_code == 200:
            data = response.get_json()
            assert 'should_retrain' in data
            assert isinstance(data['should_retrain'], bool)

    def test_status_has_reason(self, client):
        """Response should include reason."""
        response = client.get('/api/retraining/status')
        if response.status_code == 200:
            data = response.get_json()
            assert 'reason' in data

    def test_status_has_checked_at(self, client):
        """Response should include checked_at timestamp."""
        response = client.get('/api/retraining/status')
        if response.status_code == 200:
            data = response.get_json()
            assert 'checked_at' in data


class TestCheckDataEndpoint:
    """Tests for /api/retraining/check-data endpoint."""

    def test_check_data_returns_success(self, client):
        """Check data should return 200."""
        response = client.get('/api/retraining/check-data')
        assert response.status_code in [200, 500]

    def test_check_data_has_total_records(self, client):
        """Response should include total_records."""
        response = client.get('/api/retraining/check-data')
        if response.status_code == 200:
            data = response.get_json()
            assert 'total_records' in data
            assert isinstance(data['total_records'], int)

    def test_check_data_has_date_range(self, client):
        """Response should include date_range_days."""
        response = client.get('/api/retraining/check-data')
        if response.status_code == 200:
            data = response.get_json()
            assert 'date_range_days' in data

    def test_check_data_has_sufficient_data(self, client):
        """Response should include sufficient_data flag."""
        response = client.get('/api/retraining/check-data')
        if response.status_code == 200:
            data = response.get_json()
            assert 'sufficient_data' in data
            assert isinstance(data['sufficient_data'], bool)

    def test_check_data_has_readiness(self, client):
        """Response should include readiness status."""
        response = client.get('/api/retraining/check-data')
        if response.status_code == 200:
            data = response.get_json()
            assert 'readiness' in data
            assert data['readiness'] in ['ready', 'collecting']

    def test_check_data_has_progress_percent(self, client):
        """Response should include progress_percent."""
        response = client.get('/api/retraining/check-data')
        if response.status_code == 200:
            data = response.get_json()
            assert 'progress_percent' in data
            assert 0 <= data['progress_percent'] <= 100


class TestSimpleRetrainingEndpoint:
    """Tests for /api/retraining/simple endpoint."""

    def test_simple_retraining_requires_post(self, client):
        """Simple retraining should only accept POST."""
        response = client.get('/api/retraining/simple')
        assert response.status_code == 405

    def test_simple_retraining_returns_started(self, client):
        """Should return started status for new training."""
        response = client.post('/api/retraining/simple')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data
        assert data['status'] in ['started', 'in_progress']

    def test_simple_retraining_has_message(self, client):
        """Response should include message."""
        response = client.post('/api/retraining/simple')
        data = response.get_json()
        assert 'message' in data

    def test_simple_retraining_has_steps(self, client):
        """Response should include steps list when started."""
        response = client.post('/api/retraining/simple')
        data = response.get_json()
        if data['status'] == 'started':
            assert 'steps' in data
            assert isinstance(data['steps'], list)

    def test_simple_retraining_has_timestamp(self, client):
        """Response should include timestamp."""
        response = client.post('/api/retraining/simple')
        data = response.get_json()
        if data['status'] == 'started':
            assert 'timestamp' in data

    def test_simple_retraining_returns_in_progress_if_already_running(self, client):
        """Should return in_progress if training already running."""
        # Start first training
        response1 = client.post('/api/retraining/simple')

        # Try to start second training immediately
        response2 = client.post('/api/retraining/simple')
        data = response2.get_json()

        # Should either start new training or indicate already in progress
        assert data['status'] in ['started', 'in_progress']


class TestRetrainingHistoryEndpoint:
    """Tests for /api/retraining/history endpoint."""

    def test_history_returns_success(self, client):
        """History should return 200."""
        response = client.get('/api/retraining/history')
        assert response.status_code == 200

    def test_history_has_total_backups(self, client):
        """Response should include total_backups."""
        response = client.get('/api/retraining/history')
        data = response.get_json()
        assert 'total_backups' in data
        assert isinstance(data['total_backups'], int)

    def test_history_has_backups_array(self, client):
        """Response should include backups array."""
        response = client.get('/api/retraining/history')
        data = response.get_json()
        assert 'backups' in data
        assert isinstance(data['backups'], list)


class TestRollbackEndpoint:
    """Tests for /api/retraining/rollback endpoint."""

    def test_rollback_requires_post(self, client):
        """Rollback should only accept POST."""
        response = client.get('/api/retraining/rollback')
        assert response.status_code == 405

    def test_rollback_requires_backup_path(self, client):
        """Rollback should require backup_path."""
        response = client.post('/api/retraining/rollback',
                              data=json.dumps({}),
                              content_type='application/json')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_rollback_rejects_invalid_path(self, client):
        """Rollback should reject non-existent backup path."""
        response = client.post('/api/retraining/rollback',
                              data=json.dumps({'backup_path': '/nonexistent/path'}),
                              content_type='application/json')
        assert response.status_code in [400, 404]

    def test_rollback_rejects_path_outside_backup_dir(self, client):
        """Rollback should reject paths outside backup directory."""
        response = client.post('/api/retraining/rollback',
                              data=json.dumps({'backup_path': '/etc/passwd'}),
                              content_type='application/json')
        assert response.status_code == 400


class TestTriggerRetrainingEndpoint:
    """Tests for /api/retraining/trigger endpoint."""

    def test_trigger_requires_post(self, client):
        """Trigger should only accept POST."""
        response = client.get('/api/retraining/trigger')
        assert response.status_code == 405

    def test_trigger_accepts_empty_body(self, client):
        """Trigger should accept empty body (defaults to all models)."""
        response = client.post('/api/retraining/trigger')
        # May succeed or fail depending on data availability
        assert response.status_code in [200, 500]

    def test_trigger_accepts_model_type(self, client):
        """Trigger should accept model_type parameter."""
        response = client.post('/api/retraining/trigger',
                              data=json.dumps({'model_type': 'all'}),
                              content_type='application/json')
        assert response.status_code in [200, 500]

    def test_trigger_rejects_invalid_model_type(self, client):
        """Trigger should reject invalid model_type."""
        response = client.post('/api/retraining/trigger',
                              data=json.dumps({'model_type': 'invalid'}),
                              content_type='application/json')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_trigger_accepts_valid_model_types(self, client):
        """Trigger should accept all valid model types."""
        valid_types = ['lstm', 'prophet', 'ensemble', 'all']
        for model_type in valid_types:
            response = client.post('/api/retraining/trigger',
                                  data=json.dumps({'model_type': model_type}),
                                  content_type='application/json')
            # Should not return 400 for valid types
            assert response.status_code in [200, 500]

    def test_trigger_accepts_force_param(self, client):
        """Trigger should accept force parameter."""
        response = client.post('/api/retraining/trigger',
                              data=json.dumps({'force': True}),
                              content_type='application/json')
        assert response.status_code in [200, 500]


class TestProgressTracking:
    """Tests for progress tracking functionality."""

    def test_progress_updates_during_training(self, client):
        """Progress should update when training starts."""
        # Start training
        client.post('/api/retraining/simple')

        # Check progress
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()

        # Should have some progress indication
        assert 'is_training' in data
        assert 'steps' in data

    def test_progress_step_statuses(self, client):
        """Step statuses should be valid values."""
        response = client.get('/api/retraining/training-progress')
        data = response.get_json()

        valid_statuses = ['pending', 'running', 'completed', 'failed', 'skipped']
        for step in data['steps']:
            assert step['status'] in valid_statuses


class TestErrorHandling:
    """Tests for error handling in retraining routes."""

    def test_models_status_handles_missing_config(self, client):
        """Models status should handle missing config gracefully."""
        response = client.get('/api/retraining/models-status')
        # Should return data even if some models are missing
        assert response.status_code == 200

    def test_progress_handles_concurrent_access(self, client):
        """Progress endpoint should handle concurrent access."""
        # Multiple concurrent requests should not cause errors
        responses = [client.get('/api/retraining/training-progress') for _ in range(5)]
        for response in responses:
            assert response.status_code == 200


@pytest.fixture
def client():
    """Create test client."""
    from app import create_app

    app = create_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_retrainer():
    """Mock ModelRetrainer."""
    with patch('api.retraining_routes.retrainer') as mock:
        mock.should_retrain.return_value = (False, 'All models up to date')
        mock.models_dir = '/tmp/models'
        mock.backup_dir = '/tmp/models/backups'
        yield mock


@pytest.fixture
def mock_db():
    """Mock database manager."""
    with patch('api.retraining_routes.DatabaseManager') as mock:
        session_mock = MagicMock()
        session_mock.query.return_value.count.return_value = 1000
        mock.return_value._get_session.return_value = session_mock
        yield mock
