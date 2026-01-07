#!/usr/bin/env python3
"""
Test Accuracy Tracker functionality.
Run: python -m pytest tests/test_accuracy_tracker.py -v
"""

import sys
sys.path.append('.')

import os
import tempfile
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from models.accuracy_tracker import AccuracyTracker, PredictionRecord, get_tracker


class TestAccuracyTrackerInit:
    """Test AccuracyTracker initialization."""

    def test_creates_database_file(self):
        """Test that tracker creates database file on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_accuracy.db')
            tracker = AccuracyTracker(db_path=db_path)

            assert os.path.exists(db_path)
            assert tracker.db_path == db_path

    def test_initializes_with_default_params(self):
        """Test default initialization parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            assert tracker.window_size == 100
            assert tracker.drift_threshold == 0.25
            assert tracker.baseline_window == 500

    def test_initializes_prediction_buffers(self):
        """Test that prediction buffers are initialized for all horizons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            assert '1h' in tracker.predictions
            assert '4h' in tracker.predictions
            assert '24h' in tracker.predictions


class TestRecordPrediction:
    """Test prediction recording functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            yield AccuracyTracker(db_path=db_path)

    def test_record_prediction_returns_id(self, tracker):
        """Test that recording a prediction returns an ID."""
        pred_id = tracker.record_prediction(
            horizon='1h',
            predicted=0.001,
            timestamp=datetime.now()
        )

        assert isinstance(pred_id, int)
        assert pred_id > 0

    def test_record_prediction_stores_in_database(self, tracker):
        """Test that prediction is stored in database."""
        timestamp = datetime.now()
        tracker.record_prediction(
            horizon='1h',
            predicted=0.00123,
            timestamp=timestamp
        )

        # Check database directly
        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            row = conn.execute(
                'SELECT horizon, predicted FROM predictions WHERE horizon = ?',
                ('1h',)
            ).fetchone()

        assert row is not None
        assert row[0] == '1h'
        assert abs(row[1] - 0.00123) < 0.0001

    def test_record_multiple_predictions(self, tracker):
        """Test recording multiple predictions for different horizons."""
        for horizon in ['1h', '4h', '24h']:
            tracker.record_prediction(
                horizon=horizon,
                predicted=0.001 * (1 + hash(horizon) % 10 / 10),
                timestamp=datetime.now()
            )

        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            count = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]

        assert count == 3


class TestRecordActual:
    """Test actual value recording and matching."""

    @pytest.fixture
    def tracker_with_prediction(self):
        """Create tracker with a recorded prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            # Record a prediction
            timestamp = datetime.now() - timedelta(hours=2)
            tracker.record_prediction(
                horizon='1h',
                predicted=0.001,
                timestamp=timestamp
            )

            yield tracker, timestamp

    def test_record_actual_matches_prediction(self, tracker_with_prediction):
        """Test that actual value is matched to prediction."""
        tracker, timestamp = tracker_with_prediction

        result = tracker.record_actual(
            horizon='1h',
            actual=0.00105,
            prediction_timestamp=timestamp,
            tolerance_minutes=30
        )

        assert result is True

    def test_record_actual_calculates_error(self, tracker_with_prediction):
        """Test that error is calculated correctly."""
        tracker, timestamp = tracker_with_prediction

        tracker.record_actual(
            horizon='1h',
            actual=0.00105,
            prediction_timestamp=timestamp,
            tolerance_minutes=30
        )

        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            row = conn.execute(
                'SELECT error, pct_error FROM predictions WHERE actual IS NOT NULL'
            ).fetchone()

        assert row is not None
        expected_error = 0.00105 - 0.001  # actual - predicted
        assert abs(row[0] - expected_error) < 0.0001

    def test_record_actual_with_tolerance(self):
        """Test that tolerance window works for matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            # Record prediction at exact time
            pred_time = datetime.now() - timedelta(hours=2)
            tracker.record_prediction(horizon='1h', predicted=0.001, timestamp=pred_time)

            # Try to match with slightly different timestamp (within tolerance)
            search_time = pred_time + timedelta(minutes=15)
            result = tracker.record_actual(
                horizon='1h',
                actual=0.00095,
                prediction_timestamp=search_time,
                tolerance_minutes=30
            )

            assert result is True


class TestGetCurrentMetrics:
    """Test metrics calculation."""

    @pytest.fixture
    def tracker_with_data(self):
        """Create tracker with validated predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            # Add 10 validated predictions
            base_price = 0.001
            for i in range(10):
                timestamp = datetime.now() - timedelta(hours=i)
                predicted = base_price * (1 + np.random.normal(0, 0.1))
                actual = base_price * (1 + np.random.normal(0, 0.1))

                tracker.record_prediction_with_actual(
                    horizon='1h',
                    predicted=predicted,
                    actual=actual,
                    timestamp=timestamp
                )

            yield tracker

    def test_returns_none_with_insufficient_data(self):
        """Test that metrics return None with < 5 predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            # Add only 3 predictions
            for i in range(3):
                tracker.record_prediction_with_actual(
                    horizon='1h',
                    predicted=0.001,
                    actual=0.00105,
                    timestamp=datetime.now() - timedelta(hours=i)
                )

            metrics = tracker.get_current_metrics('1h')

            assert metrics['mae'] is None
            assert metrics['rmse'] is None
            assert metrics['n'] == 3

    def test_calculates_mae(self, tracker_with_data):
        """Test MAE calculation."""
        metrics = tracker_with_data.get_current_metrics('1h')

        assert metrics['mae'] is not None
        assert metrics['mae'] >= 0

    def test_calculates_rmse(self, tracker_with_data):
        """Test RMSE calculation."""
        metrics = tracker_with_data.get_current_metrics('1h')

        assert metrics['rmse'] is not None
        assert metrics['rmse'] >= 0
        # RMSE should be >= MAE
        assert metrics['rmse'] >= metrics['mae']

    def test_calculates_r2(self, tracker_with_data):
        """Test R² calculation."""
        metrics = tracker_with_data.get_current_metrics('1h')

        assert metrics['r2'] is not None
        # R² should typically be between -1 and 1 for reasonable predictions
        assert -1 <= metrics['r2'] <= 1

    def test_calculates_directional_accuracy(self, tracker_with_data):
        """Test directional accuracy calculation."""
        metrics = tracker_with_data.get_current_metrics('1h')

        assert 'directional_accuracy' in metrics
        if metrics['directional_accuracy'] is not None:
            assert 0 <= metrics['directional_accuracy'] <= 1


class TestAutoSeedIfEmpty:
    """Test auto-seeding functionality."""

    def test_seeds_when_empty(self):
        """Test that empty database gets seeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')

            # Patch the gas collector to avoid network calls
            with patch('models.accuracy_tracker.BaseGasCollector') as mock_collector:
                mock_instance = MagicMock()
                mock_instance.get_current_gas.return_value = {'current_gas': 0.001}
                mock_collector.return_value = mock_instance

                tracker = AccuracyTracker(db_path=db_path)

            # Should have seeded data
            total = sum(len(tracker.predictions[h]) for h in ['1h', '4h', '24h'])
            assert total >= 15  # At least 5 per horizon

    def test_does_not_seed_when_sufficient_data(self):
        """Test that seeding is skipped when data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')

            # Create tracker and manually add data
            tracker = AccuracyTracker.__new__(AccuracyTracker)
            tracker.db_path = db_path
            tracker.window_size = 100
            tracker.drift_threshold = 0.25
            tracker.baseline_window = 500
            tracker.predictions = {
                '1h': [],
                '4h': [],
                '24h': []
            }
            tracker.baseline_mae = {}
            tracker._lock = __import__('threading').Lock()
            tracker._init_db()

            # Manually add 20 predictions per horizon
            for horizon in ['1h', '4h', '24h']:
                for i in range(20):
                    tracker.record_prediction_with_actual(
                        horizon=horizon,
                        predicted=0.001,
                        actual=0.00105,
                        timestamp=datetime.now() - timedelta(hours=i)
                    )

            initial_count = sum(len(tracker.predictions[h]) for h in ['1h', '4h', '24h'])

            # Now call auto_seed - should not add more
            tracker._auto_seed_if_empty()

            final_count = sum(len(tracker.predictions[h]) for h in ['1h', '4h', '24h'])
            assert final_count == initial_count


class TestDriftDetection:
    """Test drift detection functionality."""

    def test_check_drift_returns_drift_metrics(self):
        """Test that drift check returns proper metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            tracker = AccuracyTracker(db_path=db_path)

            # Add enough data for drift detection
            for i in range(150):
                tracker.record_prediction_with_actual(
                    horizon='1h',
                    predicted=0.001,
                    actual=0.00105,
                    timestamp=datetime.now() - timedelta(hours=i)
                )

            # Set baseline
            tracker.baseline_mae['1h'] = 0.00005

            drift = tracker.check_drift('1h')

            assert drift is not None
            assert hasattr(drift, 'mae_current')
            assert hasattr(drift, 'mae_baseline')
            assert hasattr(drift, 'is_drifting')


class TestGetTracker:
    """Test the get_tracker singleton function."""

    def test_returns_tracker_instance(self):
        """Test that get_tracker returns an AccuracyTracker."""
        # Reset the global tracker
        import models.accuracy_tracker as module
        module._tracker = None

        with patch.object(AccuracyTracker, '_auto_seed_if_empty'):
            tracker = get_tracker()

        assert isinstance(tracker, AccuracyTracker)

    def test_returns_same_instance(self):
        """Test that get_tracker returns singleton."""
        import models.accuracy_tracker as module
        module._tracker = None

        with patch.object(AccuracyTracker, '_auto_seed_if_empty'):
            tracker1 = get_tracker()
            tracker2 = get_tracker()

        assert tracker1 is tracker2


# Manual test runner for backwards compatibility
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.exit(result.returncode)
