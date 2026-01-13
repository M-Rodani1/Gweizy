"""
Unit tests for HybridPredictor.

Tests the spike detection and prediction capabilities of the hybrid model.

Run: pytest tests/test_hybrid_predictor.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_predictor import HybridPredictor


@pytest.fixture
def predictor():
    """Create a HybridPredictor instance."""
    return HybridPredictor(models_dir='test_models')


@pytest.fixture
def sample_gas_data():
    """Generate sample gas price data for testing."""
    timestamps = pd.date_range(
        end=datetime.utcnow(),
        periods=100,
        freq='5min'
    )

    # Generate realistic gas prices (0.001 Gwei range for Base L2)
    base_price = 0.001
    np.random.seed(42)
    prices = base_price + np.random.uniform(-0.0002, 0.0002, len(timestamps))

    return pd.DataFrame({
        'timestamp': timestamps,
        'gas_price': prices,
        'base_fee': prices * 0.8,
        'priority_fee': prices * 0.2
    })


@pytest.fixture
def spike_gas_data():
    """Generate gas data with a spike pattern."""
    timestamps = pd.date_range(
        end=datetime.utcnow(),
        periods=100,
        freq='5min'
    )

    base_price = 0.001
    prices = np.full(len(timestamps), base_price)
    # Add spike at the end
    prices[-10:] = 0.08  # Above spike threshold

    return pd.DataFrame({
        'timestamp': timestamps,
        'gas_price': prices,
        'base_fee': prices * 0.8,
        'priority_fee': prices * 0.2
    })


class TestHybridPredictorInit:
    """Tests for HybridPredictor initialization."""

    def test_init_default_models_dir(self):
        """Test default models directory is set."""
        predictor = HybridPredictor()
        assert predictor.models_dir == 'models/saved_models'
        assert predictor.loaded == False

    def test_init_custom_models_dir(self, predictor):
        """Test custom models directory."""
        assert predictor.models_dir == 'test_models'

    def test_init_empty_dicts(self, predictor):
        """Test spike detectors dict is empty on init."""
        assert predictor.spike_detectors == {}
        assert predictor.lstm_models == {}

    def test_class_constants(self, predictor):
        """Test class constants are defined correctly."""
        assert predictor.NORMAL_THRESHOLD == 0.01
        assert predictor.ELEVATED_THRESHOLD == 0.05
        assert predictor.CLASS_NAMES == ['normal', 'elevated', 'spike']
        assert len(predictor.CLASS_EMOJIS) == 3
        assert len(predictor.CLASS_COLORS) == 3


class TestCreateSpikeFeatures:
    """Tests for spike feature creation."""

    def test_creates_time_features(self, predictor, sample_gas_data):
        """Test time-based features are created."""
        features_df = predictor.create_spike_features(sample_gas_data)

        assert 'hour' in features_df.columns
        assert 'day_of_week' in features_df.columns
        assert 'is_weekend' in features_df.columns
        assert 'is_business_hours' in features_df.columns

    def test_creates_volatility_features(self, predictor, sample_gas_data):
        """Test volatility features are created."""
        features_df = predictor.create_spike_features(sample_gas_data)

        for window in [6, 12, 24, 48]:
            assert f'volatility_{window}' in features_df.columns
            assert f'range_{window}' in features_df.columns
            assert f'mean_{window}' in features_df.columns

    def test_creates_rate_of_change_features(self, predictor, sample_gas_data):
        """Test rate of change features are created."""
        features_df = predictor.create_spike_features(sample_gas_data)

        for lag in [1, 2, 3, 6, 12]:
            assert f'pct_change_{lag}' in features_df.columns
            assert f'diff_{lag}' in features_df.columns

    def test_recent_spike_indicator(self, predictor, spike_gas_data):
        """Test recent spike indicator for spike data."""
        features_df = predictor.create_spike_features(spike_gas_data)

        assert 'recent_spike' in features_df.columns
        # Last rows should have spike indicator = 1
        assert features_df['recent_spike'].iloc[-1] == 1

    def test_handles_nan_values(self, predictor, sample_gas_data):
        """Test NaN values are handled."""
        # Introduce NaN
        sample_gas_data.loc[5, 'gas_price'] = np.nan

        features_df = predictor.create_spike_features(sample_gas_data)

        # Should not have any NaN in output
        assert not features_df.isna().any().any()

    def test_handles_inf_values(self, predictor, sample_gas_data):
        """Test infinite values are handled."""
        # Introduce inf by having zero values that cause div by zero
        sample_gas_data.loc[0, 'gas_price'] = 0

        features_df = predictor.create_spike_features(sample_gas_data)

        # Should not have any inf in output
        assert not np.isinf(features_df.values).any()


class TestPredict:
    """Tests for prediction functionality."""

    @patch.object(HybridPredictor, 'load_models')
    def test_loads_models_if_not_loaded(self, mock_load, predictor, sample_gas_data):
        """Test models are loaded if not already loaded."""
        mock_load.return_value = False

        with pytest.raises(ValueError, match="Models not loaded"):
            predictor.predict(sample_gas_data)

        mock_load.assert_called_once()

    def test_predict_with_mock_model(self, predictor, sample_gas_data):
        """Test prediction with mocked model."""
        # Mock the spike detector
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])  # Normal class
        mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)

        assert '1h' in predictions
        assert 'classification' in predictions['1h']
        assert predictions['1h']['classification']['class'] == 'normal'

    def test_predict_returns_all_horizons(self, predictor, sample_gas_data):
        """Test all horizons are returned when models exist."""
        # Mock models for all horizons
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

        for horizon in ['1h', '4h', '24h']:
            predictor.spike_detectors[horizon] = {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)

        assert '1h' in predictions
        assert '4h' in predictions
        assert '24h' in predictions

    def test_classification_structure(self, predictor, sample_gas_data):
        """Test classification output structure."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])  # Elevated class
        mock_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)
        classification = predictions['1h']['classification']

        assert 'class' in classification
        assert 'class_id' in classification
        assert 'emoji' in classification
        assert 'color' in classification
        assert 'confidence' in classification
        assert 'probabilities' in classification

        assert classification['class'] == 'elevated'
        assert classification['class_id'] == 1
        assert classification['color'] == 'yellow'

    def test_prediction_structure(self, predictor, sample_gas_data):
        """Test prediction output structure."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)
        prediction = predictions['1h']['prediction']

        assert 'price' in prediction
        assert 'lower_bound' in prediction
        assert 'upper_bound' in prediction
        assert 'unit' in prediction

        # Lower bound should be less than price, upper bound greater
        assert prediction['lower_bound'] < prediction['price']
        assert prediction['upper_bound'] > prediction['price']

    def test_recommendation_for_normal_class(self, predictor, sample_gas_data):
        """Test recommendation for normal classification."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])  # Normal
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)
        rec = predictions['1h']['recommendation']

        assert rec['action'] == 'transact'
        assert 'optimal' in rec['message'].lower() or 'good' in rec['message'].lower()

    def test_recommendation_for_spike_class(self, predictor, sample_gas_data):
        """Test recommendation for spike classification."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])  # Spike
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)
        rec = predictions['1h']['recommendation']

        assert rec['action'] == 'wait'
        assert 'wait' in rec['message'].lower() or 'delay' in rec['message'].lower()

    def test_alert_for_spike(self, predictor, sample_gas_data):
        """Test alert is shown for spike classification."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])  # Spike
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

        predictor.spike_detectors = {
            '1h': {
                'model': mock_model,
                'feature_names': ['hour', 'day_of_week', 'volatility_6', 'mean_6']
            }
        }
        predictor.loaded = True

        predictions = predictor.predict(sample_gas_data)
        alert = predictions['1h']['alert']

        assert alert['show_alert'] == True
        assert alert['severity'] == 'high'


class TestLoadModels:
    """Tests for model loading functionality."""

    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    @patch('pickle.load')
    def test_load_models_success(self, mock_pickle, mock_open, mock_exists, predictor):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_pickle.return_value = {
            'model': MagicMock(),
            'feature_names': ['feature1', 'feature2']
        }

        result = predictor.load_models()

        assert result == True
        assert predictor.loaded == True

    @patch('os.path.exists')
    def test_load_models_no_files(self, mock_exists, predictor):
        """Test loading when no model files exist."""
        mock_exists.return_value = False

        result = predictor.load_models()

        # Should return False or handle gracefully
        assert predictor.loaded == False or result == False


class TestClassThresholds:
    """Tests for classification threshold constants."""

    def test_normal_threshold(self, predictor):
        """Test normal threshold value."""
        assert predictor.NORMAL_THRESHOLD == 0.01
        assert predictor.NORMAL_THRESHOLD < predictor.ELEVATED_THRESHOLD

    def test_elevated_threshold(self, predictor):
        """Test elevated threshold value."""
        assert predictor.ELEVATED_THRESHOLD == 0.05
        assert predictor.ELEVATED_THRESHOLD > predictor.NORMAL_THRESHOLD

    def test_class_names_order(self, predictor):
        """Test class names are in correct order."""
        assert predictor.CLASS_NAMES[0] == 'normal'
        assert predictor.CLASS_NAMES[1] == 'elevated'
        assert predictor.CLASS_NAMES[2] == 'spike'


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, predictor):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=['timestamp', 'gas_price', 'base_fee', 'priority_fee'])

        # Should handle gracefully or raise appropriate error
        with pytest.raises((KeyError, IndexError, ValueError)):
            predictor.create_spike_features(empty_df)

    def test_single_row_dataframe(self, predictor):
        """Test handling of single row dataframe."""
        single_row = pd.DataFrame({
            'timestamp': [datetime.utcnow()],
            'gas_price': [0.001],
            'base_fee': [0.0008],
            'priority_fee': [0.0002]
        })

        features_df = predictor.create_spike_features(single_row)
        assert len(features_df) == 1

    def test_negative_gas_prices(self, predictor):
        """Test handling of negative gas prices (invalid data)."""
        timestamps = pd.date_range(end=datetime.utcnow(), periods=50, freq='5min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'gas_price': [-0.001] * 50,  # Invalid negative prices
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

        # Should not crash
        features_df = predictor.create_spike_features(data)
        assert len(features_df) == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
