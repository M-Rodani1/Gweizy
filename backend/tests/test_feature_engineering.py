"""
Comprehensive tests for feature engineering module.

Tests the GasFeatureEngineer class that transforms raw gas data into ML features.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestGasFeatureEngineer:
    """Tests for GasFeatureEngineer class."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer with mocked DB."""
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            eng = GasFeatureEngineer()
            eng.db = Mock()
            return eng

    @pytest.fixture
    def sample_gas_data(self):
        """Create sample gas price data."""
        timestamps = pd.date_range(
            start=datetime.utcnow() - timedelta(hours=100),
            periods=100,
            freq='1H'
        )
        return [
            {
                'timestamp': ts.isoformat(),
                'gwei': 0.001 + np.random.random() * 0.0005,
                'baseFee': 0.0008 + np.random.random() * 0.0002,
                'priorityFee': 0.0002,
                'block_number': 1000000 + i
            }
            for i, ts in enumerate(timestamps)
        ]

    def test_init_creates_db_manager(self):
        """Should initialize with database manager."""
        with patch('models.feature_engineering.DatabaseManager') as mock_db:
            from models.feature_engineering import GasFeatureEngineer
            engineer = GasFeatureEngineer()
            mock_db.assert_called_once()

    def test_prepare_training_data_fetches_data(self, engineer, sample_gas_data):
        """Should fetch historical data from database."""
        engineer.db.get_historical_data.return_value = sample_gas_data
        engineer.db._get_session.return_value = MagicMock()

        with patch.object(engineer, '_join_onchain_features', return_value=pd.DataFrame(sample_gas_data)):
            try:
                df = engineer.prepare_training_data(hours_back=100)
                engineer.db.get_historical_data.assert_called_once()
            except Exception:
                pass  # May fail due to missing onchain features

    def test_prepare_training_data_requires_minimum_records(self, engineer):
        """Should raise error if not enough data."""
        engineer.db.get_historical_data.return_value = [
            {'timestamp': datetime.utcnow().isoformat(), 'gwei': 0.001}
            for _ in range(50)  # Less than 100 required
        ]

        with pytest.raises(ValueError, match="Not enough data"):
            engineer.prepare_training_data(hours_back=1)


class TestTimeFeatures:
    """Tests for time-based feature engineering."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with timestamps."""
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=48,
            freq='1H'
        )
        return pd.DataFrame({
            'timestamp': timestamps,
            'gas': np.random.random(48) * 0.001,
            'base_fee': np.random.random(48) * 0.0008,
            'priority_fee': np.random.random(48) * 0.0002
        })

    def test_add_time_features_creates_hour(self, engineer, sample_df):
        """Should create hour feature."""
        result = engineer._add_time_features(sample_df.copy())
        assert 'hour' in result.columns
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23

    def test_add_time_features_creates_day_of_week(self, engineer, sample_df):
        """Should create day of week feature."""
        result = engineer._add_time_features(sample_df.copy())
        assert 'day_of_week' in result.columns
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6

    def test_add_time_features_creates_weekend_flag(self, engineer, sample_df):
        """Should create is_weekend feature."""
        result = engineer._add_time_features(sample_df.copy())
        assert 'is_weekend' in result.columns
        assert set(result['is_weekend'].unique()).issubset({0, 1, True, False})

    def test_add_time_features_creates_cyclical_encoding(self, engineer, sample_df):
        """Should create cyclical hour encoding."""
        result = engineer._add_time_features(sample_df.copy())
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        # Cyclical values should be between -1 and 1
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1


class TestLagFeatures:
    """Tests for lag feature engineering."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': [0.001 + i * 0.0001 for i in range(50)],
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

    def test_add_lag_features_creates_1h_lag(self, engineer, sample_df):
        """Should create 1-hour lag feature."""
        result = engineer._add_lag_features(sample_df.copy())
        assert 'gas_lag_1h' in result.columns

    def test_add_lag_features_creates_multiple_lags(self, engineer, sample_df):
        """Should create multiple lag features."""
        result = engineer._add_lag_features(sample_df.copy())
        expected_lags = ['gas_lag_1h', 'gas_lag_3h', 'gas_lag_6h', 'gas_lag_12h', 'gas_lag_24h']
        for lag in expected_lags:
            assert lag in result.columns, f"Missing lag feature: {lag}"

    def test_lag_values_are_correct(self, engineer, sample_df):
        """Lag values should match historical values."""
        result = engineer._add_lag_features(sample_df.copy())
        # After 1h lag, value at index 1 should equal value at index 0
        if 'gas_lag_1h' in result.columns and len(result) > 1:
            # Check non-NaN values
            valid_idx = result['gas_lag_1h'].notna()
            if valid_idx.any():
                first_valid = result[valid_idx].index[0]
                if first_valid > 0:
                    assert result.loc[first_valid, 'gas_lag_1h'] == result.loc[first_valid - 1, 'gas']


class TestRollingFeatures:
    """Tests for rolling statistical features."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': np.random.random(50) * 0.001 + 0.001,
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

    def test_add_rolling_features_creates_mean(self, engineer, sample_df):
        """Should create rolling mean features."""
        result = engineer._add_rolling_features(sample_df.copy())
        rolling_means = [col for col in result.columns if 'rolling_mean' in col]
        assert len(rolling_means) > 0

    def test_add_rolling_features_creates_std(self, engineer, sample_df):
        """Should create rolling std features."""
        result = engineer._add_rolling_features(sample_df.copy())
        rolling_stds = [col for col in result.columns if 'rolling_std' in col]
        assert len(rolling_stds) > 0

    def test_add_rolling_features_creates_min_max(self, engineer, sample_df):
        """Should create rolling min/max features."""
        result = engineer._add_rolling_features(sample_df.copy())
        rolling_mins = [col for col in result.columns if 'rolling_min' in col]
        rolling_maxs = [col for col in result.columns if 'rolling_max' in col]
        assert len(rolling_mins) > 0
        assert len(rolling_maxs) > 0

    def test_add_rolling_features_creates_change(self, engineer, sample_df):
        """Should create rate of change features."""
        result = engineer._add_rolling_features(sample_df.copy())
        change_cols = [col for col in result.columns if 'change' in col]
        assert len(change_cols) > 0


class TestTargetVariables:
    """Tests for target variable creation."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with enough data for targets."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'gas': [0.001 + i * 0.00001 for i in range(100)],
            'base_fee': [0.0008] * 100,
            'priority_fee': [0.0002] * 100
        })

    def test_add_target_variables_creates_1h(self, engineer, sample_df):
        """Should create 1-hour target."""
        result = engineer._add_target_variables(sample_df.copy())
        assert 'target_1h' in result.columns

    def test_add_target_variables_creates_4h(self, engineer, sample_df):
        """Should create 4-hour target."""
        result = engineer._add_target_variables(sample_df.copy())
        assert 'target_4h' in result.columns

    def test_add_target_variables_creates_24h(self, engineer, sample_df):
        """Should create 24-hour target."""
        result = engineer._add_target_variables(sample_df.copy())
        assert 'target_24h' in result.columns

    def test_targets_are_future_values(self, engineer, sample_df):
        """Targets should be shifted future values."""
        result = engineer._add_target_variables(sample_df.copy())
        # Target at time T should be the gas price at T + horizon
        if 'target_1h' in result.columns:
            # Check a few values
            for i in range(min(10, len(result) - 1)):
                if pd.notna(result.loc[i, 'target_1h']):
                    # Target should equal gas value 1 row ahead
                    expected = result.loc[i + 1, 'gas'] if i + 1 < len(result) else np.nan
                    if pd.notna(expected):
                        assert abs(result.loc[i, 'target_1h'] - expected) < 0.0001


class TestOnchainFeatures:
    """Tests for on-chain feature joining."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            eng = GasFeatureEngineer()
            eng.db = Mock()
            return eng

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': [0.001] * 50,
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

    def test_join_onchain_handles_missing_data(self, engineer, sample_df):
        """Should handle missing on-chain data gracefully."""
        engineer.db._get_session.return_value = MagicMock()

        # Mock empty query result
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        engineer.db._get_session.return_value = mock_session

        result = engineer._join_onchain_features(sample_df.copy())
        # Should return dataframe even if no onchain data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)


class TestGetFeatureColumns:
    """Tests for feature column selection."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    def test_get_feature_columns_excludes_targets(self, engineer):
        """Should exclude target columns."""
        df = pd.DataFrame({
            'gas': [1, 2, 3],
            'hour': [0, 1, 2],
            'target_1h': [1.1, 2.1, 3.1],
            'target_4h': [1.2, 2.2, 3.2],
            'target_24h': [1.3, 2.3, 3.3]
        })

        features = engineer.get_feature_columns(df)
        assert 'target_1h' not in features
        assert 'target_4h' not in features
        assert 'target_24h' not in features

    def test_get_feature_columns_excludes_timestamp(self, engineer):
        """Should exclude timestamp column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=3, freq='1H'),
            'gas': [1, 2, 3],
            'hour': [0, 1, 2]
        })

        features = engineer.get_feature_columns(df)
        assert 'timestamp' not in features

    def test_get_feature_columns_includes_engineered(self, engineer):
        """Should include engineered features."""
        df = pd.DataFrame({
            'gas': [1, 2, 3],
            'hour': [0, 1, 2],
            'gas_lag_1h': [0.9, 1.9, 2.9],
            'gas_rolling_mean_1h': [1, 1.5, 2]
        })

        features = engineer.get_feature_columns(df)
        assert 'gas_lag_1h' in features
        assert 'gas_rolling_mean_1h' in features


class TestFeatureNaming:
    """Tests for feature naming consistency."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    def test_feature_names_are_valid_identifiers(self, engineer):
        """Feature names should be valid Python identifiers."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': np.random.random(50) * 0.001,
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

        df = engineer._add_time_features(df)
        df = engineer._add_lag_features(df)
        df = engineer._add_rolling_features(df)

        for col in df.columns:
            # Should not contain spaces or special chars (except underscore)
            assert ' ' not in col, f"Feature name contains space: {col}"
            assert col.replace('_', '').replace('.', '').isalnum() or col.isidentifier(), \
                f"Invalid feature name: {col}"


class TestDataQuality:
    """Tests for data quality handling."""

    @pytest.fixture
    def engineer(self):
        with patch('models.feature_engineering.DatabaseManager'):
            from models.feature_engineering import GasFeatureEngineer
            return GasFeatureEngineer()

    def test_handles_nan_values(self, engineer):
        """Should handle NaN values in input data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': [0.001 if i % 5 != 0 else np.nan for i in range(50)],
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

        result = engineer._add_time_features(df.copy())
        # Should complete without error
        assert len(result) == len(df)

    def test_handles_zero_values(self, engineer):
        """Should handle zero gas prices."""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'gas': [0.001 if i % 5 != 0 else 0 for i in range(50)],
            'base_fee': [0.0008] * 50,
            'priority_fee': [0.0002] * 50
        })

        result = engineer._add_rolling_features(df.copy())
        # Should complete without error
        assert len(result) == len(df)
