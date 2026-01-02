"""
Comprehensive tests for ML components
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

# Import components to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_engineering import GasFeatureEngineer
from models.feature_selector import SHAPFeatureSelector
from models.hybrid_predictor import HybridPredictor


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering"""
    
    def setUp(self):
        """Set up test data"""
        self.engineer = GasFeatureEngineer()
        
        # Create sample data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'gas_price': np.random.uniform(0.001, 0.1, 100),
            'base_fee': np.random.uniform(0.001, 0.05, 100),
            'priority_fee': np.random.uniform(0.001, 0.05, 100),
            'block_number': range(100, 200)
        })
    
    def test_create_features(self):
        """Test feature creation"""
        features = self.engineer.create_features(self.df)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 10)  # Should have many features
        self.assertEqual(len(features), len(self.df))
    
    def test_feature_names(self):
        """Test that feature names are consistent"""
        features1 = self.engineer.create_features(self.df)
        features2 = self.engineer.create_features(self.df)
        
        self.assertEqual(list(features1.columns), list(features2.columns))
    
    def test_handles_missing_data(self):
        """Test handling of missing data"""
        df_missing = self.df.copy()
        df_missing.loc[10:20, 'gas_price'] = np.nan
        
        features = self.engineer.create_features(df_missing)
        
        # Should handle missing data gracefully
        self.assertIsInstance(features, pd.DataFrame)


class TestFeatureSelector(unittest.TestCase):
    """Test SHAP feature selector"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y = np.random.randn(n_samples)
    
    def test_feature_selection(self):
        """Test feature selection"""
        selector = SHAPFeatureSelector(n_features=10)
        selector.fit(self.X, self.y, verbose=False)
        
        X_selected = selector.transform(self.X)
        
        self.assertEqual(X_selected.shape[1], 10)
        self.assertLessEqual(len(selector.selected_features), 10)
    
    def test_save_and_load(self):
        """Test saving and loading feature selector"""
        selector = SHAPFeatureSelector(n_features=10)
        selector.fit(self.X, self.y, verbose=False)
        
        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            selector.save(temp_path)
            
            # Load
            loaded = SHAPFeatureSelector.load(temp_path)
            
            self.assertEqual(len(loaded.selected_features), len(selector.selected_features))
            self.assertTrue(loaded.fitted)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestModelPrediction(unittest.TestCase):
    """Test model prediction functionality"""
    
    def setUp(self):
        """Set up test model"""
        # Create a simple trained model
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        self.scaler = RobustScaler()
        self.scaler.fit(X_train)
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        X_test = np.random.randn(10, 10)
        X_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_scaled)
        
        self.assertEqual(len(predictions), 10)
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_prediction_range(self):
        """Test that predictions are reasonable"""
        X_test = np.random.randn(10, 10)
        X_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_scaled)
        
        # Predictions should be finite
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestDataValidation(unittest.TestCase):
    """Test data validation"""
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        data = np.array([0.001, 0.002, 0.003, 0.004, 100.0])  # Last value is outlier
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        self.assertGreater(len(outliers), 0)
        self.assertIn(100.0, outliers)
    
    def test_data_completeness(self):
        """Test data completeness check"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
        df = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100)
        })
        
        # Remove some rows
        df_incomplete = df.drop([10, 20, 30])
        
        expected_count = 100
        actual_count = len(df_incomplete)
        completeness = (actual_count / expected_count) * 100
        
        self.assertLess(completeness, 100)
        self.assertGreater(completeness, 90)


if __name__ == '__main__':
    unittest.main()

