"""
Comprehensive tests for model versioning system
"""

import unittest
import os
import tempfile
import shutil
import joblib
from datetime import datetime
from models.model_registry import ModelRegistry
from config import Config


class TestModelRegistry(unittest.TestCase):
    """Test model versioning and registry"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_register_first_version(self):
        """Test registering first model version"""
        # Create dummy model file
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        metrics = {'r2': 0.85, 'mae': 0.001, 'directional_accuracy': 0.75, 'mape': 5.0}
        
        version = self.registry.register_model(
            horizon='1h',
            model_path=model_path,
            metrics=metrics,
            chain_id=8453
        )
        
        self.assertEqual(version, 'v1.0.0')
        self.assertTrue(self.registry.get_active_version('1h', 8453) is not None)
    
    def test_version_increment(self):
        """Test version incrementing"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        metrics1 = {'r2': 0.85, 'mae': 0.001, 'directional_accuracy': 0.75, 'mape': 5.0}
        metrics2 = {'r2': 0.86, 'mae': 0.0009, 'directional_accuracy': 0.76, 'mape': 4.8}
        
        v1 = self.registry.register_model('1h', model_path, metrics1, chain_id=8453)
        v2 = self.registry.register_model('1h', model_path, metrics2, chain_id=8453)
        
        self.assertEqual(v1, 'v1.0.0')
        self.assertEqual(v2, 'v1.0.1')
    
    def test_auto_activation(self):
        """Test automatic activation of better models"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        # First version (lower score)
        metrics1 = {'r2': 0.70, 'mae': 0.002, 'directional_accuracy': 0.65, 'mape': 8.0}
        v1 = self.registry.register_model('1h', model_path, metrics1, chain_id=8453)
        
        # Second version (higher score - should auto-activate)
        metrics2 = {'r2': 0.90, 'mae': 0.0005, 'directional_accuracy': 0.85, 'mape': 3.0}
        v2 = self.registry.register_model('1h', model_path, metrics2, chain_id=8453)
        
        active = self.registry.get_active_version('1h', 8453)
        self.assertEqual(active['version'], v2)
    
    def test_manual_activation(self):
        """Test manually activating a version"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        metrics = {'r2': 0.85, 'mae': 0.001, 'directional_accuracy': 0.75, 'mape': 5.0}
        v1 = self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        v2 = self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        
        # Manually activate v1
        self.registry.activate_version('1h', v1, 8453)
        active = self.registry.get_active_version('1h', 8453)
        self.assertEqual(active['version'], v1)
    
    def test_rollback(self):
        """Test rolling back to previous version"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        metrics = {'r2': 0.85, 'mae': 0.001, 'directional_accuracy': 0.75, 'mape': 5.0}
        v1 = self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        v2 = self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        v3 = self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        
        # Rollback 1 step
        new_version = self.registry.rollback('1h', 8453, steps=1)
        self.assertEqual(new_version, v2)
        
        active = self.registry.get_active_version('1h', 8453)
        self.assertEqual(active['version'], v2)
    
    def test_get_versions(self):
        """Test getting all versions"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        metrics = {'r2': 0.85, 'mae': 0.001, 'directional_accuracy': 0.75, 'mape': 5.0}
        self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        self.registry.register_model('1h', model_path, metrics, chain_id=8453)
        
        versions = self.registry.get_versions('1h', 8453)
        self.assertEqual(len(versions), 2)
    
    def test_performance_score_calculation(self):
        """Test performance score calculation"""
        model_path = os.path.join(self.test_dir, 'test_model.pkl')
        joblib.dump({'test': 'data'}, model_path)
        
        # Good metrics
        good_metrics = {'r2': 0.90, 'mae': 0.0005, 'directional_accuracy': 0.85, 'mape': 3.0}
        v1 = self.registry.register_model('1h', model_path, good_metrics, chain_id=8453)
        
        # Poor metrics
        poor_metrics = {'r2': 0.50, 'mae': 0.005, 'directional_accuracy': 0.55, 'mape': 15.0}
        v2 = self.registry.register_model('1h', model_path, poor_metrics, chain_id=8453)
        
        versions = self.registry.get_versions('1h', 8453)
        score1 = versions[0]['performance_score']
        score2 = versions[1]['performance_score']
        
        self.assertGreater(score1, score2)


if __name__ == '__main__':
    unittest.main()

