"""
Tests for monitoring service
"""

import unittest
from unittest.mock import Mock, patch
from services.monitoring_service import MonitoringService


class TestMonitoringService(unittest.TestCase):
    """Test monitoring service"""
    
    def setUp(self):
        """Set up test environment"""
        with patch('services.monitoring_service.DatabaseManager'), \
             patch('services.monitoring_service.get_tracker'), \
             patch('services.monitoring_service.get_registry'):
            self.service = MonitoringService()
            self.service.db = Mock()
            self.service.accuracy_tracker = Mock()
            self.service.registry = Mock()
    
    def test_get_system_health(self):
        """Test system health check"""
        # Mock active versions
        self.service.registry.get_active_version.side_effect = [
            {'version': 'v1.0.0'},  # 1h
            {'version': 'v1.0.0'},  # 4h
            {'version': 'v1.0.0'}   # 24h
        ]
        
        # Mock data quality
        with patch.object(self.service, 'get_data_quality_report', return_value={'overall_status': 'healthy'}):
            # Mock drift check
            self.service.accuracy_tracker.check_drift.return_value = {'drift_detected': False}
            
            health = self.service._get_system_health()
            
            self.assertIn('status', health)
            self.assertIn('models_loaded', health)
            self.assertIn('data_quality', health)
    
    def test_data_quality_report(self):
        """Test data quality reporting"""
        # Mock database responses
        self.service.db.get_historical_data.return_value = [
            {'timestamp': '2025-01-01T00:00:00', 'gas_price': 0.001}
        ] * 1000
        
        self.service.db.get_onchain_features.return_value = [
            {'timestamp': '2025-01-01T00:00:00', 'tx_count': 100}
        ] * 1000
        
        report = self.service.get_data_quality_report()
        
        self.assertIn('gas_prices', report)
        self.assertIn('onchain_features', report)
        self.assertIn('overall_status', report)
    
    def test_detect_data_gaps(self):
        """Test data gap detection"""
        from datetime import datetime, timedelta
        
        data = [
            {'timestamp': datetime.now() - timedelta(minutes=10)},
            {'timestamp': datetime.now() - timedelta(minutes=5)},
            {'timestamp': datetime.now() - timedelta(minutes=1)}  # Gap of 4 minutes
        ]
        
        gaps = self.service._detect_data_gaps(data, 'timestamp', max_gap_minutes=2)
        
        # Should detect gap between first two entries
        self.assertGreater(len(gaps), 0)


if __name__ == '__main__':
    unittest.main()

