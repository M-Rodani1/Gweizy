"""
Unit Tests for API Endpoints
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class APITestCase(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('service', data)
        self.assertIn('endpoints', data)
    
    def test_current_gas_endpoint(self):
        """Test current gas price endpoint"""
        response = self.app.get('/api/current')
        # May fail if RPC is unavailable, but should return proper structure
        self.assertIn(response.status_code, [200, 500])
        if response.status_code == 200:
            data = response.get_json()
            self.assertEqual(data['status'], 'success')
    
    def test_historical_endpoint(self):
        """Test historical gas prices endpoint"""
        response = self.app.get('/api/historical?hours=24')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data['data'])
    
    def test_historical_endpoint_invalid_hours(self):
        """Test historical endpoint with invalid hours"""
        response = self.app.get('/api/historical?hours=1000')
        self.assertEqual(response.status_code, 400)
    
    def test_predictions_endpoint(self):
        """Test predictions endpoint"""
        response = self.app.get('/api/predictions')
        # May fail if RPC is unavailable
        self.assertIn(response.status_code, [200, 500])
        if response.status_code == 200:
            data = response.get_json()
            self.assertEqual(data['status'], 'success')
            self.assertIn('predictions', data['data'])
    
    def test_prediction_by_horizon(self):
        """Test prediction by horizon endpoint"""
        response = self.app.get('/api/predictions/1h')
        self.assertIn(response.status_code, [200, 500])
        
        # Test invalid horizon
        response = self.app.get('/api/predictions/2h')
        self.assertEqual(response.status_code, 400)
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = self.app.get('/api/stats?hours=24')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)


if __name__ == '__main__':
    unittest.main()

