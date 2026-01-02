"""
Enhanced Monitoring Service

Provides comprehensive monitoring for model performance, data quality, and system health.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from data.database import DatabaseManager
from models.accuracy_tracker import get_tracker
from models.model_registry import get_registry
from config import Config
from utils.logger import logger


class MonitoringService:
    """Comprehensive monitoring for ML system"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.accuracy_tracker = get_tracker()
        self.registry = get_registry()
    
    def get_model_performance_dashboard(self) -> Dict:
        """Get comprehensive model performance dashboard"""
        try:
            # Get active model versions
            active_models = {}
            for horizon in ['1h', '4h', '24h']:
                active = self.registry.get_active_version(horizon)
                if active:
                    active_models[horizon] = {
                        'version': active['version'],
                        'metrics': active['metrics'],
                        'performance_score': active.get('performance_score', 0),
                        'registered_at': active.get('registered_at')
                    }
            
            # Get accuracy tracking metrics
            accuracy_metrics = {}
            for horizon in ['1h', '4h', '24h']:
                try:
                    metrics = self.accuracy_tracker.get_current_metrics(horizon)
                    accuracy_metrics[horizon] = metrics
                except:
                    accuracy_metrics[horizon] = None
            
            # Get drift status
            drift_status = {}
            for horizon in ['1h', '4h', '24h']:
                try:
                    drift = self.accuracy_tracker.check_drift(horizon)
                    drift_status[horizon] = drift
                except:
                    drift_status[horizon] = None
            
            # Get recent validation results
            recent_validations = self._get_recent_validations()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'active_models': active_models,
                'accuracy_metrics': accuracy_metrics,
                'drift_status': drift_status,
                'recent_validations': recent_validations,
                'system_health': self._get_system_health()
            }
        except Exception as e:
            logger.error(f"Error getting performance dashboard: {e}")
            return {'error': str(e)}
    
    def get_data_quality_report(self) -> Dict:
        """Get data quality metrics"""
        try:
            # Get recent data counts
            try:
                recent_gas = self.db.get_historical_data(hours=24, chain_id=8453)
            except:
                recent_gas = []
            
            try:
                recent_onchain = self.db.get_onchain_features(hours=24, chain_id=8453)
            except:
                recent_onchain = []
            
            # Calculate data quality metrics
            gas_count = len(recent_gas) if recent_gas else 0
            onchain_count = len(recent_onchain) if recent_onchain else 0
            
            # Expected counts (5 second intervals = 720 per hour)
            expected_per_hour = 720
            expected_24h = expected_per_hour * 24
            
            gas_completeness = (gas_count / expected_24h * 100) if expected_24h > 0 else 0
            onchain_completeness = (onchain_count / expected_24h * 100) if expected_24h > 0 else 0
            
            # Check for gaps
            gas_gaps = self._detect_data_gaps(recent_gas, 'timestamp') if recent_gas else []
            onchain_gaps = self._detect_data_gaps(recent_onchain, 'timestamp') if recent_onchain else []
            
            # Check data freshness
            latest_gas = recent_gas[-1]['timestamp'] if recent_gas else None
            latest_onchain = recent_onchain[-1]['timestamp'] if recent_onchain else None
            
            gas_freshness_minutes = None
            onchain_freshness_minutes = None
            
            if latest_gas:
                if isinstance(latest_gas, str):
                    latest_gas = datetime.fromisoformat(latest_gas.replace('Z', '+00:00'))
                gas_freshness_minutes = (datetime.now() - latest_gas.replace(tzinfo=None)).total_seconds() / 60
            
            if latest_onchain:
                if isinstance(latest_onchain, str):
                    latest_onchain = datetime.fromisoformat(latest_onchain.replace('Z', '+00:00'))
                onchain_freshness_minutes = (datetime.now() - latest_onchain.replace(tzinfo=None)).total_seconds() / 60
            
            return {
                'timestamp': datetime.now().isoformat(),
                'gas_prices': {
                    'count_24h': gas_count,
                    'expected_24h': expected_24h,
                    'completeness_percent': round(gas_completeness, 2),
                    'gaps': len(gas_gaps),
                    'freshness_minutes': round(gas_freshness_minutes, 1) if gas_freshness_minutes else None,
                    'status': 'healthy' if gas_completeness > 90 and (gas_freshness_minutes is None or gas_freshness_minutes < 10) else 'degraded'
                },
                'onchain_features': {
                    'count_24h': onchain_count,
                    'expected_24h': expected_24h,
                    'completeness_percent': round(onchain_completeness, 2),
                    'gaps': len(onchain_gaps),
                    'freshness_minutes': round(onchain_freshness_minutes, 1) if onchain_freshness_minutes else None,
                    'status': 'healthy' if onchain_completeness > 90 and (onchain_freshness_minutes is None or onchain_freshness_minutes < 10) else 'degraded'
                },
                'overall_status': 'healthy' if gas_completeness > 90 and onchain_completeness > 90 else 'degraded'
            }
        except Exception as e:
            logger.error(f"Error getting data quality report: {e}")
            return {'error': str(e)}
    
    def _detect_data_gaps(self, data: List[Dict], timestamp_field: str, max_gap_minutes: int = 10) -> List[Dict]:
        """Detect gaps in time series data"""
        if not data or len(data) < 2:
            return []
        
        gaps = []
        for i in range(1, len(data)):
            try:
                ts1 = data[i-1][timestamp_field]
                ts2 = data[i][timestamp_field]
                
                if isinstance(ts1, str):
                    ts1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
                if isinstance(ts2, str):
                    ts2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
                
                gap_minutes = (ts2.replace(tzinfo=None) - ts1.replace(tzinfo=None)).total_seconds() / 60
                
                if gap_minutes > max_gap_minutes:
                    gaps.append({
                        'from': ts1.isoformat(),
                        'to': ts2.isoformat(),
                        'gap_minutes': round(gap_minutes, 1)
                    })
            except Exception as e:
                logger.warning(f"Error detecting gap: {e}")
                continue
        
        return gaps
    
    def _get_recent_validations(self, limit: int = 10) -> List[Dict]:
        """Get recent validation results"""
        # This would query validation history from database
        # For now, return empty list
        return []
    
    def _get_system_health(self) -> Dict:
        """Get overall system health status"""
        try:
            # Check if models are loaded
            models_loaded = True
            for horizon in ['1h', '4h', '24h']:
                active = self.registry.get_active_version(horizon)
                if not active:
                    models_loaded = False
                    break
            
            # Check data quality
            data_quality = self.get_data_quality_report()
            data_healthy = data_quality.get('overall_status') == 'healthy'
            
            # Check for drift
            has_drift = False
            for horizon in ['1h', '4h', '24h']:
                try:
                    drift = self.accuracy_tracker.check_drift(horizon)
                    if drift and drift.get('drift_detected'):
                        has_drift = True
                        break
                except:
                    pass
            
            overall_status = 'healthy'
            if not models_loaded:
                overall_status = 'critical'
            elif not data_healthy or has_drift:
                overall_status = 'degraded'
            
            return {
                'status': overall_status,
                'models_loaded': models_loaded,
                'data_quality': data_healthy,
                'drift_detected': has_drift,
                'checks': {
                    'models': models_loaded,
                    'data': data_healthy,
                    'drift': not has_drift
                }
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def get_monitoring_summary(self) -> Dict:
        """Get comprehensive monitoring summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.get_model_performance_dashboard(),
            'data_quality': self.get_data_quality_report(),
            'system_health': self._get_system_health()
        }


# Global instance
_monitoring_instance: Optional['MonitoringService'] = None


def get_monitoring_service() -> MonitoringService:
    """Get global monitoring service instance"""
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = MonitoringService()
    return _monitoring_instance

