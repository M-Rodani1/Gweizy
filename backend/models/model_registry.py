"""
Model Versioning and Registry System

Tracks model versions, enables rollback, and manages model lifecycle.
"""

import os
import json
import shutil
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from config import Config
from utils.logger import logger


class ModelRegistry:
    """Manages model versions, rollback, and metadata"""
    
    def __init__(self, registry_dir: str = None):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory for registry metadata (defaults to Config.MODELS_DIR)
        """
        try:
            if registry_dir is None:
                registry_dir = Config.MODELS_DIR
            
            self.registry_dir = registry_dir
            self.versions_dir = os.path.join(registry_dir, 'versions')
            self.registry_file = os.path.join(registry_dir, 'model_registry.json')
            
            # Create directories (with error handling)
            try:
                os.makedirs(self.versions_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create versions directory: {e}")
                # Fallback to local directory
                self.versions_dir = os.path.join('backend/models/saved_models', 'versions')
                os.makedirs(self.versions_dir, exist_ok=True)
            
            # Load or create registry
            self.registry = self._load_registry()
        except Exception as e:
            logger.error(f"Error initializing ModelRegistry: {e}")
            # Initialize with empty registry to prevent crashes
            self.registry = {
                'models': {},
                'active_versions': {},
                'rollback_history': []
            }
            self.registry_dir = registry_dir or 'backend/models/saved_models'
            self.versions_dir = os.path.join(self.registry_dir, 'versions')
            self.registry_file = os.path.join(self.registry_dir, 'model_registry.json')
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {
            'models': {},
            'active_versions': {},
            'rollback_history': []
        }
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        horizon: str,
        model_path: str,
        metrics: Dict,
        metadata: Dict = None,
        chain_id: int = 8453
    ) -> str:
        """
        Register a new model version
        
        Args:
            horizon: Prediction horizon ('1h', '4h', '24h')
            model_path: Path to model file
            metrics: Model performance metrics
            metadata: Additional metadata (training params, feature info, etc.)
            chain_id: Chain ID
            
        Returns:
            Version string (e.g., 'v1.2.3')
        """
        model_key = f"{chain_id}_{horizon}"
        
        # Get next version
        if model_key not in self.registry['models']:
            self.registry['models'][model_key] = []
            version = 'v1.0.0'
        else:
            versions = self.registry['models'][model_key]
            if versions:
                last_version = versions[-1]['version']
                # Increment patch version
                parts = last_version[1:].split('.')
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                patch += 1
                version = f"v{major}.{minor}.{patch}"
            else:
                version = 'v1.0.0'
        
        # Create version directory
        version_dir = os.path.join(self.versions_dir, model_key, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model to version directory
        version_model_path = os.path.join(version_dir, f'model_{horizon}.pkl')
        if os.path.exists(model_path):
            shutil.copy2(model_path, version_model_path)
        
        # Create version metadata
        version_data = {
            'version': version,
            'horizon': horizon,
            'chain_id': chain_id,
            'registered_at': datetime.now().isoformat(),
            'model_path': version_model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'is_active': False,
            'performance_score': self._calculate_performance_score(metrics)
        }
        
        # Add to registry
        self.registry['models'][model_key].append(version_data)
        
        # Auto-activate if this is the first version or if it's better
        if not self.registry['active_versions'].get(model_key) or \
           version_data['performance_score'] > self._get_active_score(model_key):
            self.activate_version(horizon, version, chain_id)
        
        self._save_registry()
        
        logger.info(f"✅ Registered {model_key} version {version}")
        logger.info(f"   Performance score: {version_data['performance_score']:.4f}")
        
        return version
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """
        Calculate overall performance score from metrics
        Higher is better
        """
        r2 = metrics.get('r2', 0)
        directional_acc = metrics.get('directional_accuracy', 0)
        mape = metrics.get('mape', 100)
        
        # Normalize MAPE (lower is better, so invert)
        mape_score = max(0, 1 - (mape / 100))
        
        # Weighted combination
        score = (r2 * 0.4) + (directional_acc * 0.4) + (mape_score * 0.2)
        return score
    
    def _get_active_score(self, model_key: str) -> float:
        """Get performance score of currently active version"""
        active_version = self.registry['active_versions'].get(model_key)
        if not active_version:
            return -1
        
        for version_data in self.registry['models'].get(model_key, []):
            if version_data['version'] == active_version:
                return version_data.get('performance_score', 0)
        return -1
    
    def activate_version(self, horizon: str, version: str, chain_id: int = 8453):
        """
        Activate a specific model version
        
        Args:
            horizon: Prediction horizon
            version: Version string (e.g., 'v1.2.3')
            chain_id: Chain ID
        """
        model_key = f"{chain_id}_{horizon}"
        
        # Find version
        version_data = None
        for v in self.registry['models'].get(model_key, []):
            if v['version'] == version:
                version_data = v
                break
        
        if not version_data:
            raise ValueError(f"Version {version} not found for {model_key}")
        
        # Deactivate current version
        if model_key in self.registry['active_versions']:
            old_version = self.registry['active_versions'][model_key]
            for v in self.registry['models'].get(model_key, []):
                if v['version'] == old_version:
                    v['is_active'] = False
                    break
        
        # Activate new version
        version_data['is_active'] = True
        self.registry['active_versions'][model_key] = version
        
        # Copy to active location
        active_path = os.path.join(Config.MODELS_DIR, f'model_{horizon}.pkl')
        if os.path.exists(version_data['model_path']):
            shutil.copy2(version_data['model_path'], active_path)
            logger.info(f"✅ Activated {model_key} version {version}")
        else:
            logger.warning(f"⚠️ Model file not found: {version_data['model_path']}")
        
        self._save_registry()
    
    def rollback(self, horizon: str, chain_id: int = 8453, steps: int = 1) -> str:
        """
        Rollback to previous version
        
        Args:
            horizon: Prediction horizon
            chain_id: Chain ID
            steps: Number of versions to rollback (default: 1)
            
        Returns:
            New active version string
        """
        model_key = f"{chain_id}_{horizon}"
        current_version = self.registry['active_versions'].get(model_key)
        
        if not current_version:
            raise ValueError(f"No active version for {model_key}")
        
        versions = self.registry['models'].get(model_key, [])
        if len(versions) <= steps:
            raise ValueError(f"Not enough versions to rollback {steps} steps")
        
        # Find current version index
        current_idx = None
        for i, v in enumerate(versions):
            if v['version'] == current_version:
                current_idx = i
                break
        
        if current_idx is None or current_idx < steps:
            raise ValueError(f"Cannot rollback {steps} steps from {current_version}")
        
        # Get previous version
        previous_version = versions[current_idx - steps]
        new_version = previous_version['version']
        
        # Activate previous version
        self.activate_version(horizon, new_version, chain_id)
        
        # Record rollback
        self.registry['rollback_history'].append({
            'model_key': model_key,
            'from_version': current_version,
            'to_version': new_version,
            'rolled_back_at': datetime.now().isoformat(),
            'reason': f'Manual rollback {steps} steps'
        })
        self._save_registry()
        
        logger.warning(f"⚠️ Rolled back {model_key} from {current_version} to {new_version}")
        
        return new_version
    
    def get_versions(self, horizon: str, chain_id: int = 8453) -> List[Dict]:
        """Get all versions for a model"""
        model_key = f"{chain_id}_{horizon}"
        return self.registry['models'].get(model_key, [])
    
    def get_active_version(self, horizon: str, chain_id: int = 8453) -> Optional[Dict]:
        """Get currently active version"""
        model_key = f"{chain_id}_{horizon}"
        active_version = self.registry['active_versions'].get(model_key)
        
        if not active_version:
            return None
        
        for version_data in self.registry['models'].get(model_key, []):
            if version_data['version'] == active_version:
                return version_data
        
        return None
    
    def get_registry_summary(self) -> Dict:
        """Get summary of all models and versions"""
        summary = {
            'total_models': len(self.registry['active_versions']),
            'total_versions': sum(len(v) for v in self.registry['models'].values()),
            'active_models': {},
            'rollback_count': len(self.registry['rollback_history'])
        }
        
        for model_key, active_version in self.registry['active_versions'].items():
            version_data = self.get_active_version(
                model_key.split('_')[1],  # horizon
                int(model_key.split('_')[0])  # chain_id
            )
            if version_data:
                summary['active_models'][model_key] = {
                    'version': active_version,
                    'performance_score': version_data.get('performance_score', 0),
                    'metrics': version_data.get('metrics', {}),
                    'registered_at': version_data.get('registered_at')
                }
        
        return summary


# Global registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance

