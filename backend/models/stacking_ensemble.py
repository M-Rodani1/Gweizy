"""
Week 1 Quick Win #5: Stacking Ensemble

Implements a proper stacking ensemble that uses a meta-learner to combine
base models (RandomForest, GradientBoosting, Ridge) for improved accuracy.

Expected improvement: +0.10 R² over single best model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble for gas price prediction
    
    Uses a two-level approach:
    1. Base models (RandomForest, GradientBoosting, Ridge) make predictions
    2. Meta-learner (Ridge) learns to combine base model predictions optimally
    """
    
    def __init__(self):
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        self.meta_learner = Ridge(alpha=0.1, random_state=42)
        self.scaler = RobustScaler()
        self.trained = False
        self.meta_scaler = RobustScaler()  # For meta-features
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train stacking ensemble
        
        Args:
            X_train: Training features (already scaled)
            y_train: Training targets
            X_val: Validation features (optional, for out-of-fold predictions)
            y_val: Validation targets (optional)
        """
        print("\n📊 Training Stacking Ensemble (Week 1 Quick Win #5)")
        print("="*60)
        
        # If no validation set provided, use 20% of training for meta-learner
        if X_val is None or y_val is None:
            split_idx = int(len(X_train) * 0.8)
            X_base = X_train[:split_idx]
            y_base = y_train[:split_idx]
            X_meta = X_train[split_idx:]
            y_meta = y_train[split_idx:]
        else:
            X_base = X_train
            y_base = y_train
            X_meta = X_val
            y_meta = y_val
        
        print(f"   Base models training set: {len(X_base)} samples")
        print(f"   Meta-learner training set: {len(X_meta)} samples")
        
        # Step 1: Train base models on base training set
        print("\n📊 Step 1: Training base models...")
        base_predictions = {}
        
        for name, model in self.base_models.items():
            print(f"   Training {name}...")
            model.fit(X_base, y_base)
            
            # Get predictions on meta training set (out-of-fold)
            meta_pred = model.predict(X_meta)
            base_predictions[name] = meta_pred
            
            # Evaluate base model
            base_pred_train = model.predict(X_base)
            base_r2 = r2_score(y_base, base_pred_train)
            print(f"      Base {name} R²: {base_r2:.4f}")
        
        # Step 2: Create meta-features from base model predictions
        print("\n📊 Step 2: Creating meta-features...")
        meta_features = np.column_stack([
            base_predictions['random_forest'],
            base_predictions['gradient_boosting'],
            base_predictions['ridge']
        ])
        
        # Scale meta-features
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        
        # Step 3: Train meta-learner on meta-features
        print("📊 Step 3: Training meta-learner...")
        self.meta_learner.fit(meta_features_scaled, y_meta)
        
        # Evaluate meta-learner
        meta_pred = self.meta_learner.predict(meta_features_scaled)
        meta_r2 = r2_score(y_meta, meta_pred)
        meta_mae = mean_absolute_error(y_meta, meta_pred)
        
        print(f"   Meta-learner R²: {meta_r2:.4f}")
        print(f"   Meta-learner MAE: {meta_mae:.6f}")
        
        # Step 4: Retrain base models on full training set for final predictions
        print("\n📊 Step 4: Retraining base models on full dataset...")
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
        
        self.trained = True
        
        return {
            'base_models': {name: r2_score(y_base, self.base_models[name].predict(X_base)) 
                          for name in self.base_models.keys()},
            'meta_learner_r2': meta_r2,
            'meta_learner_mae': meta_mae
        }
    
    def predict(self, X):
        """
        Make predictions using stacking ensemble
        
        Args:
            X: Features (should be scaled with same scaler as training)
            
        Returns:
            Predictions from meta-learner
        """
        if not self.trained:
            raise ValueError("Ensemble must be trained first")
        
        # Get predictions from all base models
        base_predictions = {}
        for name, model in self.base_models.items():
            base_predictions[name] = model.predict(X)
        
        # Create meta-features
        meta_features = np.column_stack([
            base_predictions['random_forest'],
            base_predictions['gradient_boosting'],
            base_predictions['ridge']
        ])
        
        # Scale meta-features
        meta_features_scaled = self.meta_scaler.transform(meta_features)
        
        # Meta-learner makes final prediction
        predictions = self.meta_learner.predict(meta_features_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble on test set"""
        y_pred = self.predict(X_test)
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'directional_accuracy': self._directional_accuracy(y_test, y_pred)
        }
    
    def _directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy"""
        if len(y_true) < 2:
            return 0.0
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        return np.mean(actual_direction == pred_direction)
    
    def save(self, output_dir='models/saved_models', horizon='1h'):
        """Save stacking ensemble"""
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f'stacking_ensemble_{horizon}.pkl')
        
        save_data = {
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'meta_scaler': self.meta_scaler,
            'trained_at': datetime.now().isoformat(),
            'horizon': horizon
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved stacking ensemble to {filepath}")
        print(f"💾 Saved stacking ensemble to {filepath}")
    
    @staticmethod
    def load(horizon='1h', model_dir='models/saved_models'):
        """Load stacking ensemble"""
        filepath = os.path.join(model_dir, f'stacking_ensemble_{horizon}.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stacking ensemble not found: {filepath}")
        
        data = joblib.load(filepath)
        ensemble = StackingEnsemble()
        ensemble.base_models = data['base_models']
        ensemble.meta_learner = data['meta_learner']
        ensemble.meta_scaler = data['meta_scaler']
        ensemble.trained = True
        
        return ensemble
