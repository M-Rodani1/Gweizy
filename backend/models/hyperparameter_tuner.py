"""
Hyperparameter Tuning for Gas Price Prediction Models

Uses Optuna for systematic optimization of model hyperparameters with
time-series cross-validation to prevent overfitting and ensure robust performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - hyperparameter tuning disabled")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge


class HyperparameterTuner:
    """
    Systematic hyperparameter optimization for ML models using Optuna.
    
    Uses time-series cross-validation to ensure proper evaluation
    without future data leakage.
    """
    
    def __init__(self, n_trials: int = 50, cv_splits: int = 3, random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            n_trials: Number of optimization trials (higher = better but slower)
            cv_splits: Number of time-series CV splits
            random_state: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}
    
    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            scores = self._cross_validate(RandomForestRegressor(**params), X, y)
            return scores['r2_mean']
        
        study = optuna.create_study(direction='maximize', study_name='random_forest')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params['RandomForest'] = study.best_params
        self.best_scores['RandomForest'] = study.best_value
        
        logger.info(f"Random Forest best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize Gradient Boosting hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
            
            scores = self._cross_validate(GradientBoostingRegressor(**params), X, y)
            return scores['r2_mean']
        
        study = optuna.create_study(direction='maximize', study_name='gradient_boosting')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params['GradientBoosting'] = study.best_params
        self.best_scores['GradientBoosting'] = study.best_value
        
        logger.info(f"Gradient Boosting best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available - skipping optimization")
            return {}
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1,
                'objective': 'regression',
                'metric': 'rmse'
            }
            
            scores = self._cross_validate(lgb.LGBMRegressor(**params), X, y)
            return scores['r2_mean']
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params['LightGBM'] = study.best_params
        self.best_scores['LightGBM'] = study.best_value
        
        logger.info(f"LightGBM best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available - skipping optimization")
            return {}
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'n_jobs': -1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }
            
            scores = self._cross_validate(xgb.XGBRegressor(**params), X, y)
            return scores['r2_mean']
        
        study = optuna.create_study(direction='maximize', study_name='xgboost')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params['XGBoost'] = study.best_params
        self.best_scores['XGBoost'] = study.best_value
        
        logger.info(f"XGBoost best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def optimize_ridge(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize Ridge Regression hyperparameters."""
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
            
            params = {
                'alpha': alpha,
                'random_state': self.random_state
            }
            
            scores = self._cross_validate(Ridge(**params), X, y)
            return scores['r2_mean']
        
        study = optuna.create_study(direction='maximize', study_name='ridge')
        study.optimize(objective, n_trials=20, show_progress_bar=False)  # Fewer trials for simple model
        
        self.best_params['Ridge'] = study.best_params
        self.best_scores['Ridge'] = study.best_value
        
        logger.info(f"Ridge best R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Model instance to evaluate
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        r2_scores = []
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            r2 = r2_score(y_val_fold, y_pred)
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            r2_scores.append(r2)
            mae_scores.append(mae)
        
        return {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores)
        }
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series, 
                           model_types: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all model types.
        
        Args:
            X: Feature matrix
            y: Target values
            model_types: List of model types to optimize. If None, optimizes all available.
            
        Returns:
            Dictionary mapping model names to best hyperparameters
        """
        if model_types is None:
            model_types = ['RandomForest', 'GradientBoosting', 'Ridge']
            if LIGHTGBM_AVAILABLE:
                model_types.append('LightGBM')
            if XGBOOST_AVAILABLE:
                model_types.append('XGBoost')
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing {model_type}...")
            logger.info(f"{'='*60}")
            
            try:
                if model_type == 'RandomForest':
                    results[model_type] = self.optimize_random_forest(X, y)
                elif model_type == 'GradientBoosting':
                    results[model_type] = self.optimize_gradient_boosting(X, y)
                elif model_type == 'LightGBM':
                    results[model_type] = self.optimize_lightgbm(X, y)
                elif model_type == 'XGBoost':
                    results[model_type] = self.optimize_xgboost(X, y)
                elif model_type == 'Ridge':
                    results[model_type] = self.optimize_ridge(X, y)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
                continue
        
        return results


# Example usage
if __name__ == "__main__":
    from models.feature_engineering import GasFeatureEngineer
    
    # Prepare data
    engineer = GasFeatureEngineer()
    df = engineer.prepare_training_data(hours_back=720)
    
    feature_cols = engineer.get_feature_columns(df)
    X = df[feature_cols]
    y = df['target_1h']
    
    # Remove NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Optimize
    tuner = HyperparameterTuner(n_trials=30, cv_splits=3)
    best_params = tuner.optimize_all_models(X, y)
    
    print("\n✅ Hyperparameter optimization complete!")
    print(f"Best parameters: {best_params}")
