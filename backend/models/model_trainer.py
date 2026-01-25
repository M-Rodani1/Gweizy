import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit  # Week 1 Quick Win #4: Time-series CV
from sklearn.preprocessing import RobustScaler  # Week 1 Quick Win #3: RobustScaler for outlier handling
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


class GasModelTrainer:
    def __init__(self, chain_id: int = 8453):
        """
        Initialize model trainer for a specific chain.
        
        Args:
            chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
        """
        self.chain_id = chain_id
        self.models = {}
        self.best_models = {}
        self.scalers = {}  # Store RobustScalers for each horizon (Week 1 Quick Win #3)
        self.feature_names = []
        
    def train_all_models(self, X, y_1h, y_4h, y_24h, chain_id: int = None):
        """
        Train models for all prediction horizons for a specific chain.
        
        Args:
            X: Feature matrix
            y_1h: Target values for 1h horizon
            y_4h: Target values for 4h horizon
            y_24h: Target values for 24h horizon
            chain_id: Chain ID (defaults to self.chain_id)
        
        Returns: Dictionary of trained models and metrics
        """
        if chain_id is not None:
            self.chain_id = chain_id
        
        horizons = {
            '1h': y_1h,
            '4h': y_4h,
            '24h': y_24h
        }
        
        results = {}
        
        for horizon, y in horizons.items():
            print(f"\n{'='*60}")
            print(f"🎯 Training models for {horizon} prediction horizon")
            print(f"{'='*60}")
            
            if len(y) == 0 or y.isna().all():
                print(f"⚠️  No target data for {horizon}, skipping...")
                continue
            
            # Align X with y (y is already filtered for this horizon via shift operations)
            # Get common indices
            common_idx = X.index.intersection(y.index)
            
            if len(common_idx) == 0:
                print(f"⚠️  No overlapping indices for {horizon}, skipping...")
                continue
            
            X_aligned = X.loc[common_idx]
            y_aligned = y.loc[common_idx]

            if not self.feature_names:
                self.feature_names = list(X_aligned.columns)
            
            print(f"   After alignment: {len(X_aligned)} samples")
            
            # Remove NaN values - be lenient with enhanced features
            # Only require core features and target to be non-NaN
            # Enhanced features can be 0 (already filled)
            y_valid = ~y_aligned.isna()
            
            # Check for NaN in core features (not enhanced features)
            # Enhanced features are: pending_tx_count, unique_addresses, etc.
            enhanced_feature_prefixes = ['pending_', 'unique_', 'tx_per_', 'gas_utilization_ratio', 
                                        'avg_tx_gas', 'large_tx_ratio', 'congestion_level', 'is_highly_congested']
            core_features = [col for col in X_aligned.columns 
                           if not any(col.startswith(prefix) for prefix in enhanced_feature_prefixes)]
            
            if core_features:
                X_core_valid = ~X_aligned[core_features].isna().any(axis=1)
            else:
                X_core_valid = pd.Series([True] * len(X_aligned), index=X_aligned.index)
            
            valid_idx = y_valid & X_core_valid
            X_clean = X_aligned[valid_idx]
            y_clean = y_aligned[valid_idx]
            
            print(f"   After NaN removal: {len(X_clean)} samples")
            
            if len(X_clean) < 50:
                print(f"⚠️  Not enough valid data for {horizon} ({len(X_clean)} samples), skipping...")
                continue
            
            # Week 1 Quick Win #4: Use time-series cross-validation for better evaluation
            # Time-series data requires temporal ordering - can't shuffle!
            # Split: 80% train, 20% test (maintain temporal order)
            split_idx = int(len(X_clean) * 0.8)
            X_train = X_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_train = y_clean.iloc[:split_idx]
            y_test = y_clean.iloc[split_idx:]
            
            # Week 1 Quick Win #3: Scale features with RobustScaler (handles outliers better)
            # RobustScaler uses median and IQR instead of mean/std, making it robust to gas price spikes
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler for this horizon
            self.scalers[horizon] = scaler
            
            print(f"✅ Features scaled with RobustScaler (robust to outliers)")
            horizon_params = self._get_horizon_hyperparams(horizon)
            
            # Week 1 Quick Win #4: Time-series cross-validation for model evaluation
            print(f"📊 Running time-series cross-validation...")
            cv_scores = self._time_series_cross_validate(X_train, y_train, horizon=horizon)
            print(f"   CV R² scores: {cv_scores['r2_mean']:.4f} ± {cv_scores['r2_std']:.4f}")
            
            # Train multiple model types (on scaled features)
            models = self._train_model_variants(X_train_scaled, y_train, X_test_scaled, y_test, horizon=horizon)
            
            # Week 1 Quick Win #5: Train stacking ensemble
            print("\n📊 Training Stacking Ensemble...")
            from models.stacking_ensemble import StackingEnsemble
            stacking = StackingEnsemble(base_model_params=horizon_params.get('stacking'))
            stacking.train(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Evaluate stacking ensemble
            stacking_metrics = stacking.evaluate(X_test_scaled, y_test)
            print(f"   Stacking Ensemble R²: {stacking_metrics['r2']:.4f}")
            print(f"   Stacking Ensemble MAE: {stacking_metrics['mae']:.6f}")
            
            # Add stacking to models list
            models.append({
                'name': 'StackingEnsemble',
                'model': stacking,
                'metrics': stacking_metrics
            })
            
            # Save stacking ensemble
            stacking.save(horizon=horizon)
            
            # Save best model (may be stacking or single model)
            best_model = self._select_best_model(models, horizon)
            
            results[horizon] = {
                'models': models,
                'best': best_model,
                'stacking': {
                    'metrics': stacking_metrics,
                    'ensemble': stacking
                }
            }
            
            print(f"✅ Best model for {horizon}: {best_model['name']}")
            print(f"   MAE: {best_model['metrics']['mae']:.6f}")
            print(f"   RMSE: {best_model['metrics']['rmse']:.6f}")
            print(f"   R²: {best_model['metrics']['r2']:.4f}")
        
        return results
    
    def _get_horizon_hyperparams(self, horizon):
        """Return horizon-specific hyperparameters for tree-based models."""
        configs = {
            '1h': {
                'random_forest': {'n_estimators': 200, 'max_depth': 10},
                'gradient_boosting': {'n_estimators': 180, 'max_depth': 3},
                'lightgbm': {'n_estimators': 300, 'max_depth': 6, 'num_leaves': 31},
                'xgboost': {'n_estimators': 280, 'max_depth': 6},
                'stacking': {
                    'random_forest': {'n_estimators': 200, 'max_depth': 10},
                    'gradient_boosting': {'n_estimators': 180, 'max_depth': 3},
                },
                'cv_random_forest': {'n_estimators': 60, 'max_depth': 8},
            },
            '4h': {
                'random_forest': {'n_estimators': 100, 'max_depth': 15},
                'gradient_boosting': {'n_estimators': 100, 'max_depth': 5},
                'lightgbm': {'n_estimators': 200, 'max_depth': 10, 'num_leaves': 31},
                'xgboost': {'n_estimators': 200, 'max_depth': 10},
                'stacking': {
                    'random_forest': {'n_estimators': 100, 'max_depth': 15},
                    'gradient_boosting': {'n_estimators': 100, 'max_depth': 5},
                },
                'cv_random_forest': {'n_estimators': 50, 'max_depth': 10},
            },
            '24h': {
                'random_forest': {'n_estimators': 80, 'max_depth': 20},
                'gradient_boosting': {'n_estimators': 80, 'max_depth': 6},
                'lightgbm': {'n_estimators': 120, 'max_depth': 14, 'num_leaves': 63},
                'xgboost': {'n_estimators': 120, 'max_depth': 14},
                'stacking': {
                    'random_forest': {'n_estimators': 80, 'max_depth': 20},
                    'gradient_boosting': {'n_estimators': 80, 'max_depth': 6},
                },
                'cv_random_forest': {'n_estimators': 40, 'max_depth': 12},
            },
        }
        return configs.get(horizon, configs['4h'])

    def _train_model_variants(self, X_train, y_train, X_test, y_test, horizon='4h'):
        """
        Train multiple model architectures
        
        Note: X_train and X_test are already scaled with RobustScaler
        """
        models = []
        horizon_params = self._get_horizon_hyperparams(horizon)
        
        # 1. Random Forest
        print("\n📊 Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=horizon_params['random_forest']['n_estimators'],
            max_depth=horizon_params['random_forest']['max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models.append({
            'name': 'RandomForest',
            'model': rf,
            'metrics': self._evaluate_model(rf, X_test, y_test)
        })
        
        # 2. Gradient Boosting
        print("📊 Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=horizon_params['gradient_boosting']['n_estimators'],
            max_depth=horizon_params['gradient_boosting']['max_depth'],
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_train, y_train)
        models.append({
            'name': 'GradientBoosting',
            'model': gb,
            'metrics': self._evaluate_model(gb, X_test, y_test)
        })
        
        # 3. Ridge Regression (baseline)
        print("📊 Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        models.append({
            'name': 'Ridge',
            'model': ridge,
            'metrics': self._evaluate_model(ridge, X_test, y_test)
        })
        
        # 4. LightGBM (excellent for time series)
        try:
            print("📊 Training LightGBM...")
            import lightgbm as lgb
            lgbm = lgb.LGBMRegressor(
                n_estimators=horizon_params['lightgbm']['n_estimators'],
                max_depth=horizon_params['lightgbm']['max_depth'],
                learning_rate=0.05,
                num_leaves=horizon_params['lightgbm']['num_leaves'],
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                objective='regression',
                metric='rmse'
            )
            lgbm.fit(X_train, y_train)
            models.append({
                'name': 'LightGBM',
                'model': lgbm,
                'metrics': self._evaluate_model(lgbm, X_test, y_test)
            })
        except ImportError:
            print("⚠️  LightGBM not available - skipping")
        except Exception as e:
            print(f"⚠️  LightGBM training failed: {e}")
        
        # 5. XGBoost (robust gradient boosting)
        try:
            print("📊 Training XGBoost...")
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor(
                n_estimators=horizon_params['xgboost']['n_estimators'],
                max_depth=horizon_params['xgboost']['max_depth'],
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
            xgb_model.fit(X_train, y_train)
            models.append({
                'name': 'XGBoost',
                'model': xgb_model,
                'metrics': self._evaluate_model(xgb_model, X_test, y_test)
            })
        except ImportError:
            print("⚠️  XGBoost not available - skipping")
        except Exception as e:
            print(f"⚠️  XGBoost training failed: {e}")
        
        return models
    
    def _evaluate_model(self, model, X_test, y_test):
        """Calculate evaluation metrics"""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy (did we predict up/down correctly?)
        if len(y_test) > 1:
            y_diff_actual = np.diff(y_test.values)
            y_diff_pred = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(y_diff_pred))
        else:
            directional_accuracy = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def _time_series_cross_validate(self, X, y, horizon='4h', n_splits=5):
        """
        Week 1 Quick Win #4: Time-series cross-validation
        
        Uses TimeSeriesSplit to properly evaluate time-series models
        without data leakage from future to past.
        
        Args:
            X: Feature matrix (already scaled)
            y: Target values
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        horizon_params = self._get_horizon_hyperparams(horizon)
        cv_params = horizon_params.get('cv_random_forest', {'n_estimators': 50, 'max_depth': 10})
        
        # Test with a simple model to get CV scores
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(
                n_estimators=cv_params['n_estimators'],  # Smaller for faster CV
                max_depth=cv_params['max_depth'],
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        cv_r2_scores = cross_val_score(
            pipeline, X, y,
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_mae_scores = -cross_val_score(
            pipeline, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        return {
            'r2_mean': np.mean(cv_r2_scores),
            'r2_std': np.std(cv_r2_scores),
            'mae_mean': np.mean(cv_mae_scores),
            'mae_std': np.std(cv_mae_scores),
            'n_splits': n_splits
        }
    
    def _select_best_model(self, models, horizon):
        """Select best model based on MAE"""
        best = min(models, key=lambda m: m['metrics']['mae'])
        
        # Save the best model
        self.best_models[horizon] = best
        
        return best
    
    def save_models(self, output_dir='models/saved_models', chain_id: int = None):
        """
        Save all best models to disk with RobustScaler for a specific chain.
        
        Week 1 Quick Win #3: RobustScaler is saved with each model
        for consistent scaling during prediction.
        
        Args:
            output_dir: Directory to save models
            chain_id: Chain ID (defaults to self.chain_id)
        """
        if chain_id is not None:
            self.chain_id = chain_id
        
        # Create chain-specific subdirectory
        chain_dir = os.path.join(output_dir, f'chain_{self.chain_id}')
        os.makedirs(chain_dir, exist_ok=True)
        
        for horizon, model_info in self.best_models.items():
            filepath = os.path.join(chain_dir, f'model_{horizon}.pkl')
            
            # Get scaler for this horizon
            scaler = self.scalers.get(horizon)
            
            # Save model + metadata + scaler
            save_data = {
                'model': model_info['model'],
                'model_name': model_info['name'],
                'metrics': model_info['metrics'],
                'trained_at': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'feature_scaler': scaler,  # Week 1 Quick Win #3: RobustScaler
                'scaler_type': 'RobustScaler',  # For compatibility checking
            }
            
            joblib.dump(save_data, filepath)
            print(f"💾 Saved {horizon} model to {filepath}")
            if scaler:
                print(f"   ✅ Included RobustScaler for feature scaling")
    
    @staticmethod
    def load_model(horizon, model_dir='models/saved_models', chain_id: int = 8453):
        """
        Load a trained model for a specific chain.
        
        Args:
            horizon: Prediction horizon ('1h', '4h', '24h')
            model_dir: Base directory for models
            chain_id: Chain ID (defaults to 8453 for Base)
        
        Returns:
            Loaded model data with scaler
        """
        # Try chain-specific directory first
        chain_dir = os.path.join(model_dir, f'chain_{chain_id}')
        filepath = os.path.join(chain_dir, f'model_{horizon}.pkl')
        
        # Fallback to old location for backward compatibility
        if not os.path.exists(filepath):
            filepath = os.path.join(model_dir, f'model_{horizon}.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        data = joblib.load(filepath)
        return data


# Example usage
if __name__ == "__main__":
    from models.feature_engineering import GasFeatureEngineer
    
    # Prepare data
    engineer = GasFeatureEngineer()
    df = engineer.prepare_training_data(hours_back=720)
    
    # Get features and targets
    feature_cols = engineer.get_feature_columns(df)
    X = df[feature_cols]
    y_1h = df['target_1h']
    y_4h = df['target_4h']
    y_24h = df['target_24h']
    
    # Train models
    trainer = GasModelTrainer()
    results = trainer.train_all_models(X, y_1h, y_4h, y_24h)
    
    # Save best models
    trainer.save_models()
