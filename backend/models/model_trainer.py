import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


class GasModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        
    def train_all_models(self, X, y_1h, y_4h, y_24h):
        """
        Train models for all prediction horizons
        Returns: Dictionary of trained models and metrics
        """
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
            
            # Remove NaN values
            valid_idx = ~(y.isna() | X.isna().any(axis=1))
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            if len(X_clean) < 50:
                print(f"⚠️  Not enough valid data for {horizon}, skipping...")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, shuffle=False  # Time series - don't shuffle!
            )
            
            # Train multiple model types
            models = self._train_model_variants(X_train, y_train, X_test, y_test)
            
            # Save best model
            best_model = self._select_best_model(models, horizon)
            
            results[horizon] = {
                'models': models,
                'best': best_model
            }
            
            print(f"✅ Best model for {horizon}: {best_model['name']}")
            print(f"   MAE: {best_model['metrics']['mae']:.6f}")
            print(f"   RMSE: {best_model['metrics']['rmse']:.6f}")
            print(f"   R²: {best_model['metrics']['r2']:.4f}")
        
        return results
    
    def _train_model_variants(self, X_train, y_train, X_test, y_test):
        """Train multiple model architectures"""
        models = []
        
        # 1. Random Forest
        print("\n📊 Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
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
            n_estimators=100,
            max_depth=5,
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
    
    def _select_best_model(self, models, horizon):
        """Select best model based on MAE"""
        best = min(models, key=lambda m: m['metrics']['mae'])
        
        # Save the best model
        self.best_models[horizon] = best
        
        return best
    
    def save_models(self, output_dir='models/saved_models'):
        """Save all best models to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for horizon, model_info in self.best_models.items():
            filepath = os.path.join(output_dir, f'model_{horizon}.pkl')
            
            # Save model + metadata
            save_data = {
                'model': model_info['model'],
                'model_name': model_info['name'],
                'metrics': model_info['metrics'],
                'trained_at': datetime.now().isoformat()
            }
            
            joblib.dump(save_data, filepath)
            print(f"💾 Saved {horizon} model to {filepath}")
    
    @staticmethod
    def load_model(horizon, model_dir='models/saved_models'):
        """Load a trained model"""
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

