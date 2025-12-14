import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble model combining multiple ML algorithms for robust predictions
    with confidence intervals
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        self.trained = False
        self.feature_names = None
        
    def train(self, X, y, feature_names):
        """Train all models in the ensemble"""
        self.feature_names = feature_names
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            model.fit(X, y)
            
            # Evaluate
            train_pred = model.predict(X)
            mae = mean_absolute_error(y, train_pred)
            r2 = r2_score(y, train_pred)
            
            logger.info(f"{model_name} - MAE: {mae:.6f}, R²: {r2:.4f}")
        
        self.trained = True
        
    def predict_with_confidence(self, X):
        """
        Generate predictions with confidence intervals
        
        Returns:
            dict with:
                - prediction: mean prediction
                - lower_bound: lower 95% confidence bound
                - upper_bound: upper 95% confidence bound
                - confidence_score: 0-1 score (higher = more confident)
                - individual_predictions: dict of predictions from each model
        """
        if not self.trained:
            raise ValueError("Models must be trained first")
        
        predictions = {}
        pred_values = []
        
        # Get prediction from each model
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions[model_name] = pred
            pred_values.append(pred)
        
        # Convert to numpy array for calculations
        pred_array = np.array(pred_values)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # 95% confidence interval (1.96 standard deviations)
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        # Ensure bounds are non-negative (gas prices can't be negative)
        lower_bound = np.maximum(lower_bound, 0)
        
        # Confidence score (inverse of coefficient of variation)
        # Higher when predictions agree, lower when they disagree
        cv = std_pred / (mean_pred + 1e-10)  # Avoid division by zero
        confidence_score = 1 / (1 + cv)
        
        return {
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred,
            'confidence_score': confidence_score,
            'individual_predictions': predictions
        }
    
    def get_confidence_level(self, confidence_score):
        """
        Convert confidence score to human-readable level
        
        Returns: ('high'|'medium'|'low', emoji, color)
        """
        if isinstance(confidence_score, np.ndarray):
            confidence_score = confidence_score[0] if len(confidence_score) > 0 else 0.5
        
        if confidence_score >= 0.8:
            return ('high', '🟢', 'green')
        elif confidence_score >= 0.6:
            return ('medium', '🟡', 'yellow')
        else:
            return ('low', '🔴', 'red')
    
    def save_models(self, base_path='models/saved_models'):
        """Save all models in the ensemble"""
        os.makedirs(base_path, exist_ok=True)
        for model_name, model in self.models.items():
            filepath = f"{base_path}/ensemble_{model_name}.pkl"
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")
    
    def load_models(self, base_path='models/saved_models'):
        """Load all models in the ensemble"""
        loaded = False
        for model_name in self.models.keys():
            filepath = f"{base_path}/ensemble_{model_name}.pkl"
            try:
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded {model_name} from {filepath}")
                    loaded = True
                else:
                    logger.warning(f"Model file not found: {filepath}")
            except Exception as e:
                logger.warning(f"Error loading {model_name}: {e}")
        
        if loaded:
            self.trained = True
        return loaded


# Singleton instance
ensemble_predictor = EnsemblePredictor()

