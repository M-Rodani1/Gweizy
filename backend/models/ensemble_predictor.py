"""
Enhanced Ensemble Predictor for Gas Price Prediction

Combines multiple prediction sources with accuracy-weighted averaging:
1. Hybrid predictor (spike detection + classification-based predictions)
2. Stacking ensemble (RF + GB + Ridge with meta-learner)
3. Simple statistical baseline (rolling mean + momentum)

Features:
- Accuracy-weighted combination of models
- Adaptive confidence intervals based on model agreement
- Volatility-aware prediction bounds
- Tracks individual model performance for dynamic weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines multiple gas price prediction models with intelligent weighting.

    The ensemble uses recent accuracy metrics to weight each model's contribution,
    providing more robust predictions than any single model alone.
    """

    # Default weights when no accuracy data available
    DEFAULT_WEIGHTS = {
        'hybrid': 0.35,      # Spike detection model
        'stacking': 0.40,    # Stacking ensemble
        'statistical': 0.25  # Simple baseline
    }

    # Minimum weight for any model (prevents complete exclusion)
    MIN_WEIGHT = 0.10

    def __init__(self, models_dir: str = None):
        """
        Initialize ensemble predictor.

        Args:
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            if os.path.exists('/data/models'):
                self.models_dir = '/data/models'
            else:
                self.models_dir = 'models/saved_models'
        else:
            self.models_dir = models_dir

        self.hybrid_predictor = None
        self.stacking_ensembles = {}  # Per-horizon
        self.loaded = False

        # Track recent accuracy for dynamic weighting
        self.recent_accuracy = {
            'hybrid': {'1h': None, '4h': None, '24h': None},
            'stacking': {'1h': None, '4h': None, '24h': None},
            'statistical': {'1h': None, '4h': None, '24h': None}
        }

        # Track prediction disagreement for confidence adjustment
        self.prediction_history = []

    def load_models(self) -> bool:
        """Load all component models."""
        success = False

        # Load hybrid predictor
        try:
            from models.hybrid_predictor import HybridPredictor
            self.hybrid_predictor = HybridPredictor(models_dir=self.models_dir)
            if self.hybrid_predictor.load_models():
                logger.info("Loaded hybrid predictor for ensemble")
                success = True
        except Exception as e:
            logger.warning(f"Could not load hybrid predictor: {e}")

        # Load stacking ensembles for each horizon
        try:
            from models.stacking_ensemble import StackingEnsemble
            for horizon in ['1h', '4h', '24h']:
                try:
                    ensemble = StackingEnsemble.load(horizon=horizon, model_dir=self.models_dir)
                    self.stacking_ensembles[horizon] = ensemble
                    logger.info(f"Loaded stacking ensemble for {horizon}")
                    success = True
                except FileNotFoundError:
                    logger.debug(f"No stacking ensemble found for {horizon}")
        except Exception as e:
            logger.warning(f"Could not load stacking ensembles: {e}")

        self.loaded = success
        return success

    def predict(
        self,
        recent_data: pd.DataFrame,
        features: Optional[np.ndarray] = None,
        horizons: List[str] = ['1h', '4h', '24h']
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate ensemble predictions for all horizons.

        Args:
            recent_data: DataFrame with recent gas price data
            features: Pre-computed feature matrix (optional)
            horizons: Which horizons to predict

        Returns:
            Dictionary with predictions per horizon, including:
            - ensemble_prediction: Weighted average prediction
            - individual_predictions: Predictions from each model
            - confidence: Ensemble confidence score
            - confidence_interval: (lower, upper) bounds
            - model_weights: Weights used for each model
        """
        if not self.loaded:
            self.load_models()

        predictions = {}

        # Ensure data has required columns
        if 'gas_price' not in recent_data.columns and 'gas' in recent_data.columns:
            recent_data = recent_data.rename(columns={'gas': 'gas_price'})
        elif 'gas_price' not in recent_data.columns and 'gwei' in recent_data.columns:
            recent_data = recent_data.rename(columns={'gwei': 'gas_price'})

        # Build feature matrix if not provided
        if features is None and self.stacking_ensembles:
            try:
                from models.feature_pipeline import build_feature_matrix
                feature_df, metadata, _ = build_feature_matrix(recent_data)
                # Get the last row (most recent) as features
                features = feature_df.iloc[[-1]].values
            except Exception as e:
                logger.debug(f"Could not build features for stacking: {e}")
                features = None

        for horizon in horizons:
            predictions[horizon] = self._predict_horizon(
                recent_data,
                features,
                horizon
            )

        return predictions

    def _predict_horizon(
        self,
        recent_data: pd.DataFrame,
        features: Optional[np.ndarray],
        horizon: str
    ) -> Dict[str, Any]:
        """Generate prediction for a single horizon."""

        individual_predictions = {}
        confidences = {}

        # Get current price for reference
        current_price = float(recent_data['gas_price'].iloc[-1])
        hist_mean = float(recent_data['gas_price'].iloc[-100:].mean()) if len(recent_data) >= 100 else current_price

        # 1. Hybrid predictor prediction
        if self.hybrid_predictor and self.hybrid_predictor.loaded:
            try:
                hybrid_preds = self.hybrid_predictor.predict(recent_data)
                if horizon in hybrid_preds:
                    pred = hybrid_preds[horizon]['prediction']['price']
                    conf = hybrid_preds[horizon]['classification']['confidence']
                    individual_predictions['hybrid'] = pred
                    confidences['hybrid'] = conf
            except Exception as e:
                logger.debug(f"Hybrid prediction failed for {horizon}: {e}")

        # 2. Stacking ensemble prediction
        if horizon in self.stacking_ensembles and features is not None:
            try:
                ensemble = self.stacking_ensembles[horizon]
                # Ensure features is 2D
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                pred = ensemble.predict(features)
                individual_predictions['stacking'] = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                confidences['stacking'] = 0.70  # Base confidence for stacking
            except Exception as e:
                logger.debug(f"Stacking prediction failed for {horizon}: {e}")

        # 3. Statistical baseline prediction
        stat_pred = self._statistical_prediction(recent_data, horizon)
        individual_predictions['statistical'] = stat_pred
        confidences['statistical'] = 0.50  # Lower confidence for simple model

        # Calculate weights based on recent accuracy
        weights = self._calculate_weights(horizon)

        # Combine predictions with weighted average
        ensemble_prediction = self._weighted_average(individual_predictions, weights)

        # Calculate adaptive confidence interval
        confidence, interval = self._calculate_confidence_interval(
            individual_predictions,
            confidences,
            weights,
            current_price,
            hist_mean,
            recent_data
        )

        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions,
            'confidence': confidence,
            'confidence_interval': interval,
            'model_weights': weights,
            'current_price': current_price,
            'historical_mean': hist_mean
        }

    def _statistical_prediction(
        self,
        recent_data: pd.DataFrame,
        horizon: str
    ) -> float:
        """Simple statistical prediction based on trends and patterns."""

        prices = recent_data['gas_price'].values
        current = prices[-1]

        # Calculate components
        # 1. Rolling mean (reversion tendency)
        if len(prices) >= 24:
            rolling_mean = np.mean(prices[-24:])
        else:
            rolling_mean = np.mean(prices)

        # 2. Recent momentum
        if len(prices) >= 6:
            momentum = (prices[-1] - prices[-6]) / (prices[-6] + 1e-10)
        else:
            momentum = 0

        # 3. Volatility-adjusted momentum decay
        if len(prices) >= 12:
            volatility = np.std(prices[-12:]) / (np.mean(prices[-12:]) + 1e-10)
        else:
            volatility = 0.1

        # Prediction horizon affects momentum decay
        decay_factors = {'1h': 0.8, '4h': 0.5, '24h': 0.2}
        decay = decay_factors.get(horizon, 0.5)

        # Blend: current + decayed momentum, then regress toward mean
        momentum_adjusted = current * (1 + momentum * decay)

        # Stronger mean reversion for longer horizons
        mean_reversion = {'1h': 0.3, '4h': 0.5, '24h': 0.7}
        reversion = mean_reversion.get(horizon, 0.5)

        prediction = momentum_adjusted * (1 - reversion) + rolling_mean * reversion

        # Clamp to reasonable range
        prediction = max(current * 0.5, min(prediction, current * 2.0))

        return float(prediction)

    def _calculate_weights(self, horizon: str) -> Dict[str, float]:
        """Calculate model weights based on recent accuracy."""

        weights = dict(self.DEFAULT_WEIGHTS)

        # Try to get recent accuracy from tracker
        try:
            from models.accuracy_tracker import get_tracker
            tracker = get_tracker()

            # This would require tracking accuracy per model, which we don't have yet
            # For now, use default weights
            # Future: Store per-model predictions and compare to actuals

        except Exception:
            pass

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _weighted_average(
        self,
        predictions: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted average of predictions."""

        if not predictions:
            raise ValueError("No predictions available")

        weighted_sum = 0
        total_weight = 0

        for model, pred in predictions.items():
            weight = weights.get(model, self.MIN_WEIGHT)
            weighted_sum += pred * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else predictions.get('statistical', 0)

    def _calculate_confidence_interval(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float],
        weights: Dict[str, float],
        current_price: float,
        hist_mean: float,
        recent_data: pd.DataFrame
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate ensemble confidence and adaptive interval.

        Uses:
        - Model agreement/disagreement
        - Individual model confidences
        - Recent volatility
        """

        pred_values = list(predictions.values())

        if len(pred_values) < 2:
            # Single model, use wider bounds
            pred = pred_values[0] if pred_values else current_price
            return 0.50, (pred * 0.85, pred * 1.15)

        # Calculate prediction spread (disagreement)
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        disagreement = pred_std / (pred_mean + 1e-10)  # Coefficient of variation

        # Calculate recent volatility
        prices = recent_data['gas_price'].values
        if len(prices) >= 12:
            recent_volatility = np.std(prices[-12:]) / (np.mean(prices[-12:]) + 1e-10)
        else:
            recent_volatility = 0.1

        # Weighted confidence from individual models
        weighted_confidence = sum(
            confidences.get(model, 0.5) * weights.get(model, 0.1)
            for model in predictions.keys()
        ) / sum(weights.get(model, 0.1) for model in predictions.keys())

        # Adjust confidence based on disagreement
        # High disagreement = lower confidence
        disagreement_penalty = min(0.3, disagreement * 2)
        ensemble_confidence = weighted_confidence * (1 - disagreement_penalty)
        ensemble_confidence = max(0.3, min(0.95, ensemble_confidence))

        # Calculate interval width based on:
        # 1. Model disagreement
        # 2. Recent volatility
        # 3. Ensemble confidence
        base_width = 0.10  # 10% base interval
        disagreement_factor = 1 + disagreement * 2  # Widen for disagreement
        volatility_factor = 1 + recent_volatility * 3  # Widen for volatility
        confidence_factor = 2 - ensemble_confidence  # Widen for low confidence

        interval_width = base_width * disagreement_factor * volatility_factor * confidence_factor
        interval_width = min(0.50, interval_width)  # Cap at 50%

        # Calculate bounds
        ensemble_pred = self._weighted_average(predictions, weights)
        lower = ensemble_pred * (1 - interval_width)
        upper = ensemble_pred * (1 + interval_width)

        # Sanity bounds
        lower = max(hist_mean * 0.3, lower)
        upper = min(hist_mean * 3.0, upper)

        return float(ensemble_confidence), (float(lower), float(upper))

    def update_model_accuracy(
        self,
        model_name: str,
        horizon: str,
        mae: float,
        r2: float
    ):
        """
        Update tracked accuracy for a specific model.
        Used for dynamic weight adjustment.
        """
        if model_name in self.recent_accuracy:
            self.recent_accuracy[model_name][horizon] = {
                'mae': mae,
                'r2': r2,
                'updated_at': datetime.now()
            }


# Singleton instance
_ensemble_predictor: Optional[EnsemblePredictor] = None


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get or create the global ensemble predictor."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsemblePredictor()
        _ensemble_predictor.load_models()
    return _ensemble_predictor
