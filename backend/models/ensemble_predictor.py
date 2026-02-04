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
import json

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

        # Tail risk caps from training (±3σ bounds)
        self.tail_risk_caps = {}
        self.monitoring_config = {}
        self.bias_tracker = None

        # Track recent accuracy for dynamic weighting
        self.recent_accuracy = {
            'hybrid': {'1h': None, '4h': None, '24h': None},
            'stacking': {'1h': None, '4h': None, '24h': None},
            'statistical': {'1h': None, '4h': None, '24h': None}
        }

        # Track prediction disagreement for confidence adjustment
        self.prediction_history = []

    def _load_tail_risk_caps(self):
        """Load tail risk caps from training metadata."""
        try:
            metadata_paths = [
                os.path.join(self.models_dir, 'training_metadata.json'),
                '/data/models/training_metadata.json',
                'models/saved_models/training_metadata.json'
            ]

            for path in metadata_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        metadata = json.load(f)

                    if 'tail_risk_caps' in metadata:
                        self.tail_risk_caps = metadata['tail_risk_caps']
                        logger.info(f"Loaded tail risk caps for horizons: {list(self.tail_risk_caps.keys())}")

                    if 'monitoring_config' in metadata:
                        self.monitoring_config = metadata['monitoring_config']
                        logger.info("Loaded monitoring config")

                    break
        except Exception as e:
            logger.debug(f"Could not load tail risk caps: {e}")

    def _apply_tail_risk_cap(self, prediction: float, horizon: str) -> float:
        """Apply tail risk capping to prediction (±3σ from training mean)."""
        if horizon not in self.tail_risk_caps:
            return prediction

        caps = self.tail_risk_caps[horizon]
        lower = caps.get('lower_cap', 0)
        upper = caps.get('upper_cap', float('inf'))

        # Also use global caps as safety net
        global_lower = caps.get('global_lower', 0)
        global_upper = caps.get('global_upper', float('inf'))

        # Use more conservative of the two bounds
        effective_lower = max(lower, global_lower * 0.8)  # Allow 20% below global
        effective_upper = min(upper, global_upper * 1.2)  # Allow 20% above global

        original = prediction
        capped = max(effective_lower, min(prediction, effective_upper))

        if capped != original:
            logger.debug(f"Tail risk cap applied for {horizon}: {original:.4f} -> {capped:.4f}")

        return capped

    def load_models(self) -> bool:
        """Load all component models."""
        success = False

        # Load tail risk caps from training metadata
        self._load_tail_risk_caps()

        # Initialize online bias tracker if monitoring is enabled
        if self.monitoring_config.get('enable_online_bias', False):
            try:
                from utils.prediction_validator import OnlineBiasTracker
                self.bias_tracker = OnlineBiasTracker(
                    window_hours=self.monitoring_config.get('bias_window_hours', 24),
                    max_correction=self.monitoring_config.get('max_bias_correction', 0.15)
                )
                logger.info("Initialized online bias tracker")
            except Exception as e:
                logger.debug(f"Could not initialize bias tracker: {e}")

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
        horizons: List[str] = ['1h', '4h', '24h'],
        include_mempool: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate ensemble predictions for all horizons.

        Args:
            recent_data: DataFrame with recent gas price data
            features: Pre-computed feature matrix (optional)
            horizons: Which horizons to predict
            include_mempool: Include mempool features in prediction

        Returns:
            Dictionary with predictions per horizon, including:
            - ensemble_prediction: Weighted average prediction
            - individual_predictions: Predictions from each model
            - confidence: Ensemble confidence score
            - confidence_interval: (lower, upper) bounds
            - model_weights: Weights used for each model
            - mempool_status: Current mempool metrics (if available)
        """
        if not self.loaded:
            self.load_models()

        predictions = {}
        mempool_status = None

        # Ensure data has required columns
        if 'gas_price' not in recent_data.columns and 'gas' in recent_data.columns:
            recent_data = recent_data.rename(columns={'gas': 'gas_price'})
        elif 'gas_price' not in recent_data.columns and 'gwei' in recent_data.columns:
            recent_data = recent_data.rename(columns={'gwei': 'gas_price'})

        # Get mempool status for confidence adjustment
        if include_mempool:
            mempool_status = self._get_mempool_status()

        # Build feature matrix if not provided
        if features is None and self.stacking_ensembles:
            try:
                from models.feature_pipeline import build_feature_matrix
                feature_df, metadata, _ = build_feature_matrix(
                    recent_data,
                    include_mempool_features=include_mempool
                )
                # Get the last row (most recent) as features
                features = feature_df.iloc[[-1]].values
            except Exception as e:
                logger.debug(f"Could not build features for stacking: {e}")
                features = None

        for horizon in horizons:
            predictions[horizon] = self._predict_horizon(
                recent_data,
                features,
                horizon,
                mempool_status
            )

        return predictions

    def _get_mempool_status(self) -> Optional[Dict[str, Any]]:
        """Get current mempool metrics for prediction adjustment."""
        try:
            from data.mempool_collector import get_mempool_collector
            collector = get_mempool_collector()
            features = collector.get_current_features()

            return {
                'pending_count': features.get('mempool_pending_count', 0),
                'avg_gas_price': features.get('mempool_avg_gas_price', 0),
                'is_congested': features.get('mempool_is_congested', 0) > 0,
                'arrival_rate': features.get('mempool_arrival_rate', 0),
                'count_momentum': features.get('mempool_count_momentum', 0),
                'gas_momentum': features.get('mempool_gas_momentum', 0),
                'available': True
            }
        except Exception as e:
            logger.debug(f"Mempool status unavailable: {e}")
            return {'available': False}

    def _predict_horizon(
        self,
        recent_data: pd.DataFrame,
        features: Optional[np.ndarray],
        horizon: str,
        mempool_status: Optional[Dict[str, Any]] = None
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

        # Apply tail risk capping (±3σ from training mean)
        ensemble_prediction = self._apply_tail_risk_cap(ensemble_prediction, horizon)

        # Apply online bias correction if available
        bias_correction_info = None
        if self.bias_tracker:
            try:
                # Get hour from recent data if available
                hour = None
                if hasattr(recent_data.index, 'hour'):
                    hour = recent_data.index[-1].hour
                elif 'timestamp' in recent_data.columns:
                    hour = recent_data['timestamp'].iloc[-1].hour

                ensemble_prediction, bias_correction_info = self.bias_tracker.apply_bias_correction(
                    ensemble_prediction, horizon, hour
                )
            except Exception as e:
                logger.debug(f"Bias correction failed for {horizon}: {e}")

        # Calculate adaptive confidence interval
        confidence, interval = self._calculate_confidence_interval(
            individual_predictions,
            confidences,
            weights,
            current_price,
            hist_mean,
            recent_data,
            mempool_status
        )

        result = {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions,
            'confidence': confidence,
            'confidence_interval': interval,
            'model_weights': weights,
            'current_price': current_price,
            'historical_mean': hist_mean
        }

        # Add mempool status if available
        if mempool_status and mempool_status.get('available'):
            result['mempool_status'] = {
                'pending_count': mempool_status.get('pending_count', 0),
                'is_congested': mempool_status.get('is_congested', False),
                'gas_momentum': mempool_status.get('gas_momentum', 0)
            }

        # Add bias correction info if applied
        if bias_correction_info and bias_correction_info.get('applied'):
            result['bias_correction'] = bias_correction_info

        return result

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
        """
        Calculate model weights based on recent R² performance.
        
        Uses dynamic weighting: models with better recent R² scores get higher weights.
        Falls back to default weights if no accuracy data is available.
        """
        weights = {}
        
        # Collect recent R² scores for each model
        r2_scores = {}
        for model_name in ['hybrid', 'stacking', 'statistical']:
            if model_name in self.recent_accuracy:
                acc_data = self.recent_accuracy[model_name].get(horizon)
                if acc_data and acc_data.get('r2') is not None:
                    r2 = acc_data['r2']
                    # Ensure R² is positive (negative = worse than baseline)
                    r2_scores[model_name] = max(0.0, r2)
                else:
                    r2_scores[model_name] = None
        
        # Calculate weights based on R² performance
        if any(score is not None and score > 0 for score in r2_scores.values()):
            # Dynamic weighting: weight = recent_r2 / sum(all_recent_r2)
            # For models without recent data, use small default weight
            total_r2 = sum(score for score in r2_scores.values() if score is not None and score > 0)
            
            if total_r2 > 0:
                for model_name in ['hybrid', 'stacking', 'statistical']:
                    if r2_scores.get(model_name) is not None and r2_scores[model_name] > 0:
                        # Proportional to R², but ensure minimum weight
                        weight = r2_scores[model_name] / total_r2
                        weights[model_name] = max(self.MIN_WEIGHT, weight)
                    else:
                        # Model without recent data gets minimum weight
                        weights[model_name] = self.MIN_WEIGHT
                
                # Normalize to sum to 1.0
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
                else:
                    weights = dict(self.DEFAULT_WEIGHTS)
            else:
                # All R² scores are 0 or negative, use default weights
                weights = dict(self.DEFAULT_WEIGHTS)
        else:
            # No recent accuracy data available, use default weights
            weights = dict(self.DEFAULT_WEIGHTS)
        
        # Ensure all models have weights
        for model_name in ['hybrid', 'stacking', 'statistical']:
            if model_name not in weights:
                weights[model_name] = self.DEFAULT_WEIGHTS.get(model_name, self.MIN_WEIGHT)
        
        # Final normalization
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights

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
        recent_data: pd.DataFrame,
        mempool_status: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate ensemble confidence and adaptive interval.

        Uses:
        - Model agreement/disagreement
        - Individual model confidences
        - Recent volatility
        - Mempool congestion status (leading indicator)
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

        # Mempool-based confidence adjustment
        mempool_factor = 1.0
        if mempool_status and mempool_status.get('available'):
            # High congestion = more uncertainty
            if mempool_status.get('is_congested'):
                mempool_factor = 1.3  # Widen intervals during congestion
            # Strong gas momentum = directional confidence
            gas_momentum = abs(mempool_status.get('gas_momentum', 0))
            if gas_momentum > 0.1:  # Significant momentum
                mempool_factor *= 1.1  # More uncertainty during rapid changes

        ensemble_confidence = max(0.3, min(0.95, ensemble_confidence))

        # Calculate interval width based on:
        # 1. Model disagreement
        # 2. Recent volatility
        # 3. Ensemble confidence
        # 4. Mempool status (leading indicator)
        base_width = 0.10  # 10% base interval
        disagreement_factor = 1 + disagreement * 2  # Widen for disagreement
        volatility_factor = 1 + recent_volatility * 3  # Widen for volatility
        confidence_factor = 2 - ensemble_confidence  # Widen for low confidence

        interval_width = base_width * disagreement_factor * volatility_factor * confidence_factor * mempool_factor
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
