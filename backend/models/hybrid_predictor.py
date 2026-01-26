"""
Hybrid Gas Price Predictor

Combines spike detection with LSTM/Prophet regression for improved predictions.
Strategy:
1. Classify upcoming period as Normal/Elevated/Spike
2. Use appropriate prediction strategy based on classification
3. Provide confidence intervals and alerts
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Hybrid gas price predictor combining spike classification with price regression.

    This predictor uses a two-stage approach:
    1. **Classification Stage**: Classifies the upcoming period as Normal, Elevated, or Spike
       using gradient boosting models trained on historical patterns.
    2. **Prediction Stage**: Generates price predictions and confidence intervals based
       on the classification and current market conditions.

    The hybrid approach provides several advantages:
    - Better handling of extreme events (spikes) that regression alone misses
    - Actionable classifications for users (wait vs. transact recommendations)
    - Confidence intervals that adapt to market conditions

    Classification Thresholds:
        - Normal: < 0.01 Gwei (low activity, optimal for transactions)
        - Elevated: 0.01 - 0.05 Gwei (moderate activity)
        - Spike: > 0.05 Gwei (high activity, recommend waiting)

    Attributes:
        models_dir (str): Directory containing trained model files.
        spike_detectors (dict): Loaded spike detection models keyed by horizon.
        loaded (bool): Whether models have been successfully loaded.

    Example:
        >>> predictor = HybridPredictor()
        >>> predictor.load_models()
        >>> predictions = predictor.predict(recent_data_df)
        >>> print(predictions['1h']['classification']['class'])
        'normal'

    Model Files Required:
        - spike_detector_1h.pkl: 1-hour horizon classifier
        - spike_detector_4h.pkl: 4-hour horizon classifier
        - spike_detector_24h.pkl: 24-hour horizon classifier
    """

    # Thresholds (in Gwei) - must match training script
    NORMAL_THRESHOLD = 0.01
    ELEVATED_THRESHOLD = 0.05

    # Class names
    CLASS_NAMES = ['normal', 'elevated', 'spike']
    CLASS_EMOJIS = ['🟢', '🟡', '🔴']
    CLASS_COLORS = ['green', 'yellow', 'red']

    def __init__(self, models_dir='models/saved_models'):
        self.models_dir = models_dir
        self.spike_detectors = {}
        self.lstm_models = {}
        self.prophet_models = {}
        self.scalers = {}
        self.loaded = False

    def load_models(self):
        """
        Load trained spike detection models from disk.

        Searches for model files in multiple locations (priority order):
        1. Persistent storage (/data/models) - used in Railway deployment
        2. Config.MODELS_DIR - if defined in configuration
        3. self.models_dir - local development fallback
        4. backend/models/saved_models - alternative local path

        Returns:
            bool: True if at least one model was loaded successfully.

        Raises:
            No exceptions raised - failures are logged as warnings.

        Side Effects:
            Sets self.loaded to True if successful.
            Populates self.spike_detectors dict with loaded models.
        """
        try:
            # Load spike detectors for all horizons
            for horizon in ['1h', '4h', '24h']:
                # Try multiple paths in priority order: persistent storage first, then fallbacks
                possible_paths = []
                
                # Priority 1: Persistent storage (Railway)
                if os.path.exists('/data/models'):
                    possible_paths.append(os.path.join('/data/models', f'spike_detector_{horizon}.pkl'))
                
                # Priority 2: Config.MODELS_DIR if available
                try:
                    from config import Config
                    if hasattr(Config, 'MODELS_DIR') and os.path.exists(Config.MODELS_DIR):
                        possible_paths.append(os.path.join(Config.MODELS_DIR, f'spike_detector_{horizon}.pkl'))
                except:
                    pass
                
                # Priority 3: Original paths (fallback)
                possible_paths.extend([
                    os.path.join(self.models_dir, f'spike_detector_{horizon}.pkl'),
                    os.path.join('backend', self.models_dir, f'spike_detector_{horizon}.pkl')
                ])
                
                detector_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        detector_path = path
                        break

                if detector_path and os.path.exists(detector_path):
                    self.spike_detectors[horizon] = joblib.load(detector_path)
                    logger.info(f"✓ Loaded spike detector for {horizon} from {detector_path}")
                else:
                    # Debug level - expected during initial setup before training
                    logger.debug(f"Spike detector not found for {horizon}")

            self.loaded = len(self.spike_detectors) > 0

            if not self.loaded:
                # Only log once per instance
                if not hasattr(self, '_warned_no_detectors'):
                    logger.info("Spike detectors not available - using statistical fallback. Run training to enable.")
                    self._warned_no_detectors = True

            return self.loaded

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def create_spike_features(self, recent_data):
        """
        Create features for spike detection from recent gas price data

        Args:
            recent_data: DataFrame with columns [timestamp, gas_price, base_fee, priority_fee]
                        Should contain at least last 48 5-minute intervals (4 hours)
        """
        df = recent_data.copy()
        df = df.sort_values('timestamp')

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

        # Recent volatility
        for window in [6, 12, 24, 48]:
            df[f'volatility_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).std()
            df[f'range_{window}'] = (
                df['gas_price'].rolling(window=window, min_periods=1).max() -
                df['gas_price'].rolling(window=window, min_periods=1).min()
            )
            df[f'mean_{window}'] = df['gas_price'].rolling(window=window, min_periods=1).mean()
            df[f'is_rising_{window}'] = (
                df['gas_price'] > df[f'mean_{window}']
            ).astype(int)

        # Rate of change
        for lag in [1, 2, 3, 6, 12]:
            df[f'pct_change_{lag}'] = df['gas_price'].pct_change(lag).fillna(0)
            df[f'diff_{lag}'] = df['gas_price'].diff(lag).fillna(0)

        # Recent spike indicator
        df['recent_spike'] = (
            df['gas_price'].rolling(window=24, min_periods=1).max() > self.ELEVATED_THRESHOLD
        ).astype(int)

        # Replace inf/-inf with 0
        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        return df

    def predict(self, recent_data):
        """
        Generate gas price predictions for multiple time horizons.

        Uses spike detection models to classify expected market conditions
        and generate price predictions with confidence intervals.

        Args:
            recent_data (pd.DataFrame): Recent gas price data with columns:
                - timestamp: datetime of each observation
                - gas_price: gas price in Gwei
                - base_fee: base fee component (optional)
                - priority_fee: priority fee component (optional)
                Should contain at least 4 hours of data (~48 records at 5-min intervals).

        Returns:
            dict: Predictions keyed by horizon ('1h', '4h', '24h'), each containing:
                - classification: {class, class_id, emoji, color, confidence, probabilities}
                - prediction: {price, lower_bound, upper_bound, unit}
                - alert: {show_alert, message, severity}
                - recommendation: {action, message, suggested_gas}

        Raises:
            ValueError: If models are not loaded and cannot be loaded.

        Example:
            >>> predictions = predictor.predict(recent_data)
            >>> print(predictions['1h']['classification'])
            {'class': 'normal', 'confidence': 0.85, ...}
            >>> print(predictions['1h']['recommendation'])
            {'action': 'transact', 'message': 'Optimal time to submit transactions'}
        """
        if not self.loaded:
            if not self.load_models():
                raise ValueError("Models not loaded")

        # Create features
        features_df = self.create_spike_features(recent_data)
        latest_features = features_df.iloc[[-1]]  # Most recent row

        predictions = {}

        for horizon in ['1h', '4h', '24h']:
            if horizon not in self.spike_detectors:
                continue

            detector_data = self.spike_detectors[horizon]
            model = detector_data['model']
            feature_names = detector_data['feature_names']

            # Prepare features in correct order
            X = latest_features[feature_names].values

            # Get spike classification and probabilities
            spike_class = model.predict(X)[0]
            spike_probs = model.predict_proba(X)[0]

            # Convert class to name
            class_name = self.CLASS_NAMES[int(spike_class)]
            class_emoji = self.CLASS_EMOJIS[int(spike_class)]
            class_color = self.CLASS_COLORS[int(spike_class)]

            # Generate price prediction based on classification
            # Use blend of current price and historical mean for stability
            current_price = recent_data['gas_price'].iloc[-1]
            hist_mean = recent_data['gas_price'].iloc[-100:].mean()
            # 70% historical mean, 30% current - more stable predictions
            base_price = hist_mean * 0.7 + current_price * 0.3

            if spike_class == 0:  # Normal
                # Expect stable to slight decrease
                predicted_price = base_price * 0.98
                lower_bound = base_price * 0.85
                upper_bound = base_price * 1.1
                confidence = spike_probs[0]

            elif spike_class == 1:  # Elevated
                # Expect moderate increase (capped at 20%)
                predicted_price = base_price * 1.12
                lower_bound = base_price * 1.0
                upper_bound = base_price * 1.3
                confidence = spike_probs[1]

            else:  # Spike
                # Expect significant but bounded increase (capped at 50%)
                predicted_price = base_price * 1.35
                lower_bound = base_price * 1.15
                upper_bound = base_price * 1.6
                confidence = spike_probs[2]

            # Final sanity check: clamp prediction within 0.5x to 2x of historical mean
            predicted_price = max(hist_mean * 0.5, min(predicted_price, hist_mean * 2.0))
            lower_bound = max(hist_mean * 0.4, min(lower_bound, hist_mean * 1.5))
            upper_bound = max(hist_mean * 0.6, min(upper_bound, hist_mean * 2.5))

            predictions[horizon] = {
                'classification': {
                    'class': class_name,
                    'class_id': int(spike_class),
                    'emoji': class_emoji,
                    'color': class_color,
                    'confidence': float(confidence),
                    'probabilities': {
                        'normal': float(spike_probs[0]),
                        'elevated': float(spike_probs[1]),
                        'spike': float(spike_probs[2])
                    }
                },
                'prediction': {
                    'price': float(predicted_price),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'unit': 'gwei'
                },
                'alert': {
                    'show_alert': bool(spike_class >= 2),
                    'message': self._get_alert_message(class_name, confidence),
                    'severity': class_name
                },
                'recommendation': self._get_recommendation(class_name, confidence)
            }

        return predictions

    def _get_alert_message(self, class_name, confidence):
        """Generate alert message based on classification"""
        if class_name == 'spike':
            if confidence > 0.8:
                return "High probability of gas price spike detected. Consider waiting or using higher gas limits."
            else:
                return "Possible gas price spike ahead. Monitor prices closely."
        elif class_name == 'elevated':
            return "Gas prices elevated. Consider waiting for normal prices if not urgent."
        else:
            return "Gas prices normal. Good time to transact."

    def _get_recommendation(self, class_name, confidence):
        """Generate user recommendation based on classification"""
        if class_name == 'normal':
            return {
                'action': 'transact',
                'message': 'Optimal time to submit transactions',
                'suggested_gas': 'standard'
            }
        elif class_name == 'elevated':
            return {
                'action': 'wait_or_proceed',
                'message': 'Prices are elevated. Wait if not urgent, or use higher gas for faster confirmation.',
                'suggested_gas': 'fast'
            }
        else:  # spike
            return {
                'action': 'wait',
                'message': 'High gas prices detected. Strongly recommend waiting unless urgent.',
                'suggested_gas': 'rapid'
            }

    def get_current_status(self, recent_data):
        """
        Get current gas price status and short-term outlook

        Returns a simple status for dashboard display
        """
        current_price = recent_data['gas_price'].iloc[-1]

        # Determine current status
        if current_price < self.NORMAL_THRESHOLD:
            status = 'normal'
            emoji = '🟢'
            color = 'green'
        elif current_price < self.ELEVATED_THRESHOLD:
            status = 'elevated'
            emoji = '🟡'
            color = 'yellow'
        else:
            status = 'spike'
            emoji = '🔴'
            color = 'red'

        # Get predictions
        predictions = self.predict(recent_data)

        # Check if any horizon predicts spike
        upcoming_spike = any(
            pred['classification']['class'] == 'spike'
            for pred in predictions.values()
        )

        return {
            'current': {
                'price': float(current_price),
                'status': status,
                'emoji': emoji,
                'color': color
            },
            'outlook': {
                'upcoming_spike': upcoming_spike,
                'next_1h': predictions.get('1h', {}).get('classification', {}).get('class', 'unknown'),
                'next_4h': predictions.get('4h', {}).get('classification', {}).get('class', 'unknown'),
                'next_24h': predictions.get('24h', {}).get('classification', {}).get('class', 'unknown')
            },
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
hybrid_predictor = HybridPredictor()
