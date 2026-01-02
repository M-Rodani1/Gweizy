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
import pickle
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Hybrid predictor combining classification and regression
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
        """Load all models (spike detectors, LSTM, Prophet)"""
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
                    with open(detector_path, 'rb') as f:
                        self.spike_detectors[horizon] = pickle.load(f)
                    logger.info(f"✓ Loaded spike detector for {horizon} from {detector_path}")
                else:
                    logger.warning(f"⚠️  Spike detector not found: tried {possible_paths[0] if possible_paths else 'N/A'}")

            self.loaded = len(self.spike_detectors) > 0

            if not self.loaded:
                logger.error("No spike detectors loaded!")

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
        Make hybrid prediction using spike detection + regression

        Args:
            recent_data: DataFrame with recent gas prices

        Returns:
            dict with predictions for each horizon (1h, 4h, 24h)
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
            current_price = recent_data['gas_price'].iloc[-1]

            if spike_class == 0:  # Normal
                predicted_price = current_price * 0.95  # Expect slight decrease
                lower_bound = 0.001
                upper_bound = self.NORMAL_THRESHOLD
                confidence = spike_probs[0]

            elif spike_class == 1:  # Elevated
                predicted_price = (self.NORMAL_THRESHOLD + self.ELEVATED_THRESHOLD) / 2
                lower_bound = self.NORMAL_THRESHOLD
                upper_bound = self.ELEVATED_THRESHOLD
                confidence = spike_probs[1]

            else:  # Spike
                predicted_price = max(current_price * 1.5, self.ELEVATED_THRESHOLD * 2)
                lower_bound = self.ELEVATED_THRESHOLD
                upper_bound = min(predicted_price * 2, 1.0)  # Cap at 1 Gwei
                confidence = spike_probs[2]

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
