"""
Prediction Routes
Endpoints for ML-powered gas price predictions and explanations.
"""

from flask import Blueprint, jsonify, request
from data.collector import BaseGasCollector
from data.multichain_collector import MultiChainGasCollector
from data.database import DatabaseManager
from models.feature_engineering import GasFeatureEngineer
from models.accuracy_tracker import get_tracker
from utils.logger import logger, log_error_with_context
from utils.prediction_validator import PredictionValidator
from api.cache import cached
from api.model_state import models, scalers, feature_names
from datetime import datetime, timedelta
import traceback
import numpy as np
import os


prediction_bp = Blueprint('prediction', __name__)

# Shared instances
collector = BaseGasCollector()
multichain_collector = MultiChainGasCollector()
db = DatabaseManager()
engineer = GasFeatureEngineer()
validator = PredictionValidator()
accuracy_tracker = get_tracker()


@prediction_bp.route('/predictions', methods=['GET'])
@cached(ttl=60)  # Cache for 1 minute
def get_predictions():
    """Get ML-powered gas price predictions using hybrid spike detection for a specific chain"""
    try:
        # Get chain_id from query params (default to Base)
        chain_id = request.args.get('chain_id', 8453, type=int)

        # Get current gas for the specified chain
        current = multichain_collector.get_current_gas(chain_id)

        # If we can't get current gas, try to get a fallback value from recent data
        if not current or not current.get('current_gas'):
            logger.warning(f"Could not fetch current gas for chain {chain_id}, using fallback from recent data")
            # Try to get a recent gas price from the database as fallback
            recent_data = db.get_historical_data(hours=1, chain_id=chain_id, limit=1, order='desc')
            if recent_data and len(recent_data) > 0:
                # Get the most recent entry
                fallback_gas = recent_data[0].get('gwei') or recent_data[0].get('current_gas', 0.01)
                current = {
                    'chain_id': chain_id,
                    'current_gas': fallback_gas,
                    'base_fee': fallback_gas * 0.9,
                    'priority_fee': fallback_gas * 0.1,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Last resort: use a default value
                current = {
                    'chain_id': chain_id,
                    'current_gas': 0.01,  # Default 0.01 gwei
                    'base_fee': 0.009,
                    'priority_fee': 0.001,
                    'timestamp': datetime.now().isoformat()
                }
                logger.warning(f"Using default gas value for chain {chain_id}")

        # Try hybrid predictor first (spike detection + classification)
        try:
            from models.hybrid_predictor import hybrid_predictor
            import pandas as pd
            from dateutil import parser

            # Get recent data for hybrid predictor (needs at least 50 points) for specific chain
            recent_data = db.get_historical_data(hours=48, chain_id=chain_id)

            if len(recent_data) >= 50:
                # Convert to DataFrame format for hybrid predictor
                recent_df = []
                for d in recent_data:
                    timestamp = d.get('timestamp', '')
                    if isinstance(timestamp, str):
                        try:
                            dt = parser.parse(timestamp)
                        except (ValueError, TypeError):
                            dt = datetime.now()
                    else:
                        dt = timestamp if hasattr(timestamp, 'hour') else datetime.now()

                    gas_price = d.get('gwei', 0) or d.get('current_gas', 0)
                    if gas_price is None or gas_price <= 0:
                        continue  # Skip invalid data points

                    recent_df.append({
                        'timestamp': dt,
                        'gas_price': gas_price,
                        'base_fee': d.get('baseFee', 0) or d.get('base_fee', 0),
                        'priority_fee': d.get('priorityFee', 0) or d.get('priority_fee', 0)
                    })

                if len(recent_df) < 50:
                    raise ValueError(f"Not enough valid data points: {len(recent_df)} < 50")

                df = pd.DataFrame(recent_df)

                # Ensure DataFrame is sorted by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)

                # Validate DataFrame has required columns
                if 'gas_price' not in df.columns or len(df) == 0:
                    raise ValueError("Invalid DataFrame format for hybrid predictor")

                # Use hybrid predictor (faster than ensemble for real-time responses)
                try:
                    hybrid_preds = hybrid_predictor.predict(df)

                    if not hybrid_preds or len(hybrid_preds) == 0:
                        raise ValueError("Hybrid predictor returned empty predictions")

                    # Format for API response
                    prediction_data = {}
                    for horizon, pred in hybrid_preds.items():
                        classification = pred['classification']
                        prediction = pred['prediction']
                        alert = pred['alert']
                        recommendation = pred['recommendation']

                        # Map confidence value to level (frontend expects 'high'/'medium'/'low')
                        conf_value = classification['confidence']
                        if conf_value >= 0.7:
                            conf_level = 'high'
                        elif conf_value >= 0.5:
                            conf_level = 'medium'
                        else:
                            conf_level = 'low'

                        prediction_data[horizon] = [{
                            'time': horizon,
                            'predictedGwei': prediction['price'],
                            'lowerBound': prediction['lower_bound'],
                            'upperBound': prediction['upper_bound'],
                            'confidence': conf_value,
                            'confidenceLevel': conf_level,
                            'confidenceEmoji': classification['emoji'],
                            'confidenceColor': classification['color'],
                            'classification': {
                                'class': classification['class'],
                                'emoji': classification['emoji'],
                                'probabilities': classification['probabilities']
                            },
                            'alert': alert,
                            'recommendation': recommendation
                        }]

                except Exception as hybrid_err:
                    logger.warning(f"Hybrid predictor failed: {hybrid_err}")
                    raise

                # Format historical data for graph
                historical = []
                for d in recent_data[-100:]:
                    timestamp = d.get('timestamp', '')
                    if isinstance(timestamp, str):
                        try:
                            dt = parser.parse(timestamp)
                            time_str = dt.strftime('%H:%M')
                        except (ValueError, TypeError):
                            time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                    else:
                        time_str = str(timestamp)[:5]

                    historical.append({
                        'time': time_str,
                        'gwei': round(d.get('gwei', 0) or d.get('current_gas', 0), 4)
                    })

                prediction_data['historical'] = historical

                # Track predictions for accuracy monitoring (use ensemble prediction if available)
                for horizon in ['1h', '4h', '24h']:
                    try:
                        if accuracy_tracker is not None and horizon in prediction_data:
                            pred_value = prediction_data[horizon][0]['predictedGwei']
                            accuracy_tracker.record_prediction(
                                horizon=horizon,
                                predicted=pred_value,
                                timestamp=datetime.now()
                            )
                            logger.debug(f"Recorded {horizon} prediction: {pred_value:.6f} gwei")
                    except Exception as track_err:
                        logger.warning(f"Could not track prediction: {track_err}")

                # Determine model type for info
                model_type = 'ensemble' if 'ensemble_info' in prediction_data.get('1h', [{}])[0] else 'hybrid'

                # Get pattern matching insights (disabled for performance)
                pattern_data = None

                response = {
                    'chain_id': chain_id,
                    'current': current,
                    'predictions': prediction_data,
                    'model_info': {
                        'type': model_type,
                        'version': 'ensemble_v1' if model_type == 'ensemble' else 'spike_detection_v1',
                        'description': 'Ensemble prediction (Hybrid + Stacking + Statistical)' if model_type == 'ensemble'
                                      else 'Classification-based prediction (Normal/Elevated/Spike)',
                        'chain_id': chain_id
                    }
                }

                if pattern_data and pattern_data.get('available'):
                    response['pattern_analysis'] = pattern_data

                return jsonify(response)
            else:
                logger.warning(f"Not enough data for hybrid predictor: {len(recent_data)} records")

        except Exception as e:
            logger.warning(f"Hybrid predictor failed: {e}, falling back to legacy models")
            import traceback
            logger.warning(traceback.format_exc())
            # Continue to fallback - don't return here

        # Fallback to legacy models if hybrid fails
        # Get recent historical data for the specified chain (if not already fetched)
        if 'recent_data' not in locals():
            recent_data = db.get_historical_data(hours=48, chain_id=chain_id)

        if not models:
            logger.warning("Models not loaded, using fallback predictions")
            # Format historical data for graph even when models aren't loaded
            historical = []
            hist_prices = []
            for d in recent_data[-100:] if len(recent_data) > 0 else []:
                timestamp = d.get('timestamp', '')
                if isinstance(timestamp, str):
                    try:
                        from dateutil import parser
                        dt = parser.parse(timestamp)
                        time_str = dt.strftime('%H:%M')
                    except (ValueError, TypeError):
                        time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                else:
                    time_str = str(timestamp)[:5]
                price = d.get('gwei', 0) or d.get('current_gas', 0)
                if price and price > 0:
                    hist_prices.append(price)
                historical.append({
                    'time': time_str,
                    'gwei': round(price, 4)
                })

            # Use historical mean blended with current for more stable predictions
            hist_mean = np.mean(hist_prices) if hist_prices else current['current_gas']
            base_price = hist_mean * 0.7 + current['current_gas'] * 0.3

            return jsonify({
                'chain_id': chain_id,
                'current': current,
                'predictions': {
                    '1h': [{'time': '1h', 'predictedGwei': round(base_price * 1.02, 6)}],
                    '4h': [{'time': '4h', 'predictedGwei': round(base_price * 1.05, 6)}],
                    '24h': [{'time': '24h', 'predictedGwei': round(base_price * 1.08, 6)}],
                    'historical': historical,
                },
                'note': 'Using fallback predictions. Train models for ML predictions.'
            })

        # Get recent historical data for the specified chain (if not already fetched)
        if 'recent_data' not in locals():
            recent_data = db.get_historical_data(hours=48, chain_id=chain_id)

        if len(recent_data) < 100:
            logger.warning(f"Not enough data: {len(recent_data)} records, using fallback predictions")
            # Use fallback instead of returning 500 error
            fallback_predictions = {}

            # Gather historical prices and format for graph
            historical = []
            hist_prices = []
            for d in recent_data[-100:] if len(recent_data) > 0 else []:
                timestamp = d.get('timestamp', '')
                if isinstance(timestamp, str):
                    try:
                        from dateutil import parser
                        dt = parser.parse(timestamp)
                        time_str = dt.strftime('%H:%M')
                    except (ValueError, TypeError):
                        time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                else:
                    time_str = str(timestamp)[:5]

                price = d.get('gwei', 0) or d.get('current_gas', 0)
                if price and price > 0:
                    hist_prices.append(price)
                historical.append({
                    'time': time_str,
                    'gwei': round(price, 4)
                })

            # Use historical mean blended with current for stable predictions
            hist_mean = np.mean(hist_prices) if hist_prices else current['current_gas']
            base_price = hist_mean * 0.7 + current['current_gas'] * 0.3

            for horizon in ['1h', '4h', '24h']:
                # Smaller multipliers for more conservative predictions
                multiplier = 1.02 if horizon == '1h' else 1.05 if horizon == '4h' else 1.08
                pred_value = round(base_price * multiplier, 6)

                fallback_predictions[horizon] = [{
                    'time': horizon,
                    'predictedGwei': pred_value,
                    'lowerBound': round(pred_value * 0.9, 6),
                    'upperBound': round(pred_value * 1.1, 6),
                    'confidence': 0.5,
                    'confidenceLevel': 'low',
                    'confidenceEmoji': '🔴',
                    'confidenceColor': 'red'
                }]

            fallback_predictions['historical'] = historical

            return jsonify({
                'chain_id': chain_id,
                'current': current,
                'predictions': fallback_predictions,
                'model_info': {
                    'warning': 'Insufficient data for ML predictions - using fallback',
                    'data_points': len(recent_data),
                    'chain_id': chain_id
                }
            })

        # Wrap entire ML prediction pipeline in try-catch for graceful fallback
        try:
            from models.feature_pipeline import build_feature_matrix

            features, _, _ = build_feature_matrix(recent_data, include_external_features=True)
            features = features.fillna(0)

            # Make predictions with standard models
            prediction_data = {}
            model_info = {}

            for horizon in ['1h', '4h', '24h']:
                if horizon in models:
                    # Fallback to standard models
                    model_data = models[horizon]
                    model = model_data['model']

                    # Scale features if scaler is available
                    features_to_predict = features
                    if horizon in scalers:
                        try:
                            # Get expected features from model_data (these are already SELECTED features)
                            expected_features = model_data.get('feature_names', [])
                            if not expected_features and horizon in feature_names:
                                expected_features = feature_names[horizon]

                            if expected_features and len(expected_features) > 0:
                                # Reorder features to match training order
                                available_features = list(features.columns)
                                missing_features = [f for f in expected_features if f not in available_features]

                                if missing_features:
                                    logger.warning(f"Missing features for {horizon}: {missing_features[:5]}...")
                                    # Add missing features as zeros
                                    for f in missing_features:
                                        features[f] = 0

                                # Select and order features (model expects these specific features)
                                features_to_predict = features[expected_features]
                                logger.debug(f"Selected {len(features_to_predict.columns)} features for {horizon}")
                            else:
                                features_to_predict = features
                                logger.warning(f"No expected features for {horizon}, using all {len(features.columns)} features")

                            # Scale features
                            features_scaled = scalers[horizon].transform(features_to_predict)
                            pred = model.predict(features_scaled)[0]
                        except ValueError as ve:
                            # Feature mismatch - use simple fallback
                            logger.warning(f"Feature mismatch for {horizon}: {ve}. Using fallback prediction")
                            pred = current['current_gas'] * (1.05 if horizon == '1h' else 1.1 if horizon == '4h' else 1.15)
                        except Exception as e:
                            logger.warning(f"Prediction failed for {horizon}: {e}, using fallback")
                            pred = current['current_gas'] * (1.05 if horizon == '1h' else 1.1 if horizon == '4h' else 1.15)
                    else:
                        try:
                            pred = model.predict(features)[0]
                        except ValueError as ve:
                            # Feature mismatch - use simple fallback
                            logger.warning(f"Feature mismatch for {horizon}: {ve}. Using fallback prediction")
                            pred = current['current_gas'] * (1.05 if horizon == '1h' else 1.1 if horizon == '4h' else 1.15)

                    # Check if model predicts percentage change, absolute price, or log scale
                    predicts_pct_change = model_data.get('predicts_percentage_change', False)
                    uses_log_scale = model_data.get('uses_log_scale', False)
                    target_scaler = model_data.get('target_scaler')

                    # If target_scaler not in model_data, try to load from separate file
                    if target_scaler is None:
                        try:
                            import joblib
                            target_scaler_path = f'backend/models/saved_models/target_scaler_{horizon}.pkl'
                            if os.path.exists(target_scaler_path):
                                target_scaler = joblib.load(target_scaler_path)
                                logger.info(f"Loaded target_scaler from {target_scaler_path}")
                        except Exception as e:
                            logger.warning(f"Could not load target_scaler: {e}")

                    if uses_log_scale and target_scaler is not None:
                        # Model predicts in log space with scaling
                        pred_log_scaled = np.array([[float(pred)]])
                        pred_log = target_scaler.inverse_transform(pred_log_scaled)[0][0]
                        pred_value = round(np.exp(pred_log) - 1e-8, 6)  # Inverse log
                        logger.info(f"Prediction for {horizon} (log scale): {pred_log:.6f} -> {pred_value:.6f} gwei")
                    elif target_scaler is not None:
                        # Model was trained with target scaling - inverse transform needed
                        pred_scaled = np.array([[float(pred)]])
                        pred_value = target_scaler.inverse_transform(pred_scaled)[0][0]
                        pred_value = round(pred_value, 6)
                        logger.info(f"Prediction for {horizon} (scaled): {pred:.6f} -> {pred_value:.6f} gwei")
                    elif predicts_pct_change:
                        # Model predicts percentage change, convert to absolute price
                        pct_change = float(pred)

                        # Calculate historical mean as anchor (more stable reference)
                        hist_prices = [d.get('gwei', 0) or d.get('current_gas', 0) for d in recent_data[-100:]]
                        hist_prices = [p for p in hist_prices if p and p > 0]
                        hist_mean = np.mean(hist_prices) if hist_prices else current['current_gas']

                        # MUCH tighter clamp: gas prices don't change dramatically short-term
                        horizon_clamps = {'1h': 30, '4h': 50, '24h': 80}
                        max_clamp = horizon_clamps.get(horizon, 50)
                        pct_change = max(-max_clamp, min(max_clamp, pct_change))

                        # Blend: 70% historical mean, 30% current price
                        base_price = hist_mean * 0.7 + current['current_gas'] * 0.3
                        pred_value = round(base_price * (1 + pct_change / 100), 6)

                        # Final sanity check: prediction must be within 2x of historical mean
                        pred_value = max(hist_mean * 0.5, min(pred_value, hist_mean * 2.0))
                        pred_value = round(pred_value, 6)
                    else:
                        # Model predicts absolute price directly
                        pred_value = round(float(pred), 6)

                        # Sanity check absolute predictions against historical mean
                        hist_prices = [d.get('gwei', 0) or d.get('current_gas', 0) for d in recent_data[-100:]]
                        hist_prices = [p for p in hist_prices if p and p > 0]
                        hist_mean = np.mean(hist_prices) if hist_prices else current['current_gas']

                        # Clamp to reasonable range (0.5x to 2x historical mean)
                        pred_value = max(hist_mean * 0.5, min(pred_value, hist_mean * 2.0))
                        pred_value = round(pred_value, 6)

                    # Estimate confidence (simple heuristic)
                    confidence = 0.75  # Default medium confidence
                    lower_bound = pred_value * 0.9
                    upper_bound = pred_value * 1.1
                    conf_level, emoji, color = 'medium', '🟡', 'yellow'

                    prediction_data[horizon] = [{
                        'time': horizon,
                        'predictedGwei': pred_value,
                        'lowerBound': lower_bound,
                        'upperBound': upper_bound,
                        'confidence': confidence,
                        'confidenceLevel': conf_level,
                        'confidenceEmoji': emoji,
                        'confidenceColor': color
                    }]

                    model_info[horizon] = {
                        'name': model_data['model_name'],
                        'mae': model_data['metrics']['mae']
                    }

                    db.save_prediction(
                        horizon=horizon,
                        predicted_gas=pred_value,
                        model_version=model_data['model_name'],
                        chain_id=chain_id
                    )

                    # Log prediction for validation
                    horizon_hours = {'1h': 1, '4h': 4, '24h': 24}[horizon]
                    target_time = datetime.now() + timedelta(hours=horizon_hours)
                    validator.log_prediction(
                        horizon=horizon,
                        predicted_gas=pred_value,
                        target_time=target_time,
                        model_version=model_data['model_name']
                    )

                    # Track prediction for accuracy monitoring
                    try:
                        if accuracy_tracker is not None:
                            accuracy_tracker.record_prediction(
                                horizon=horizon,
                                predicted=pred_value,
                                timestamp=datetime.now()
                            )
                            logger.debug(f"Recorded {horizon} prediction: {pred_value:.6f} gwei")
                        else:
                            logger.warning("Accuracy tracker not available - predictions not being tracked")
                    except Exception as track_err:
                        logger.warning(f"Could not track prediction: {track_err}")
                        import traceback
                        logger.debug(traceback.format_exc())

            # Format historical data for graph
            historical = []
            for d in recent_data[-100:]:
                timestamp = d.get('timestamp', '')
                if isinstance(timestamp, str):
                    from dateutil import parser
                    try:
                        dt = parser.parse(timestamp)
                        time_str = dt.strftime('%H:%M')
                    except (ValueError, TypeError):
                        time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                else:
                    time_str = str(timestamp)[:5]

                historical.append({
                    'time': time_str,
                    'gwei': round(d.get('gwei', 0) or d.get('current_gas', 0), 4)
                })

            prediction_data['historical'] = historical

            logger.info(f"Predictions with confidence: 1h={prediction_data.get('1h', [{}])[0].get('predictedGwei', 0)}")

            return jsonify({
                'chain_id': chain_id,
                'current': current,
                'predictions': prediction_data,
                'model_info': model_info
            })

        except Exception as ml_error:
            # ML pipeline failed - use simple fallback predictions
            logger.error(f"ML prediction pipeline failed: {ml_error}")
            logger.error(traceback.format_exc())
            logger.info("Using fallback predictions based on current gas price")

            # Simple fallback: slight increases based on horizon
            fallback_predictions = {}
            for horizon in ['1h', '4h', '24h']:
                multiplier = 1.05 if horizon == '1h' else 1.1 if horizon == '4h' else 1.15
                pred_value = round(current['current_gas'] * multiplier, 6)

                fallback_predictions[horizon] = [{
                    'time': horizon,
                    'predictedGwei': pred_value,
                    'lowerBound': round(pred_value * 0.9, 6),
                    'upperBound': round(pred_value * 1.1, 6),
                    'confidence': 0.5,
                    'confidenceLevel': 'low',
                    'confidenceEmoji': '🔴',
                    'confidenceColor': 'red'
                }]

            # Format historical data for graph
            historical = []
            for d in recent_data[-100:]:
                timestamp = d.get('timestamp', '')
                if isinstance(timestamp, str):
                    from dateutil import parser
                    try:
                        dt = parser.parse(timestamp)
                        time_str = dt.strftime('%H:%M')
                    except (ValueError, TypeError):
                        time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                else:
                    time_str = str(timestamp)[:5]

                historical.append({
                    'time': time_str,
                    'gwei': round(d.get('gwei', 0) or d.get('current_gas', 0), 4)
                })

            fallback_predictions['historical'] = historical

            return jsonify({
                'chain_id': chain_id,
                'current': current,
                'predictions': fallback_predictions,
                'model_info': {
                    'warning': 'Using fallback predictions - ML models need retraining',
                    'error': str(ml_error),
                    'chain_id': chain_id
                }
            })

    except Exception as e:
        logger.error(f"Error in /predictions: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/explain/<horizon>', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def explain_prediction(horizon):
    """
    Get explanation for a prediction

    Args:
        horizon: '1h', '4h', or '24h'
    """
    try:
        from models.explainer import explainer, initialize_explainer

        if horizon not in ['1h', '4h', '24h']:
            return jsonify({'error': 'Invalid horizon'}), 400

        # Get current gas
        current_gas = collector.get_current_gas()

        # Get recent historical data
        recent_data = db.get_historical_data(hours=48)

        if len(recent_data) < 100:
            return jsonify({'error': 'Not enough historical data'}), 500

        # Prepare features
        import pandas as pd
        from dateutil import parser

        recent_df = []
        for d in recent_data:
            timestamp = d.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    dt = parser.parse(timestamp)
                except (ValueError, TypeError):
                    dt = datetime.now()
            else:
                dt = timestamp if hasattr(timestamp, 'hour') else datetime.now()

            recent_df.append({
                'timestamp': dt,
                'gas': d.get('gwei', 0) or d.get('current_gas', 0),
                'base_fee': d.get('baseFee', 0) or d.get('base_fee', 0),
                'priority_fee': d.get('priorityFee', 0) or d.get('priority_fee', 0),
                'block_number': 0
            })

        features = engineer.prepare_prediction_features(recent_df)

        # Get model for this horizon
        if horizon not in models:
            return jsonify({'error': f'Model for {horizon} not available'}), 404

        model_data = models[horizon]
        model = model_data['model']

        # Get prediction first
        prediction = model.predict(features)[0]

        # Get historical data for comparison
        historical = db.get_historical_data(hours=168)  # Last week
        historical_data = [{
            'timestamp': h.get('timestamp', ''),
            'gas_price': h.get('gwei', 0) or h.get('current_gas', 0)
        } for h in historical]

        # Initialize explainer if needed
        from models.explainer import initialize_explainer
        current_explainer = explainer

        if current_explainer is None:
            # Get feature names from the features DataFrame
            try:
                feat_names = list(features.columns)
            except (AttributeError, TypeError):
                try:
                    # Create a sample DataFrame to get feature columns
                    import pandas as pd
                    sample_df = pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'gas': current_gas['current_gas'],
                        'base_fee': 0,
                        'priority_fee': 0,
                        'block_number': 0
                    }])
                    sample_df = engineer._add_time_features(sample_df)
                    sample_df = engineer._add_lag_features(sample_df)
                    sample_df = engineer._add_rolling_features(sample_df)
                    feat_names = list(engineer.get_feature_columns(sample_df))
                except Exception as e:
                    logger.warning(f"Could not get feature names: {e}")
                    # Fallback feature names
                    feat_names = ['hour', 'day_of_week', 'trend_1h', 'trend_3h', 'avg_last_24h']

            current_explainer = initialize_explainer(model, feat_names)

        # Generate explanation (now includes LLM explanation)
        try:
            if current_explainer is None:
                raise ValueError("Explainer not initialized")

            # Convert features to proper format
            features_for_explainer = features.values if hasattr(features, 'values') else features

            explanation = current_explainer.explain_prediction(
                features_for_explainer,
                prediction,
                historical_data,
                current_gas=current_gas['current_gas']  # Pass current gas for LLM context
            )
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}, using fallback")
            # Fallback explanation if explainer fails
            explanation = {
                'llm_explanation': f'Prediction of {prediction:.4f} gwei based on current gas price of {current_gas["current_gas"]:.4f} gwei and historical patterns. The model analyzed time-of-day patterns, recent trends, and network activity to generate this forecast.',
                'technical_explanation': f'Prediction of {prediction:.4f} gwei based on current gas price of {current_gas["current_gas"]:.4f} gwei.',
                'technical_details': {
                    'feature_importance': {},
                    'increasing_factors': [],
                    'decreasing_factors': [],
                    'similar_cases': historical_data[:3] if len(historical_data) > 0 else []
                },
                'prediction': float(prediction),
                'current_gas': current_gas['current_gas']
            }

        # Add horizon
        explanation['horizon'] = horizon

        return jsonify(explanation)

    except Exception as e:
        logger.error(f"Error in /explain: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
