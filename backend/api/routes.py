"""
API Routes
All API endpoints for the gas price prediction system
"""

from flask import Blueprint, jsonify, request, send_file, abort
from data.collector import BaseGasCollector
from data.multichain_collector import MultiChainGasCollector
from data.database import DatabaseManager
from models.feature_engineering import GasFeatureEngineer
# model_trainer removed - training done via notebooks/train_all_models.ipynb
from models.accuracy_tracker import get_tracker
from utils.base_scanner import BaseScanner
from utils.logger import logger, log_error_with_context
import sys
from utils.prediction_validator import PredictionValidator
from api.cache import cached, clear_cache
from datetime import datetime, timedelta
import traceback
import numpy as np
import threading
import os


api_bp = Blueprint('api', __name__)


collector = BaseGasCollector()
multichain_collector = MultiChainGasCollector()
db = DatabaseManager()
engineer = GasFeatureEngineer()
scanner = BaseScanner()
validator = PredictionValidator()
accuracy_tracker = get_tracker()


# Load trained models (global state)
models = {}
scalers = {}
feature_names = {}
_models_lock = threading.Lock()  # Thread lock for safe model reloading


def load_models():
    """
    Load models from disk into global state.
    Thread-safe and can be called to reload models after training.
    
    Returns:
        dict: Summary of loaded models
    """
    global models, scalers, feature_names
    
    with _models_lock:
        # Clear existing models
        models.clear()
        scalers.clear()
        feature_names.clear()
        
        try:
            import joblib
            import os
            
            # Get models directory from config (persistent on Railway)
            from config import Config
            models_dir = Config.MODELS_DIR
            
            for horizon in ['1h', '4h', '24h']:
                # Try persistent storage first, then fallback paths
                model_path = os.path.join(models_dir, f'model_{horizon}.pkl')
                if not os.path.exists(model_path):
                    model_path = f'models/saved_models/model_{horizon}.pkl'
                if not os.path.exists(model_path):
                    model_path = f'backend/models/saved_models/model_{horizon}.pkl'

                scaler_path = os.path.join(models_dir, f'scaler_{horizon}.pkl')
                if not os.path.exists(scaler_path):
                    scaler_path = f'models/saved_models/scaler_{horizon}.pkl'
                if not os.path.exists(scaler_path):
                    scaler_path = f'backend/models/saved_models/scaler_{horizon}.pkl'
                
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    
                    # Handle both old and new model formats
                    if isinstance(model_data, dict):
                        models[horizon] = {
                            'model': model_data.get('model'),
                            'model_name': model_data.get('model_name', 'Unknown'),
                            'metrics': model_data.get('metrics', {}),
                            'feature_names': model_data.get('feature_names', []),  # Store directly in model_data
                            'feature_scaler': model_data.get('feature_scaler'),
                            'target_scaler': model_data.get('target_scaler'),
                            'feature_selector': model_data.get('feature_selector'),
                            'predicts_percentage_change': model_data.get('predicts_percentage_change', False),
                            'uses_log_scale': model_data.get('uses_log_scale', False)
                        }
                        
                        # Load scaler if available (prefer feature_scaler)
                        if 'feature_scaler' in model_data:
                            scalers[horizon] = model_data['feature_scaler']
                        elif 'scaler' in model_data:
                            scalers[horizon] = model_data['scaler']
                        elif os.path.exists(scaler_path):
                            scalers[horizon] = joblib.load(scaler_path)
                        
                        # Load feature names if available
                        if 'feature_names' in model_data:
                            feature_names[horizon] = model_data['feature_names']
                    else:
                        # Old format - model is the object itself
                        models[horizon] = {
                            'model': model_data,
                            'model_name': 'Legacy',
                            'metrics': {}
                        }
                else:
                    logger.debug(f"Model file not found for {horizon}")
            
            # Load global feature names if available
            feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
            if not os.path.exists(feature_names_path):
                feature_names_path = 'models/saved_models/feature_names.pkl'
            if not os.path.exists(feature_names_path):
                feature_names_path = 'backend/models/saved_models/feature_names.pkl'

            if os.path.exists(feature_names_path):
                global_feature_names = joblib.load(feature_names_path)
                for horizon in ['1h', '4h', '24h']:
                    if horizon not in feature_names:
                        feature_names[horizon] = global_feature_names
            
            logger.info(f"✅ Loaded {len(models)} ML models successfully")
            if scalers:
                logger.info(f"✅ Loaded {len(scalers)} scalers")
            
            return {
                'success': True,
                'models_loaded': len(models),
                'scalers_loaded': len(scalers),
                'horizons': list(models.keys())
            }
        except Exception as e:
            log_error_with_context(
                e,
                "Model loading operation",
                context={
                    'current_dir': os.getcwd(),
                    'models_cleared': True,
                    'python_version': sys.version
                }
            )
            return {
                'success': False,
                'error': str(e),
                'models_loaded': 0,
                'scalers_loaded': 0,
                'horizons': []
            }


def reload_models():
    """
    Reload models from disk. Thread-safe wrapper around load_models().
    Use this after training completes to load new models without restarting.
    
    Returns:
        dict: Summary of reload operation
    """
    logger.info("🔄 Reloading models from disk...")
    result = load_models()
    if result['success']:
        logger.info(f"✅ Models reloaded successfully: {result['models_loaded']} models, {result['scalers_loaded']} scalers")
        # Clear prediction cache since models changed
        try:
            clear_cache()
            logger.info("✅ Cleared prediction cache")
        except Exception as e:
            logger.debug(f"Could not clear cache: {e}")
    else:
        logger.error(f"❌ Failed to reload models: {result.get('error')}")
    return result


# Lazy model loading - don't block startup
# Models will be loaded on first prediction request or can be loaded via /models/reload
# This ensures the app starts quickly for health checks
def _lazy_load_models():
    """Load models in background thread to avoid blocking startup"""
    import threading
    def load_in_background():
        thread_name = threading.current_thread().name
        logger.info(f"🔄 [Thread: {thread_name}] Starting background model loading...")
        try:
            result = load_models()
            if result.get('success'):
                logger.info(f"✅ [Thread: {thread_name}] Models loaded successfully: {result.get('models_loaded')} models")
            else:
                logger.error(f"❌ [Thread: {thread_name}] Model loading failed: {result.get('error')}")
                if result.get('failed_horizons'):
                    logger.error(f"   Failed horizons: {result.get('failed_horizons')}")
        except Exception as e:
            log_error_with_context(
                e,
                "Background model loading thread",
                context={
                    'thread_name': thread_name,
                    'is_daemon': threading.current_thread().daemon
                }
            )

    # Start loading in background (don't wait for completion)
    thread = threading.Thread(target=load_in_background, name="ModelLoader", daemon=True)
    thread.start()
    logger.info(f"✓ Model loading started in background thread (non-blocking, thread: {thread.name})")

# Start lazy loading in background
_lazy_load_models()


@api_bp.route('/models/reload', methods=['POST'])
def reload_models_endpoint():
    """
    Manually reload models from disk.
    Useful after training completes to use new models without restarting.
    
    Returns:
        Summary of reload operation
    """
    try:
        result = reload_models()
        return jsonify({
            'success': result['success'],
            'message': 'Models reloaded successfully' if result['success'] else 'Failed to reload models',
            'models_loaded': result['models_loaded'],
            'scalers_loaded': result['scalers_loaded'],
            'horizons': result.get('horizons', []),
            'error': result.get('error')
        }), 200 if result['success'] else 500
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint - must respond quickly for deployment health checks"""
    try:
        # Quick response - don't block on model loading
        # Just verify the app is running
        health_status = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': False,
            'hybrid_predictor_loaded': False,
            'legacy_models_loaded': False,
            'database_connected': True,
            'cache_stats': {},
            'warnings': []
        }
        
        # Check hybrid predictor (non-blocking)
        try:
            from models.hybrid_predictor import hybrid_predictor
            health_status['hybrid_predictor_loaded'] = hybrid_predictor.loaded
        except Exception as hybrid_error:
            log_error_with_context(
                hybrid_error,
                "Health check: Checking hybrid predictor",
                level='warning'
            )
            health_status['warnings'].append(f"Hybrid predictor check failed: {str(hybrid_error)}")

        # Check legacy models (non-blocking, just check if any loaded)
        try:
            health_status['legacy_models_loaded'] = len(models) > 0
            health_status['models_loaded'] = health_status['legacy_models_loaded'] or health_status['hybrid_predictor_loaded']
        except Exception as models_error:
            log_error_with_context(
                models_error,
                "Health check: Checking legacy models",
                level='warning'
            )
            health_status['warnings'].append(f"Legacy models check failed: {str(models_error)}")

        # Get cache statistics (non-blocking)
        try:
            from api.cache import get_cache_stats
            health_status['cache_stats'] = get_cache_stats()
        except Exception as cache_error:
            log_error_with_context(
                cache_error,
                "Health check: Getting cache statistics",
                level='warning'
            )
            health_status['warnings'].append(f"Cache stats check failed: {str(cache_error)}")

        # Check database connection (non-blocking)
        try:
            from data.database import DatabaseManager
            db = DatabaseManager()
            # Just check if we can get a session, don't run a query
            session = db._get_session()
            session.close()
            health_status['database_connected'] = True
        except Exception as db_error:
            log_error_with_context(
                db_error,
                "Health check: Verifying database connection",
                level='warning'
            )
            health_status['database_connected'] = False
            health_status['warnings'].append(f"Database check failed: {str(db_error)}")

        # Log warnings if any
        if health_status['warnings']:
            logger.warning(f"Health check completed with {len(health_status['warnings'])} warnings")
            for warning in health_status['warnings']:
                logger.warning(f"   - {warning}")

        return jsonify(health_status), 200
    except Exception as e:
        # Even if there's an error, return 200 so deployment doesn't fail
        # The app is running even if some components aren't ready
        log_error_with_context(
            e,
            "Health check endpoint",
            context={
                'endpoint': '/api/health',
                'method': 'GET'
            },
            level='warning'
        )
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'warning': str(e),
            'models_loaded': False,
            'database_connected': False,
            'error_in_health_check': True
        }), 200


@api_bp.route('/cache/stats', methods=['GET'])
def get_cache_statistics():
    """Get cache statistics endpoint"""
    try:
        from api.cache import get_cache_stats
        stats = get_cache_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/current', methods=['GET'])
@cached(ttl=30)  # Cache for 30 seconds
def current_gas():
    """Get current Base gas price"""
    try:
        data = collector.get_current_gas()
        if data:
            # Try to save to database, but don't fail if database is locked
            try:
                db.save_gas_price(data)
            except Exception as db_error:
                # Log database error but still return data to user
                logger.warning(f"Could not save to database (database may be locked): {db_error}")

            logger.info(f"Current gas: {data['current_gas']} gwei")
            return jsonify(data)
        return jsonify({'error': 'Failed to fetch gas data'}), 500
    except Exception as e:
        logger.error(f"Error in /current: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/eth-price', methods=['GET'])
@cached(ttl=60)  # Cache for 1 minute
def eth_price():
    """Proxy endpoint for ETH price from CoinGecko (avoids CORS issues)"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Configure retry strategy for transient errors
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.get(
            'https://api.coingecko.com/api/v3/simple/price',
            params={
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            },
            headers={
                'User-Agent': 'Gweizy/1.0',
                'Accept': 'application/json'
            },
            timeout=(5, 10)  # (connect timeout, read timeout)
        )

        if response.ok:
            data = response.json()
            return jsonify(data), 200
        else:
            logger.warning(f"CoinGecko API returned status {response.status_code}")
            return jsonify({'error': 'Failed to fetch ETH price'}), response.status_code
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error fetching ETH price (will retry on next request): {e}")
        return jsonify({'error': 'ETH price service temporarily unavailable'}), 503
    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout fetching ETH price: {e}")
        return jsonify({'error': 'ETH price request timed out'}), 504
    except Exception as e:
        logger.error(f"Error fetching ETH price: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@api_bp.route('/historical', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def historical():
    """Get historical gas prices"""
    try:
        hours = request.args.get('hours', 168, type=int)  # Default 7 days
        timeframe = request.args.get('timeframe', 'hourly')  # hourly, daily
        
        data = db.get_historical_data(hours=hours)
        
        if not data:
            return jsonify({'error': 'No historical data available'}), 404
        
        # Format for frontend
        # data is now a list of dicts, not ORM objects
        formatted_data = []
        for d in data:
            # Handle timestamp - could be string or datetime
            timestamp = d.get('timestamp', '')
            if isinstance(timestamp, str):
                from dateutil import parser
                try:
                    timestamp = parser.parse(timestamp)
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            formatted_data.append({
                'time': timestamp.strftime('%Y-%m-%d %H:%M') if isinstance(timestamp, datetime) else str(timestamp),
                'gwei': round(d.get('gwei', 0), 4),
                'baseFee': round(d.get('baseFee', 0), 4),
                'priorityFee': round(d.get('priorityFee', 0), 4)
            })
        
        logger.info(f"Returned {len(formatted_data)} historical records")
        return jsonify({
            'data': formatted_data,
            'count': len(formatted_data),
            'timeframe': timeframe
        })
        
    except Exception as e:
        logger.error(f"Error in /historical: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/gas/patterns', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def gas_patterns():
    """Get hourly and daily gas price patterns for heatmap views."""
    try:
        hours = request.args.get('hours', 168, type=int)
        data = db.get_historical_data(hours=hours)

        if not data:
            return jsonify({'error': 'No historical data available'}), 404

        from dateutil import parser

        hourly_groups = {hour: [] for hour in range(24)}
        daily_groups = {day: [] for day in range(7)}
        all_values = []

        for entry in data:
            timestamp = entry.get('timestamp', '')
            if not timestamp:
                continue
            try:
                dt = parser.parse(timestamp) if isinstance(timestamp, str) else timestamp
            except Exception:
                continue

            gwei = entry.get('gwei')
            if gwei is None:
                gwei = entry.get('current_gas')
            if gwei is None:
                continue

            all_values.append(gwei)
            hourly_groups[dt.hour].append(gwei)
            daily_groups[dt.weekday()].append(gwei)

        if not all_values:
            return jsonify({'success': False, 'error': 'No valid gas data available'}), 404

        overall_avg = sum(all_values) / len(all_values)

        hourly = []
        for hour in range(24):
            samples = hourly_groups[hour]
            if samples:
                avg_gwei = sum(samples) / len(samples)
                min_gwei = min(samples)
                max_gwei = max(samples)
            else:
                avg_gwei = overall_avg
                min_gwei = overall_avg
                max_gwei = overall_avg

            hourly.append({
                'hour': hour,
                'avg_gwei': round(avg_gwei, 8),
                'min_gwei': round(min_gwei, 8),
                'max_gwei': round(max_gwei, 8),
                'sample_count': len(samples)
            })

        daily = []
        for day in range(7):
            samples = daily_groups[day]
            if samples and len(samples) > 0:
                try:
                    avg_gwei = sum(samples) / len(samples)
                    min_gwei = min(samples)
                    max_gwei = max(samples)
                except (ValueError, TypeError):
                    # Fallback in case of calculation error
                    avg_gwei = overall_avg
                    min_gwei = overall_avg
                    max_gwei = overall_avg
            else:
                # Use overall average for missing days
                avg_gwei = overall_avg
                min_gwei = overall_avg
                max_gwei = overall_avg

            daily.append({
                'day': day,
                'avg_gwei': round(avg_gwei, 8),
                'min_gwei': round(min_gwei, 8),
                'max_gwei': round(max_gwei, 8)
            })

        # Get cheapest/most expensive hours and days (with safety checks)
        cheapest_hour = min(hourly, key=lambda h: h['avg_gwei'])['hour'] if hourly else 0
        most_expensive_hour = max(hourly, key=lambda h: h['avg_gwei'])['hour'] if hourly else 23
        cheapest_day = min(daily, key=lambda d: d['avg_gwei'])['day'] if daily else 0
        most_expensive_day = max(daily, key=lambda d: d['avg_gwei'])['day'] if daily else 6

        return jsonify({
            'success': True,
            'data': {
                'hourly': hourly,
                'daily': daily,
                'overall_avg': round(overall_avg, 8),
                'cheapest_hour': cheapest_hour,
                'most_expensive_hour': most_expensive_hour,
                'cheapest_day': cheapest_day,
                'most_expensive_day': most_expensive_day
            }
        })

    except Exception as e:
        logger.error(f"Error in /gas/patterns: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/predictions', methods=['GET'])
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
                # Pattern matching is resource-intensive; skip for real-time responses
                # Uncomment to enable:
                # try:
                #     from models.pattern_matcher import get_pattern_matcher
                #     pattern_matcher = get_pattern_matcher()
                #     matches = pattern_matcher.find_similar_patterns(df, df)
                #     if matches:
                #         current_price = current.get('current_gas', 0.01)
                #         pattern_data = pattern_matcher.predict_from_patterns(matches, current_price)
                # except Exception as pattern_err:
                #     logger.debug(f"Pattern matching unavailable: {pattern_err}")

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
                                # Note: feature_selector was already applied during training, so these are the selected features
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
                            target_scaler_path = f'backend/models/saved_models/target_scaler_{horizon}.pkl'
                            if os.path.exists(target_scaler_path):
                                target_scaler = joblib.load(target_scaler_path)
                                logger.info(f"Loaded target_scaler from {target_scaler_path}")
                        except Exception as e:
                            logger.warning(f"Could not load target_scaler: {e}")

                    if uses_log_scale and target_scaler is not None:
                        # Model predicts in log space with scaling
                        # pred is in scaled log space, need to inverse transform then exp
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
                        # 1h: ±30%, 4h: ±50%, 24h: ±80%
                        horizon_clamps = {'1h': 30, '4h': 50, '24h': 80}
                        max_clamp = horizon_clamps.get(horizon, 50)
                        pct_change = max(-max_clamp, min(max_clamp, pct_change))

                        # Blend: 70% historical mean, 30% current price - makes predictions more stable
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


@api_bp.route('/explain/<horizon>', methods=['GET'])
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
                feature_names = list(features.columns)
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
                    feature_names = list(engineer.get_feature_columns(sample_df))
                except Exception as e:
                    logger.warning(f"Could not get feature names: {e}")
                    # Fallback feature names
                    feature_names = ['hour', 'day_of_week', 'trend_1h', 'trend_3h', 'avg_last_24h']
            
            current_explainer = initialize_explainer(model, feature_names)
        
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


@api_bp.route('/transactions', methods=['GET'])
@cached(ttl=30)  # Cache for 30 seconds
def get_transactions():
    """Get recent Base transactions"""
    try:
        limit = request.args.get('limit', 10, type=int)
        transactions = scanner.get_recent_transactions(limit=limit)
        
        logger.info(f"Returned {len(transactions)} transactions")
        return jsonify({
            'transactions': transactions,
            'count': len(transactions)
        })
        
    except Exception as e:
        logger.error(f"Error in /transactions: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/accuracy', methods=['GET'])
@cached(ttl=3600)  # Cache for 1 hour
def get_accuracy():
    """Get real-time model accuracy metrics from AccuracyTracker"""
    try:
        from datetime import datetime, timedelta
        
        # Get real-time metrics from AccuracyTracker
        if accuracy_tracker is None:
            logger.warning("Accuracy tracker not available, using fallback")
            # Fallback to hardcoded if tracker unavailable
            return jsonify({
                'mae': 0.000275,
                'rmse': 0.000442,
                'r2': 0.0709,
                'directional_accuracy': 0.5983,
                'recent_predictions': [],
                'last_updated': datetime.now().isoformat(),
                'warning': 'Using fallback metrics - accuracy tracker unavailable'
            }), 200
        
        # Get metrics for 1h horizon (primary metric)
        metrics_1h = accuracy_tracker.get_current_metrics('1h')
        
        # Check if we have enough data
        if metrics_1h.get('n', 0) < 5:
            logger.warning(f"Insufficient accuracy data: {metrics_1h.get('n', 0)} predictions")
            # Return metrics with warning if insufficient data
            return jsonify({
                'mae': metrics_1h.get('mae'),
                'rmse': metrics_1h.get('rmse'),
                'r2': metrics_1h.get('r2', 0),
                'directional_accuracy': metrics_1h.get('directional_accuracy', 0),
                'recent_predictions': [],
                'last_updated': datetime.now().isoformat(),
                'warning': f'Limited data: only {metrics_1h.get("n", 0)} predictions available',
                'n_predictions': metrics_1h.get('n', 0)
            }), 200
        
        # Extract real metrics
        mae = metrics_1h.get('mae', 0)
        rmse = metrics_1h.get('rmse', 0)
        r2 = metrics_1h.get('r2', 0)
        directional_accuracy = metrics_1h.get('directional_accuracy', 0)
        n_predictions = metrics_1h.get('n', 0)
        
        # Get recent predictions vs actuals for chart
        recent_predictions = []
        try:
            # Get accuracy history for recent predictions
            history = accuracy_tracker.get_accuracy_history(hours_back=24, resolution='hourly')
            history_1h = history.get('1h', [])
            
            # Get actual prediction records for detailed chart
            records = list(accuracy_tracker.predictions['1h'])
            records = [r for r in records if r.actual is not None]
            recent_records = records[-24:] if len(records) >= 24 else records
            
            for record in recent_records:
                recent_predictions.append({
                    'timestamp': record.timestamp.isoformat(),
                    'predicted': round(record.predicted, 6),
                    'actual': round(record.actual, 6),
                    'error': round(abs(record.error), 6) if record.error is not None else 0
                })
        except Exception as e:
            logger.warning(f"Could not get recent predictions: {e}")
            recent_predictions = []
        
        # Convert to percentages for logging
        r2_percent = r2 * 100
        directional_accuracy_percent = directional_accuracy * 100
        
        result = {
            'mae': float(mae) if mae is not None else 0,
            'rmse': float(rmse) if rmse is not None else 0,
            'r2': float(r2) if r2 is not None else 0,  # Return as decimal (0.0-1.0), frontend will convert to %
            'directional_accuracy': float(directional_accuracy) if directional_accuracy is not None else 0,  # Return as decimal (0.0-1.0), frontend will convert to %
            'recent_predictions': recent_predictions[:24],  # Last 24 hours
            'last_updated': datetime.now().isoformat(),
            'n_predictions': n_predictions,
            'source': 'real-time'  # Indicate these are real metrics
        }
        
        logger.info(f"Real-time accuracy metrics - R²: {r2:.4f} ({r2_percent:.1f}%), Directional: {directional_accuracy:.4f} ({directional_accuracy_percent:.1f}%), N={n_predictions}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /accuracy: {traceback.format_exc()}")
        # Return fallback metrics on error
        from datetime import datetime
        return jsonify({
            'mae': 0.000275,
            'rmse': 0.000442,
            'r2': 0.0709,
            'directional_accuracy': 0.5983,
            'recent_predictions': [],
            'last_updated': datetime.now().isoformat(),
            'error': str(e),
            'warning': 'Error calculating metrics, using fallback values'
        }), 200  # Still return 200 so frontend can display


@api_bp.route('/user-history/<address>', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_user_history(address):
    """Get user transaction history and savings analysis"""
    try:
        import requests
        from datetime import datetime, timedelta
        from config import Config
        
        # BaseScan API endpoint
        basescan_api_key = Config.BASESCAN_API_KEY
        basescan_url = f"https://api.basescan.org/api"
        
        # Get transactions from last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_block = int(start_date.timestamp())
        end_block = int(end_date.timestamp())
        
        # Fetch transactions from BaseScan
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': 100,
            'sort': 'desc'
        }
        
        if basescan_api_key:
            params['apikey'] = basescan_api_key
        
        response = requests.get(basescan_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != '1' or not data.get('result'):
            # Return default recommendations even if no transactions
            current_hour = datetime.now().hour
            default_recommendations = {
                'usual_time': f"{current_hour}:00 UTC",
                'best_time': "7:00 UTC",
                'avg_savings': 35
            }
            return jsonify({
                'transactions': [],
                'total_transactions': 0,
                'total_gas_paid': 0,
                'potential_savings': 0,
                'savings_percentage': 0,
                'recommendations': default_recommendations
            })
        
        transactions = data['result']
        
        # Filter to last 30 days and Base network
        recent_transactions = []
        total_gas_paid = 0
        potential_savings = 0
        transaction_times = []
        
        ETH_PRICE = 3000  # USD per ETH
        
        for tx in transactions[:50]:  # Limit to 50 most recent
            tx_timestamp = int(tx.get('timeStamp', 0))
            tx_date = datetime.fromtimestamp(tx_timestamp)
            
            if tx_date < start_date:
                continue
            
            gas_used = int(tx.get('gasUsed', 0))
            gas_price = int(tx.get('gasPrice', 0))
            
            if gas_used == 0 or gas_price == 0:
                continue
            
            # Calculate cost
            gas_price_gwei = gas_price / 1e9
            cost_eth = (gas_price * gas_used) / 1e18
            cost_usd = cost_eth * ETH_PRICE
            total_gas_paid += cost_usd
            
            # Estimate optimal cost (using current best prediction)
            current_gas = collector.get_current_gas()
            if current_gas:
                current_gas_price = current_gas.get('current_gas', 0)
                # Assume could have saved 30% on average (this would use actual predictions)
                optimal_cost = cost_usd * 0.7  # 30% savings estimate
                potential_savings += (cost_usd - optimal_cost)
            
            transaction_times.append(tx_date.hour)
            
            recent_transactions.append({
                'hash': tx.get('hash', ''),
                'timestamp': tx_timestamp,
                'gasUsed': gas_used,
                'gasPrice': gas_price,
                'value': tx.get('value', '0'),
                'from': tx.get('from', ''),
                'to': tx.get('to', ''),
                'method': tx.get('methodId', '0x')[:10] if tx.get('methodId') else 'Transfer'
            })
        
        # Calculate recommendations
        from collections import Counter
        recommendations = {}
        
        if transaction_times and len(transaction_times) > 0:
            time_counts = Counter(transaction_times)
            most_common_hour = time_counts.most_common(1)[0][0]
            recommendations['usual_time'] = f"{most_common_hour}:00 UTC"
            # Suggest opposite time (when gas is typically lower)
            best_hour = (most_common_hour + 6) % 24
            recommendations['best_time'] = f"{best_hour}:00 UTC"
            recommendations['avg_savings'] = 40  # Estimated average savings
        else:
            # Default recommendations if no transaction history
            current_hour = datetime.now().hour
            recommendations['usual_time'] = f"{current_hour}:00 UTC"
            # Suggest early morning (typically lower gas)
            recommendations['best_time'] = "7:00 UTC"
            recommendations['avg_savings'] = 35  # Estimated average savings
        
        savings_percentage = (potential_savings / total_gas_paid * 100) if total_gas_paid > 0 else 0
        
        result = {
            'transactions': recent_transactions[:10],  # Return last 10
            'total_transactions': len(recent_transactions),
            'total_gas_paid': round(total_gas_paid, 4),
            'potential_savings': round(potential_savings, 4),
            'savings_percentage': round(savings_percentage, 2),
            'recommendations': recommendations
        }
        
        logger.info(f"Returned user history for {address[:10]}...")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /user-history: {traceback.format_exc()}")
        # Always return recommendations, even on error
        current_hour = datetime.now().hour
        default_recommendations = {
            'usual_time': f"{current_hour}:00 UTC",
            'best_time': "7:00 UTC",
            'avg_savings': 35
        }
        return jsonify({
            'error': str(e),
            'transactions': [],
            'total_transactions': 0,
            'total_gas_paid': 0,
            'potential_savings': 0,
            'savings_percentage': 0,
            'recommendations': default_recommendations
        }), 200  # Return 200 so frontend can handle gracefully


@api_bp.route('/leaderboard', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_leaderboard():
    """Get savings leaderboard from real user transaction data"""
    try:
        from data.database import UserTransaction
        from sqlalchemy import func
        
        chain_id = request.args.get('chain_id', type=int, default=8453)
        period = request.args.get('period', 'week')  # week, month, all
        
        session = db._get_session()
        try:
            # Calculate date filter based on period
            from datetime import datetime, timedelta
            if period == 'week':
                cutoff_date = datetime.now() - timedelta(days=7)
            elif period == 'month':
                cutoff_date = datetime.now() - timedelta(days=30)
            else:
                cutoff_date = datetime.min  # All time
            
            # Get user savings aggregated from transactions
            user_savings = session.query(
                UserTransaction.user_address,
                func.sum(UserTransaction.saved_by_waiting).label('total_savings'),
                func.count(UserTransaction.id).label('transaction_count')
            ).filter(
                UserTransaction.chain_id == chain_id,
                UserTransaction.status == 'success',
                UserTransaction.saved_by_waiting.isnot(None),
                UserTransaction.saved_by_waiting > 0,
                UserTransaction.timestamp >= cutoff_date
            ).group_by(
                UserTransaction.user_address
            ).order_by(
                func.sum(UserTransaction.saved_by_waiting).desc()
            ).limit(100).all()
            
            # Build leaderboard
            leaderboard = []
            for rank, (address, total_savings, tx_count) in enumerate(user_savings, 1):
                leaderboard.append({
                    'address': address,
                    'savings': round(float(total_savings or 0), 2),
                    'rank': rank,
                    'transaction_count': tx_count
                })
            
            # Get user rank if address provided
            user_address = request.args.get('address')
            user_rank = None
            if user_address:
                user_address_lower = user_address.lower()
                # Find user in leaderboard
                user_entry = next((e for e in leaderboard if e['address'].lower() == user_address_lower), None)
                if user_entry:
                    user_rank = user_entry['rank']
                else:
                    # Calculate user's rank even if not in top 100
                    user_total = session.query(
                        func.sum(UserTransaction.saved_by_waiting)
                    ).filter(
                        UserTransaction.user_address == user_address_lower,
                        UserTransaction.chain_id == chain_id,
                        UserTransaction.status == 'success',
                        UserTransaction.saved_by_waiting.isnot(None),
                        UserTransaction.saved_by_waiting > 0,
                        UserTransaction.timestamp >= cutoff_date
                    ).scalar() or 0
                    
                    if user_total > 0:
                        # Count users with more savings
                        users_ahead = session.query(func.count(func.distinct(UserTransaction.user_address))).filter(
                            UserTransaction.chain_id == chain_id,
                            UserTransaction.status == 'success',
                            UserTransaction.saved_by_waiting.isnot(None),
                            UserTransaction.saved_by_waiting > user_total,
                            UserTransaction.timestamp >= cutoff_date
                        ).scalar() or 0
                        user_rank = users_ahead + 1
            
            return jsonify({
                'leaderboard': leaderboard,
                'user_rank': user_rank,
                'period': period,
                'chain_id': chain_id,
                'total_users': len(leaderboard)
            })
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"Error in /leaderboard: {traceback.format_exc()}")
        return jsonify({
            'leaderboard': [],
            'user_rank': None,
            'error': str(e)
        }), 200


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Base platform configuration"""
    return jsonify({
        'name': 'Base Gas Optimizer',
        'description': 'ML-powered gas price predictions for Base network',
        'chainId': 8453,
        'version': '1.0.0',
        'features': [
            'Real-time gas tracking',
            'ML predictions (1h, 4h, 24h)',
            'Transaction history',
            'Model accuracy metrics'
        ]
    })


@api_bp.route('/cache/clear', methods=['POST'])
def clear_all_cache():
    """Clear API cache (admin endpoint)"""
    clear_cache()
    logger.info("Cache cleared by request")
    return jsonify({'message': 'Cache cleared'})


@api_bp.route('/stats', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def stats():
    """Get statistics about gas prices"""
    try:
        hours = request.args.get('hours', 24, type=int)
        data = db.get_historical_data(hours)
        
        if not data:
            return jsonify({
                'hours': hours,
                'count': 0,
                'stats': None
            })
        
        gas_prices = [d.get('gwei', 0) for d in data]

        stats = {
            'hours': hours,
            'count': len(gas_prices),
            'min': round(min(gas_prices), 6),
            'max': round(max(gas_prices), 6),
            'avg': round(sum(gas_prices) / len(gas_prices), 6),
            'latest': round(gas_prices[-1], 6) if gas_prices else None
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in /stats: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/export', methods=['GET'])
def export_data():
    """
    Export historical gas price data as CSV or JSON.

    Query params:
        format: 'csv' or 'json' (default: json)
        hours: Number of hours of data (default: 24, max: 720 = 30 days)

    Returns:
        CSV file download or JSON array
    """
    try:
        export_format = request.args.get('format', 'json').lower()
        hours = min(request.args.get('hours', 24, type=int), 720)  # Max 30 days

        data = db.get_historical_data(hours)

        if not data:
            return jsonify({'error': 'No data available'}), 404

        # Format data for export
        export_data = []
        for record in data:
            export_data.append({
                'timestamp': record.get('timestamp', ''),
                'gas_price_gwei': record.get('gwei', record.get('current_gas', 0)),
                'base_fee_gwei': record.get('base_fee', 0),
                'priority_fee_gwei': record.get('priority_fee', 0),
                'block_number': record.get('block_number', ''),
                'chain_id': record.get('chain_id', 8453)
            })

        if export_format == 'csv':
            import csv
            import io

            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)

            response = api_bp.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=gas_data_{hours}h.csv'}
            )
            return response

        # Default: JSON
        return jsonify({
            'success': True,
            'hours': hours,
            'count': len(export_data),
            'data': export_data
        })

    except Exception as e:
        logger.error(f"Error in /export: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/patterns', methods=['GET'])
@cached(ttl=30)
def get_pattern_analysis():
    """
    Get detailed historical pattern matching analysis.

    Returns similar patterns from history and what happened after them.
    Query params:
        - hours: Hours of historical data to search (default: 168 = 1 week)
    """
    try:
        import pandas as pd
        from dateutil import parser
        from models.pattern_matcher import get_pattern_matcher

        hours = int(request.args.get('hours', 168))
        chain_id = request.args.get('chain_id', 8453, type=int)

        # Get historical data
        historical_data = db.get_historical_data(hours=hours, chain_id=chain_id)

        if len(historical_data) < 50:
            return jsonify({
                'available': False,
                'reason': f'Not enough historical data: {len(historical_data)} records (need 50+)',
                'data_points': len(historical_data)
            }), 200

        # Convert to DataFrame
        df_data = []
        for d in historical_data:
            timestamp = d.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    dt = parser.parse(timestamp)
                except (ValueError, TypeError):
                    dt = datetime.now()
            else:
                dt = timestamp if hasattr(timestamp, 'hour') else datetime.now()

            gas_price = d.get('gwei', 0) or d.get('current_gas', 0)
            if gas_price and gas_price > 0:
                df_data.append({
                    'timestamp': dt,
                    'gas_price': gas_price
                })

        if len(df_data) < 50:
            return jsonify({
                'available': False,
                'reason': f'Not enough valid data: {len(df_data)} records',
                'data_points': len(df_data)
            }), 200

        df = pd.DataFrame(df_data).sort_values('timestamp').reset_index(drop=True)

        # Find patterns
        pattern_matcher = get_pattern_matcher()
        matches = pattern_matcher.find_similar_patterns(df, df)

        if not matches:
            return jsonify({
                'available': True,
                'match_count': 0,
                'reason': 'No similar patterns found in history',
                'data_points': len(df)
            }), 200

        # Generate predictions from patterns
        current_price = df['gas_price'].iloc[-1]
        predictions = pattern_matcher.predict_from_patterns(matches, current_price)

        # Format detailed match information
        match_details = []
        for match in matches[:5]:  # Top 5 matches
            match_details.append({
                'timestamp': match.timestamp.isoformat() + 'Z',
                'correlation': round(match.correlation, 4),
                'time_similarity': round(match.time_similarity, 4),
                'combined_score': round(match.combined_score, 4),
                'outcome': {
                    '1h_change': round(match.outcome_change_1h * 100, 2),
                    '4h_change': round(match.outcome_change_4h * 100, 2)
                }
            })

        return jsonify({
            'available': True,
            'current_price': round(current_price, 6),
            'data_points': len(df),
            'search_period_hours': hours,
            'match_count': len(matches),
            'predictions': predictions,
            'top_matches': match_details,
            'timestamp': datetime.now().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error in pattern analysis: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/database/info', methods=['GET'])
def database_info():
    """
    Get information about the database contents.
    
    Returns:
        JSON with database statistics including record counts, date ranges, etc.
    """
    try:
        from sqlalchemy import func, text
        from data.database import GasPrice
        from datetime import datetime
        
        session = db._get_session()
        
        # Get total record count
        total_records = session.query(func.count(GasPrice.id)).scalar() or 0
        
        # Get date range
        min_date = session.query(func.min(GasPrice.timestamp)).scalar()
        max_date = session.query(func.max(GasPrice.timestamp)).scalar()
        
        # Get recent records (last 24 hours)
        from datetime import timedelta
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        recent_count = session.query(func.count(GasPrice.id)).filter(
            GasPrice.timestamp >= twenty_four_hours_ago
        ).scalar() or 0
        
        # Get average gas price
        avg_gas = session.query(func.avg(GasPrice.current_gas)).scalar()
        
        # Get database file info
        db_path = None
        file_size_mb = None
        if Config.DATABASE_URL.startswith('sqlite'):
            db_path = Config.DATABASE_URL.replace('sqlite:///', '')
            if db_path.startswith('/'):
                if os.path.exists(db_path):
                    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            else:
                # Relative path
                full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)
                if os.path.exists(full_path):
                    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    db_path = full_path
        
        session.close()
        
        return jsonify({
            'success': True,
            'database': {
                'total_records': total_records,
                'recent_records_24h': recent_count,
                'date_range': {
                    'earliest': min_date.isoformat() if min_date else None,
                    'latest': max_date.isoformat() if max_date else None
                },
                'average_gas_price_gwei': round(avg_gas, 6) if avg_gas else None,
                'file_info': {
                    'path': db_path,
                    'size_mb': round(file_size_mb, 2) if file_size_mb else None,
                    'exists': os.path.exists(db_path) if db_path else False
                },
                'database_url': Config.DATABASE_URL,
                'has_data': total_records > 0,
                'ready_for_training': total_records >= 1000,
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting database info: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to get database info: {str(e)}'}), 500


@api_bp.route('/database/download', methods=['GET'])
def download_database():
    """
    Download the gas_data.db database file.
    
    This endpoint allows downloading the database for training models on Colab.
    The database is located at /data/gas_data.db on Railway or gas_data.db locally.
    
    Query Parameters:
        token (optional): Simple token for basic security (set via DB_DOWNLOAD_TOKEN env var)
    
    Returns:
        Database file as download, or error message
    """
    try:
        # Optional token-based security
        download_token = os.getenv('DB_DOWNLOAD_TOKEN', '')
        if download_token:
            provided_token = request.args.get('token', '')
            if provided_token != download_token:
                logger.warning(f"Unauthorized database download attempt from {request.remote_addr}")
                return jsonify({'error': 'Unauthorized. Token required.'}), 401
        
        # Determine database path (same logic as config.py)
        if os.path.exists('/data'):
            db_path = '/data/gas_data.db'
        else:
            # Try local path
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gas_data.db')
            if not os.path.exists(db_path):
                # Try current directory
                db_path = 'gas_data.db'
        
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return jsonify({
                'error': 'Database file not found',
                'searched_paths': [
                    '/data/gas_data.db' if os.path.exists('/data') else None,
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gas_data.db'),
                    'gas_data.db'
                ]
            }), 404
        
        # Get file size for logging
        file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        logger.info(f"Database download requested: {db_path} ({file_size_mb:.2f} MB) from {request.remote_addr}")
        
        # Send file as download
        return send_file(
            db_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='gas_data.db'
        )
        
    except Exception as e:
        logger.error(f"Error downloading database: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to download database: {str(e)}'}), 500
