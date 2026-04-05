"""
Shared Model State
Global state for ML models, scalers, and feature names.
Used by routes.py, prediction_routes.py, and other modules that need model access.
"""

import os
import sys
import threading

from utils.logger import logger, log_error_with_context
from api.cache import clear_cache


# Global model state
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
                            'feature_names': model_data.get('feature_names', []),
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

            logger.info(f"Loaded {len(models)} ML models successfully")
            if scalers:
                logger.info(f"Loaded {len(scalers)} scalers")

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
    logger.info("Reloading models from disk...")
    result = load_models()
    if result['success']:
        logger.info(f"Models reloaded successfully: {result['models_loaded']} models, {result['scalers_loaded']} scalers")
        # Clear prediction cache since models changed
        try:
            clear_cache()
            logger.info("Cleared prediction cache")
        except Exception as e:
            logger.debug(f"Could not clear cache: {e}")
    else:
        logger.error(f"Failed to reload models: {result.get('error')}")
    return result


def _lazy_load_models():
    """Load models in background thread to avoid blocking startup"""
    def load_in_background():
        thread_name = threading.current_thread().name
        logger.info(f"[Thread: {thread_name}] Starting background model loading...")
        try:
            result = load_models()
            if result.get('success'):
                logger.info(f"[Thread: {thread_name}] Models loaded successfully: {result.get('models_loaded')} models")
            else:
                logger.error(f"[Thread: {thread_name}] Model loading failed: {result.get('error')}")
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
    logger.info(f"Model loading started in background thread (non-blocking, thread: {thread.name})")


# Start lazy loading in background on module import
_lazy_load_models()
