"""
API routes for model accuracy tracking and feature selection.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

accuracy_bp = Blueprint('accuracy', __name__)


def get_tracker():
    """Get or create accuracy tracker."""
    try:
        from models.accuracy_tracker import get_tracker
        return get_tracker()
    except Exception as e:
        logger.error(f"Failed to load accuracy tracker: {e}")
        return None


def get_feature_selector():
    """Load saved feature selector."""
    try:
        from models.feature_selector import SHAPFeatureSelector
        return SHAPFeatureSelector.load()
    except Exception as e:
        logger.warning(f"Feature selector not loaded: {e}")
        return None


@accuracy_bp.route('/metrics', methods=['GET'])
def get_accuracy_metrics():
    """
    Get current accuracy metrics for all horizons.

    Returns:
        {
            "1h": {"mae": 0.001, "rmse": 0.002, "r2": 0.85, ...},
            "4h": {...},
            "24h": {...}
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        metrics = {}
        for horizon in ['1h', '4h', '24h']:
            metrics[horizon] = tracker.get_current_metrics(horizon)

        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/drift', methods=['GET'])
def check_drift():
    """
    Check for model drift across all horizons.

    Returns:
        {
            "1h": {"is_drifting": false, "drift_ratio": 0.05, ...},
            "4h": {...},
            "24h": {...},
            "should_retrain": false
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        drift_info = {}
        for horizon in ['1h', '4h', '24h']:
            drift = tracker.check_drift(horizon)
            drift_info[horizon] = {
                'is_drifting': drift.is_drifting,
                'drift_ratio': drift.drift_ratio,
                'mae_current': drift.mae_current,
                'mae_baseline': drift.mae_baseline,
                'confidence': drift.confidence,
                'sample_size': drift.sample_size
            }

        should_retrain, reasons = tracker.should_retrain()

        return jsonify({
            'success': True,
            'drift': drift_info,
            'should_retrain': should_retrain,
            'retrain_reasons': reasons
        })
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/record', methods=['POST'])
def record_prediction():
    """
    Record a prediction for later accuracy tracking.

    Request body:
        {
            "horizon": "1h",
            "predicted": 0.00123,
            "actual": 0.00125  (optional, for immediate recording)
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        data = request.get_json()
        horizon = data.get('horizon', '1h')
        predicted = data.get('predicted')
        actual = data.get('actual')

        if predicted is None:
            return jsonify({
                'success': False,
                'error': 'predicted value required'
            }), 400

        if actual is not None:
            # Record both together
            tracker.record_prediction_with_actual(
                horizon=horizon,
                predicted=predicted,
                actual=actual
            )
            return jsonify({
                'success': True,
                'message': 'Prediction and actual recorded'
            })
        else:
            # Record prediction only
            pred_id = tracker.record_prediction(
                horizon=horizon,
                predicted=predicted
            )
            return jsonify({
                'success': True,
                'prediction_id': pred_id,
                'message': 'Prediction recorded'
            })
    except Exception as e:
        logger.error(f"Error recording prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/summary', methods=['GET'])
def get_summary():
    """
    Get comprehensive accuracy tracking summary.
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        summary = tracker.get_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/report', methods=['GET'])
def generate_report():
    """
    Generate accuracy report for a specific horizon and time period.

    Query params:
        horizon: '1h', '4h', or '24h' (default: '1h')
        hours_back: Number of hours to look back (default: 24)
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        horizon = request.args.get('horizon', '1h')
        hours_back = int(request.args.get('hours_back', 24))

        report = tracker.generate_accuracy_report(horizon, hours_back)

        if report is None:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for report'
            }), 400

        return jsonify({
            'success': True,
            'report': {
                'horizon': report.horizon,
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'n_predictions': report.n_predictions,
                'mae': report.mae,
                'rmse': report.rmse,
                'mape': report.mape,
                'directional_accuracy': report.directional_accuracy,
                'r2': report.r2,
                'best_prediction_error': report.best_prediction_error,
                'worst_prediction_error': report.worst_prediction_error
            }
        })
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Feature Selection endpoints
@accuracy_bp.route('/features', methods=['GET'])
def get_selected_features():
    """
    Get the list of SHAP-selected features.
    """
    selector = get_feature_selector()

    if selector is None:
        return jsonify({
            'success': False,
            'error': 'Feature selector not trained yet'
        }), 404

    try:
        return jsonify({
            'success': True,
            'n_features': len(selector.selected_features),
            'selected_features': selector.selected_features,
            'top_10_importance': {
                f: selector.feature_importances[f]
                for f in selector.selected_features[:10]
            }
        })
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/features/importance', methods=['GET'])
def get_feature_importance():
    """
    Get full feature importance report.
    """
    selector = get_feature_selector()

    if selector is None:
        return jsonify({
            'success': False,
            'error': 'Feature selector not trained yet'
        }), 404

    try:
        report = selector.get_feature_report()
        top_n = int(request.args.get('top_n', 30))

        return jsonify({
            'success': True,
            'total_features': len(report),
            'selected_count': report['selected'].sum(),
            'importance_report': report.head(top_n).to_dict('records')
        })
    except Exception as e:
        logger.error(f"Error getting importance: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/history', methods=['GET'])
def get_accuracy_history():
    """
    Get historical accuracy metrics for charting.

    Query params:
        hours_back: Number of hours to look back (default: 168, max: 720)
        resolution: 'hourly' or 'daily' (default: 'hourly')
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        hours_back = min(int(request.args.get('hours_back', 168)), 720)
        resolution = request.args.get('resolution', 'hourly')

        # Get historical data from tracker
        history = tracker.get_accuracy_history(hours_back=hours_back, resolution=resolution)

        return jsonify({
            'success': True,
            'history': history,
            'hours_back': hours_back,
            'resolution': resolution
        })
    except Exception as e:
        logger.error(f"Error getting accuracy history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/features/train', methods=['POST'])
def train_feature_selector():
    """
    Trigger feature selector training with current data.

    Request body (optional):
        {
            "n_features": 30,
            "hours_back": 720
        }
    """
    try:
        from models.feature_engineering import GasFeatureEngineer
        from models.feature_selector import SHAPFeatureSelector

        data = request.get_json() or {}
        n_features = data.get('n_features', 30)
        hours_back = data.get('hours_back', 720)

        # Get training data
        engineer = GasFeatureEngineer()
        df = engineer.prepare_training_data(hours_back=hours_back)
        feature_cols = engineer.get_feature_columns(df)

        X = df[feature_cols]
        y = df['gas']

        # Train selector
        selector = SHAPFeatureSelector(n_features=n_features)
        selector.fit(X, y)
        selector.save()

        return jsonify({
            'success': True,
            'message': f'Feature selector trained with {n_features} features',
            'selected_features': selector.selected_features[:10],
            'total_evaluated': len(feature_cols)
        })
    except Exception as e:
        logger.error(f"Error training feature selector: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
