"""
Prediction Validation API Routes

Endpoints for accessing prediction accuracy metrics, validation status,
and model performance monitoring.
"""

from flask import Blueprint, jsonify, request
from utils.prediction_validator import PredictionValidator
from utils.logger import logger
from api.cache import cached
from api.middleware import require_admin_auth
from datetime import datetime

validation_bp = Blueprint('validation', __name__)
validator = PredictionValidator()


@validation_bp.route('/validation/summary', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_validation_summary():
    """
    Get summary of prediction validation status

    Returns:
        - Total predictions logged
        - Number validated
        - Pending validations
        - Recent performance metrics
    """
    try:
        summary = validator.get_validation_summary()
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"Error getting validation summary: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/metrics', methods=['GET'])
@cached(ttl=300)
def get_validation_metrics():
    """
    Get performance metrics for validated predictions

    Query params:
        - horizon: '1h', '4h', '24h', or 'all' (default: 'all')
        - days: Number of days to look back (default: 7)
        - model_version: Model version filter (optional)

    Returns:
        Performance metrics including MAE, RMSE, directional accuracy
    """
    try:
        horizon = request.args.get('horizon', None)
        days = int(request.args.get('days', 7))
        model_version = request.args.get('model_version', None)

        # Get metrics for each horizon
        if horizon and horizon != 'all':
            metrics = validator.calculate_metrics(
                horizon=horizon,
                days=days,
                model_version=model_version
            )
            return jsonify(metrics), 200
        else:
            # Get metrics for all horizons
            all_metrics = {}
            for h in ['1h', '4h', '24h']:
                all_metrics[h] = validator.calculate_metrics(
                    horizon=h,
                    days=days,
                    model_version=model_version
                )

            return jsonify({
                'horizons': all_metrics,
                'days': days,
                'timestamp': datetime.now().isoformat()
            }), 200

    except Exception as e:
        logger.error(f"Error getting validation metrics: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/trends', methods=['GET'])
@cached(ttl=300)
def get_performance_trends():
    """
    Get performance trend over time

    Query params:
        - horizon: '1h', '4h', or '24h' (required)
        - days: Number of days to look back (default: 30)

    Returns:
        Time series of daily performance metrics
    """
    try:
        horizon = request.args.get('horizon', '1h')
        days = int(request.args.get('days', 30))

        if horizon not in ['1h', '4h', '24h']:
            return jsonify({'error': 'Invalid horizon. Must be 1h, 4h, or 24h'}), 400

        trends = validator.get_performance_trends(horizon=horizon, days=days)

        return jsonify({
            'horizon': horizon,
            'days': days,
            'trends': trends,
            'count': len(trends)
        }), 200

    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/health', methods=['GET'])
@cached(ttl=60)  # Cache for 1 minute
def check_model_health():
    """
    Check model health status

    Returns alerts if:
    - MAE exceeds threshold
    - Directional accuracy is too low
    - No recent validations
    """
    try:
        threshold_mae = float(request.args.get('threshold_mae', 2.0))

        health = validator.check_model_health(threshold_mae=threshold_mae)

        status_code = 200 if health['healthy'] else 503  # Service Degraded

        return jsonify(health), status_code

    except Exception as e:
        logger.error(f"Error checking model health: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/validate', methods=['POST'])
def trigger_validation():
    """
    Manually trigger prediction validation

    This normally runs automatically via scheduler, but can be triggered manually
    for testing or immediate validation.

    Query params:
        - max_age_hours: Only validate predictions younger than this (default: 48)

    Returns:
        Validation results
    """
    try:
        max_age_hours = int(request.args.get('max_age_hours', 48))

        results = validator.validate_predictions(max_age_hours=max_age_hours)

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error triggering validation: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/metrics/daily', methods=['POST'])
@require_admin_auth
def save_daily_metrics():
    """
    Save daily aggregated metrics

    This should be called daily (via cron or scheduler) to save
    performance metrics for trending analysis.

    Requires admin authentication (X-Admin-API-Key header).
    """
    try:
        results = validator.save_daily_metrics()

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error saving daily metrics: {e}")
        return jsonify({'error': str(e)}), 500


@validation_bp.route('/validation/log-prediction', methods=['POST'])
def log_prediction():
    """
    Manually log a prediction (normally done automatically in /predictions endpoint)

    Body:
        {
            "horizon": "1h",
            "predicted_gas": 0.002345,
            "target_time": "2025-12-17T12:00:00",
            "model_version": "v1"
        }

    Returns:
        Prediction log ID
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['horizon', 'predicted_gas', 'target_time']
        missing = [f for f in required_fields if f not in data]

        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Parse target time
        target_time = datetime.fromisoformat(data['target_time'].replace('Z', '+00:00'))

        # Log prediction
        log_id = validator.log_prediction(
            horizon=data['horizon'],
            predicted_gas=float(data['predicted_gas']),
            target_time=target_time,
            model_version=data.get('model_version', 'v1')
        )

        return jsonify({
            'log_id': log_id,
            'message': 'Prediction logged successfully',
            'timestamp': datetime.now().isoformat()
        }), 201

    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        return jsonify({'error': str(e)}), 500


# Admin endpoint to get detailed validation logs
@validation_bp.route('/validation/logs', methods=['GET'])
@require_admin_auth
def get_validation_logs():
    """
    Get detailed prediction logs for debugging

    Query params:
        - limit: Number of logs to return (default: 100)
        - validated: Filter by validation status (true/false)
        - horizon: Filter by horizon

    Requires admin authentication (X-Admin-API-Key header).

    Returns:
        List of prediction logs
    """
    try:
        limit = int(request.args.get('limit', 100))
        validated_filter = request.args.get('validated', None)
        horizon_filter = request.args.get('horizon', None)

        from utils.prediction_validator import PredictionLog
        from data.database import DatabaseManager

        db = DatabaseManager()
        session = db._get_session()

        try:
            query = session.query(PredictionLog)

            if validated_filter is not None:
                validated_bool = validated_filter.lower() == 'true'
                query = query.filter(PredictionLog.validated == validated_bool)

            if horizon_filter:
                query = query.filter(PredictionLog.horizon == horizon_filter)

            logs = query.order_by(
                PredictionLog.prediction_time.desc()
            ).limit(limit).all()

            return jsonify({
                'logs': [{
                    'id': log.id,
                    'prediction_time': log.prediction_time.isoformat(),
                    'target_time': log.target_time.isoformat(),
                    'horizon': log.horizon,
                    'predicted_gas': log.predicted_gas,
                    'actual_gas': log.actual_gas,
                    'absolute_error': log.absolute_error,
                    'direction_correct': log.direction_correct,
                    'model_version': log.model_version,
                    'validated': log.validated,
                    'validated_at': log.validated_at.isoformat() if log.validated_at else None
                } for log in logs],
                'count': len(logs),
                'filters': {
                    'validated': validated_filter,
                    'horizon': horizon_filter,
                    'limit': limit
                }
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error getting validation logs: {e}")
        return jsonify({'error': str(e)}), 500
