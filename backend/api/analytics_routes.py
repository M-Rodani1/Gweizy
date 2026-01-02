"""
Analytics API Routes

Provides real-time analytics, model performance tracking, and prediction accuracy metrics.
"""

from flask import Blueprint, jsonify, request
from utils.prediction_validator import PredictionValidator
from data.database import DatabaseManager
from utils.logger import logger
from api.cache import cached
from datetime import datetime, timedelta
import traceback
import numpy as np


analytics_bp = Blueprint('analytics', __name__)
validator = PredictionValidator()
db = DatabaseManager()


@analytics_bp.route('/performance', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_performance_metrics():
    """
    Get current model performance metrics

    Query params:
        horizon: '1h', '4h', '24h' (optional, default: all)
        days: Number of days to look back (default: 7)
    """
    try:
        horizon = request.args.get('horizon')
        days = request.args.get('days', 7, type=int)

        if horizon and horizon not in ['1h', '4h', '24h']:
            return jsonify({'error': 'Invalid horizon. Must be 1h, 4h, or 24h'}), 400

        if horizon:
            # Single horizon metrics
            metrics = validator.calculate_metrics(horizon=horizon, days=days)
            return jsonify(metrics)
        else:
            # All horizons
            all_metrics = {}
            for h in ['1h', '4h', '24h']:
                all_metrics[h] = validator.calculate_metrics(horizon=h, days=days)

            return jsonify({
                'metrics': all_metrics,
                'days': days,
                'timestamp': datetime.now().isoformat()
            })

    except Exception as e:
        logger.error(f"Error in /analytics/performance: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/trends', methods=['GET'])
@cached(ttl=600)  # Cache for 10 minutes
def get_performance_trends():
    """
    Get performance trends over time

    Query params:
        horizon: '1h', '4h', '24h' (required)
        days: Number of days to look back (default: 30)
    """
    try:
        horizon = request.args.get('horizon')
        days = request.args.get('days', 30, type=int)

        if not horizon or horizon not in ['1h', '4h', '24h']:
            return jsonify({'error': 'horizon parameter required (1h, 4h, or 24h)'}), 400

        trends = validator.get_performance_trends(horizon=horizon, days=days)

        return jsonify({
            'horizon': horizon,
            'days': days,
            'trends': trends,
            'count': len(trends),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in /analytics/trends: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/validation-summary', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_validation_summary():
    """Get summary of prediction validation status"""
    try:
        summary = validator.get_validation_summary()
        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error in /analytics/validation-summary: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/model-health', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_model_health():
    """
    Check model health and get alerts for degraded performance

    Query params:
        threshold: MAE threshold for alerts (default: 0.001)
    """
    try:
        threshold = request.args.get('threshold', 0.001, type=float)
        health = validator.check_model_health(threshold_mae=threshold)

        return jsonify(health)

    except Exception as e:
        logger.error(f"Error in /analytics/model-health: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/collection-stats', methods=['GET'])
@cached(ttl=60)  # Cache for 1 minute
def get_collection_stats():
    """
    Get statistics about data collection

    Query params:
        hours: Number of hours to analyze (default: 24)
    """
    try:
        hours = request.args.get('hours', 24, type=int)

        # Get historical data
        data = db.get_historical_data(hours=hours)

        if not data:
            return jsonify({
                'error': 'No data available',
                'hours': hours
            }), 404

        # Extract gas prices
        gas_prices = [d.get('gwei', 0) for d in data]
        timestamps = [d.get('timestamp', '') for d in data]

        # Calculate statistics
        stats = {
            'hours': hours,
            'total_records': len(data),
            'expected_records': hours * 60,  # 1 per minute
            'collection_rate': len(data) / (hours * 60) if hours > 0 else 0,
            'gas_price': {
                'current': gas_prices[-1] if gas_prices else None,
                'min': min(gas_prices) if gas_prices else None,
                'max': max(gas_prices) if gas_prices else None,
                'avg': np.mean(gas_prices) if gas_prices else None,
                'median': np.median(gas_prices) if gas_prices else None,
                'std': np.std(gas_prices) if gas_prices else None,
            },
            'volatility': {
                'coefficient_of_variation': (np.std(gas_prices) / np.mean(gas_prices)) if gas_prices and np.mean(gas_prices) > 0 else None,
                'price_range': (max(gas_prices) - min(gas_prices)) if gas_prices else None,
                'spikes_detected': sum(1 for p in gas_prices if p > np.mean(gas_prices) + 2 * np.std(gas_prices)) if len(gas_prices) > 10 else 0
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Collection stats: {len(data)} records over {hours}h ({stats['collection_rate']:.1%} rate)")
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error in /analytics/collection-stats: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/dashboard', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def get_analytics_dashboard():
    """
    Comprehensive analytics dashboard data

    Returns all key metrics in one endpoint for dashboard display
    """
    try:
        # Get performance metrics for all horizons (last 7 days)
        performance_metrics = {}
        for horizon in ['1h', '4h', '24h']:
            performance_metrics[horizon] = validator.calculate_metrics(horizon=horizon, days=7)

        # Get validation summary
        validation = validator.get_validation_summary()

        # Get model health
        health = validator.check_model_health()

        # Get collection stats
        collection_24h = db.get_historical_data(hours=24)
        collection_stats = {
            'records_24h': len(collection_24h),
            'expected_records': 24 * 60,  # 1 per minute
            'collection_rate': len(collection_24h) / (24 * 60) if collection_24h else 0
        }

        # Calculate data quality score (0-100)
        data_quality_score = min(100, (
            collection_stats['collection_rate'] * 40 +  # 40% weight on collection rate
            (validation['validation_rate'] * 30) +  # 30% weight on validation rate
            ((1 - len(health['alerts']) / 6) * 30)  # 30% weight on health (max 6 alerts)
        ))

        dashboard = {
            'performance': performance_metrics,
            'validation': {
                'total_predictions': validation['total_predictions'],
                'validated': validation['validated'],
                'pending': validation['pending'],
                'validation_rate': validation['validation_rate']
            },
            'health': {
                'healthy': health['healthy'],
                'alerts_count': len(health['alerts']),
                'alerts': health['alerts']
            },
            'collection': collection_stats,
            'data_quality_score': round(data_quality_score, 1),
            'summary': {
                'models_trained': len([m for m in performance_metrics.values() if m.get('sample_size', 0) > 0]),
                'best_horizon': max(performance_metrics.items(), key=lambda x: x[1].get('directional_accuracy', 0))[0] if any(m.get('sample_size', 0) > 0 for m in performance_metrics.values()) else None,
                'overall_accuracy': np.mean([m.get('directional_accuracy', 0) for m in performance_metrics.values() if m.get('sample_size', 0) > 0]) if any(m.get('sample_size', 0) > 0 for m in performance_metrics.values()) else None
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Dashboard: Quality score {data_quality_score:.1f}/100, {validation['validated']} validated predictions")
        return jsonify(dashboard)

    except Exception as e:
        logger.error(f"Error in /analytics/dashboard: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/recent-predictions', methods=['GET'])
@cached(ttl=60)  # Cache for 1 minute
def get_recent_predictions():
    """
    Get recent predictions with validation status

    Query params:
        limit: Number of predictions to return (default: 20)
        validated_only: Only return validated predictions (default: false)
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        validated_only = request.args.get('validated_only', 'false').lower() == 'true'

        session = db._get_session()
        try:
            from utils.prediction_validator import PredictionLog

            query = session.query(PredictionLog)

            if validated_only:
                query = query.filter(PredictionLog.validated == True)

            predictions = query.order_by(
                PredictionLog.prediction_time.desc()
            ).limit(limit).all()

            results = [{
                'id': p.id,
                'prediction_time': p.prediction_time.isoformat(),
                'target_time': p.target_time.isoformat(),
                'horizon': p.horizon,
                'predicted_gas': round(p.predicted_gas, 6),
                'actual_gas': round(p.actual_gas, 6) if p.actual_gas else None,
                'error': round(p.absolute_error, 6) if p.absolute_error else None,
                'error_percentage': round((p.absolute_error / p.actual_gas * 100), 2) if p.actual_gas and p.absolute_error else None,
                'direction_correct': p.direction_correct,
                'validated': p.validated,
                'model_version': p.model_version
            } for p in predictions]

            return jsonify({
                'predictions': results,
                'count': len(results),
                'validated_only': validated_only,
                'timestamp': datetime.now().isoformat()
            })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error in /analytics/recent-predictions: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
