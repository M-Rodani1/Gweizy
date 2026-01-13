"""
Analytics API Routes

Provides real-time analytics, model performance tracking, and prediction accuracy metrics.
Includes rate limiting for compute-intensive endpoints.
"""

from flask import Blueprint, jsonify, request
from utils.prediction_validator import PredictionValidator
from data.database import DatabaseManager
from utils.logger import logger, capture_exception
from api.cache import cached
from api.middleware import limiter, get_rate_limit
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
        capture_exception(e, {'endpoint': '/analytics/performance'})
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
        capture_exception(e, {'endpoint': '/analytics/trends'})
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
        capture_exception(e, {'endpoint': '/analytics/validation-summary'})
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
        capture_exception(e, {'endpoint': '/analytics/model-health'})
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
        capture_exception(e, {'endpoint': '/analytics/collection-stats'})
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
        capture_exception(e, {'endpoint': '/analytics/dashboard'})
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
        capture_exception(e, {'endpoint': '/analytics/recent-predictions'})
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Advanced Analytics Endpoints
# ============================================================================

@analytics_bp.route('/volatility', methods=['GET'])
@limiter.limit(get_rate_limit('analytics_volatility'))
@cached(ttl=60)
def get_volatility_index():
    """
    Calculate Gas Volatility Index (GVI) - similar to VIX for gas prices.

    Returns a 0-100 score where:
    - 0-20: Very Low volatility (stable)
    - 20-40: Low volatility
    - 40-60: Moderate volatility
    - 60-80: High volatility
    - 80-100: Extreme volatility

    Query params:
        hours: Lookback period (default: 24)
    """
    try:
        hours = min(int(request.args.get('hours', 24)), 168)
        data = db.get_historical_data(hours=hours)

        if not data or len(data) < 10:
            return jsonify({
                'available': False,
                'reason': 'Insufficient data for volatility calculation'
            }), 200

        prices = [d['gas_price'] for d in data if d.get('gas_price')]

        if len(prices) < 10:
            return jsonify({
                'available': False,
                'reason': 'Insufficient price data'
            }), 200

        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]

        std_returns = np.std(returns) * 100
        recent_returns = returns[-12:] if len(returns) >= 12 else returns
        recent_vol = np.std(recent_returns) * 100
        cv = (np.std(prices) / np.mean(prices)) * 100
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices) * 100

        raw_index = (std_returns * 0.4 + recent_vol * 0.3 + cv * 0.2 + price_range * 0.1)
        volatility_index = min(100, raw_index * 5)

        if volatility_index < 20:
            level, description, color = 'very_low', 'Gas prices are very stable', 'green'
        elif volatility_index < 40:
            level, description, color = 'low', 'Gas prices are relatively stable', 'green'
        elif volatility_index < 60:
            level, description, color = 'moderate', 'Normal gas price fluctuations', 'yellow'
        elif volatility_index < 80:
            level, description, color = 'high', 'Gas prices are volatile - consider timing carefully', 'orange'
        else:
            level, description, color = 'extreme', 'Extreme volatility - high uncertainty', 'red'

        trend = 'stable'
        if len(returns) >= 6:
            recent_trend = np.mean(returns[-6:])
            trend = 'increasing' if recent_trend > 0.02 else 'decreasing' if recent_trend < -0.02 else 'stable'

        return jsonify({
            'available': True,
            'volatility_index': round(volatility_index, 1),
            'level': level,
            'description': description,
            'color': color,
            'trend': trend,
            'metrics': {
                'std_returns': round(std_returns, 4),
                'recent_volatility': round(recent_vol, 4),
                'coefficient_of_variation': round(cv, 2),
                'price_range_pct': round(price_range, 2),
                'current_price': round(float(prices[-1]), 6),
                'avg_price': round(float(np.mean(prices)), 6)
            },
            'data_points': len(prices),
            'period_hours': hours,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error calculating volatility index: {e}")
        capture_exception(e, {'endpoint': '/analytics/volatility'})
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/whales', methods=['GET'])
@limiter.limit(get_rate_limit('analytics_whales'))
@cached(ttl=45)  # 45 seconds - balance between freshness and compute cost
def get_whale_activity():
    """
    Monitor large transaction (whale) activity.

    Whales are transactions with gas > 500k, indicating:
    - Contract deployments
    - Large DeFi operations
    - NFT batch mints
    """
    try:
        from data.mempool_collector import get_mempool_collector, is_collector_ready

        whale_threshold = 500_000
        collector = get_mempool_collector(timeout=2.0)

        current_whale_count = 0
        total_whale_txs = 0
        avg_whale_txs = 0

        if collector and is_collector_ready():
            snapshots = collector.snapshot_history[-20:]
            large_tx_counts = [s.large_tx_count for s in snapshots if hasattr(s, 'large_tx_count')]
            total_whale_txs = sum(large_tx_counts)
            avg_whale_txs = np.mean(large_tx_counts) if large_tx_counts else 0
            latest = collector.get_latest_snapshot()
            current_whale_count = latest.large_tx_count if latest else 0

        if current_whale_count == 0:
            activity_level, description, impact = 'none', 'No whale activity detected', 'neutral'
        elif current_whale_count <= 2:
            activity_level, description, impact = 'low', 'Light whale activity', 'minimal'
        elif current_whale_count <= 5:
            activity_level, description, impact = 'moderate', 'Moderate whale activity', 'moderate'
        else:
            activity_level, description, impact = 'high', 'Heavy whale activity', 'significant'

        estimated_impact_pct = min(current_whale_count * 2, 20) if current_whale_count > 0 else 0

        return jsonify({
            'available': True,
            'current': {
                'whale_count': current_whale_count,
                'activity_level': activity_level,
                'description': description,
                'estimated_price_impact_pct': estimated_impact_pct,
                'impact': impact
            },
            'recent': {
                'total_whale_txs': total_whale_txs,
                'avg_whale_txs_per_snapshot': round(avg_whale_txs, 1)
            },
            'threshold': {'gas_limit': whale_threshold},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error getting whale activity: {e}")
        capture_exception(e, {'endpoint': '/analytics/whales'})
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/anomalies', methods=['GET'])
@limiter.limit(get_rate_limit('analytics_anomalies'))
@cached(ttl=45)  # 45 seconds - balance between freshness and compute cost
def get_anomaly_detection():
    """
    Detect anomalies in gas price patterns using z-score analysis.

    Query params:
        hours: Lookback period (default: 24)
        sensitivity: 1-3 (default: 2)
    """
    try:
        hours = min(int(request.args.get('hours', 24)), 168)
        sensitivity = min(max(int(request.args.get('sensitivity', 2)), 1), 3)
        z_threshold = {1: 3.0, 2: 2.5, 3: 2.0}[sensitivity]

        data = db.get_historical_data(hours=hours)

        if not data or len(data) < 20:
            return jsonify({'available': False, 'reason': 'Insufficient data'}), 200

        prices = np.array([d['gas_price'] for d in data if d.get('gas_price')])
        if len(prices) < 20:
            return jsonify({'available': False, 'reason': 'Insufficient price data'}), 200

        mean_price = np.mean(prices)
        std_price = np.std(prices)
        current_price = prices[-1]
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

        anomalies = []

        if abs(z_score) > z_threshold:
            anomaly_type = 'spike' if z_score > 0 else 'drop'
            anomalies.append({
                'type': anomaly_type,
                'severity': 'high' if abs(z_score) > z_threshold + 1 else 'medium',
                'z_score': round(z_score, 2),
                'deviation_pct': round((current_price - mean_price) / mean_price * 100, 1),
                'message': f"Gas price {anomaly_type} detected"
            })

        if len(prices) >= 5:
            recent_change = (prices[-1] - prices[-5]) / prices[-5] * 100
            if abs(recent_change) > 30:
                anomalies.append({
                    'type': 'rapid_change',
                    'severity': 'high' if abs(recent_change) > 50 else 'medium',
                    'change_pct': round(recent_change, 1),
                    'message': f"Rapid {'increase' if recent_change > 0 else 'decrease'} of {abs(round(recent_change, 1))}%"
                })

        if len(prices) >= 10:
            recent_std = np.std(prices[-10:])
            vol_ratio = recent_std / std_price if std_price > 0 else 1
            if vol_ratio > 2:
                anomalies.append({
                    'type': 'volatility_spike',
                    'severity': 'high' if vol_ratio > 3 else 'medium',
                    'volatility_ratio': round(vol_ratio, 2),
                    'message': f"Volatility is {round(vol_ratio, 1)}x higher than normal"
                })

        if not anomalies:
            status, status_color = 'normal', 'green'
        elif any(a['severity'] == 'high' for a in anomalies):
            status, status_color = 'alert', 'red'
        else:
            status, status_color = 'warning', 'yellow'

        return jsonify({
            'available': True,
            'status': status,
            'status_color': status_color,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'current_analysis': {
                'price': round(float(current_price), 6),
                'z_score': round(z_score, 2),
                'vs_average_pct': round((current_price - mean_price) / mean_price * 100, 1)
            },
            'baseline': {
                'mean': round(float(mean_price), 6),
                'std': round(float(std_price), 6),
                'data_points': len(prices)
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        capture_exception(e, {'endpoint': '/analytics/anomalies'})
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/ensemble', methods=['GET'])
@limiter.limit(get_rate_limit('analytics_ensemble'))
@cached(ttl=60)
def get_ensemble_weights():
    """Get model ensemble weights and contributions."""
    try:
        from models.ensemble_predictor import get_ensemble_predictor
        from models.hybrid_predictor import hybrid_predictor

        ensemble = get_ensemble_predictor()
        hybrid = hybrid_predictor

        models = []

        if hasattr(ensemble, 'models') and ensemble.models:
            for name, model_info in ensemble.models.items():
                models.append({
                    'name': name,
                    'type': 'ensemble_member',
                    'loaded': True,
                    'weight': model_info.get('weight', 1.0) if isinstance(model_info, dict) else 1.0
                })

        hybrid_loaded = hybrid.loaded if hybrid else False
        spike_count = len(hybrid.spike_detectors) if hybrid and hasattr(hybrid, 'spike_detectors') else 0

        models.extend([
            {'name': 'hybrid_predictor', 'type': 'primary', 'loaded': hybrid_loaded, 'description': 'Main ML model'},
            {'name': 'spike_detectors', 'type': 'classifier', 'loaded': spike_count > 0, 'count': spike_count},
            {'name': 'pattern_matcher', 'type': 'statistical', 'loaded': True},
            {'name': 'fallback_predictor', 'type': 'heuristic', 'loaded': True}
        ])

        loaded_count = sum(1 for m in models if m.get('loaded'))
        health_pct = (loaded_count / len(models)) * 100

        if health_pct >= 80:
            health_status, health_color = 'healthy', 'green'
        elif health_pct >= 50:
            health_status, health_color = 'degraded', 'yellow'
        else:
            health_status, health_color = 'limited', 'red'

        primary_model = 'hybrid_predictor' if hybrid_loaded else 'fallback_predictor'
        prediction_mode = 'ML-based predictions' if hybrid_loaded else 'Heuristic predictions'

        return jsonify({
            'available': True,
            'health': {
                'status': health_status,
                'color': health_color,
                'loaded_models': loaded_count,
                'total_models': len(models),
                'health_pct': round(health_pct, 1)
            },
            'primary_model': primary_model,
            'prediction_mode': prediction_mode,
            'models': models,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error getting ensemble weights: {e}")
        capture_exception(e, {'endpoint': '/analytics/ensemble'})
        return jsonify({'error': str(e)}), 500
