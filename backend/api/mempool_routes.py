"""
Mempool API Routes

Endpoints for accessing real-time mempool data as leading indicators
for gas price prediction.
"""

from flask import Blueprint, jsonify, request
from utils.logger import logger
from api.cache import cached
from datetime import datetime

mempool_bp = Blueprint('mempool', __name__)


@mempool_bp.route('/mempool/status', methods=['GET'])
@cached(ttl=10)  # Cache for 10 seconds (mempool changes rapidly)
def get_mempool_status():
    """
    Get current mempool status and metrics.

    Returns pending transaction count, average gas prices,
    congestion indicators, and momentum signals.
    """
    try:
        from data.mempool_collector import get_mempool_collector

        collector = get_mempool_collector()
        features = collector.get_current_features()
        latest_snapshot = collector.get_latest_snapshot()

        # Determine congestion level
        pending_count = features.get('mempool_pending_count', 0)
        is_congested = features.get('mempool_is_congested', 0) > 0

        if pending_count == 0:
            congestion_level = 'unknown'
        elif is_congested:
            congestion_level = 'high'
        elif pending_count > 50:
            congestion_level = 'moderate'
        else:
            congestion_level = 'low'

        response = {
            'status': 'active' if latest_snapshot else 'inactive',
            'metrics': {
                'pending_count': int(features.get('mempool_pending_count', 0)),
                'avg_gas_price': round(features.get('mempool_avg_gas_price', 0), 4),
                'median_gas_price': round(features.get('mempool_median_gas_price', 0), 4),
                'p90_gas_price': round(features.get('mempool_p90_gas_price', 0), 4),
                'gas_price_spread': round(features.get('mempool_gas_price_spread', 0), 4),
                'large_tx_ratio': round(features.get('mempool_large_tx_ratio', 0), 4),
                'arrival_rate': round(features.get('mempool_arrival_rate', 0), 4)
            },
            'signals': {
                'is_congested': is_congested,
                'congestion_level': congestion_level,
                'count_momentum': round(features.get('mempool_count_momentum', 0), 4),
                'gas_momentum': round(features.get('mempool_gas_momentum', 0), 4)
            },
            'interpretation': _get_mempool_interpretation(features),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        if latest_snapshot:
            response['snapshot_time'] = latest_snapshot.timestamp.isoformat() + 'Z'

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting mempool status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@mempool_bp.route('/mempool/history', methods=['GET'])
@cached(ttl=30)
def get_mempool_history():
    """
    Get recent mempool snapshot history.

    Query params:
        - minutes: Number of minutes to look back (default: 60, max: 240)

    Returns time series of mempool metrics.
    """
    try:
        from data.mempool_collector import get_mempool_collector
        from datetime import timedelta

        minutes = min(int(request.args.get('minutes', 60)), 240)
        collector = get_mempool_collector()

        # Get snapshots from history
        snapshots = collector.snapshot_history
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        history = []
        for snapshot in snapshots:
            if snapshot.timestamp >= cutoff:
                history.append({
                    'timestamp': snapshot.timestamp.isoformat() + 'Z',
                    'pending_count': snapshot.pending_count,
                    'avg_gas_price': round(snapshot.avg_gas_price, 4),
                    'median_gas_price': round(snapshot.median_gas_price, 4),
                    'p90_gas_price': round(snapshot.p90_gas_price, 4),
                    'large_tx_count': snapshot.large_tx_count
                })

        # Calculate summary stats
        if history:
            avg_pending = sum(h['pending_count'] for h in history) / len(history)
            max_pending = max(h['pending_count'] for h in history)
            avg_gas = sum(h['avg_gas_price'] for h in history) / len(history)
        else:
            avg_pending = max_pending = avg_gas = 0

        return jsonify({
            'minutes': minutes,
            'snapshot_count': len(history),
            'history': history,
            'summary': {
                'avg_pending_count': round(avg_pending, 1),
                'max_pending_count': max_pending,
                'avg_gas_price': round(avg_gas, 4)
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error getting mempool history: {e}")
        return jsonify({'error': str(e)}), 500


@mempool_bp.route('/mempool/features', methods=['GET'])
@cached(ttl=10)
def get_mempool_features():
    """
    Get current mempool-derived features for ML prediction.

    These are the exact features used in the prediction pipeline.
    """
    try:
        from data.mempool_collector import get_mempool_collector

        collector = get_mempool_collector()
        features = collector.get_current_features()

        return jsonify({
            'features': features,
            'feature_count': len(features),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200

    except Exception as e:
        logger.error(f"Error getting mempool features: {e}")
        return jsonify({'error': str(e)}), 500


@mempool_bp.route('/mempool/collector/start', methods=['POST'])
def start_collector():
    """Start the background mempool collection thread."""
    try:
        from data.mempool_collector import get_mempool_collector

        collector = get_mempool_collector()

        if collector.running:
            return jsonify({
                'status': 'already_running',
                'message': 'Mempool collector is already running'
            }), 200

        collector.start_background_collection()

        return jsonify({
            'status': 'started',
            'message': 'Mempool collector started',
            'interval_seconds': collector.collection_interval
        }), 200

    except Exception as e:
        logger.error(f"Error starting mempool collector: {e}")
        return jsonify({'error': str(e)}), 500


@mempool_bp.route('/mempool/collector/stop', methods=['POST'])
def stop_collector():
    """Stop the background mempool collection thread."""
    try:
        from data.mempool_collector import get_mempool_collector

        collector = get_mempool_collector()
        collector.stop_background_collection()

        return jsonify({
            'status': 'stopped',
            'message': 'Mempool collector stopped'
        }), 200

    except Exception as e:
        logger.error(f"Error stopping mempool collector: {e}")
        return jsonify({'error': str(e)}), 500


def _get_mempool_interpretation(features: dict) -> dict:
    """Generate human-readable interpretation of mempool state."""
    pending = features.get('mempool_pending_count', 0)
    is_congested = features.get('mempool_is_congested', 0) > 0
    count_momentum = features.get('mempool_count_momentum', 0)
    gas_momentum = features.get('mempool_gas_momentum', 0)

    # Determine trend
    if count_momentum > 0.1:
        trend = 'increasing'
        trend_desc = 'Pending transactions are growing'
    elif count_momentum < -0.1:
        trend = 'decreasing'
        trend_desc = 'Pending transactions are clearing'
    else:
        trend = 'stable'
        trend_desc = 'Mempool is stable'

    # Generate recommendation
    if is_congested:
        if gas_momentum > 0:
            recommendation = 'High congestion with rising gas. Consider waiting.'
        else:
            recommendation = 'High congestion but gas stabilizing. May improve soon.'
    else:
        if gas_momentum > 0.05:
            recommendation = 'Low congestion but gas rising. Good time to transact now.'
        else:
            recommendation = 'Low congestion and stable gas. Excellent time to transact.'

    return {
        'trend': trend,
        'trend_description': trend_desc,
        'recommendation': recommendation
    }
