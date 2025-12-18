"""
On-Chain Features API Routes

Endpoints for accessing blockchain-specific features that influence gas prices.
"""

from flask import Blueprint, jsonify, request
from utils.onchain_features import OnChainFeatureExtractor, feature_cache
from utils.logger import logger
from api.cache import cached
from datetime import datetime

onchain_bp = Blueprint('onchain', __name__)
extractor = OnChainFeatureExtractor()


@onchain_bp.route('/onchain/network-state', methods=['GET'])
@cached(ttl=30)  # Cache for 30 seconds
def get_network_state():
    """
    Get current network state with on-chain features

    Returns:
        Current blockchain congestion, utilization, and metrics
    """
    try:
        state = extractor.get_current_network_state()

        if not state:
            return jsonify({'error': 'Could not fetch network state'}), 500

        return jsonify({
            'network_state': state,
            'interpretation': {
                'congestion_level': _get_congestion_level(state['avg_utilization']),
                'gas_trend': 'increasing' if state['base_fee_trend'] > 0 else 'decreasing',
                'recommendation': _get_recommendation(state)
            },
            'timestamp': state['timestamp'].isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting network state: {e}")
        return jsonify({'error': str(e)}), 500


@onchain_bp.route('/onchain/block-features/<int:block_number>', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes (historical data doesn't change)
def get_block_features(block_number):
    """
    Get on-chain features for a specific block

    Args:
        block_number: Block number to analyze

    Returns:
        Detailed on-chain features for the block
    """
    try:
        features = feature_cache.get_features(block_number)

        if not features:
            return jsonify({'error': 'Could not extract features for block'}), 500

        return jsonify({
            'block_number': block_number,
            'features': features,
            'timestamp': features['timestamp'].isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting block features: {e}")
        return jsonify({'error': str(e)}), 500


@onchain_bp.route('/onchain/congestion-history', methods=['GET'])
@cached(ttl=300)
def get_congestion_history():
    """
    Get historical congestion levels

    Query params:
        - hours: Number of hours to look back (default: 24)

    Returns:
        Time series of network congestion metrics
    """
    try:
        from data.database import DatabaseManager, OnChainFeatures
        from datetime import timedelta

        hours = int(request.args.get('hours', 24))
        db = DatabaseManager()
        session = db._get_session()

        try:
            cutoff = datetime.now() - timedelta(hours=hours)

            features = session.query(OnChainFeatures).filter(
                OnChainFeatures.timestamp >= cutoff
            ).order_by(OnChainFeatures.timestamp.asc()).all()

            if not features:
                return jsonify({
                    'message': 'No historical on-chain data available',
                    'hours': hours
                }), 200

            congestion_data = [{
                'timestamp': f.timestamp.isoformat(),
                'block_number': f.block_number,
                'utilization': f.block_utilization,
                'tx_count': f.tx_count,
                'base_fee': f.base_fee,
                'congestion_level': _get_congestion_level(f.block_utilization)
            } for f in features]

            # Calculate summary stats
            avg_utilization = sum(d['utilization'] for d in congestion_data) / len(congestion_data)
            max_utilization = max(d['utilization'] for d in congestion_data)
            high_congestion_periods = sum(1 for d in congestion_data if d['utilization'] > 0.7)

            return jsonify({
                'hours': hours,
                'data': congestion_data,
                'summary': {
                    'avg_utilization': avg_utilization,
                    'max_utilization': max_utilization,
                    'high_congestion_periods': high_congestion_periods,
                    'total_periods': len(congestion_data)
                }
            }), 200

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error getting congestion history: {e}")
        return jsonify({'error': str(e)}), 500


def _get_congestion_level(utilization: float) -> str:
    """
    Convert utilization percentage to congestion level

    Args:
        utilization: Block utilization (0-1)

    Returns:
        Congestion level string
    """
    if utilization < 0.3:
        return 'low'
    elif utilization < 0.6:
        return 'moderate'
    elif utilization < 0.8:
        return 'high'
    else:
        return 'critical'


def _get_recommendation(state: dict) -> str:
    """
    Generate transaction recommendation based on network state

    Args:
        state: Network state dictionary

    Returns:
        User-friendly recommendation
    """
    if state['is_congested']:
        if state['base_fee_trend'] > 0:
            return "Network is congested and gas prices are rising. Consider waiting if not urgent."
        else:
            return "Network is congested but gas prices are falling. May improve soon."
    else:
        if state['base_fee_trend'] > 0:
            return "Network is clear but gas prices are rising. Good time to transact before prices increase."
        else:
            return "Network is clear and gas prices are falling. Excellent time to transact!"
