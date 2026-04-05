"""
API Routes
Core API endpoints for the gas price prediction system.

Gas data routes (current, eth-price, historical, gas-stats, gas/patterns) -> gas_routes.py
Prediction routes (predictions, explain) -> prediction_routes.py
Data routes (export, database/info, database/download) -> data_routes.py
Model state (models, scalers, load_models, reload_models) -> model_state.py
"""

from flask import Blueprint, jsonify, request
from data.collector import BaseGasCollector
from data.database import DatabaseManager
from models.accuracy_tracker import get_tracker
from utils.base_scanner import BaseScanner
from utils.logger import logger, log_error_with_context
from api.cache import cached, clear_cache
from api.middleware import require_admin_auth
from datetime import datetime, timedelta
import traceback

# Re-export model state for backward compatibility
# (e.g. `from api.routes import load_models, reload_models` still works)
from api.model_state import (  # noqa: F401
    models,
    scalers,
    feature_names,
    load_models,
    reload_models,
)


api_bp = Blueprint('api', __name__)


collector = BaseGasCollector()
db = DatabaseManager()
scanner = BaseScanner()
accuracy_tracker = get_tracker()


@api_bp.route('/models/reload', methods=['POST'])
@require_admin_auth
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
            db_check = DatabaseManager()
            # Just check if we can get a session, don't run a query
            session = db_check._get_session()
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


@api_bp.route('/cache/clear', methods=['POST'])
@require_admin_auth
def clear_all_cache():
    """Clear API cache (admin endpoint)"""
    clear_cache()
    logger.info("Cache cleared by request")
    return jsonify({'message': 'Cache cleared'})


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
            logger.warning("Accuracy tracker not available")
            return jsonify({
                'mae': None,
                'rmse': None,
                'r2': None,
                'directional_accuracy': None,
                'recent_predictions': [],
                'last_updated': datetime.now().isoformat(),
                'available': False,
                'warning': 'Accuracy tracker unavailable'
            }), 503

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
            'r2': float(r2) if r2 is not None else 0,
            'directional_accuracy': float(directional_accuracy) if directional_accuracy is not None else 0,
            'recent_predictions': recent_predictions[:24],
            'last_updated': datetime.now().isoformat(),
            'n_predictions': n_predictions,
            'source': 'real-time'
        }

        logger.info(f"Real-time accuracy metrics - R2: {r2:.4f} ({r2_percent:.1f}%), Directional: {directional_accuracy:.4f} ({directional_accuracy_percent:.1f}%), N={n_predictions}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /accuracy: {traceback.format_exc()}")
        from datetime import datetime
        return jsonify({
            'mae': None,
            'rmse': None,
            'r2': None,
            'directional_accuracy': None,
            'recent_predictions': [],
            'last_updated': datetime.now().isoformat(),
            'available': False,
            'error': str(e),
        }), 503


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
