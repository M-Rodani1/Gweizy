"""
Gas Data Routes
Endpoints for current gas prices, ETH price, historical data, gas stats, and patterns.
"""

from flask import Blueprint, jsonify, request
from data.collector import BaseGasCollector
from data.database import DatabaseManager
from utils.logger import logger
from api.cache import cached
from datetime import datetime
import traceback


gas_bp = Blueprint('gas', __name__)

# Shared instances
collector = BaseGasCollector()
db = DatabaseManager()


@gas_bp.route('/current', methods=['GET'])
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


@gas_bp.route('/eth-price', methods=['GET'])
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


@gas_bp.route('/historical', methods=['GET'])
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


@gas_bp.route('/gas/patterns', methods=['GET'])
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
                    avg_gwei = overall_avg
                    min_gwei = overall_avg
                    max_gwei = overall_avg
            else:
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


@gas_bp.route('/gas-stats', methods=['GET'])
@cached(ttl=300)  # Cache for 5 minutes
def stats():
    """Get historical gas price statistics (min/max/avg). For global platform stats see /stats."""
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
