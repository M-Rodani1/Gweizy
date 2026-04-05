"""
Data Routes
Endpoints for data export, database info, and database download.
"""

from flask import Blueprint, jsonify, request, send_file
from data.database import DatabaseManager
from utils.logger import logger
from config import Config
from datetime import datetime
import traceback
import os


data_bp = Blueprint('data', __name__)

# Shared instances
db = DatabaseManager()


@data_bp.route('/export', methods=['GET'])
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
        rows = []
        for record in data:
            rows.append({
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
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            response = data_bp.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=gas_data_{hours}h.csv'}
            )
            return response

        # Default: JSON
        return jsonify({
            'success': True,
            'hours': hours,
            'count': len(rows),
            'data': rows
        })

    except Exception as e:
        logger.error(f"Error in /export: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@data_bp.route('/database/info', methods=['GET'])
def database_info():
    """
    Get information about the database contents.

    Returns:
        JSON with database statistics including record counts, date ranges, etc.
    """
    try:
        from sqlalchemy import func
        from data.database import GasPrice
        from datetime import timedelta

        session = db._get_session()

        # Get total record count
        total_records = session.query(func.count(GasPrice.id)).scalar() or 0

        # Get date range
        min_date = session.query(func.min(GasPrice.timestamp)).scalar()
        max_date = session.query(func.max(GasPrice.timestamp)).scalar()

        # Get recent records (last 24 hours)
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


@data_bp.route('/database/download', methods=['GET'])
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
