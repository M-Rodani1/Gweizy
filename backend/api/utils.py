"""
API Utility Functions
Helper functions for API endpoints
"""

from datetime import datetime
from flask import jsonify


def success_response(data, message="Success"):
    """Create a successful JSON response"""
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })


def error_response(message, status_code=400):
    """Create an error JSON response"""
    return jsonify({
        'status': 'error',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }), status_code


def validate_horizon(horizon):
    """Validate prediction horizon"""
    valid_horizons = ['1h', '4h', '24h']
    return horizon in valid_horizons


def format_gas_data(gas_price_obj):
    """Format GasPrice object to dictionary"""
    return {
        'id': gas_price_obj.id,
        'timestamp': gas_price_obj.timestamp.isoformat(),
        'current_gas': gas_price_obj.current_gas,
        'base_fee': gas_price_obj.base_fee,
        'priority_fee': gas_price_obj.priority_fee,
        'block_number': gas_price_obj.block_number
    }


def format_prediction_data(prediction_obj):
    """Format Prediction object to dictionary"""
    return {
        'id': prediction_obj.id,
        'timestamp': prediction_obj.timestamp.isoformat(),
        'horizon': prediction_obj.horizon,
        'predicted_gas': prediction_obj.predicted_gas,
        'actual_gas': prediction_obj.actual_gas,
        'model_version': prediction_obj.model_version
    }

