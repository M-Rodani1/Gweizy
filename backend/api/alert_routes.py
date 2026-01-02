"""
Alert API Routes
Endpoints for managing gas price alerts
"""

from flask import Blueprint, request, jsonify
from services.alert_service import AlertService
from utils.logger import logger

alert_bp = Blueprint('alerts', __name__)
alert_service = AlertService()


@alert_bp.route('/alerts', methods=['POST'])
def create_alert():
    """Create a new gas price alert"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['user_id', 'alert_type', 'threshold_gwei']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        alert = alert_service.create_alert(
            user_id=data['user_id'],
            alert_type=data['alert_type'],
            threshold_gwei=float(data['threshold_gwei']),
            notification_method=data.get('notification_method', 'browser'),
            notification_target=data.get('notification_target', ''),
            cooldown_minutes=data.get('cooldown_minutes', 30)
        )

        return jsonify({
            'success': True,
            'alert': alert
        }), 201

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        return jsonify({'error': 'Failed to create alert'}), 500


@alert_bp.route('/alerts/<user_id>', methods=['GET'])
def get_user_alerts(user_id):
    """Get all alerts for a user"""
    try:
        alerts = alert_service.get_user_alerts(user_id)

        return jsonify({
            'success': True,
            'alerts': alerts,
            'count': len(alerts)
        })

    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        return jsonify({'error': 'Failed to fetch alerts'}), 500


@alert_bp.route('/alerts/<int:alert_id>', methods=['PATCH'])
def update_alert(alert_id):
    """Update an alert (activate/deactivate)"""
    try:
        data = request.json

        alert = alert_service.update_alert(
            alert_id=alert_id,
            is_active=data.get('is_active')
        )

        return jsonify({
            'success': True,
            'alert': alert
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error updating alert: {e}")
        return jsonify({'error': 'Failed to update alert'}), 500


@alert_bp.route('/alerts/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Delete an alert"""
    try:
        # Get user_id from query params or body
        user_id = request.args.get('user_id') or request.json.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        success = alert_service.delete_alert(alert_id, user_id)

        if not success:
            return jsonify({'error': 'Alert not found'}), 404

        return jsonify({
            'success': True,
            'message': 'Alert deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        return jsonify({'error': 'Failed to delete alert'}), 500


@alert_bp.route('/alerts/check', methods=['POST'])
def check_alerts():
    """Check if any alerts should be triggered (called by data collector)"""
    try:
        data = request.json
        current_gas_gwei = float(data.get('current_gas_gwei', 0))

        if current_gas_gwei <= 0:
            return jsonify({'error': 'Invalid current_gas_gwei'}), 400

        chain_id = data.get('chain_id', 8453)
        triggered = alert_service.check_alerts(current_gas_gwei, chain_id=chain_id)

        return jsonify({
            'success': True,
            'triggered_count': len(triggered),
            'triggered_alerts': triggered
        })

    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        return jsonify({'error': 'Failed to check alerts'}), 500
