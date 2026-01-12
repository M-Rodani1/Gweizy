"""
WebSocket Event Emitters for Real-Time Updates

Provides functions to emit real-time updates to connected clients:
- Gas price updates
- Prediction updates
- Mempool status
- Accuracy metrics
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global reference to socketio instance
_socketio = None


def init_socketio(socketio_instance):
    """Initialize the global socketio reference."""
    global _socketio
    _socketio = socketio_instance
    logger.info("WebSocket events initialized")


def get_socketio():
    """Get the socketio instance."""
    return _socketio


def emit_gas_update(gas_data: Dict[str, Any]):
    """
    Emit a gas price update to all connected clients.

    Args:
        gas_data: Dict with current_gas, base_fee, priority_fee, timestamp
    """
    if not _socketio:
        return

    try:
        _socketio.emit('gas_update', {
            'type': 'gas_price',
            'data': {
                'current_gas': gas_data.get('current_gas'),
                'base_fee': gas_data.get('base_fee'),
                'priority_fee': gas_data.get('priority_fee'),
                'block_number': gas_data.get('block_number'),
                'timestamp': gas_data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
            }
        })
        logger.debug("Emitted gas_update event")
    except Exception as e:
        logger.warning(f"Failed to emit gas_update: {e}")


def emit_prediction_update(predictions: Dict[str, Any], current_price: float):
    """
    Emit prediction updates to all connected clients.

    Args:
        predictions: Dict with horizon predictions (1h, 4h, 24h)
        current_price: Current gas price for reference
    """
    if not _socketio:
        return

    try:
        # Format predictions for frontend
        formatted = {
            'current_price': current_price,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'horizons': {}
        }

        for horizon, pred_data in predictions.items():
            if isinstance(pred_data, dict):
                formatted['horizons'][horizon] = {
                    'prediction': pred_data.get('ensemble_prediction') or pred_data.get('prediction', {}).get('price'),
                    'confidence': pred_data.get('confidence'),
                    'confidence_interval': pred_data.get('confidence_interval'),
                    'direction': 'up' if (pred_data.get('ensemble_prediction', 0) or 0) > current_price else 'down'
                }

        _socketio.emit('prediction_update', {
            'type': 'predictions',
            'data': formatted
        })
        logger.debug("Emitted prediction_update event")
    except Exception as e:
        logger.warning(f"Failed to emit prediction_update: {e}")


def emit_mempool_update(mempool_status: Dict[str, Any]):
    """
    Emit mempool status update to all connected clients.

    Args:
        mempool_status: Dict with pending_count, is_congested, gas_momentum, etc.
    """
    if not _socketio:
        return

    try:
        _socketio.emit('mempool_update', {
            'type': 'mempool',
            'data': {
                'pending_count': mempool_status.get('pending_count', 0),
                'avg_gas_price': mempool_status.get('avg_gas_price', 0),
                'is_congested': mempool_status.get('is_congested', False),
                'gas_momentum': mempool_status.get('gas_momentum', 0),
                'count_momentum': mempool_status.get('count_momentum', 0),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        })
        logger.debug("Emitted mempool_update event")
    except Exception as e:
        logger.warning(f"Failed to emit mempool_update: {e}")


def emit_accuracy_update(metrics: Dict[str, Any]):
    """
    Emit accuracy metrics update to all connected clients.

    Args:
        metrics: Dict with MAE, RMSE, R2, directional accuracy per horizon
    """
    if not _socketio:
        return

    try:
        _socketio.emit('accuracy_update', {
            'type': 'accuracy',
            'data': {
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        })
        logger.debug("Emitted accuracy_update event")
    except Exception as e:
        logger.warning(f"Failed to emit accuracy_update: {e}")


def emit_alert(alert_type: str, message: str, data: Optional[Dict] = None):
    """
    Emit an alert/notification to all connected clients.

    Args:
        alert_type: Type of alert (spike, drop, congestion, etc.)
        message: Human-readable message
        data: Optional additional data
    """
    if not _socketio:
        return

    try:
        _socketio.emit('alert', {
            'type': alert_type,
            'message': message,
            'data': data or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        logger.debug(f"Emitted alert event: {alert_type}")
    except Exception as e:
        logger.warning(f"Failed to emit alert: {e}")


def emit_combined_update(
    gas_data: Optional[Dict] = None,
    predictions: Optional[Dict] = None,
    mempool_status: Optional[Dict] = None
):
    """
    Emit a combined update with all available data.

    This is efficient for clients that want all data at once.
    """
    if not _socketio:
        return

    try:
        update = {
            'type': 'combined',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        if gas_data:
            update['gas'] = {
                'current_gas': gas_data.get('current_gas'),
                'base_fee': gas_data.get('base_fee'),
                'priority_fee': gas_data.get('priority_fee')
            }

        if predictions:
            update['predictions'] = {}
            current = gas_data.get('current_gas', 0) if gas_data else 0
            for horizon, pred in predictions.items():
                if isinstance(pred, dict):
                    update['predictions'][horizon] = {
                        'price': pred.get('ensemble_prediction') or pred.get('prediction', {}).get('price'),
                        'confidence': pred.get('confidence'),
                        'interval': pred.get('confidence_interval')
                    }

        if mempool_status:
            update['mempool'] = {
                'pending_count': mempool_status.get('pending_count', 0),
                'is_congested': mempool_status.get('is_congested', False),
                'gas_momentum': mempool_status.get('gas_momentum', 0)
            }

        _socketio.emit('update', update)
        logger.debug("Emitted combined update event")
    except Exception as e:
        logger.warning(f"Failed to emit combined update: {e}")
