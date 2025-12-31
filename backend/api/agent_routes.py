"""
RL Agent API Routes

Endpoints for transaction timing recommendations using the trained RL agent.
"""

from flask import Blueprint, jsonify, request
from services.agent_service import get_agent_service
from data.collector import BaseGasCollector
from data.database import DatabaseManager
from utils.logger import logger
from api.cache import cached
from datetime import datetime
import traceback

agent_bp = Blueprint('agent', __name__)

collector = BaseGasCollector()
db = DatabaseManager()


@agent_bp.route('/agent/recommend', methods=['POST'])
def get_recommendation():
    """
    Get agent's recommendation for transaction timing

    Request body:
    {
        "urgency": 0.5,           // 0-1, how urgent is the transaction
        "gas_limit": 21000,       // Optional: gas limit for the transaction
        "value_usd": 100          // Optional: transaction value for context
    }

    Response:
    {
        "success": true,
        "recommendation": {
            "action": "WAIT",              // WAIT, SUBMIT_NOW, SUBMIT_LOW, SUBMIT_HIGH
            "confidence": 0.85,            // 0-1 confidence score
            "recommended_gas": 0.0095,     // Recommended gas price in gwei
            "expected_savings": 0.0005,    // Expected savings in gwei
            "reasoning": "...",            // Human-readable explanation
            "q_values": {...}              // Raw Q-values for all actions
        },
        "context": {
            "current_gas": 0.01,
            "predictions": {...},
            "timestamp": "..."
        }
    }
    """
    try:
        data = request.get_json() or {}
        urgency = float(data.get('urgency', 0.5))
        urgency = max(0.0, min(1.0, urgency))  # Clamp to 0-1

        # Get current gas price
        current_data = collector.get_current_gas()
        if not current_data:
            return jsonify({
                'success': False,
                'error': 'No current gas data available'
            }), 503

        current_gas = current_data.get('current_gas', 0.01)

        # Get predictions
        predictions = _get_predictions()

        # Get agent recommendation
        agent = get_agent_service()

        # Update agent's statistics with recent data
        recent_prices = db.get_historical_data(hours=24)
        if recent_prices:
            gas_prices = [r.get('gwei') or r.get('current_gas') or r.get('gas_price', 0.01) for r in recent_prices]
            agent.update_statistics(gas_prices)

        recommendation = agent.get_recommendation(
            current_gas=current_gas,
            predictions=predictions,
            urgency=urgency
        )

        return jsonify({
            'success': True,
            'recommendation': {
                'action': recommendation.action,
                'confidence': round(recommendation.confidence, 3),
                'recommended_gas': round(recommendation.recommended_gas, 8),
                'expected_savings': round(recommendation.expected_savings, 8),
                'reasoning': recommendation.reasoning,
                'q_values': {k: round(v, 4) for k, v in recommendation.q_values.items()},
                'urgency_factor': recommendation.urgency_factor
            },
            'context': {
                'current_gas': current_gas,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Agent recommendation error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@agent_bp.route('/agent/recommend', methods=['GET'])
@cached(ttl=30)
def get_recommendation_simple():
    """
    Simple GET endpoint for agent recommendation with default urgency

    Query params:
    - urgency: float (0-1), default 0.5

    Returns same format as POST endpoint
    """
    try:
        urgency = float(request.args.get('urgency', 0.5))
        urgency = max(0.0, min(1.0, urgency))

        # Get current gas price
        current_data = collector.get_current_gas()
        if not current_data:
            return jsonify({
                'success': False,
                'error': 'No current gas data available'
            }), 503

        current_gas = current_data.get('current_gas', 0.01)

        # Get predictions
        predictions = _get_predictions()

        # Get agent recommendation
        agent = get_agent_service()
        recommendation = agent.get_recommendation(
            current_gas=current_gas,
            predictions=predictions,
            urgency=urgency
        )

        return jsonify({
            'success': True,
            'recommendation': {
                'action': recommendation.action,
                'confidence': round(recommendation.confidence, 3),
                'recommended_gas': round(recommendation.recommended_gas, 8),
                'expected_savings': round(recommendation.expected_savings, 8),
                'reasoning': recommendation.reasoning,
                'urgency_factor': recommendation.urgency_factor
            },
            'context': {
                'current_gas': current_gas,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Agent recommendation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@agent_bp.route('/agent/status', methods=['GET'])
@cached(ttl=60)
def get_agent_status():
    """
    Get agent status and health information

    Response:
    {
        "success": true,
        "status": {
            "loaded": true,
            "error": null,
            "torch_available": true,
            "statistics": {...},
            "agent_metrics": {...}
        }
    }
    """
    try:
        agent = get_agent_service()
        status = agent.get_status()

        return jsonify({
            'success': True,
            'status': status
        })

    except Exception as e:
        logger.error(f"Agent status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@agent_bp.route('/agent/actions', methods=['GET'])
def get_available_actions():
    """
    Get list of available agent actions with descriptions

    Response:
    {
        "success": true,
        "actions": [
            {
                "id": "WAIT",
                "name": "Wait",
                "description": "..."
            },
            ...
        ]
    }
    """
    actions = [
        {
            'id': 'WAIT',
            'name': 'Wait',
            'description': 'Hold off on submitting. The agent expects better gas prices soon.',
            'risk': 'low',
            'speed': 'slow'
        },
        {
            'id': 'SUBMIT_NOW',
            'name': 'Submit Now',
            'description': 'Submit transaction at current gas price. Good timing detected.',
            'risk': 'low',
            'speed': 'normal'
        },
        {
            'id': 'SUBMIT_LOW',
            'name': 'Submit Low',
            'description': 'Submit at 10% below current price. Saves gas but may fail (~15% risk).',
            'risk': 'medium',
            'speed': 'slow'
        },
        {
            'id': 'SUBMIT_HIGH',
            'name': 'Submit High',
            'description': 'Submit at 10% above current price. Faster confirmation, higher cost.',
            'risk': 'very_low',
            'speed': 'fast'
        }
    ]

    return jsonify({
        'success': True,
        'actions': actions
    })


@agent_bp.route('/agent/simulate', methods=['POST'])
def simulate_episode():
    """
    Simulate an episode with the agent (for testing/visualization)

    Request body:
    {
        "steps": 20,           // Number of steps to simulate
        "urgency": 0.5         // Transaction urgency
    }

    Response:
    {
        "success": true,
        "simulation": {
            "steps": [...],
            "final_action": "...",
            "total_reward": 0.5
        }
    }
    """
    try:
        data = request.get_json() or {}
        num_steps = min(int(data.get('steps', 20)), 60)  # Cap at 60
        urgency = float(data.get('urgency', 0.5))

        agent = get_agent_service()
        if not agent.is_loaded:
            return jsonify({
                'success': False,
                'error': 'Agent not loaded'
            }), 503

        # Get historical data for simulation
        historical = db.get_historical_data(hours=2)
        if not historical or len(historical) < num_steps:
            return jsonify({
                'success': False,
                'error': 'Not enough historical data for simulation'
            }), 400

        predictions = _get_predictions()
        steps = []
        final_action = None

        for i in range(num_steps):
            gas_price = historical[i].get('gwei') or historical[i].get('current_gas') or historical[i].get('gas_price', 0.01)

            recommendation = agent.get_recommendation(
                current_gas=gas_price,
                predictions=predictions,
                urgency=urgency
            )

            step_data = {
                'step': i,
                'gas_price': gas_price,
                'action': recommendation.action,
                'confidence': round(recommendation.confidence, 3)
            }
            steps.append(step_data)

            if recommendation.action != 'WAIT':
                final_action = recommendation.action
                break

        return jsonify({
            'success': True,
            'simulation': {
                'steps': steps,
                'final_action': final_action or 'WAIT',
                'steps_taken': len(steps),
                'urgency': urgency
            }
        })

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _get_predictions() -> dict:
    """Get current predictions from the prediction models"""
    try:
        # Import here to avoid circular imports
        from api.routes import models, scalers, engineer

        current_data = collector.get_current_gas()
        if not current_data:
            return {'1h': 0.01, '4h': 0.01, '24h': 0.01}

        current_gas = current_data.get('current_gas', 0.01)

        # Try to get actual predictions
        historical = db.get_historical_data(hours=24)
        if not historical or not models:
            return {'1h': current_gas, '4h': current_gas, '24h': current_gas}

        from models.feature_pipeline import build_feature_matrix
        import numpy as np

        X, _, _ = build_feature_matrix(historical, include_external_features=True)

        predictions = {}
        for horizon in ['1h', '4h', '24h']:
            if horizon in models and models[horizon].get('model'):
                model_data = models[horizon]
                model = model_data['model']
                scaler = scalers.get(horizon)

                X_latest = X.iloc[-1:].copy()

                if scaler:
                    X_scaled = scaler.transform(X_latest)
                else:
                    X_scaled = X_latest.values

                pred = model.predict(X_scaled)[0]

                # Handle log-scale predictions
                if model_data.get('uses_log_scale'):
                    pred = np.exp(pred) - 1e-8

                predictions[horizon] = float(pred)
            else:
                predictions[horizon] = current_gas

        return predictions

    except Exception as e:
        logger.warning(f"Could not get predictions: {e}")
        current_data = collector.get_current_gas()
        current_gas = current_data.get('current_gas', 0.01) if current_data else 0.01
        return {'1h': current_gas, '4h': current_gas, '24h': current_gas}
