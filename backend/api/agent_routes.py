"""
RL Agent API routes for transaction timing recommendations.
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
import numpy as np
import os

from utils.logger import logger

agent_bp = Blueprint('agent', __name__)

# Global DQN agent instance
_dqn_agent = None
_agent_loaded = False


def get_dqn_agent():
    """Lazy load the DQN agent with improved path resolution."""
    global _dqn_agent, _agent_loaded
    
    if _agent_loaded:
        return _dqn_agent
    
    try:
        from rl.agents.dqn import DQNAgent
        from rl.state import StateBuilder
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try multiple possible model paths (in order of preference)
        possible_paths = [
            os.path.join(base_dir, 'models', 'rl_agents', 'dqn_final.pkl'),  # New training pipeline
            os.path.join(base_dir, 'models', 'rl_agents', 'dqn_best.pkl'),   # Best checkpoint
            os.path.join(base_dir, 'models', 'saved_models', 'dqn_agent.pkl'),  # Old path
            os.path.join(base_dir, 'models', 'dqn_agent.pkl'),  # Alternate old path
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            state_builder = StateBuilder(history_length=24)
            _dqn_agent = DQNAgent(
                state_dim=state_builder.get_state_dim(),
                action_dim=2
            )
            _dqn_agent.load(model_path)
            logger.info(f"✓ DQN agent loaded from {model_path}")
            logger.info(f"  Training steps: {_dqn_agent.training_steps}, Epsilon: {_dqn_agent.epsilon:.4f}")
        else:
            logger.warning(f"DQN model not found. Tried paths:")
            for path in possible_paths:
                logger.warning(f"  - {path}")
            logger.warning("Using heuristic fallback. Run training pipeline to train a model.")
            _dqn_agent = None
    except Exception as e:
        logger.error(f"Failed to load DQN agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _dqn_agent = None
    
    _agent_loaded = True
    return _dqn_agent


@agent_bp.route('/recommend', methods=['GET', 'POST'])
def get_recommendation():
    """
    Get AI recommendation for transaction timing.

    GET params or POST body:
        {
            "tx_type": "swap",  # Transaction type
            "gas_amount": 150000,  # Gas units
            "urgency": 0.5,  # 0-1 urgency level
            "current_gas": 0.001,  # Current gas price (optional)
            "price_history": [...]  # Recent prices (optional)
        }

    Returns:
        {
            "recommendation": "wait" | "execute",
            "confidence": 0.85,
            "reason": "Gas prices expected to drop 15% in next 2 hours",
            "predicted_savings": 0.15,
            "optimal_time": "2 hours",
            "q_values": {"wait": 0.5, "execute": 0.3}
        }
    """
    try:
        # Support both GET params and POST body
        if request.method == 'GET':
            data = {
                'tx_type': request.args.get('tx_type', 'swap'),
                'gas_amount': request.args.get('gas_amount', 150000, type=int),
                'urgency': request.args.get('urgency', 0.5, type=float),
                'current_gas': request.args.get('current_gas', type=float)
            }
        else:
            data = request.get_json() or {}
        
        tx_type = data.get('tx_type', 'swap')
        gas_amount = data.get('gas_amount', 150000)
        urgency = min(1.0, max(0.0, float(data.get('urgency', 0.5))))
        
        # Get current gas price
        current_gas = data.get('current_gas')
        if current_gas is None:
            from data.collector import BaseGasCollector
            collector = BaseGasCollector()
            gas_data = collector.get_current_gas()
            current_gas = gas_data.get('current_gas', 0.001) if gas_data else 0.001
        
        # Get price history
        price_history = data.get('price_history', [])
        if not price_history:
            from data.database import DatabaseManager
            db = DatabaseManager()
            historical = db.get_historical_data(hours=24)
            price_history = [h.get('gwei', current_gas) for h in historical[-24:]]
        
        # Try DQN agent first
        agent = get_dqn_agent()
        
        if agent is not None:
            # Build state for DQN
            from rl.state import StateBuilder, GasState
            
            state_builder = StateBuilder(history_length=24)
            
            # Calculate volatility and momentum
            if len(price_history) >= 2:
                volatility = np.std(price_history) / (np.mean(price_history) + 1e-8)
                momentum = (price_history[-1] - price_history[0]) / (price_history[0] + 1e-8)
            else:
                volatility = 0.0
                momentum = 0.0
            
            gas_state = GasState(
                current_price=current_gas,
                price_history=price_history,
                hour=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                volatility=volatility,
                momentum=momentum,
                urgency=urgency,
                time_waiting=0
            )
            
            price_stats = {
                'mean': np.mean(price_history) if price_history else current_gas,
                'std': np.std(price_history) if len(price_history) > 1 else current_gas * 0.1,
                'min': min(price_history) if price_history else current_gas * 0.8,
                'max': max(price_history) if price_history else current_gas * 1.2
            }
            
            state = state_builder.build_state(gas_state, price_stats)
            
            # Get DQN recommendation
            result = agent.get_recommendation(state, threshold=0.6)
            
            # Calculate potential savings
            if result['action'] == 'wait':
                # Estimate savings based on Q-value difference
                q_diff = result['q_values']['wait'] - result['q_values']['execute']
                predicted_savings = min(0.3, max(0, q_diff * 0.1))
                optimal_time = "1-2 hours" if predicted_savings > 0.1 else "30 minutes"
            else:
                predicted_savings = 0
                optimal_time = "now"
            
            # Generate reason
            if result['action'] == 'wait':
                if urgency < 0.3:
                    reason = f"Low urgency detected. Gas prices may drop {predicted_savings*100:.0f}% in the next {optimal_time}."
                else:
                    reason = f"AI predicts {predicted_savings*100:.0f}% potential savings if you wait {optimal_time}."
            else:
                if urgency > 0.7:
                    reason = "High urgency - executing now is recommended."
                elif result['confidence'] > 0.8:
                    reason = "Current gas price is near predicted minimum. Good time to execute."
                else:
                    reason = "No significant savings expected from waiting."
            
            return jsonify({
                'recommendation': result['action'],
                'confidence': round(result['confidence'], 3),
                'reason': reason,
                'predicted_savings': round(predicted_savings, 3),
                'optimal_time': optimal_time,
                'q_values': {
                    'wait': round(result['q_values']['wait'], 4),
                    'execute': round(result['q_values']['execute'], 4)
                },
                'model': 'dqn',
                'current_gas': current_gas,
                'urgency': urgency
            })
        
        else:
            # Fallback to heuristic recommendation
            return _heuristic_recommendation(current_gas, price_history, urgency, gas_amount)
    
    except Exception as e:
        logger.error(f"Error in /agent/recommend: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'recommendation': 'execute',
            'confidence': 0.5,
            'reason': 'Error occurred, defaulting to execute'
        }), 200


def _heuristic_recommendation(current_gas, price_history, urgency, gas_amount):
    """Fallback heuristic when DQN is not available."""
    
    if not price_history:
        return jsonify({
            'recommendation': 'execute',
            'confidence': 0.5,
            'reason': 'Insufficient data for prediction',
            'predicted_savings': 0,
            'optimal_time': 'now',
            'model': 'heuristic'
        })
    
    avg_price = np.mean(price_history)
    min_price = min(price_history)
    
    # Simple heuristics
    price_vs_avg = (current_gas - avg_price) / avg_price if avg_price > 0 else 0
    price_vs_min = (current_gas - min_price) / min_price if min_price > 0 else 0
    
    # Time-based heuristic (gas typically lower at night)
    hour = datetime.now().hour
    is_low_gas_time = 2 <= hour <= 7
    
    # Decision logic
    if urgency > 0.8:
        recommendation = 'execute'
        confidence = 0.9
        reason = 'High urgency - execute now'
        predicted_savings = 0
        optimal_time = 'now'
    elif price_vs_avg < -0.1:  # 10% below average
        recommendation = 'execute'
        confidence = 0.8
        reason = f'Current price is {abs(price_vs_avg)*100:.0f}% below average - good time to execute'
        predicted_savings = 0
        optimal_time = 'now'
    elif price_vs_avg > 0.15 and not is_low_gas_time:  # 15% above average
        recommendation = 'wait'
        confidence = 0.7
        reason = f'Current price is {price_vs_avg*100:.0f}% above average - consider waiting'
        predicted_savings = min(0.15, price_vs_avg)
        optimal_time = '2-4 hours'
    elif is_low_gas_time:
        recommendation = 'execute'
        confidence = 0.75
        reason = 'Currently in low-gas time window (2-7 AM UTC)'
        predicted_savings = 0
        optimal_time = 'now'
    else:
        recommendation = 'execute'
        confidence = 0.6
        reason = 'No significant savings expected'
        predicted_savings = 0
        optimal_time = 'now'
    
    return jsonify({
        'recommendation': recommendation,
        'confidence': round(confidence, 3),
        'reason': reason,
        'predicted_savings': round(predicted_savings, 3),
        'optimal_time': optimal_time,
        'model': 'heuristic',
        'current_gas': current_gas,
        'urgency': urgency
    })


@agent_bp.route('/status', methods=['GET'])
def agent_status():
    """Get DQN agent status with detailed information."""
    agent = get_dqn_agent()
    
    status = {
        'dqn_loaded': agent is not None,
        'model_type': 'DQN' if agent else 'Heuristic',
        'state_dim': agent.state_dim if agent else 34,
        'action_dim': agent.action_dim if agent else 2
    }
    
    if agent:
        buffer_capacity = 0
        if hasattr(agent.replay_buffer, 'buffer') and hasattr(agent.replay_buffer.buffer, 'maxlen'):
            buffer_capacity = agent.replay_buffer.buffer.maxlen or 0
        
        status.update({
            'training_steps': agent.training_steps,
            'epsilon': round(agent.epsilon, 4),
            'replay_buffer_size': len(agent.replay_buffer),
            'replay_buffer_capacity': buffer_capacity,
            'episode_rewards_count': len(agent.episode_rewards) if hasattr(agent, 'episode_rewards') else 0
        })
        
        # Add average reward if available
        if hasattr(agent, 'episode_rewards') and len(agent.episode_rewards) > 0:
            import numpy as np
            recent_rewards = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
            status['avg_reward_last_100'] = round(float(np.mean(recent_rewards)), 3)
    else:
        status.update({
            'training_steps': 0,
            'epsilon': 1.0,
            'replay_buffer_size': 0,
            'replay_buffer_capacity': 0,
            'episode_rewards_count': 0
        })
    
    return jsonify(status)


@agent_bp.route('/train', methods=['POST'])
def trigger_training():
    """Trigger DQN training (admin endpoint)."""
    try:
        data = request.get_json() or {}
        episodes = data.get('episodes', 500)
        use_diverse = data.get('use_diverse_episodes', True)
        
        # Import and run training
        from rl.train import train_dqn
        
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'rl_agents'
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Run training (this may take a while)
        agent = train_dqn(
            num_episodes=episodes,
            save_path=os.path.join(model_dir, 'dqn_final.pkl'),
            checkpoint_dir=model_dir,
            checkpoint_freq=100,
            verbose=True,
            use_diverse_episodes=use_diverse
        )
        
        # Reload the agent
        global _dqn_agent, _agent_loaded
        _agent_loaded = False
        get_dqn_agent()
        
        return jsonify({
            'success': True,
            'episodes_trained': episodes,
            'final_epsilon': round(agent.epsilon, 4),
            'training_steps': agent.training_steps,
            'model_path': os.path.join(model_dir, 'dqn_final.pkl')
        })
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
