"""
RL Agent API routes for transaction timing recommendations.

Supports both single DQN agent and production ensemble for robust predictions.
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
import numpy as np
import os
import pickle

from utils.logger import logger
from api.cache import cached
from data.database import DatabaseManager
from data.multichain_collector import MultiChainGasCollector

agent_bp = Blueprint('agent', __name__)

# Import ensemble service
try:
    from services.ensemble_service import get_ensemble_service, ProductionEnsembleService
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logger.warning("Ensemble service not available")

# Shared helpers for request handlers
db = DatabaseManager()
multichain_collector = MultiChainGasCollector()

# Cache for chain-specific agents
_chain_agents = {}
_chain_agents_loaded = {}


def _is_torch_checkpoint(path: str) -> bool:
    """Detect Torch zip checkpoint by magic header."""
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'
    except Exception:
        return False


def _load_torch_agent(model_path: str, state_dim: int, action_dim: int):
    """Load a PyTorch DQN agent from a checkpoint with hidden_dims inference."""
    try:
        from rl.agents.dqn_torch import DQNAgent as TorchDQNAgent, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            return None
        import torch  # type: ignore
    except Exception:
        return None

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    hidden_dims = state_dict.get('hidden_dims', [128, 64])
    # Heuristic: default dueling unless keys indicate otherwise
    q_keys = state_dict.get('q_network_state_dict', {}).keys()
    use_dueling = any('advantage' in k or 'value' in k for k in q_keys)

    agent = TorchDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_dueling=use_dueling
    )
    agent.load_state_dict(state_dict)
    return agent


def get_dqn_agent(chain_id: int = 8453):
    """Lazy load the DQN agent for a specific chain with improved path resolution."""
    global _chain_agents, _chain_agents_loaded
    
    # Check if already loaded for this chain
    if chain_id in _chain_agents_loaded and _chain_agents_loaded[chain_id]:
        return _chain_agents.get(chain_id)
    
    try:
        # Import numpy agent (always available)
        from rl.agents.dqn import DQNAgent as NumpyDQNAgent
        # Torch availability
        try:
            from rl.agents.dqn_torch import TORCH_AVAILABLE
        except Exception:
            TORCH_AVAILABLE = False

        from rl.state import StateBuilder

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try persistent storage first, then fallback paths
        # Priority: /data/models (Railway persistent) > local paths
        possible_paths = []
        
        # Persistent storage paths (Railway)
        if os.path.exists('/data'):
            possible_paths.extend([
                os.path.join('/data', 'models', 'rl_agents', f'chain_{chain_id}', 'dqn_final.pkl'),
                os.path.join('/data', 'models', 'rl_agents', f'chain_{chain_id}', 'dqn_best.pkl'),
                os.path.join('/data', 'models', 'rl_agents', 'chain_8453', 'dqn_final.pkl'),
                os.path.join('/data', 'models', 'rl_agents', 'chain_8453', 'dqn_best.pkl'),
            ])
        
        # Local paths (fallback)
        possible_paths.extend([
            # Chain-specific paths
            os.path.join(base_dir, 'models', 'rl_agents', f'chain_{chain_id}', 'dqn_final.pkl'),
            os.path.join(base_dir, 'models', 'rl_agents', f'chain_{chain_id}', 'dqn_best.pkl'),
            # Fallback to Base chain (8453) if chain-specific not found
            os.path.join(base_dir, 'models', 'rl_agents', 'chain_8453', 'dqn_final.pkl'),
            os.path.join(base_dir, 'models', 'rl_agents', 'chain_8453', 'dqn_best.pkl'),
            # Legacy paths (for backward compatibility)
            os.path.join(base_dir, 'models', 'rl_agents', 'dqn_final.pkl'),
            os.path.join(base_dir, 'models', 'rl_agents', 'dqn_best.pkl'),
            os.path.join(base_dir, 'models', 'saved_models', 'dqn_agent.pkl'),
            os.path.join(base_dir, 'models', 'dqn_agent.pkl'),
        ])
        
        existing_paths = [path for path in possible_paths if os.path.exists(path)]

        if existing_paths:
            state_builder = StateBuilder(history_length=40)
            state_dim = state_builder.get_state_dim()
            action_dim = 2

            for model_path in existing_paths:
                is_torch = _is_torch_checkpoint(model_path)
                if is_torch and not TORCH_AVAILABLE:
                    logger.warning(
                        f"Found PyTorch checkpoint at {model_path} but torch is not available. Skipping."
                    )
                    continue

                try:
                    if is_torch and TORCH_AVAILABLE:
                        agent = _load_torch_agent(model_path, state_dim=state_dim, action_dim=action_dim)
                        if agent is None:
                            logger.warning(f"Failed to initialize PyTorch agent for {model_path}.")
                            continue
                        logger.debug("Using PyTorch DQN agent")
                    else:
                        agent = NumpyDQNAgent(
                            state_dim=state_dim,
                            action_dim=action_dim,
                            hidden_dims=[64, 64]  # Match training configuration
                        )
                        try:
                            agent.load(model_path)
                        except pickle.UnpicklingError as e:
                            # Likely a torch legacy checkpoint; try torch loader if available.
                            if TORCH_AVAILABLE:
                                torch_agent = _load_torch_agent(
                                    model_path, state_dim=state_dim, action_dim=action_dim
                                )
                                if torch_agent is not None:
                                    agent = torch_agent
                                    logger.debug("Recovered using PyTorch DQN agent after pickle error")
                                else:
                                    raise e
                            else:
                                raise e
                        logger.debug("Using numpy DQN agent")

                    # Cache the agent for this chain
                    _chain_agents[chain_id] = agent
                    _chain_agents_loaded[chain_id] = True

                    from data.multichain_collector import CHAINS
                    chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')
                    logger.info(f"✓ DQN agent loaded from {model_path} for {chain_name} (Chain ID: {chain_id})")
                    if hasattr(agent, 'training_steps'):
                        logger.info(f"  Training steps: {agent.training_steps}, Epsilon: {agent.epsilon:.4f}")
                    return agent
                except pickle.UnpicklingError as e:
                    logger.warning(f"Pickle load failed for {model_path}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load DQN agent from {model_path}: {e}")
                    continue

            logger.warning(f"DQN model(s) found for chain {chain_id} but none could be loaded - using heuristic fallback")
            _chain_agents[chain_id] = None
            _chain_agents_loaded[chain_id] = True
            return None
        else:
            # Only log detailed paths once, then use brief message
            if not hasattr(get_dqn_agent, '_warned_chains'):
                get_dqn_agent._warned_chains = set()

            if chain_id not in get_dqn_agent._warned_chains:
                logger.info(f"DQN model not found for chain {chain_id} - using heuristic fallback")
                logger.debug(f"Searched paths: {possible_paths[:3]}...")
                get_dqn_agent._warned_chains.add(chain_id)

            _chain_agents[chain_id] = None
            _chain_agents_loaded[chain_id] = True
            return None
    except Exception as e:
        logger.error(f"Failed to load DQN agent for chain {chain_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _chain_agents[chain_id] = None
        _chain_agents_loaded[chain_id] = True
        return None


@agent_bp.route('/recommend', methods=['GET', 'POST'])
@cached(ttl=30, key_prefix='agent_recommend')
def get_recommendation():
    """
    Get AI recommendation for transaction timing.

    Cached for 30 seconds to improve response times.
    Cache key includes request parameters (urgency, chain_id, tx_type).

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
        
        # Get chain_id from request (default to Base)
        chain_id = data.get('chain_id', request.args.get('chain_id', 8453, type=int))
        
        # Get current gas price for the specified chain
        current_gas = data.get('current_gas')
        if current_gas is None:
            gas_data = multichain_collector.get_current_gas(chain_id)
            current_gas = gas_data.get('current_gas', 0.001) if gas_data else 0.001
        
        # Get price history for the specified chain
        price_history = data.get('price_history', [])
        if not price_history:
            historical = db.get_historical_data(hours=24, chain_id=chain_id, limit=24, order='desc')
            price_history = [h.get('gwei', current_gas) for h in reversed(historical)]
        
        # Try chain-specific DQN agent first
        agent = get_dqn_agent(chain_id=chain_id)
        
        if agent is not None:
            # Build state for DQN
            from rl.state import StateBuilder, GasState

            state_builder = StateBuilder(history_length=40)
            
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
                'success': True,
                'recommendation': {
                    'action': result['action'].upper(),
                    'confidence': round(result['confidence'], 3),
                    'reason': reason,
                    'predicted_savings': round(predicted_savings, 3),
                    'optimal_time': optimal_time,
                    'q_values': {
                        'wait': round(result['q_values']['wait'], 4),
                        'execute': round(result['q_values']['execute'], 4)
                    }
                },
                'model': 'dqn',
                'chain_id': chain_id,
                'current_gas': current_gas,
                'urgency': urgency,
                'generated_at': datetime.utcnow().isoformat() + 'Z'
            })
        
        else:
            # Fallback to heuristic recommendation
            return _heuristic_recommendation(current_gas, price_history, urgency, gas_amount)
    
    except Exception as e:
        logger.error(f"Error in /agent/recommend: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'recommendation': {
                'action': 'EXECUTE',
                'confidence': 0.5,
                'reason': 'Error occurred, defaulting to execute',
                'predicted_savings': 0,
                'optimal_time': 'now',
                'q_values': {
                    'wait': 0.3,
                    'execute': 0.5
                }
            }
        }), 200


def _heuristic_recommendation(current_gas, price_history, urgency, gas_amount):
    """Fallback heuristic when DQN is not available."""
    
    if not price_history:
        return jsonify({
            'success': True,
            'recommendation': {
                'action': 'EXECUTE',
                'confidence': 0.5,
                'reason': 'Insufficient data for prediction',
                'predicted_savings': 0,
                'optimal_time': 'now',
                'q_values': {
                    'wait': 0.3,
                    'execute': 0.5
                }
            },
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
        'success': True,
        'recommendation': {
            'action': recommendation.upper(),
            'confidence': round(confidence, 3),
            'reason': reason,
            'predicted_savings': round(predicted_savings, 3),
            'optimal_time': optimal_time,
            'q_values': {
                'wait': 0.3 if recommendation == 'wait' else 0.5,
                'execute': 0.5 if recommendation == 'execute' else 0.3
            }
        },
        'model': 'heuristic',
        'current_gas': current_gas,
        'urgency': urgency,
        'generated_at': datetime.utcnow().isoformat() + 'Z'
    })


@agent_bp.route('/ensemble/recommend', methods=['GET', 'POST'])
@cached(ttl=15, key_prefix='ensemble_recommend')
def get_ensemble_recommendation():
    """
    Get AI recommendation using production ensemble (recommended).

    The ensemble provides more robust predictions with:
    - 75% positive rate
    - 29% median savings
    - 95% confidence from multi-agent agreement

    GET params or POST body:
        {
            "urgency": 0.5,           # 0-1 urgency level
            "chain_id": 8453,         # Chain ID (default: Base)
            "current_gas": 0.001,     # Current gas price (optional)
            "time_waiting": 0         # Steps already waited (optional)
        }

    Returns:
        {
            "recommendation": {
                "action": "WAIT" | "EXECUTE",
                "confidence": 0.95,
                "reason": "...",
                "predicted_savings": 0.15,
                "optimal_time": "1-2 hours"
            },
            "model": "ensemble",
            "metrics": {...}
        }
    """
    if not ENSEMBLE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Ensemble service not available',
            'fallback': 'Use /agent/recommend for single-agent recommendations'
        }), 503

    try:
        # Parse request
        if request.method == 'GET':
            data = {
                'urgency': request.args.get('urgency', 0.5, type=float),
                'chain_id': request.args.get('chain_id', 8453, type=int),
                'current_gas': request.args.get('current_gas', type=float),
                'time_waiting': request.args.get('time_waiting', 0, type=int)
            }
        else:
            data = request.get_json() or {}

        urgency = min(1.0, max(0.0, float(data.get('urgency', 0.5))))
        chain_id = data.get('chain_id', 8453)
        time_waiting = data.get('time_waiting', 0)

        # Get current gas price
        current_gas = data.get('current_gas')
        if current_gas is None:
            gas_data = multichain_collector.get_current_gas(chain_id)
            current_gas = gas_data.get('current_gas', 0.001) if gas_data else 0.001

        # Get predictions (optional)
        predictions = data.get('predictions', {})

        # Get ensemble service
        service = get_ensemble_service(chain_id)

        # Get recommendation
        rec = service.get_recommendation(
            current_price=current_gas,
            predictions=predictions,
            urgency=urgency,
            time_waiting=time_waiting
        )

        return jsonify({
            'success': True,
            'recommendation': {
                'action': rec.action,
                'confidence': round(rec.confidence, 3),
                'reason': rec.reasoning,
                'predicted_savings': round(rec.expected_savings, 3),
                'optimal_time': rec.wait_time_estimate,
                'q_values': {k: round(v, 4) for k, v in rec.q_values.items()}
            },
            'model': 'ensemble',
            'chain_id': chain_id,
            'current_gas': current_gas,
            'urgency': urgency,
            'metrics': rec.metrics,
            'agent_agreement': rec.agent_agreement,
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        })

    except Exception as e:
        logger.error(f"Error in /agent/ensemble/recommend: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'recommendation': {
                'action': 'EXECUTE',
                'confidence': 0.5,
                'reason': 'Error occurred, defaulting to execute',
                'predicted_savings': 0,
                'optimal_time': 'now'
            }
        }), 200


@agent_bp.route('/ensemble/status', methods=['GET'])
@cached(ttl=30, key_prefix='ensemble_status')
def get_ensemble_status():
    """Get ensemble service status and metrics."""
    if not ENSEMBLE_AVAILABLE:
        return jsonify({
            'available': False,
            'error': 'Ensemble service not available'
        })

    try:
        chain_id = request.args.get('chain_id', 8453, type=int)
        service = get_ensemble_service(chain_id)
        status = service.get_status()

        return jsonify({
            'available': True,
            **status
        })

    except Exception as e:
        logger.error(f"Error getting ensemble status: {e}")
        return jsonify({
            'available': False,
            'error': str(e)
        })


@agent_bp.route('/status', methods=['GET'])
@cached(ttl=60, key_prefix='agent_status')
def agent_status():
    """Get DQN agent status with detailed information. Cached for 60 seconds."""
    agent = get_dqn_agent()
    
    # Convert dimensions to native Python int
    state_dim = agent.state_dim if agent else 34
    action_dim = agent.action_dim if agent else 2
    
    if hasattr(state_dim, 'item'):
        state_dim = int(state_dim.item())
    elif not isinstance(state_dim, (int, float)):
        state_dim = int(state_dim)
    
    if hasattr(action_dim, 'item'):
        action_dim = int(action_dim.item())
    elif not isinstance(action_dim, (int, float)):
        action_dim = int(action_dim)
    
    status = {
        'dqn_loaded': agent is not None,
        'model_type': 'DQN' if agent else 'Heuristic',
        'state_dim': int(state_dim),
        'action_dim': int(action_dim)
    }
    
    if agent:
        buffer_capacity = 0
        if hasattr(agent.replay_buffer, 'buffer') and hasattr(agent.replay_buffer.buffer, 'maxlen'):
            buffer_capacity = agent.replay_buffer.buffer.maxlen or 0
        
        # Convert all values to native Python types to avoid JSON serialization issues
        training_steps = agent.training_steps
        if hasattr(training_steps, 'item'):
            training_steps = int(training_steps.item())
        elif not isinstance(training_steps, (int, float)):
            training_steps = int(training_steps)
        
        replay_buffer_size = len(agent.replay_buffer)
        if hasattr(replay_buffer_size, 'item'):
            replay_buffer_size = int(replay_buffer_size.item())
        elif not isinstance(replay_buffer_size, (int, float)):
            replay_buffer_size = int(replay_buffer_size)
        
        episode_rewards_count = len(agent.episode_rewards) if hasattr(agent, 'episode_rewards') else 0
        if hasattr(episode_rewards_count, 'item'):
            episode_rewards_count = int(episode_rewards_count.item())
        elif not isinstance(episode_rewards_count, (int, float)):
            episode_rewards_count = int(episode_rewards_count)
        
        status.update({
            'training_steps': int(training_steps),
            'epsilon': float(round(agent.epsilon, 4)),
            'replay_buffer_size': int(replay_buffer_size),
            'replay_buffer_capacity': int(buffer_capacity),
            'episode_rewards_count': int(episode_rewards_count)
        })
        
        # Add average reward if available
        if hasattr(agent, 'episode_rewards') and len(agent.episode_rewards) > 0:
            import numpy as np
            recent_rewards = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
            avg_reward = np.mean(recent_rewards)
            # Convert numpy scalar to native Python float
            if hasattr(avg_reward, 'item'):
                avg_reward = float(avg_reward.item())
            else:
                avg_reward = float(avg_reward)
            status['avg_reward_last_100'] = round(avg_reward, 3)
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
        
        # Convert all values to native Python types
        training_steps = agent.training_steps
        if hasattr(training_steps, 'item'):
            training_steps = int(training_steps.item())
        elif not isinstance(training_steps, (int, float)):
            training_steps = int(training_steps)
        
        return jsonify({
            'success': True,
            'episodes_trained': int(episodes),
            'final_epsilon': float(round(agent.epsilon, 4)),
            'training_steps': int(training_steps),
            'model_path': os.path.join(model_dir, 'dqn_final.pkl')
        })
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
