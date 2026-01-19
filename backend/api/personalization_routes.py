"""
Personalized recommendations API routes.
Provides personalized gas optimization suggestions based on user transaction history.
"""
from flask import Blueprint, request, jsonify
from services.personalization_service import get_personalization_service
from utils.logger import logger
from data.database import DatabaseManager

personalization_bp = Blueprint('personalization', __name__)
db = DatabaseManager()


@personalization_bp.route('/personalization/recommendations/<user_address>', methods=['GET'])
def get_personalized_recommendations(user_address: str):
    """
    Get personalized recommendations for a user.
    
    Args:
        user_address: Wallet address
    
    Query params:
        chain_id: Chain ID (default: 8453)
    
    Returns:
        Personalized recommendations including best time to transact
    """
    try:
        chain_id = request.args.get('chain_id', 8453, type=int)
        
        service = get_personalization_service()
        recommendations = service.get_personalized_recommendation(user_address, chain_id=chain_id)
        
        return jsonify({
            'success': True,
            'user_address': user_address,
            'chain_id': chain_id,
            **recommendations
        })
    
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@personalization_bp.route('/personalization/patterns/<user_address>', methods=['GET'])
def get_user_patterns(user_address: str):
    """
    Get user transaction patterns analysis.
    
    Args:
        user_address: Wallet address
    
    Query params:
        chain_id: Chain ID (default: 8453)
    
    Returns:
        Transaction patterns and statistics
    """
    try:
        chain_id = request.args.get('chain_id', 8453, type=int)
        
        service = get_personalization_service()
        patterns = service.analyze_user_patterns(user_address, chain_id=chain_id)
        
        return jsonify({
            'success': True,
            'user_address': user_address,
            'chain_id': chain_id,
            **patterns
        })
    
    except Exception as e:
        logger.error(f"Error analyzing user patterns: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@personalization_bp.route('/personalization/savings/<user_address>', methods=['GET'])
def get_user_savings(user_address: str):
    """
    Get user savings statistics.
    
    Args:
        user_address: Wallet address
    
    Query params:
        chain_id: Chain ID (default: 8453)
    
    Returns:
        Savings statistics and potential savings
    """
    try:
        chain_id = request.args.get('chain_id', 8453, type=int)
        stats = db.get_user_savings_stats(user_address, chain_id=chain_id)
        
        return jsonify({
            'success': True,
            'user_address': user_address,
            'chain_id': chain_id,
            **stats
        })
    
    except Exception as e:
        logger.error(f"Error getting user savings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@personalization_bp.route('/personalization/track-transaction', methods=['POST'])
def track_transaction():
    """
    Track a user transaction and calculate potential savings.
    
    POST body:
        {
            "user_address": "0x...",
            "tx_hash": "0x...",
            "gas_price_gwei": 0.001,
            "gas_used": 150000,
            "chain_id": 8453
        }
    
    Returns:
        Savings analysis
    """
    try:
        data = request.json
        
        required_fields = ['user_address', 'tx_hash', 'gas_price_gwei', 'gas_used']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        service = get_personalization_service()
        savings = service.track_transaction_savings(
            user_address=data['user_address'],
            tx_hash=data['tx_hash'],
            gas_price_gwei=float(data['gas_price_gwei']),
            gas_used=int(data['gas_used']),
            chain_id=data.get('chain_id', 8453)
        )
        
        if savings:
            return jsonify({
                'success': True,
                **savings
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Could not calculate savings (insufficient historical data)'
            }), 400
    
    except Exception as e:
        logger.error(f"Error tracking transaction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
