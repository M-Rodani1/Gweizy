"""
Base app configuration endpoint
Required for Coinbase x Queen Mary hackathon
"""

from flask import Blueprint, jsonify, request

base_config_bp = Blueprint('base_config', __name__)


@base_config_bp.route('/config.json', methods=['GET'])
def get_base_config():
    """
    Base app configuration endpoint
    Returns JSON configuration for Base dev platform registration
    """
    
    config = {
        "name": "Base Gas Optimizer",
        "description": "AI-powered gas price predictions for Base network. Save 30-65% on transaction fees by timing your trades optimally.",
        "version": "1.0.0",
        "icon": "https://base-gas-optimizer.vercel.app/logo.png",
        "splash": "https://base-gas-optimizer.vercel.app/splash.png",
        "website": "https://base-gas-optimizer.vercel.app",
        "category": "defi",
        "tags": ["gas", "optimization", "ml", "prediction", "defi"],
        
        # Base-specific configuration
        "base": {
            "chain_id": 8453,  # Base mainnet
            "supported_chains": [8453, 84532],  # Base mainnet + testnet
            "rpc_url": "https://mainnet.base.org",
            "explorer": "https://basescan.org"
        },
        
        # App capabilities
        "capabilities": [
            "read_gas_prices",
            "predict_gas_trends",
            "calculate_savings",
            "send_alerts"
        ],
        
        # API endpoints
        "endpoints": {
            "predictions": "/api/predictions",
            "current_gas": "/api/current",
            "model_accuracy": "/api/accuracy",
            "explanation": "/api/explain/{horizon}"
        },
        
        # Farcaster Frame configuration
        "frame": {
            "version": "vNext",
            "image": "https://base-gas-optimizer.vercel.app/frame-image.png",
            "buttons": [
                {
                    "label": "Check Gas Price",
                    "action": "post"
                },
                {
                    "label": "Get Predictions",
                    "action": "post"
                }
            ],
            "post_url": "https://base-gas-optimizer-api.onrender.com/api/frame"
        },
        
        # Developer info
        "developer": {
            "name": "Mohamed & Team",
            "email": "contact@basegasoptimizer.com",
            "github": "https://github.com/M-Rodani1/gasFeesPrediction"
        },
        
        # Permissions requested
        "permissions": {
            "required": ["read_blockchain_data"],
            "optional": ["send_notifications", "read_wallet_transactions"]
        }
    }
    
    return jsonify(config)


@base_config_bp.route('/manifest.json', methods=['GET'])
def get_manifest():
    """
    Web app manifest for PWA support
    """
    
    manifest = {
        "name": "Base Gas Optimizer",
        "short_name": "Gas Optimizer",
        "description": "AI-powered gas predictions for Base",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1b26",
        "theme_color": "#06b6d4",
        "icons": [
            {
                "src": "/logo192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/logo512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    
    return jsonify(manifest)

