"""
Farcaster Frame API Routes

Endpoints for Farcaster Frame integration:
- Frame metadata generation
- Frame interaction handling
- Share cards with gas predictions
"""

from flask import Blueprint, jsonify, request, Response
from data.collector import BaseGasCollector
from utils.logger import logger
from datetime import datetime
import json

farcaster_bp = Blueprint('farcaster', __name__)
collector = BaseGasCollector()


@farcaster_bp.route('/frame', methods=['GET', 'POST'])
def farcaster_frame():
    """
    Farcaster Frame endpoint for gas predictions

    Supports both GET (initial load) and POST (button interactions)
    """
    try:
        # Get current gas data
        current_gas = collector.get_current_gas()

        if not current_gas:
            return jsonify({'error': 'Could not fetch gas data'}), 500

        gas_price = current_gas.get('current_gas', 0)

        # Determine gas level and color
        if gas_price < 0.002:
            level = "LOW"
            color = "#10b981"  # Green
            emoji = "🟢"
        elif gas_price < 0.003:
            level = "MODERATE"
            color = "#f59e0b"  # Yellow
            emoji = "🟡"
        else:
            level = "HIGH"
            color = "#ef4444"  # Red
            emoji = "🔴"

        # Handle POST (button interaction)
        if request.method == 'POST':
            data = request.get_json() or {}
            button_index = data.get('untrustedData', {}).get('buttonIndex', 1)

            if button_index == 2:
                # "Get Predictions" button clicked
                return generate_predictions_frame()

        # Generate Frame HTML
        frame_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta property="fc:frame" content="vNext" />
    <meta property="fc:frame:image" content="https://basegasfeesml.onrender.com/api/frame/image" />
    <meta property="fc:frame:button:1" content="Check Gas Price" />
    <meta property="fc:frame:button:2" content="Get Predictions" />
    <meta property="fc:frame:button:1:action" content="post" />
    <meta property="fc:frame:button:2:action" content="post" />
    <meta property="fc:frame:post_url" content="https://basegasfeesml.onrender.com/api/frame" />
    <meta property="og:title" content="Base Gas Optimizer" />
    <meta property="og:description" content="Current gas: {gas_price:.6f} Gwei {emoji} {level}" />
    <meta property="og:image" content="https://basegasfeesml.onrender.com/api/frame/image" />
    <title>Base Gas Optimizer</title>
</head>
<body>
    <h1>Base Gas Optimizer</h1>
    <p>Current Gas: {gas_price:.6f} Gwei {emoji}</p>
    <p>Level: {level}</p>
    <p><a href="https://basegasfeesml.netlify.app">Open Full App</a></p>
</body>
</html>
"""

        return Response(frame_html, mimetype='text/html')

    except Exception as e:
        logger.error(f"Error in Farcaster frame: {e}")
        return jsonify({'error': str(e)}), 500


def generate_predictions_frame():
    """Generate frame showing predictions"""
    try:
        from data.database import DatabaseManager
        from models.ensemble_predictor import ensemble_predictor

        db = DatabaseManager()

        # Get historical data for predictions
        recent_data = db.get_historical_data(hours=48)

        if len(recent_data) < 100:
            return jsonify({'error': 'Not enough data for predictions'}), 500

        # Get predictions (simplified)
        # In production, this would use the full prediction pipeline
        predictions_text = "1h: ↓ Decreasing\n4h: → Stable\n24h: ↑ Increasing"

        frame_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta property="fc:frame" content="vNext" />
    <meta property="fc:frame:image" content="https://basegasfeesml.onrender.com/api/frame/predictions-image" />
    <meta property="fc:frame:button:1" content="View Details" />
    <meta property="fc:frame:button:1:action" content="link" />
    <meta property="fc:frame:button:1:target" content="https://basegasfeesml.netlify.app/app" />
    <meta property="og:title" content="Gas Price Predictions" />
    <meta property="og:description" content="{predictions_text}" />
    <title>Gas Predictions</title>
</head>
<body>
    <h1>Gas Price Predictions</h1>
    <pre>{predictions_text}</pre>
</body>
</html>
"""

        return Response(frame_html, mimetype='text/html')

    except Exception as e:
        logger.error(f"Error generating predictions frame: {e}")
        return jsonify({'error': str(e)}), 500


@farcaster_bp.route('/frame/image', methods=['GET'])
def frame_image():
    """
    Generate dynamic Frame image showing current gas price

    In production, this would generate an actual image using PIL or similar
    For now, returns SVG
    """
    try:
        current_gas = collector.get_current_gas()
        gas_price = current_gas.get('current_gas', 0) if current_gas else 0

        # Determine gas level
        if gas_price < 0.002:
            level = "LOW"
            color = "#10b981"
            emoji = "🟢"
        elif gas_price < 0.003:
            level = "MODERATE"
            color = "#f59e0b"
            emoji = "🟡"
        else:
            level = "HIGH"
            color = "#ef4444"
            emoji = "🔴"

        # Generate SVG image
        svg = f"""
<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1b26;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2d3748;stop-opacity:1" />
    </linearGradient>
  </defs>

  <rect width="1200" height="630" fill="url(#grad)"/>

  <text x="600" y="150" text-anchor="middle" fill="white" font-size="48" font-weight="bold" font-family="Arial, sans-serif">
    Base Gas Optimizer
  </text>

  <text x="600" y="250" text-anchor="middle" fill="{color}" font-size="96" font-weight="bold" font-family="Arial, sans-serif">
    {gas_price:.6f}
  </text>

  <text x="600" y="320" text-anchor="middle" fill="white" font-size="36" font-family="Arial, sans-serif">
    Gwei
  </text>

  <circle cx="600" cy="420" r="50" fill="{color}" opacity="0.3"/>
  <text x="600" y="440" text-anchor="middle" fill="{color}" font-size="48" font-weight="bold" font-family="Arial, sans-serif">
    {level}
  </text>

  <text x="600" y="550" text-anchor="middle" fill="#9ca3af" font-size="24" font-family="Arial, sans-serif">
    AI-Powered Gas Price Predictions
  </text>
</svg>
"""

        return Response(svg, mimetype='image/svg+xml')

    except Exception as e:
        logger.error(f"Error generating frame image: {e}")
        return Response("Error generating image", status=500)


@farcaster_bp.route('/frame/predictions-image', methods=['GET'])
def frame_predictions_image():
    """Generate image showing predictions"""

    svg = """
<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1b26;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2d3748;stop-opacity:1" />
    </linearGradient>
  </defs>

  <rect width="1200" height="630" fill="url(#grad)"/>

  <text x="600" y="100" text-anchor="middle" fill="white" font-size="48" font-weight="bold" font-family="Arial, sans-serif">
    Gas Price Predictions
  </text>

  <text x="200" y="250" fill="#10b981" font-size="32" font-weight="bold" font-family="monospace">1h:</text>
  <text x="300" y="250" fill="#10b981" font-size="32" font-family="monospace">↓ Decreasing</text>

  <text x="200" y="350" fill="#f59e0b" font-size="32" font-weight="bold" font-family="monospace">4h:</text>
  <text x="300" y="350" fill="#f59e0b" font-size="32" font-family="monospace">→ Stable</text>

  <text x="200" y="450" fill="#ef4444" font-size="32" font-weight="bold" font-family="monospace">24h:</text>
  <text x="300" y="450" fill="#ef4444" font-size="32" font-family="monospace">↑ Increasing</text>

  <text x="600" y="570" text-anchor="middle" fill="#9ca3af" font-size="20" font-family="Arial, sans-serif">
    Click "View Details" for full predictions
  </text>
</svg>
"""

    return Response(svg, mimetype='image/svg+xml')


@farcaster_bp.route('/frame/share', methods=['POST'])
def share_frame():
    """
    Handle sharing to Farcaster

    Body:
        {
            "user_fid": 12345,
            "prediction": "1h",
            "value": 0.002345
        }
    """
    try:
        data = request.get_json()

        user_fid = data.get('user_fid')
        prediction_horizon = data.get('prediction', '1h')
        predicted_value = data.get('value', 0)

        # Log the share event (could save to database for analytics)
        logger.info(f"User {user_fid} shared {prediction_horizon} prediction: {predicted_value}")

        # Generate share URL
        share_url = f"https://basegasfeesml.netlify.app/app?shared=true&horizon={prediction_horizon}"

        return jsonify({
            'success': True,
            'share_url': share_url,
            'message': 'Shared successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error sharing frame: {e}")
        return jsonify({'error': str(e)}), 500
