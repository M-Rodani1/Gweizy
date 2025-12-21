"""
Cron Job Routes
Endpoints triggered by Cloudflare Worker scheduled events for automated maintenance
"""

from flask import Blueprint, jsonify, request
from utils.logger import logger
from utils.prediction_validator import PredictionValidator
from data.database import DatabaseManager
import subprocess
import os
from datetime import datetime
import traceback

cron_bp = Blueprint('cron', __name__)

db = DatabaseManager()
validator = PredictionValidator()


@cron_bp.route('/cron/retrain', methods=['POST'])
def cron_retrain_models():
    """
    Triggered by Cloudflare Worker cron weekly (Sunday 2 AM)
    Retrains models with latest data
    """
    try:
        logger.info("=" * 60)
        logger.info("[CRON RETRAIN] Weekly retraining triggered")
        logger.info("=" * 60)

        # Get trigger info
        data = request.get_json() or {}
        trigger_source = data.get('trigger', 'unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())

        logger.info(f"Trigger source: {trigger_source}")
        logger.info(f"Timestamp: {timestamp}")

        # Check if we have enough data
        from sqlalchemy import func
        session = db._get_session()

        try:
            gas_count = session.query(func.count()).select_from(db.GasPrice).scalar()
            onchain_count = session.query(func.count()).select_from(db.OnChainFeatures).scalar()

            logger.info(f"Data available: {gas_count} gas prices, {onchain_count} onchain features")

            # Need minimum data for retraining
            if gas_count < 1000:
                logger.warning(f"Insufficient gas price data: {gas_count} < 1000")
                return jsonify({
                    "success": False,
                    "message": "Insufficient data for retraining",
                    "gas_prices": gas_count,
                    "required": 1000
                }), 400

            # Run training script
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'scripts',
                'train_with_current_data.py'
            )

            logger.info(f"Running training script: {script_path}")

            result = subprocess.run(
                ['python3', script_path],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Model retraining completed successfully")
                logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars

                return jsonify({
                    "success": True,
                    "message": "Models retrained successfully",
                    "timestamp": datetime.now().isoformat(),
                    "data_used": {
                        "gas_prices": gas_count,
                        "onchain_features": onchain_count
                    }
                })
            else:
                logger.error("❌ Model retraining failed")
                logger.error(f"Error: {result.stderr}")

                return jsonify({
                    "success": False,
                    "message": "Retraining failed",
                    "error": result.stderr[-500:]  # Last 500 chars
                }), 500

        finally:
            session.close()

    except subprocess.TimeoutExpired:
        logger.error("❌ Retraining timed out (30 minutes)")
        return jsonify({
            "success": False,
            "message": "Retraining timed out"
        }), 500

    except Exception as e:
        logger.error(f"❌ Error during retraining: {e}")
        logger.error(traceback.format_exc())

        return jsonify({
            "success": False,
            "message": "Internal error during retraining",
            "error": str(e)
        }), 500


@cron_bp.route('/cron/health-check', methods=['POST'])
def cron_health_check():
    """
    Triggered by Cloudflare Worker cron every 6 hours
    Checks model performance and alerts if degraded
    """
    try:
        logger.info("[CRON HEALTH] Running model health check")

        # Use the existing prediction validator
        health = validator.check_model_health(threshold_mae=0.01)

        if health['healthy']:
            logger.info("✅ Model health: GOOD")
        else:
            logger.warning("⚠️ Model health: DEGRADED")
            logger.warning(f"Alerts: {health['alerts']}")

            # TODO: Send notification (email/Discord/Slack)
            # For now, just log

        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            **health
        })

    except Exception as e:
        logger.error(f"❌ Error during health check: {e}")
        logger.error(traceback.format_exc())

        return jsonify({
            "success": False,
            "healthy": False,
            "error": str(e),
            "alerts": [{
                "severity": "critical",
                "message": f"Health check failed: {str(e)}"
            }]
        }), 500


@cron_bp.route('/model-stats', methods=['GET'])
def model_stats():
    """
    Get current model performance metrics for dashboard
    Returns R², MAE, directional accuracy, data collection status, and last trained time
    """
    try:
        session = db._get_session()

        try:
            # Get latest performance metrics from database
            from sqlalchemy import desc, func
            latest_metrics = session.query(db.ModelPerformanceMetrics)\
                .order_by(desc(db.ModelPerformanceMetrics.timestamp))\
                .first()

            # Get data collection stats
            gas_count = session.query(func.count()).select_from(db.GasPrice).scalar()
            onchain_count = session.query(func.count()).select_from(db.OnChainFeatures).scalar()

            # Get last training time from model files
            import glob
            model_files = glob.glob('models/saved_models/model_*.pkl')
            if not model_files:
                model_files = glob.glob('backend/models/saved_models/model_*.pkl')

            last_trained = None
            if model_files:
                last_trained = max(os.path.getmtime(f) for f in model_files)
                last_trained = datetime.fromtimestamp(last_trained)

            # Build response
            performance = {}
            for horizon in ['1h', '4h', '24h']:
                if latest_metrics:
                    r2 = getattr(latest_metrics, f'r2_{horizon}', 0)
                    mae = getattr(latest_metrics, f'mae_{horizon}', 0)
                    dir_acc = getattr(latest_metrics, f'directional_accuracy_{horizon}', 0)
                else:
                    r2 = mae = dir_acc = 0

                performance[horizon] = {
                    "r2": float(r2) if r2 else 0,
                    "mae": float(mae) if mae else 0,
                    "directional_accuracy": float(dir_acc) if dir_acc else 0
                }

            return jsonify({
                "performance": performance,
                "data_collection": {
                    "gas_prices": gas_count,
                    "onchain_features": onchain_count
                },
                "last_trained": last_trained.isoformat() if last_trained else None,
                "metrics_timestamp": latest_metrics.timestamp.isoformat() if latest_metrics else None
            })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "performance": {
                "1h": {"r2": 0, "mae": 0, "directional_accuracy": 0},
                "4h": {"r2": 0, "mae": 0, "directional_accuracy": 0},
                "24h": {"r2": 0, "mae": 0, "directional_accuracy": 0}
            },
            "data_collection": {"gas_prices": 0, "onchain_features": 0},
            "last_trained": None,
            "error": str(e)
        }), 500


@cron_bp.route('/cron/status', methods=['GET'])
def cron_status():
    """
    Get status of automated maintenance system
    Shows last retrain, last health check, etc.
    """
    try:
        session = db._get_session()

        try:
            # Get data collection stats
            from sqlalchemy import func
            gas_count = session.query(func.count()).select_from(db.GasPrice).scalar()
            onchain_count = session.query(func.count()).select_from(db.OnChainFeatures).scalar()

            # Get latest gas price timestamp
            latest_gas = session.query(func.max(db.GasPrice.timestamp)).scalar()

            # Get latest onchain feature timestamp
            latest_onchain = session.query(func.max(db.OnChainFeatures.timestamp)).scalar()

            # Check model files
            import glob
            model_files = glob.glob('models/saved_models/model_*.pkl')
            if not model_files:
                model_files = glob.glob('backend/models/saved_models/model_*.pkl')

            last_retrain = None
            if model_files:
                # Get most recent model file modification time
                last_retrain = max(os.path.getmtime(f) for f in model_files)
                last_retrain = datetime.fromtimestamp(last_retrain).isoformat()

            return jsonify({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "data_collection": {
                    "gas_prices": gas_count,
                    "onchain_features": onchain_count,
                    "latest_gas_timestamp": latest_gas.isoformat() if latest_gas else None,
                    "latest_onchain_timestamp": latest_onchain.isoformat() if latest_onchain else None,
                    "collection_active": (
                        latest_onchain and
                        (datetime.now() - latest_onchain).total_seconds() < 300  # Within 5 minutes
                    ) if latest_onchain else False
                },
                "models": {
                    "count": len(model_files),
                    "last_trained": last_retrain
                }
            })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error getting cron status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
