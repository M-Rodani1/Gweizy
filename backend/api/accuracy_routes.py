"""
API routes for model accuracy tracking and feature selection.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

accuracy_bp = Blueprint('accuracy', __name__)


def get_tracker():
    """Get or create accuracy tracker."""
    try:
        from models.accuracy_tracker import get_tracker
        return get_tracker()
    except Exception as e:
        logger.error(f"Failed to load accuracy tracker: {e}")
        return None


def get_feature_selector():
    """Load saved feature selector."""
    try:
        from models.feature_selector import SHAPFeatureSelector
        return SHAPFeatureSelector.load()
    except Exception as e:
        logger.warning(f"Feature selector not loaded: {e}")
        return None


@accuracy_bp.route('/metrics', methods=['GET'])
def get_accuracy_metrics():
    """
    Get current accuracy metrics for all horizons.

    Returns:
        {
            "1h": {"mae": 0.001, "rmse": 0.002, "r2": 0.85, ...},
            "4h": {...},
            "24h": {...}
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        # Try to validate any pending predictions before returning metrics
        # This is more aggressive - validate ALL ready predictions, not just one
        try:
            from data.collector import BaseGasCollector
            from datetime import timedelta
            import sqlite3
            
            collector = BaseGasCollector()
            current_data = collector.get_current_gas()
            
            if current_data:
                actual_gas = current_data.get('current_gas', 0)
                now = datetime.now()
                db_path = tracker.db_path
                validated_count = 0
                
                with sqlite3.connect(db_path) as conn:
                    for horizon, hours_back in [('1h', 1), ('4h', 4), ('24h', 24)]:
                        # Find ALL pending predictions that are old enough
                        min_age = timedelta(hours=hours_back * 0.8)
                        cutoff_time = (now - min_age).isoformat()
                        
                        rows = conn.execute('''
                            SELECT timestamp FROM predictions
                            WHERE horizon = ? 
                            AND actual IS NULL
                            AND timestamp <= ?
                            ORDER BY timestamp ASC
                            LIMIT 10
                        ''', (horizon, cutoff_time)).fetchall()
                        
                        for row in rows:
                            from dateutil import parser
                            pred_time = parser.parse(row[0])
                            try:
                                success = tracker.record_actual(
                                    horizon=horizon,
                                    actual=actual_gas,
                                    prediction_timestamp=pred_time,
                                    tolerance_minutes=60
                                )
                                if success:
                                    validated_count += 1
                            except Exception as e:
                                logger.debug(f"Could not validate {horizon} prediction: {e}")
                
                if validated_count > 0:
                    logger.info(f"Auto-validated {validated_count} predictions")
        except Exception as e:
            logger.debug(f"Auto-validation failed (non-critical): {e}")
        
        metrics = {}
        for horizon in ['1h', '4h', '24h']:
            metrics[horizon] = tracker.get_current_metrics(horizon)

        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/drift', methods=['GET'])
def check_drift():
    """
    Check for model drift across all horizons.

    Returns:
        {
            "1h": {"is_drifting": false, "drift_ratio": 0.05, ...},
            "4h": {...},
            "24h": {...},
            "should_retrain": false
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        drift_info = {}
        for horizon in ['1h', '4h', '24h']:
            drift = tracker.check_drift(horizon)
            drift_info[horizon] = {
                'is_drifting': drift.is_drifting,
                'drift_ratio': drift.drift_ratio,
                'mae_current': drift.mae_current,
                'mae_baseline': drift.mae_baseline,
                'confidence': drift.confidence,
                'sample_size': drift.sample_size
            }

        should_retrain, reasons = tracker.should_retrain()

        return jsonify({
            'success': True,
            'drift': drift_info,
            'should_retrain': should_retrain,
            'retrain_reasons': reasons
        })
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/status', methods=['GET'])
def get_prediction_status():
    """
    Get status of predictions - how many pending, validated, etc.
    """
    tracker = get_tracker()
    
    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503
    
    try:
        import sqlite3
        from datetime import datetime, timedelta
        db_path = tracker.db_path
        
        status = {}
        now = datetime.now()
        
        with sqlite3.connect(db_path) as conn:
            for horizon in ['1h', '4h', '24h']:
                # Count total predictions
                total = conn.execute('''
                    SELECT COUNT(*) FROM predictions WHERE horizon = ?
                ''', (horizon,)).fetchone()[0]
                
                # Count validated predictions
                validated = conn.execute('''
                    SELECT COUNT(*) FROM predictions 
                    WHERE horizon = ? AND actual IS NOT NULL
                ''', (horizon,)).fetchone()[0]
                
                # Count pending predictions
                pending = total - validated
                
                # Count pending predictions that are old enough to validate
                hours_back = {'1h': 1, '4h': 4, '24h': 24}[horizon]
                min_age = timedelta(hours=hours_back * 0.8)
                cutoff_time = (now - min_age).isoformat()
                
                ready_to_validate = conn.execute('''
                    SELECT COUNT(*) FROM predictions
                    WHERE horizon = ? 
                    AND actual IS NULL
                    AND timestamp <= ?
                ''', (horizon, cutoff_time)).fetchone()[0]
                
                # Get oldest pending prediction timestamp
                oldest_pending = conn.execute('''
                    SELECT timestamp FROM predictions
                    WHERE horizon = ? AND actual IS NULL
                    ORDER BY timestamp ASC
                    LIMIT 1
                ''', (horizon,)).fetchone()
                
                oldest_timestamp = oldest_pending[0] if oldest_pending else None
                
                status[horizon] = {
                    'total': total,
                    'validated': validated,
                    'pending': pending,
                    'ready_to_validate': ready_to_validate,
                    'oldest_pending': oldest_timestamp
                }
        
        return jsonify({
            'success': True,
            'status': status,
            'message': 'Use /accuracy/validate-pending to manually validate ready predictions'
        })
    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/validate-pending', methods=['POST'])
def validate_pending_predictions():
    """
    Manually validate all pending predictions by comparing with actual gas prices.
    This can be called to immediately update metrics.
    """
    tracker = get_tracker()
    
    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503
    
    try:
        from data.collector import BaseGasCollector
        from datetime import timedelta
        import sqlite3
        
        collector = BaseGasCollector()
        current_data = collector.get_current_gas()
        
        if not current_data:
            return jsonify({
                'success': False,
                'error': 'Could not fetch current gas price'
            }), 500
        
        actual_gas = current_data.get('current_gas', 0)
        now = datetime.now()
        validated_count = 0
        
        # Validate all pending predictions for each horizon
        db_path = tracker.db_path
        with sqlite3.connect(db_path) as conn:
            for horizon, hours_back in [('1h', 1), ('4h', 4), ('24h', 24)]:
                # Find all pending predictions that are old enough
                min_age = timedelta(hours=hours_back * 0.8)  # At least 80% of horizon
                cutoff_time = (now - min_age).isoformat()
                
                rows = conn.execute('''
                    SELECT timestamp FROM predictions
                    WHERE horizon = ? 
                    AND actual IS NULL
                    AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (horizon, cutoff_time)).fetchall()
                
                for row in rows:
                    from dateutil import parser
                    pred_time = parser.parse(row[0])
                    try:
                        success = tracker.record_actual(
                            horizon=horizon,
                            actual=actual_gas,
                            prediction_timestamp=pred_time,
                            tolerance_minutes=60
                        )
                        if success:
                            validated_count += 1
                    except Exception as e:
                        logger.debug(f"Could not validate {horizon} prediction at {pred_time}: {e}")
        
        # Get updated metrics
        metrics = {}
        for horizon in ['1h', '4h', '24h']:
            metrics[horizon] = tracker.get_current_metrics(horizon)
        
        return jsonify({
            'success': True,
            'validated': validated_count,
            'metrics': metrics,
            'message': f'Validated {validated_count} predictions'
        })
    except Exception as e:
        logger.error(f"Error validating predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/record', methods=['POST'])
def record_prediction():
    """
    Record a prediction for later accuracy tracking.

    Request body:
        {
            "horizon": "1h",
            "predicted": 0.00123,
            "actual": 0.00125  (optional, for immediate recording)
        }
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        data = request.get_json()
        horizon = data.get('horizon', '1h')
        predicted = data.get('predicted')
        actual = data.get('actual')

        if predicted is None:
            return jsonify({
                'success': False,
                'error': 'predicted value required'
            }), 400

        if actual is not None:
            # Record both together
            tracker.record_prediction_with_actual(
                horizon=horizon,
                predicted=predicted,
                actual=actual
            )
            return jsonify({
                'success': True,
                'message': 'Prediction and actual recorded'
            })
        else:
            # Record prediction only
            pred_id = tracker.record_prediction(
                horizon=horizon,
                predicted=predicted
            )
            return jsonify({
                'success': True,
                'prediction_id': pred_id,
                'message': 'Prediction recorded'
            })
    except Exception as e:
        logger.error(f"Error recording prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/summary', methods=['GET'])
def get_summary():
    """
    Get comprehensive accuracy tracking summary.
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        summary = tracker.get_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/report', methods=['GET'])
def generate_report():
    """
    Generate accuracy report for a specific horizon and time period.

    Query params:
        horizon: '1h', '4h', or '24h' (default: '1h')
        hours_back: Number of hours to look back (default: 24)
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        horizon = request.args.get('horizon', '1h')
        hours_back = int(request.args.get('hours_back', 24))

        report = tracker.generate_accuracy_report(horizon, hours_back)

        if report is None:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for report'
            }), 400

        return jsonify({
            'success': True,
            'report': {
                'horizon': report.horizon,
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'n_predictions': report.n_predictions,
                'mae': report.mae,
                'rmse': report.rmse,
                'mape': report.mape,
                'directional_accuracy': report.directional_accuracy,
                'r2': report.r2,
                'best_prediction_error': report.best_prediction_error,
                'worst_prediction_error': report.worst_prediction_error
            }
        })
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Feature Selection endpoints
@accuracy_bp.route('/features', methods=['GET'])
def get_selected_features():
    """
    Get the list of SHAP-selected features.
    """
    selector = get_feature_selector()

    if selector is None:
        return jsonify({
            'success': False,
            'error': 'Feature selector not trained yet'
        }), 404

    try:
        # Convert feature importances to native Python types
        top_10_importance = {}
        for f in selector.selected_features[:10]:
            importance_val = selector.feature_importances[f]
            # Convert numpy types to native Python types
            if hasattr(importance_val, 'item'):
                importance_val = importance_val.item()
            elif not isinstance(importance_val, (int, float, str, bool, type(None))):
                importance_val = float(importance_val)
            top_10_importance[f] = importance_val
        
        return jsonify({
            'success': True,
            'n_features': int(len(selector.selected_features)),
            'selected_features': selector.selected_features,
            'top_10_importance': top_10_importance
        })
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/features/importance', methods=['GET'])
def get_feature_importance():
    """
    Get full feature importance report.
    """
    selector = get_feature_selector()

    if selector is None:
        # Return empty report instead of 404 to prevent console errors
        return jsonify({
            'success': True,
            'total_features': 0,
            'selected_count': 0,
            'importance_report': [],
            'message': 'Feature selector not trained yet'
        }), 200

    try:
        report = selector.get_feature_report()
        top_n = int(request.args.get('top_n', 30))

        # Convert DataFrame to dict and ensure all values are JSON-serializable
        importance_records = report.head(top_n).to_dict('records')
        
        # Convert numpy/pandas types to native Python types
        def convert_to_native(obj):
            """Recursively convert numpy/pandas types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                # Fallback: convert to string or native type
                try:
                    return int(obj) if isinstance(obj, (np.integer, np.int64, np.int32)) else float(obj) if isinstance(obj, (np.floating, np.float64, np.float32)) else str(obj)
                except:
                    return str(obj)
        
        importance_records = convert_to_native(importance_records)
        selected_count = int(report['selected'].sum()) if hasattr(report['selected'].sum(), 'item') else int(report['selected'].sum())

        return jsonify({
            'success': True,
            'total_features': int(len(report)),
            'selected_count': selected_count,
            'importance_report': importance_records
        })
    except Exception as e:
        logger.error(f"Error getting importance: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/history', methods=['GET'])
def get_accuracy_history():
    """
    Get historical accuracy metrics for charting.

    Query params:
        hours_back: Number of hours to look back (default: 168, max: 720)
        resolution: 'hourly' or 'daily' (default: 'hourly')
    """
    tracker = get_tracker()

    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503

    try:
        hours_back = min(int(request.args.get('hours_back', 168)), 720)
        resolution = request.args.get('resolution', 'hourly')

        # Get historical data from tracker
        history = tracker.get_accuracy_history(hours_back=hours_back, resolution=resolution)

        return jsonify({
            'success': True,
            'history': history,
            'hours_back': hours_back,
            'resolution': resolution
        })
    except Exception as e:
        logger.error(f"Error getting accuracy history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/seed-test-data', methods=['POST'])
def seed_test_data():
    """
    Seed test data for immediate metrics display (for development/testing).
    Creates validated predictions so metrics can be shown immediately.
    """
    tracker = get_tracker()
    
    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available'
        }), 503
    
    try:
        from data.collector import BaseGasCollector
        import numpy as np
        
        collector = BaseGasCollector()
        current_data = collector.get_current_gas()
        current_gas = current_data.get('current_gas', 0.001) if current_data else 0.001
        
        # Create test predictions with actuals (simulating past validated predictions)
        now = datetime.now()
        seeded_count = 0
        
        for horizon in ['1h', '4h', '24h']:
            # Create 20 test predictions going back in time
            for i in range(20, 0, -1):
                # Simulate a prediction made i hours ago
                pred_time = now - timedelta(hours=i)
                
                # Add some realistic variation
                variation = np.random.normal(0, 0.1)
                predicted = current_gas * (1 + variation)
                actual = current_gas * (1 + variation * 0.9)  # Slightly different for realism
                
                try:
                    tracker.record_prediction_with_actual(
                        horizon=horizon,
                        predicted=predicted,
                        actual=actual,
                        timestamp=pred_time
                    )
                    seeded_count += 1
                except Exception as e:
                    logger.debug(f"Could not seed {horizon} prediction: {e}")
        
        # Get updated metrics
        metrics = {}
        for horizon in ['1h', '4h', '24h']:
            metrics[horizon] = tracker.get_current_metrics(horizon)
        
        return jsonify({
            'success': True,
            'seeded': seeded_count,
            'metrics': metrics,
            'message': f'Seeded {seeded_count} test predictions. Metrics should now be visible.'
        })
    except Exception as e:
        logger.error(f"Error seeding test data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@accuracy_bp.route('/diagnostics', methods=['GET'])
def get_diagnostics():
    """
    Get diagnostic information about the accuracy tracking system.
    """
    tracker = get_tracker()
    
    if tracker is None:
        return jsonify({
            'success': False,
            'error': 'Accuracy tracker not available',
            'tracker_initialized': False
        }), 503
    
    try:
        import sqlite3
        import os
        
        diagnostics = {
            'tracker_initialized': True,
            'db_path': tracker.db_path,
            'db_exists': os.path.exists(tracker.db_path),
            'db_writable': os.access(os.path.dirname(tracker.db_path), os.W_OK) if os.path.exists(tracker.db_path) else False,
            'predictions': {},
            'metrics': {}
        }
        
        # Check database contents
        if os.path.exists(tracker.db_path):
            with sqlite3.connect(tracker.db_path) as conn:
                for horizon in ['1h', '4h', '24h']:
                    total = conn.execute('''
                        SELECT COUNT(*) FROM predictions WHERE horizon = ?
                    ''', (horizon,)).fetchone()[0]
                    
                    validated = conn.execute('''
                        SELECT COUNT(*) FROM predictions 
                        WHERE horizon = ? AND actual IS NOT NULL
                    ''', (horizon,)).fetchone()[0]
                    
                    pending = total - validated
                    
                    # Get oldest and newest prediction timestamps
                    oldest = conn.execute('''
                        SELECT timestamp FROM predictions 
                        WHERE horizon = ?
                        ORDER BY timestamp ASC LIMIT 1
                    ''', (horizon,)).fetchone()
                    
                    newest = conn.execute('''
                        SELECT timestamp FROM predictions 
                        WHERE horizon = ?
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (horizon,)).fetchone()
                    
                    diagnostics['predictions'][horizon] = {
                        'total': total,
                        'validated': validated,
                        'pending': pending,
                        'oldest': oldest[0] if oldest else None,
                        'newest': newest[0] if newest else None
                    }
                    
                    # Get current metrics
                    metrics = tracker.get_current_metrics(horizon)
                    diagnostics['metrics'][horizon] = metrics
        
        return jsonify({
            'success': True,
            'diagnostics': diagnostics
        })
    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'tracker_initialized': tracker is not None
        }), 500


@accuracy_bp.route('/features/train', methods=['POST'])
def train_feature_selector():
    """
    Trigger feature selector training with current data.

    Request body (optional):
        {
            "n_features": 30,
            "hours_back": 720
        }
    """
    try:
        from models.feature_engineering import GasFeatureEngineer
        from models.feature_selector import SHAPFeatureSelector

        data = request.get_json() or {}
        n_features = data.get('n_features', 30)
        hours_back = data.get('hours_back', 720)

        # Get training data
        engineer = GasFeatureEngineer()
        df = engineer.prepare_training_data(hours_back=hours_back)
        feature_cols = engineer.get_feature_columns(df)

        X = df[feature_cols]
        y = df['gas']

        # Train selector
        selector = SHAPFeatureSelector(n_features=n_features)
        selector.fit(X, y)
        selector.save()

        return jsonify({
            'success': True,
            'message': f'Feature selector trained with {n_features} features',
            'selected_features': selector.selected_features[:10],
            'total_evaluated': len(feature_cols)
        })
    except Exception as e:
        logger.error(f"Error training feature selector: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
