"""
Model Evaluator - Automated model quality checks and performance monitoring
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class EvaluationResult:
    """Result of a model evaluation"""
    timestamp: datetime
    horizon: str
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    health: ModelHealth
    alerts: List[Dict]
    recommendations: List[str]
    sample_size: int


class ModelEvaluator:
    """
    Evaluates model performance and provides health assessments
    """

    # Thresholds for different horizons (in gwei for Base)
    THRESHOLDS = {
        '1h': {
            'mae_warn': 0.001,      # Warning if MAE > 0.001 gwei
            'mae_critical': 0.005,   # Critical if MAE > 0.005 gwei
            'dir_acc_warn': 0.55,    # Warning if directional accuracy < 55%
            'dir_acc_critical': 0.45, # Critical if < 45%
            'r2_warn': 0.3,          # Warning if R² < 0.3
            'r2_critical': 0.1       # Critical if R² < 0.1
        },
        '4h': {
            'mae_warn': 0.002,
            'mae_critical': 0.008,
            'dir_acc_warn': 0.52,
            'dir_acc_critical': 0.42,
            'r2_warn': 0.25,
            'r2_critical': 0.08
        },
        '24h': {
            'mae_warn': 0.005,
            'mae_critical': 0.015,
            'dir_acc_warn': 0.50,
            'dir_acc_critical': 0.40,
            'r2_warn': 0.20,
            'r2_critical': 0.05
        }
    }

    def __init__(self, accuracy_tracker=None):
        self.accuracy_tracker = accuracy_tracker
        self.evaluation_history: List[EvaluationResult] = []

    def evaluate_model(self, horizon: str, predictions: List[Dict],
                       actuals: List[float]) -> EvaluationResult:
        """
        Evaluate model performance for a specific horizon

        Args:
            horizon: '1h', '4h', or '24h'
            predictions: List of prediction dicts with 'predicted' key
            actuals: List of actual values

        Returns:
            EvaluationResult with metrics and health assessment
        """
        if len(predictions) != len(actuals) or len(predictions) == 0:
            return EvaluationResult(
                timestamp=datetime.now(),
                horizon=horizon,
                mae=0,
                rmse=0,
                r2=0,
                directional_accuracy=0,
                health=ModelHealth.UNKNOWN,
                alerts=[{'severity': 'error', 'message': 'Insufficient data for evaluation'}],
                recommendations=['Collect more prediction data'],
                sample_size=0
            )

        pred_values = [p.get('predicted', 0) for p in predictions]

        # Calculate metrics
        mae = self._calculate_mae(pred_values, actuals)
        rmse = self._calculate_rmse(pred_values, actuals)
        r2 = self._calculate_r2(pred_values, actuals)
        dir_acc = self._calculate_directional_accuracy(pred_values, actuals)

        # Assess health
        health, alerts = self._assess_health(horizon, mae, rmse, r2, dir_acc)

        # Generate recommendations
        recommendations = self._generate_recommendations(horizon, mae, r2, dir_acc, health)

        result = EvaluationResult(
            timestamp=datetime.now(),
            horizon=horizon,
            mae=mae,
            rmse=rmse,
            r2=r2,
            directional_accuracy=dir_acc,
            health=health,
            alerts=alerts,
            recommendations=recommendations,
            sample_size=len(predictions)
        )

        self.evaluation_history.append(result)

        return result

    def evaluate_all_horizons(self) -> Dict[str, EvaluationResult]:
        """
        Evaluate all model horizons using data from accuracy tracker

        Returns:
            Dict mapping horizon to EvaluationResult
        """
        results = {}

        if self.accuracy_tracker is None:
            logger.warning("No accuracy tracker available")
            return results

        for horizon in ['1h', '4h', '24h']:
            try:
                # Get recent predictions with actuals from tracker
                metrics = self.accuracy_tracker.get_accuracy_metrics(horizon)

                if metrics:
                    # Create evaluation result from tracker metrics
                    health, alerts = self._assess_health(
                        horizon,
                        metrics.get('mae', 0),
                        metrics.get('rmse', 0),
                        metrics.get('r2', 0),
                        metrics.get('directional_accuracy', 0)
                    )

                    recommendations = self._generate_recommendations(
                        horizon,
                        metrics.get('mae', 0),
                        metrics.get('r2', 0),
                        metrics.get('directional_accuracy', 0),
                        health
                    )

                    results[horizon] = EvaluationResult(
                        timestamp=datetime.now(),
                        horizon=horizon,
                        mae=metrics.get('mae', 0),
                        rmse=metrics.get('rmse', 0),
                        r2=metrics.get('r2', 0),
                        directional_accuracy=metrics.get('directional_accuracy', 0),
                        health=health,
                        alerts=alerts,
                        recommendations=recommendations,
                        sample_size=metrics.get('n_samples', 0)
                    )
            except Exception as e:
                logger.error(f"Error evaluating {horizon}: {e}")
                results[horizon] = EvaluationResult(
                    timestamp=datetime.now(),
                    horizon=horizon,
                    mae=0,
                    rmse=0,
                    r2=0,
                    directional_accuracy=0,
                    health=ModelHealth.UNKNOWN,
                    alerts=[{'severity': 'error', 'message': str(e)}],
                    recommendations=['Check model and data pipeline'],
                    sample_size=0
                )

        return results

    def get_overall_health(self) -> Tuple[ModelHealth, List[Dict]]:
        """
        Get overall model health across all horizons

        Returns:
            Tuple of (overall_health, combined_alerts)
        """
        results = self.evaluate_all_horizons()

        if not results:
            return ModelHealth.UNKNOWN, [{'severity': 'warning', 'message': 'No evaluation data available'}]

        all_alerts = []
        health_scores = {'healthy': 0, 'degraded': 0, 'critical': 0, 'unknown': 0}

        for horizon, result in results.items():
            health_scores[result.health.value] += 1
            all_alerts.extend(result.alerts)

        # Determine overall health
        if health_scores['critical'] > 0:
            overall = ModelHealth.CRITICAL
        elif health_scores['degraded'] >= 2:
            overall = ModelHealth.DEGRADED
        elif health_scores['unknown'] == len(results):
            overall = ModelHealth.UNKNOWN
        else:
            overall = ModelHealth.HEALTHY

        return overall, all_alerts

    def check_for_drift(self, horizon: str, window_hours: int = 24) -> Dict:
        """
        Check for model drift by comparing recent performance to baseline

        Args:
            horizon: Model horizon to check
            window_hours: Hours to look back for recent performance

        Returns:
            Dict with drift detection results
        """
        if self.accuracy_tracker is None:
            return {'is_drifting': False, 'confidence': 0, 'message': 'No tracker available'}

        try:
            drift_status = self.accuracy_tracker.check_drift()

            if horizon in drift_status:
                return drift_status[horizon]

            return {'is_drifting': False, 'confidence': 0, 'message': 'No drift data available'}

        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return {'is_drifting': False, 'confidence': 0, 'error': str(e)}

    def should_retrain(self) -> Tuple[bool, List[str]]:
        """
        Determine if model should be retrained

        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []

        # Check overall health
        health, alerts = self.get_overall_health()

        if health == ModelHealth.CRITICAL:
            reasons.append("Model health is critical")

        # Check for drift in each horizon
        for horizon in ['1h', '4h', '24h']:
            drift = self.check_for_drift(horizon)
            if drift.get('is_drifting', False):
                reasons.append(f"{horizon} model showing significant drift")

        # Check if any critical alerts
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        if len(critical_alerts) >= 2:
            reasons.append("Multiple critical performance alerts")

        return len(reasons) > 0, reasons

    def _calculate_mae(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        if len(predictions) == 0:
            return 0.0
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        return sum(errors) / len(errors)

    def _calculate_rmse(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate Root Mean Squared Error"""
        if len(predictions) == 0:
            return 0.0
        squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
        return np.sqrt(sum(squared_errors) / len(squared_errors))

    def _calculate_r2(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate R² score"""
        if len(predictions) < 2:
            return 0.0

        mean_actual = sum(actuals) / len(actuals)
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for p, a in zip(predictions, actuals))

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def _calculate_directional_accuracy(self, predictions: List[float],
                                         actuals: List[float]) -> float:
        """Calculate directional accuracy (% of correct direction predictions)"""
        if len(predictions) < 2:
            return 0.0

        correct = 0
        total = len(predictions) - 1

        for i in range(1, len(predictions)):
            pred_direction = predictions[i] > predictions[i-1]
            actual_direction = actuals[i] > actuals[i-1]
            if pred_direction == actual_direction:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _assess_health(self, horizon: str, mae: float, rmse: float,
                       r2: float, dir_acc: float) -> Tuple[ModelHealth, List[Dict]]:
        """Assess model health based on metrics"""
        alerts = []
        thresholds = self.THRESHOLDS.get(horizon, self.THRESHOLDS['1h'])

        is_critical = False
        is_degraded = False

        # Check MAE
        if mae > thresholds['mae_critical']:
            alerts.append({
                'severity': 'critical',
                'message': f'MAE ({mae:.6f}) exceeds critical threshold ({thresholds["mae_critical"]})',
                'metric': 'mae'
            })
            is_critical = True
        elif mae > thresholds['mae_warn']:
            alerts.append({
                'severity': 'warning',
                'message': f'MAE ({mae:.6f}) exceeds warning threshold ({thresholds["mae_warn"]})',
                'metric': 'mae'
            })
            is_degraded = True

        # Check directional accuracy
        if dir_acc < thresholds['dir_acc_critical']:
            alerts.append({
                'severity': 'critical',
                'message': f'Directional accuracy ({dir_acc:.1%}) below critical threshold ({thresholds["dir_acc_critical"]:.1%})',
                'metric': 'directional_accuracy'
            })
            is_critical = True
        elif dir_acc < thresholds['dir_acc_warn']:
            alerts.append({
                'severity': 'warning',
                'message': f'Directional accuracy ({dir_acc:.1%}) below warning threshold ({thresholds["dir_acc_warn"]:.1%})',
                'metric': 'directional_accuracy'
            })
            is_degraded = True

        # Check R²
        if r2 < thresholds['r2_critical']:
            alerts.append({
                'severity': 'warning',
                'message': f'R² score ({r2:.4f}) below threshold ({thresholds["r2_critical"]})',
                'metric': 'r2'
            })
            is_degraded = True

        # Determine overall health
        if is_critical:
            health = ModelHealth.CRITICAL
        elif is_degraded:
            health = ModelHealth.DEGRADED
        else:
            health = ModelHealth.HEALTHY

        return health, alerts

    def _generate_recommendations(self, horizon: str, mae: float, r2: float,
                                   dir_acc: float, health: ModelHealth) -> List[str]:
        """Generate actionable recommendations based on evaluation"""
        recommendations = []

        if health == ModelHealth.CRITICAL:
            recommendations.append("Consider immediate model retraining")
            recommendations.append("Check for data pipeline issues or external factors")

        if health == ModelHealth.DEGRADED:
            recommendations.append("Monitor closely and prepare for retraining if degradation continues")

        if r2 < 0.1:
            recommendations.append("Model explains very little variance - consider feature engineering")

        if dir_acc < 0.5:
            recommendations.append("Direction prediction is worse than random - review training data")

        if mae > 0.005:
            recommendations.append("Consider adjusting model architecture or hyperparameters")

        if not recommendations:
            recommendations.append("Model performing within acceptable parameters")

        return recommendations

    def get_evaluation_summary(self) -> Dict:
        """
        Get a summary of model evaluations

        Returns:
            Dict with evaluation summary
        """
        results = self.evaluate_all_horizons()
        overall_health, alerts = self.get_overall_health()
        should_retrain, retrain_reasons = self.should_retrain()

        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health.value,
            'horizons': {
                horizon: {
                    'health': result.health.value,
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'r2': result.r2,
                    'directional_accuracy': result.directional_accuracy,
                    'sample_size': result.sample_size,
                    'alerts': result.alerts,
                    'recommendations': result.recommendations
                }
                for horizon, result in results.items()
            },
            'alerts': alerts,
            'should_retrain': should_retrain,
            'retrain_reasons': retrain_reasons
        }


# Global instance
evaluator: Optional[ModelEvaluator] = None


def get_evaluator() -> ModelEvaluator:
    """Get or create the global evaluator instance"""
    global evaluator
    if evaluator is None:
        try:
            from models.accuracy_tracker import get_tracker
            tracker = get_tracker()
            evaluator = ModelEvaluator(accuracy_tracker=tracker)
        except Exception as e:
            logger.warning(f"Could not initialize evaluator with tracker: {e}")
            evaluator = ModelEvaluator()
    return evaluator
