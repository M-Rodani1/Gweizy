"""
Prediction Validation System

Tracks prediction accuracy by comparing forecasts against actual gas prices
after the prediction horizon has passed. Provides real-time accuracy metrics
and model performance monitoring.

Features:
- Automatic prediction logging
- Scheduled validation against actual prices
- Performance metrics tracking (MAE, RMSE, directional accuracy)
- Model degradation alerts
- Historical accuracy analysis
"""

from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from data.database import DatabaseManager, Base, GasPrice
import numpy as np
from typing import Dict, List, Optional
import json


class PredictionLog(Base):
    """Logged prediction for validation"""
    __tablename__ = 'prediction_logs'

    id = Column(Integer, primary_key=True)
    prediction_time = Column(DateTime, default=datetime.now, index=True)
    target_time = Column(DateTime, index=True)  # When prediction is for
    horizon = Column(String)  # '1h', '4h', '24h'
    predicted_gas = Column(Float)
    actual_gas = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)
    direction_correct = Column(Boolean, nullable=True)  # UP/DOWN prediction
    model_version = Column(String)
    validated = Column(Boolean, default=False, index=True)
    validated_at = Column(DateTime, nullable=True)


class ModelPerformanceMetrics(Base):
    """Aggregated model performance metrics over time"""
    __tablename__ = 'model_performance_metrics'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now, index=True)
    horizon = Column(String, index=True)
    model_version = Column(String)

    # Metrics
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Squared Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    directional_accuracy = Column(Float)  # % correct UP/DOWN

    # Sample info
    sample_size = Column(Integer)  # Number of predictions validated

    # Metadata
    created_at = Column(DateTime, default=datetime.now)


class PredictionValidator:
    """Validates predictions and tracks model performance"""

    def __init__(self):
        self.db = DatabaseManager()
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """Create prediction validation tables if they don't exist"""
        Base.metadata.create_all(self.db.engine)

    def log_prediction(
        self,
        horizon: str,
        predicted_gas: float,
        target_time: datetime,
        model_version: str = "v1"
    ) -> int:
        """
        Log a prediction for future validation

        Args:
            horizon: Prediction horizon ('1h', '4h', '24h')
            predicted_gas: Predicted gas price in Gwei
            target_time: Datetime the prediction is for
            model_version: Model version identifier

        Returns:
            Prediction log ID
        """
        session = self.db._get_session()
        try:
            log = PredictionLog(
                prediction_time=datetime.now(),
                target_time=target_time,
                horizon=horizon,
                predicted_gas=predicted_gas,
                model_version=model_version,
                validated=False
            )
            session.add(log)
            session.commit()
            return log.id
        finally:
            session.close()

    def validate_predictions(self, max_age_hours: int = 48) -> Dict:
        """
        Validate pending predictions by comparing with actual gas prices

        Args:
            max_age_hours: Only validate predictions younger than this

        Returns:
            Dictionary with validation results
        """
        session = self.db._get_session()
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=max_age_hours)

            # Find unvalidated predictions where target_time has passed
            pending = session.query(PredictionLog).filter(
                PredictionLog.validated == False,
                PredictionLog.target_time <= now,
                PredictionLog.prediction_time >= cutoff
            ).all()

            validated_count = 0
            errors = []

            for prediction in pending:
                try:
                    # Find actual gas price at target time
                    actual_gas = self._get_actual_gas_price(
                        prediction.target_time,
                        session
                    )

                    if actual_gas is not None:
                        # Calculate error
                        error = abs(prediction.predicted_gas - actual_gas)

                        # Check directional accuracy (needs previous price)
                        direction_correct = self._check_direction_accuracy(
                            prediction,
                            actual_gas,
                            session
                        )

                        # Update prediction log
                        prediction.actual_gas = actual_gas
                        prediction.absolute_error = error
                        prediction.direction_correct = direction_correct
                        prediction.validated = True
                        prediction.validated_at = now

                        validated_count += 1

                except Exception as e:
                    errors.append(f"Error validating prediction {prediction.id}: {e}")
                    continue

            session.commit()

            return {
                'validated': validated_count,
                'pending': len(pending) - validated_count,
                'errors': errors,
                'timestamp': now.isoformat()
            }

        finally:
            session.close()

    def _get_actual_gas_price(
        self,
        target_time: datetime,
        session,
        tolerance_minutes: int = 15
    ) -> Optional[float]:
        """
        Get actual gas price closest to target time

        Args:
            target_time: Target datetime
            session: Database session
            tolerance_minutes: Max time difference to accept

        Returns:
            Actual gas price or None if not found
        """
        # Find gas price record closest to target time
        time_lower = target_time - timedelta(minutes=tolerance_minutes)
        time_upper = target_time + timedelta(minutes=tolerance_minutes)

        result = session.query(GasPrice).filter(
            GasPrice.timestamp >= time_lower,
            GasPrice.timestamp <= time_upper
        ).order_by(
            func.abs(
                func.extract('epoch', GasPrice.timestamp) -
                func.extract('epoch', target_time)
            )
        ).first()

        return result.current_gas if result else None

    def _check_direction_accuracy(
        self,
        prediction: PredictionLog,
        actual_gas: float,
        session
    ) -> Optional[bool]:
        """
        Check if directional prediction (UP/DOWN) was correct

        Args:
            prediction: Prediction log entry
            actual_gas: Actual gas price at target time
            session: Database session

        Returns:
            True if direction correct, False if wrong, None if can't determine
        """
        # Get gas price at prediction time
        prediction_time_gas = self._get_actual_gas_price(
            prediction.prediction_time,
            session,
            tolerance_minutes=30
        )

        if prediction_time_gas is None:
            return None

        # Calculate predicted and actual directions
        predicted_direction = prediction.predicted_gas > prediction_time_gas
        actual_direction = actual_gas > prediction_time_gas

        return predicted_direction == actual_direction

    def calculate_metrics(
        self,
        horizon: str = None,
        days: int = 7,
        model_version: str = None
    ) -> Dict:
        """
        Calculate performance metrics for validated predictions

        Args:
            horizon: Filter by horizon ('1h', '4h', '24h') or None for all
            days: Number of days to look back
            model_version: Filter by model version or None for all

        Returns:
            Dictionary with performance metrics
        """
        session = self.db._get_session()
        try:
            cutoff = datetime.now() - timedelta(days=days)

            # Build query
            query = session.query(PredictionLog).filter(
                PredictionLog.validated == True,
                PredictionLog.validated_at >= cutoff
            )

            if horizon:
                query = query.filter(PredictionLog.horizon == horizon)

            if model_version:
                query = query.filter(PredictionLog.model_version == model_version)

            predictions = query.all()

            if len(predictions) == 0:
                return {
                    'sample_size': 0,
                    'message': 'No validated predictions found'
                }

            # Extract data
            predicted = np.array([p.predicted_gas for p in predictions])
            actual = np.array([p.actual_gas for p in predictions])
            errors = np.array([p.absolute_error for p in predictions])
            directions = [p.direction_correct for p in predictions if p.direction_correct is not None]

            # Calculate metrics
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(errors ** 2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            directional_accuracy = (
                sum(directions) / len(directions) if directions else 0.0
            )

            return {
                'horizon': horizon or 'all',
                'model_version': model_version or 'all',
                'sample_size': len(predictions),
                'days': days,
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'mean_predicted': float(np.mean(predicted)),
                'mean_actual': float(np.mean(actual)),
                'timestamp': datetime.now().isoformat()
            }

        finally:
            session.close()

    def save_daily_metrics(self):
        """
        Calculate and save daily performance metrics for all horizons

        This should be run daily via a cron job or scheduler
        """
        session = self.db._get_session()
        try:
            horizons = ['1h', '4h', '24h']
            saved_count = 0

            for horizon in horizons:
                metrics = self.calculate_metrics(horizon=horizon, days=1)

                if metrics['sample_size'] > 0:
                    metric_record = ModelPerformanceMetrics(
                        date=datetime.now(),
                        horizon=horizon,
                        model_version='current',  # TODO: Track actual version
                        mae=metrics['mae'],
                        rmse=metrics['rmse'],
                        mape=metrics['mape'],
                        directional_accuracy=metrics['directional_accuracy'],
                        sample_size=metrics['sample_size']
                    )
                    session.add(metric_record)
                    saved_count += 1

            session.commit()
            return {
                'saved': saved_count,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            session.close()

    def get_performance_trends(
        self,
        horizon: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get performance trend over time

        Args:
            horizon: Prediction horizon
            days: Number of days to look back

        Returns:
            List of daily metrics
        """
        session = self.db._get_session()
        try:
            cutoff = datetime.now() - timedelta(days=days)

            metrics = session.query(ModelPerformanceMetrics).filter(
                ModelPerformanceMetrics.horizon == horizon,
                ModelPerformanceMetrics.date >= cutoff
            ).order_by(ModelPerformanceMetrics.date.asc()).all()

            return [{
                'date': m.date.isoformat(),
                'mae': m.mae,
                'rmse': m.rmse,
                'mape': m.mape,
                'directional_accuracy': m.directional_accuracy,
                'sample_size': m.sample_size
            } for m in metrics]

        finally:
            session.close()

    def check_model_health(self, threshold_mae: float = 0.001) -> Dict:
        """
        Check if model performance is degrading

        Args:
            threshold_mae: MAE threshold above which to alert

        Returns:
            Health status and alerts
        """
        alerts = []

        for horizon in ['1h', '4h', '24h']:
            metrics = self.calculate_metrics(horizon=horizon, days=1)

            if metrics['sample_size'] == 0:
                alerts.append({
                    'horizon': horizon,
                    'type': 'no_data',
                    'message': f'No validated predictions for {horizon} in last 24h'
                })
                continue

            # Check MAE threshold
            if metrics['mae'] > threshold_mae:
                alerts.append({
                    'horizon': horizon,
                    'type': 'high_error',
                    'message': f"MAE ({metrics['mae']:.6f}) exceeds threshold ({threshold_mae})",
                    'severity': 'warning'
                })

            # Check directional accuracy
            if metrics['directional_accuracy'] < 0.55:  # Barely better than random
                alerts.append({
                    'horizon': horizon,
                    'type': 'poor_direction',
                    'message': f"Directional accuracy ({metrics['directional_accuracy']:.2%}) near random",
                    'severity': 'critical'
                })

        return {
            'healthy': len(alerts) == 0,
            'alerts': alerts,
            'checked_at': datetime.now().isoformat()
        }

    def get_validation_summary(self) -> Dict:
        """Get summary of validation status"""
        session = self.db._get_session()
        try:
            total = session.query(func.count(PredictionLog.id)).scalar()
            validated = session.query(func.count(PredictionLog.id)).filter(
                PredictionLog.validated == True
            ).scalar()
            pending = total - validated

            # Recent accuracy
            recent_metrics = self.calculate_metrics(days=7)

            return {
                'total_predictions': total,
                'validated': validated,
                'pending': pending,
                'validation_rate': validated / total if total > 0 else 0,
                'recent_performance': recent_metrics,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            session.close()


# Scheduler integration
def scheduled_validation_job():
    """
    Job to run on schedule (e.g., every hour) to validate predictions
    """
    validator = PredictionValidator()

    # Validate pending predictions
    validation_results = validator.validate_predictions()
    print(f"[Validation] Validated {validation_results['validated']} predictions")

    # Check model health
    health = validator.check_model_health()
    if not health['healthy']:
        print(f"[Alert] Model health issues detected: {health['alerts']}")

    return validation_results


def scheduled_daily_metrics_job():
    """
    Job to run daily to save aggregated metrics
    """
    validator = PredictionValidator()
    results = validator.save_daily_metrics()
    print(f"[Metrics] Saved daily metrics for {results['saved']} horizons")
    return results


if __name__ == "__main__":
    # Example usage
    validator = PredictionValidator()

    print("=== Prediction Validation System ===\n")

    # Get summary
    summary = validator.get_validation_summary()
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Validated: {summary['validated']}")
    print(f"Pending: {summary['pending']}")

    # Run validation
    print("\nRunning validation...")
    results = validator.validate_predictions()
    print(f"Validated {results['validated']} predictions")

    # Check metrics
    print("\nPerformance metrics (last 7 days):")
    for horizon in ['1h', '4h', '24h']:
        metrics = validator.calculate_metrics(horizon=horizon, days=7)
        if metrics['sample_size'] > 0:
            print(f"\n{horizon} horizon:")
            print(f"  MAE: {metrics['mae']:.6f} Gwei")
            print(f"  RMSE: {metrics['rmse']:.6f} Gwei")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            print(f"  Sample size: {metrics['sample_size']}")
