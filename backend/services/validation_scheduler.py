"""
Prediction Validation Scheduler

Automatically validates predictions and tracks model performance.
Runs in background to continuously monitor accuracy.
"""

import time
import threading
from datetime import datetime, timedelta
from utils.prediction_validator import PredictionValidator
from utils.logger import logger


class ValidationScheduler:
    """
    Background scheduler for prediction validation

    Features:
    - Validates predictions every hour
    - Saves daily performance metrics
    - Monitors model health
    - Sends alerts for degraded performance
    """

    def __init__(self):
        self.validator = PredictionValidator()
        self.running = False

        # Schedule intervals (seconds)
        self.validation_interval = 3600  # 1 hour
        self.metrics_interval = 86400    # 24 hours
        self.health_check_interval = 3600  # 1 hour

        # State
        self.last_validation = None
        self.last_metrics_save = None
        self.last_health_check = None

        # Statistics
        self.total_validations = 0
        self.total_validated_predictions = 0
        self.health_alerts = []

    def should_run_validation(self) -> bool:
        """Check if validation should run"""
        if self.last_validation is None:
            return True
        return (datetime.now() - self.last_validation).total_seconds() >= self.validation_interval

    def should_save_metrics(self) -> bool:
        """Check if daily metrics should be saved"""
        if self.last_metrics_save is None:
            return True
        return (datetime.now() - self.last_metrics_save).total_seconds() >= self.metrics_interval

    def should_check_health(self) -> bool:
        """Check if health check should run"""
        if self.last_health_check is None:
            return True
        return (datetime.now() - self.last_health_check).total_seconds() >= self.health_check_interval

    def run_validation(self):
        """Run prediction validation"""
        try:
            logger.info("="*60)
            logger.info("Running Prediction Validation")
            logger.info("="*60)

            results = self.validator.validate_predictions(max_age_hours=48)

            self.total_validations += 1
            self.total_validated_predictions += results['validated']
            self.last_validation = datetime.now()

            logger.info(f"✓ Validated {results['validated']} predictions")
            logger.info(f"  Pending: {results['pending']}")
            if results['errors']:
                logger.warning(f"  Errors: {len(results['errors'])}")

            # Get recent metrics
            summary = self.validator.get_validation_summary()
            logger.info(f"  Total predictions: {summary['total_predictions']}")
            logger.info(f"  Validation rate: {summary['validation_rate']:.1%}")

            logger.info("="*60)

            return results

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def save_daily_metrics(self):
        """Save daily performance metrics"""
        try:
            logger.info("="*60)
            logger.info("Saving Daily Performance Metrics")
            logger.info("="*60)

            results = self.validator.save_daily_metrics()

            self.last_metrics_save = datetime.now()

            logger.info(f"✓ Saved metrics for {results['saved']} horizons")

            # Log current performance
            for horizon in ['1h', '4h', '24h']:
                metrics = self.validator.calculate_metrics(horizon=horizon, days=1)
                if metrics['sample_size'] > 0:
                    logger.info(
                        f"  {horizon}: MAE={metrics['mae']:.6f}, "
                        f"DA={metrics['directional_accuracy']:.2%}, "
                        f"n={metrics['sample_size']}"
                    )

            logger.info("="*60)

            return results

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def check_model_health(self):
        """Check model health and log alerts"""
        try:
            health = self.validator.check_model_health(threshold_mae=0.001)

            self.last_health_check = datetime.now()

            if not health['healthy']:
                logger.warning("="*60)
                logger.warning("⚠️  MODEL HEALTH ALERTS")
                logger.warning("="*60)

                for alert in health['alerts']:
                    level = alert.get('severity', 'info').upper()
                    logger.warning(f"  [{level}] {alert['horizon']}: {alert['message']}")

                # Store alerts for monitoring
                self.health_alerts = health['alerts']

                logger.warning("="*60)
            else:
                logger.info("✓ Model health check passed - all models performing well")

                # Clear previous alerts
                if self.health_alerts:
                    logger.info("  Previous alerts have been resolved")
                    self.health_alerts = []

            return health

        except Exception as e:
            logger.error(f"Error during health check: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_scheduled_tasks(self):
        """Run all scheduled tasks that are due"""
        tasks_run = []

        # Validation (hourly)
        if self.should_run_validation():
            result = self.run_validation()
            if result:
                tasks_run.append('validation')

        # Daily metrics (daily)
        if self.should_save_metrics():
            result = self.save_daily_metrics()
            if result:
                tasks_run.append('metrics')

        # Health check (hourly)
        if self.should_check_health():
            result = self.check_model_health()
            if result:
                tasks_run.append('health_check')

        return tasks_run

    def start(self):
        """Start the validation scheduler"""
        logger.info("="*60)
        logger.info("Prediction Validation Scheduler Starting")
        logger.info(f"Validation interval: {self.validation_interval}s ({self.validation_interval/3600:.1f}h)")
        logger.info(f"Metrics save interval: {self.metrics_interval}s ({self.metrics_interval/86400:.1f} days)")
        logger.info(f"Health check interval: {self.health_check_interval}s ({self.health_check_interval/3600:.1f}h)")
        logger.info("="*60)

        self.running = True

        while self.running:
            try:
                tasks_run = self.run_scheduled_tasks()

                if tasks_run:
                    logger.info(f"Scheduler: Completed {len(tasks_run)} tasks: {', '.join(tasks_run)}")

                # Sleep for 5 minutes before checking again
                time.sleep(300)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(300)

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping validation scheduler...")
        self.running = False
        self.log_stats()
        logger.info("Validation scheduler stopped")

    def log_stats(self):
        """Log scheduler statistics"""
        logger.info("="*60)
        logger.info("Validation Scheduler Statistics:")
        logger.info(f"  Total validation runs: {self.total_validations}")
        logger.info(f"  Total predictions validated: {self.total_validated_predictions}")
        logger.info(f"  Active health alerts: {len(self.health_alerts)}")
        logger.info(f"  Last validation: {self.last_validation}")
        logger.info(f"  Last metrics save: {self.last_metrics_save}")
        logger.info(f"  Last health check: {self.last_health_check}")
        logger.info("="*60)


def start_scheduler_background():
    """Start the validation scheduler in a background thread"""
    scheduler = ValidationScheduler()

    def run():
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.stop()
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            scheduler.stop()

    thread = threading.Thread(target=run, name="ValidationScheduler", daemon=True)
    thread.start()
    logger.info("✓ Validation scheduler started in background thread")

    return scheduler


def main():
    """Main entry point for standalone scheduler"""
    import signal
    import sys

    scheduler = ValidationScheduler()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        scheduler.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scheduler.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
