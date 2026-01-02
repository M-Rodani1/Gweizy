"""
Prediction Validation Scheduler

Runs scheduled jobs to:
1. Validate predictions every hour
2. Save daily metrics every day at midnight
3. Check model health every 6 hours

Usage:
    python scripts/run_scheduler.py

This should be run as a background process in production:
    nohup python scripts/run_scheduler.py > logs/scheduler.log 2>&1 &
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import schedule
import time
from datetime import datetime
from utils.prediction_validator import (
    PredictionValidator,
    scheduled_validation_job,
    scheduled_daily_metrics_job
)
from utils.model_retrainer import scheduled_retraining_check
from utils.logger import logger


def run_hourly_validation():
    """Run validation every hour"""
    try:
        logger.info("=" * 60)
        logger.info(f"[Scheduler] Running hourly validation at {datetime.now()}")
        logger.info("=" * 60)

        results = scheduled_validation_job()

        logger.info(f"Validated {results['validated']} predictions")
        logger.info(f"Pending: {results['pending']}")

        if results['errors']:
            logger.warning(f"Errors during validation: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  {error}")

    except Exception as e:
        logger.error(f"Error in hourly validation job: {e}")
        import traceback
        logger.error(traceback.format_exc())


def run_health_check():
    """Run health check every 6 hours"""
    try:
        logger.info("=" * 60)
        logger.info(f"[Scheduler] Running health check at {datetime.now()}")
        logger.info("=" * 60)

        validator = PredictionValidator()
        health = validator.check_model_health(threshold_mae=0.001)

        if health['healthy']:
            logger.info("✓ Model health: GOOD")
        else:
            logger.warning("⚠ Model health: DEGRADED")
            for alert in health['alerts']:
                severity = alert.get('severity', 'warning')
                message = alert.get('message', 'Unknown issue')
                logger.warning(f"  [{severity.upper()}] {alert['horizon']}: {message}")

    except Exception as e:
        logger.error(f"Error in health check job: {e}")
        import traceback
        logger.error(traceback.format_exc())


def run_daily_metrics():
    """Run daily metrics save at midnight"""
    try:
        logger.info("=" * 60)
        logger.info(f"[Scheduler] Saving daily metrics at {datetime.now()}")
        logger.info("=" * 60)

        results = scheduled_daily_metrics_job()

        logger.info(f"Saved metrics for {results['saved']} horizons")

    except Exception as e:
        logger.error(f"Error in daily metrics job: {e}")
        import traceback
        logger.error(traceback.format_exc())


def run_retraining_check():
    """Run retraining check (daily)"""
    try:
        logger.info("=" * 60)
        logger.info(f"[Scheduler] Checking if model retraining needed at {datetime.now()}")
        logger.info("=" * 60)

        results = scheduled_retraining_check()

        if results.get('retrained'):
            logger.info(f"Models retrained: {results.get('models_trained')}")
            if results.get('validation_passed'):
                logger.info("✓ New models validated and deployed")
            else:
                logger.error("✗ Retraining failed - rolled back")
        else:
            logger.info(f"No retraining needed: {results.get('reason')}")

    except Exception as e:
        logger.error(f"Error in retraining check job: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main scheduler loop"""
    logger.info("=" * 60)
    logger.info("Prediction Validation Scheduler Starting")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Schedule:")
    logger.info("  - Hourly validation: Every hour at :05")
    logger.info("  - Health check: Every 6 hours")
    logger.info("  - Daily metrics: Every day at 00:10")
    logger.info("  - Retraining check: Every day at 02:00")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    logger.info("")

    # Schedule jobs
    schedule.every().hour.at(":05").do(run_hourly_validation)
    schedule.every(6).hours.do(run_health_check)
    schedule.every().day.at("00:10").do(run_daily_metrics)
    schedule.every().day.at("02:00").do(run_retraining_check)  # Daily at 2 AM

    # Run initial validation
    logger.info("Running initial validation...")
    run_hourly_validation()
    run_health_check()

    # Run scheduler loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("\n\nScheduler stopped by user")
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
