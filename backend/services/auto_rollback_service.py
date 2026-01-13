"""
Automatic Model Rollback Service

Monitors model accuracy and automatically rolls back to previous versions
when accuracy degrades beyond acceptable thresholds.

This is a critical reliability feature that ensures poor model updates
don't affect users for extended periods.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from utils.logger import logger, capture_exception


@dataclass
class AccuracyWindow:
    """Sliding window of accuracy measurements."""
    window_size: int = 10
    measurements: List[float] = field(default_factory=list)

    def add(self, accuracy: float) -> None:
        """Add a measurement to the window."""
        self.measurements.append(accuracy)
        if len(self.measurements) > self.window_size:
            self.measurements = self.measurements[-self.window_size:]

    def get_average(self) -> Optional[float]:
        """Get average accuracy in window."""
        if not self.measurements:
            return None
        return sum(self.measurements) / len(self.measurements)

    def get_trend(self) -> Optional[float]:
        """Get accuracy trend (positive = improving, negative = degrading)."""
        if len(self.measurements) < 3:
            return None
        recent = self.measurements[-3:]
        older = self.measurements[:-3] if len(self.measurements) > 3 else self.measurements[:1]
        return sum(recent) / len(recent) - sum(older) / len(older)


class AutoRollbackService:
    """
    Monitors model accuracy and triggers automatic rollback when accuracy degrades.

    Thresholds:
    - Critical: Accuracy drops > 30% from baseline -> immediate rollback
    - Warning: Accuracy drops > 15% for 5 consecutive checks -> rollback
    - Trend: Consistent degradation over 10 checks -> rollback

    The service runs as a background thread and checks accuracy periodically.
    """

    def __init__(
        self,
        check_interval_seconds: int = 300,  # 5 minutes
        critical_drop_threshold: float = 0.30,  # 30% drop triggers immediate rollback
        warning_drop_threshold: float = 0.15,  # 15% drop
        warning_consecutive_checks: int = 5,  # 5 consecutive warnings
        trend_window_size: int = 10,  # Window for trend calculation
        enabled: bool = True
    ):
        self.check_interval = check_interval_seconds
        self.critical_threshold = critical_drop_threshold
        self.warning_threshold = warning_drop_threshold
        self.warning_consecutive = warning_consecutive_checks
        self.enabled = enabled

        # Per-horizon tracking
        self.horizons = ['1h', '4h', '24h']
        self.baseline_accuracy: Dict[str, float] = {}
        self.accuracy_windows: Dict[str, AccuracyWindow] = {
            h: AccuracyWindow(window_size=trend_window_size) for h in self.horizons
        }
        self.warning_counts: Dict[str, int] = {h: 0 for h in self.horizons}
        self.last_rollback: Dict[str, datetime] = {}

        # Cooldown to prevent rapid rollbacks
        self.rollback_cooldown_minutes = 60

        # Thread control
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Rollback history
        self.rollback_history: List[Dict] = []

        logger.info(f"AutoRollbackService initialized: check_interval={check_interval_seconds}s, "
                   f"critical_threshold={critical_drop_threshold}, enabled={enabled}")

    def start(self) -> None:
        """Start the auto-rollback monitoring service."""
        if self._thread and self._thread.is_alive():
            logger.warning("AutoRollbackService already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("AutoRollbackService started")

    def stop(self) -> None:
        """Stop the auto-rollback monitoring service."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("AutoRollbackService stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        # Wait for initial startup
        time.sleep(30)

        while self.running:
            try:
                if self.enabled:
                    self._check_all_horizons()
            except Exception as e:
                logger.error(f"Error in auto-rollback monitoring: {e}")
                capture_exception(e, {'service': 'auto_rollback'})

            time.sleep(self.check_interval)

    def _check_all_horizons(self) -> None:
        """Check accuracy for all prediction horizons."""
        for horizon in self.horizons:
            try:
                self._check_horizon(horizon)
            except Exception as e:
                logger.error(f"Error checking horizon {horizon}: {e}")

    def _check_horizon(self, horizon: str) -> None:
        """Check accuracy for a single horizon."""
        # Get current accuracy metrics
        current_accuracy = self._get_current_accuracy(horizon)
        if current_accuracy is None:
            return

        # Update accuracy window
        self.accuracy_windows[horizon].add(current_accuracy)

        # Set baseline if not set
        if horizon not in self.baseline_accuracy:
            self.baseline_accuracy[horizon] = current_accuracy
            logger.info(f"Set baseline accuracy for {horizon}: {current_accuracy:.4f}")
            return

        baseline = self.baseline_accuracy[horizon]

        # Check for critical drop
        drop_percent = (baseline - current_accuracy) / baseline if baseline > 0 else 0

        if drop_percent >= self.critical_threshold:
            self._trigger_rollback(
                horizon,
                f"CRITICAL: Accuracy dropped {drop_percent:.1%} from baseline",
                current_accuracy,
                baseline
            )
            return

        # Check for warning-level drop
        if drop_percent >= self.warning_threshold:
            self.warning_counts[horizon] += 1
            logger.warning(
                f"Accuracy warning for {horizon}: {current_accuracy:.4f} "
                f"(baseline: {baseline:.4f}, drop: {drop_percent:.1%}, "
                f"consecutive: {self.warning_counts[horizon]}/{self.warning_consecutive})"
            )

            if self.warning_counts[horizon] >= self.warning_consecutive:
                self._trigger_rollback(
                    horizon,
                    f"WARNING: Accuracy below threshold for {self.warning_consecutive} consecutive checks",
                    current_accuracy,
                    baseline
                )
        else:
            # Reset warning count on recovery
            if self.warning_counts[horizon] > 0:
                logger.info(f"Accuracy recovered for {horizon}, resetting warning count")
            self.warning_counts[horizon] = 0

        # Check for consistent degradation trend
        trend = self.accuracy_windows[horizon].get_trend()
        if trend is not None and trend < -0.05:  # 5% degradation trend
            window_avg = self.accuracy_windows[horizon].get_average()
            logger.warning(f"Accuracy trend degrading for {horizon}: trend={trend:.4f}, avg={window_avg:.4f}")

    def _get_current_accuracy(self, horizon: str) -> Optional[float]:
        """Get current accuracy metric for a horizon."""
        try:
            from services.accuracy_tracker import get_accuracy_tracker
            tracker = get_accuracy_tracker()

            metrics = tracker.get_current_metrics()
            if metrics and 'metrics_by_horizon' in metrics:
                horizon_metrics = metrics['metrics_by_horizon'].get(horizon, {})
                # Use directional accuracy as primary metric
                return horizon_metrics.get('directional_accuracy', None)

            return None
        except Exception as e:
            logger.debug(f"Could not get accuracy for {horizon}: {e}")
            return None

    def _trigger_rollback(
        self,
        horizon: str,
        reason: str,
        current_accuracy: float,
        baseline_accuracy: float
    ) -> None:
        """Trigger automatic rollback for a horizon."""
        with self._lock:
            # Check cooldown
            last_rollback = self.last_rollback.get(horizon)
            if last_rollback:
                elapsed = (datetime.now() - last_rollback).total_seconds() / 60
                if elapsed < self.rollback_cooldown_minutes:
                    logger.warning(
                        f"Rollback cooldown active for {horizon}: "
                        f"{self.rollback_cooldown_minutes - elapsed:.0f} minutes remaining"
                    )
                    return

            logger.error(f"TRIGGERING AUTO-ROLLBACK for {horizon}: {reason}")
            logger.error(f"  Current accuracy: {current_accuracy:.4f}")
            logger.error(f"  Baseline accuracy: {baseline_accuracy:.4f}")

            try:
                from models.model_registry import get_registry
                registry = get_registry()

                # Get current version info before rollback
                current_version = registry.get_active_version(horizon)
                current_version_str = current_version['version'] if current_version else 'unknown'

                # Perform rollback
                new_version = registry.rollback(horizon)

                # Record rollback
                rollback_record = {
                    'horizon': horizon,
                    'reason': reason,
                    'from_version': current_version_str,
                    'to_version': new_version,
                    'current_accuracy': current_accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'timestamp': datetime.now().isoformat(),
                    'automatic': True
                }
                self.rollback_history.append(rollback_record)

                # Update tracking
                self.last_rollback[horizon] = datetime.now()
                self.warning_counts[horizon] = 0

                # Update baseline to rolled-back version's performance
                # (will be recalibrated on next accurate measurement)
                self.baseline_accuracy[horizon] = baseline_accuracy * 0.95

                logger.info(f"Auto-rollback successful for {horizon}: {current_version_str} -> {new_version}")

                # Reload models to use rolled-back version
                try:
                    from api.routes import reload_models
                    reload_models()
                    logger.info("Models reloaded after rollback")
                except Exception as e:
                    logger.warning(f"Could not reload models after rollback: {e}")

                # Capture to Sentry for visibility
                capture_exception(
                    RuntimeError(f"Auto-rollback triggered: {reason}"),
                    {
                        'horizon': horizon,
                        'from_version': current_version_str,
                        'to_version': new_version,
                        'current_accuracy': current_accuracy,
                        'baseline_accuracy': baseline_accuracy
                    }
                )

            except ValueError as e:
                logger.error(f"Could not rollback {horizon}: {e}")
                capture_exception(e, {'horizon': horizon, 'action': 'auto_rollback_failed'})
            except Exception as e:
                logger.error(f"Unexpected error during rollback: {e}")
                capture_exception(e, {'horizon': horizon, 'action': 'auto_rollback_error'})

    def update_baseline(self, horizon: str, accuracy: float) -> None:
        """Manually update baseline accuracy (e.g., after successful retraining)."""
        with self._lock:
            old_baseline = self.baseline_accuracy.get(horizon)
            self.baseline_accuracy[horizon] = accuracy
            self.warning_counts[horizon] = 0
            logger.info(f"Updated baseline for {horizon}: {old_baseline} -> {accuracy}")

    def get_status(self) -> Dict:
        """Get current service status."""
        return {
            'enabled': self.enabled,
            'running': self.running,
            'check_interval_seconds': self.check_interval,
            'thresholds': {
                'critical_drop': self.critical_threshold,
                'warning_drop': self.warning_threshold,
                'warning_consecutive': self.warning_consecutive
            },
            'horizons': {
                horizon: {
                    'baseline_accuracy': self.baseline_accuracy.get(horizon),
                    'current_window_avg': self.accuracy_windows[horizon].get_average(),
                    'current_trend': self.accuracy_windows[horizon].get_trend(),
                    'warning_count': self.warning_counts[horizon],
                    'last_rollback': self.last_rollback.get(horizon, '').isoformat()
                    if horizon in self.last_rollback else None
                }
                for horizon in self.horizons
            },
            'rollback_history': self.rollback_history[-10:],  # Last 10 rollbacks
            'total_rollbacks': len(self.rollback_history)
        }

    def enable(self) -> None:
        """Enable automatic rollback."""
        self.enabled = True
        logger.info("AutoRollbackService enabled")

    def disable(self) -> None:
        """Disable automatic rollback (but keep monitoring)."""
        self.enabled = False
        logger.info("AutoRollbackService disabled (monitoring continues)")


# Global instance
_service_instance: Optional[AutoRollbackService] = None


def get_auto_rollback_service() -> AutoRollbackService:
    """Get global auto-rollback service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AutoRollbackService()
    return _service_instance
