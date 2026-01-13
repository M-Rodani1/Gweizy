"""
Circuit Breaker Pattern Implementation for External Service Calls.

Provides fault tolerance for RPC calls, external APIs, and database operations.
Prevents cascade failures by failing fast when a service is unhealthy.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is unhealthy, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed

Usage:
    from utils.circuit_breaker import circuit_breaker, CircuitBreaker

    # As decorator
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def fetch_from_rpc():
        return web3.eth.get_block('latest')

    # As context manager
    cb = CircuitBreaker('rpc_service', failure_threshold=5)
    with cb:
        result = risky_external_call()
"""

import time
import threading
from enum import Enum
from typing import Callable, Optional, Any, Dict
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from utils.logger import logger, capture_exception


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker instance."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascade failures.

    When failures exceed the threshold, the circuit opens and fails fast
    for a recovery period before attempting to test if the service recovered.

    Args:
        name: Identifier for this circuit breaker
        failure_threshold: Number of consecutive failures to open circuit
        recovery_timeout: Seconds to wait before testing recovery
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Successes needed to close circuit from half-open
        excluded_exceptions: Exception types that don't count as failures
    """

    # Registry of all circuit breakers for monitoring
    _registry: Dict[str, 'CircuitBreaker'] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        excluded_exceptions: tuple = (),
        fallback: Optional[Callable] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.excluded_exceptions = excluded_exceptions
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = datetime.now()
        self._half_open_calls = 0
        self._lock = threading.Lock()

        # Register this circuit breaker
        with CircuitBreaker._lock:
            CircuitBreaker._registry[name] = self

        logger.info(f"Circuit breaker '{name}' initialized: threshold={failure_threshold}, timeout={recovery_timeout}s")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        return self._stats

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        elapsed = (datetime.now() - self._last_state_change).total_seconds()
        return elapsed >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()
        self._stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.warning(
            f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
        )

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(f"Circuit breaker '{self.name}' recovered - closing circuit")

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.error(
                        f"Circuit breaker '{self.name}' opened after {self.failure_threshold} failures. "
                        f"Last error: {exception}"
                    )
                    capture_exception(exception, {
                        'circuit_breaker': self.name,
                        'consecutive_failures': self._stats.consecutive_failures
                    })

            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"Circuit breaker '{self.name}' reopened - service still failing")

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        current_state = self.state  # This may trigger state transitions

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            return False

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function's return value

        Raises:
            CircuitOpenError: If the circuit is open
            The original exception: If the function fails
        """
        if not self._can_execute():
            if self.fallback:
                logger.debug(f"Circuit '{self.name}' open - using fallback")
                return self.fallback(*args, **kwargs)
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Will retry in {self.recovery_timeout}s"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise

        except Exception as e:
            self._record_failure(e)
            if self.fallback and self._state == CircuitState.OPEN:
                logger.debug(f"Circuit '{self.name}' using fallback after failure")
                return self.fallback(*args, **kwargs)
            raise

    def __enter__(self):
        """Context manager entry."""
        if not self._can_execute():
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self._record_success()
        elif exc_type not in self.excluded_exceptions:
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status for monitoring."""
        return {
            'name': self.name,
            'state': self.state.value,
            'stats': {
                'total_calls': self._stats.total_calls,
                'successful_calls': self._stats.successful_calls,
                'failed_calls': self._stats.failed_calls,
                'rejected_calls': self._stats.rejected_calls,
                'consecutive_failures': self._stats.consecutive_failures,
                'consecutive_successes': self._stats.consecutive_successes,
                'state_changes': self._stats.state_changes,
                'last_failure': self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
                'last_success': self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            },
            'config': {
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'half_open_max_calls': self.half_open_max_calls,
                'success_threshold': self.success_threshold,
            },
            'last_state_change': self._last_state_change.isoformat(),
        }

    @classmethod
    def get_all_status(cls) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered circuit breakers."""
        with cls._lock:
            return {name: cb.get_status() for name, cb in cls._registry.items()}


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
    excluded_exceptions: tuple = (),
    fallback: Optional[Callable] = None
) -> Callable:
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        half_open_max_calls: Calls allowed in half-open state
        success_threshold: Successes needed to close from half-open
        excluded_exceptions: Exceptions that don't count as failures
        fallback: Function to call when circuit is open

    Returns:
        Decorated function

    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def call_external_api():
            return requests.get('https://api.example.com/data')
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or f"{func.__module__}.{func.__name__}"
        cb = CircuitBreaker(
            name=cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold,
            excluded_exceptions=excluded_exceptions,
            fallback=fallback
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        # Attach circuit breaker instance for testing/monitoring
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator


# Pre-configured circuit breakers for common services
rpc_circuit = CircuitBreaker(
    name='rpc_provider',
    failure_threshold=5,
    recovery_timeout=30,
    half_open_max_calls=2
)

owlracle_circuit = CircuitBreaker(
    name='owlracle_api',
    failure_threshold=3,
    recovery_timeout=60,
    half_open_max_calls=1
)

database_circuit = CircuitBreaker(
    name='database',
    failure_threshold=3,
    recovery_timeout=15,
    half_open_max_calls=2
)
