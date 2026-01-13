import logging
import sys
from datetime import datetime
import os


def setup_logger(name='base_gas_api'):
    """Configure application logger"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if os.getenv('DEBUG') == 'True' else logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')
    )
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def capture_exception(error: Exception, extra_context: dict = None):
    """
    Explicitly capture an exception to Sentry.

    Use this in catch blocks where you handle the error gracefully
    but still want it reported to Sentry.

    Args:
        error: The exception to capture
        extra_context: Optional dict of additional context data

    Example:
        try:
            risky_operation()
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            capture_exception(e, {'user_id': user_id})
    """
    try:
        import sentry_sdk
        if extra_context:
            with sentry_sdk.push_scope() as scope:
                for key, value in extra_context.items():
                    scope.set_extra(key, value)
                sentry_sdk.capture_exception(error)
        else:
            sentry_sdk.capture_exception(error)
    except ImportError:
        # Sentry not configured, just log
        logger.debug("Sentry not available for exception capture")
    except Exception as capture_error:
        logger.debug(f"Failed to capture exception to Sentry: {capture_error}")

