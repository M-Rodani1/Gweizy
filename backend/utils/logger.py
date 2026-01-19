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


def log_error_with_context(
    error: Exception,
    operation: str,
    context: dict = None,
    level: str = 'error',
    include_traceback: bool = True
):
    """
    Log an error with comprehensive context for easy debugging.
    
    This function provides structured error logging with:
    - Clear operation description
    - Full stack trace
    - Relevant context (state, parameters, etc.)
    - Error type and message
    - Captures to Sentry if configured
    
    Args:
        error: The exception that occurred
        operation: Description of what was being attempted (e.g., "Loading model for 1h horizon")
        context: Optional dict with relevant context (e.g., {'model_path': '/path/to/model.pkl', 'horizon': '1h'})
        level: Log level ('error', 'warning', 'critical')
        include_traceback: Whether to include full stack trace (default: True)
    
    Example:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            log_error_with_context(
                e,
                "Loading model from disk",
                context={
                    'model_path': model_path,
                    'horizon': horizon,
                    'file_exists': os.path.exists(model_path),
                    'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else None
                }
            )
    """
    import traceback
    
    # Build error message header
    error_header = f"❌ ERROR in {operation}"
    log_message_parts = [error_header]
    log_message_parts.append(f"   Error Type: {type(error).__name__}")
    log_message_parts.append(f"   Error Message: {str(error)}")
    
    # Add context if provided
    if context:
        log_message_parts.append("   Context:")
        for key, value in context.items():
            # Format value for logging (handle long values)
            if isinstance(value, (str, bytes)) and len(str(value)) > 200:
                formatted_value = f"{str(value)[:200]}... (truncated, length: {len(str(value))})"
            else:
                formatted_value = str(value)
            log_message_parts.append(f"      {key}: {formatted_value}")
    
    # Add stack trace
    if include_traceback:
        log_message_parts.append("   Stack Trace:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            if line.strip():
                log_message_parts.append(f"      {line}")
    
    # Join all parts
    full_message = '\n'.join(log_message_parts)
    
    # Log at appropriate level
    if level == 'critical':
        logger.critical(full_message)
    elif level == 'warning':
        logger.warning(full_message)
    else:
        logger.error(full_message)
    
    # Capture to Sentry with context
    capture_context = {'operation': operation}
    if context:
        capture_context.update(context)
    capture_exception(error, capture_context)
    
    return full_message

