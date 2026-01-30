"""
Safe Model Loading Utility

Provides secure model loading with path validation to mitigate pickle deserialization risks.
Pickle files can execute arbitrary code during unpickling, so this module:
1. Validates file paths are within allowed directories
2. Prefers joblib over pickle where possible
3. Logs all loading attempts for audit
4. Rejects paths with suspicious patterns (path traversal, etc.)
"""

import os
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib

from config import Config
from utils.logger import logger


# Allowed directories for model loading (relative to project root)
# Only files within these directories can be loaded
ALLOWED_MODEL_DIRECTORIES = [
    '/data/models',                    # Railway persistent storage
    '/data',                           # Railway data directory
    'models/saved_models',             # Local development
    'backend/models/saved_models',     # Alternative local path
]


class UnsafePathError(Exception):
    """Raised when attempting to load from an unauthorized path."""
    pass


def get_allowed_paths() -> List[str]:
    """
    Get list of allowed model directories, resolved to absolute paths.

    Returns:
        List of absolute paths where models can be loaded from.
    """
    allowed = []

    # Add configured models directory
    if hasattr(Config, 'MODELS_DIR') and Config.MODELS_DIR:
        allowed.append(os.path.abspath(Config.MODELS_DIR))

    # Add standard directories
    for dir_path in ALLOWED_MODEL_DIRECTORIES:
        if os.path.isabs(dir_path):
            allowed.append(dir_path)
        else:
            # Make relative paths absolute
            allowed.append(os.path.abspath(dir_path))

    return list(set(allowed))  # Remove duplicates


def is_path_allowed(file_path: str) -> Tuple[bool, str]:
    """
    Check if a file path is within allowed directories.

    Args:
        file_path: Path to validate.

    Returns:
        Tuple of (is_allowed, reason).
    """
    # Resolve to absolute path, following symlinks
    try:
        resolved_path = os.path.realpath(file_path)
    except (OSError, ValueError) as e:
        return False, f"Invalid path: {e}"

    # Check for path traversal attempts
    if '..' in file_path:
        return False, "Path contains '..' traversal"

    # Check file extension
    valid_extensions = {'.pkl', '.pickle', '.joblib'}
    if not any(resolved_path.endswith(ext) for ext in valid_extensions):
        return False, f"Invalid file extension. Allowed: {valid_extensions}"

    # Check against allowed directories
    allowed_dirs = get_allowed_paths()

    for allowed_dir in allowed_dirs:
        try:
            resolved_allowed = os.path.realpath(allowed_dir)
            if resolved_path.startswith(resolved_allowed + os.sep) or resolved_path.startswith(resolved_allowed):
                return True, f"Path is within allowed directory: {allowed_dir}"
        except (OSError, ValueError):
            continue

    return False, f"Path not in allowed directories: {allowed_dirs}"


def safe_load(
    file_path: str,
    prefer_joblib: bool = True,
    validate_path: bool = True
) -> Any:
    """
    Safely load a model from a pickle/joblib file with path validation.

    Args:
        file_path: Path to the model file.
        prefer_joblib: Whether to try joblib first (default: True).
        validate_path: Whether to validate the path is allowed (default: True).

    Returns:
        The loaded model object.

    Raises:
        UnsafePathError: If path validation fails.
        FileNotFoundError: If file doesn't exist.
        Exception: If loading fails.
    """
    # Validate path if enabled
    if validate_path:
        is_allowed, reason = is_path_allowed(file_path)
        if not is_allowed:
            logger.error(f"Blocked model load from unauthorized path: {file_path} - {reason}")
            raise UnsafePathError(f"Cannot load model from {file_path}: {reason}")

    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    # Log loading attempt
    logger.info(f"Loading model from: {file_path}")

    # Try joblib first (preferred for sklearn models)
    if prefer_joblib:
        try:
            model = joblib.load(file_path)
            logger.debug(f"Successfully loaded with joblib: {file_path}")
            return model
        except Exception as joblib_err:
            logger.debug(f"joblib.load failed, trying pickle: {joblib_err}")

    # Fall back to pickle
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Successfully loaded with pickle: {file_path}")
        return model
    except Exception as pickle_err:
        logger.error(f"Failed to load model from {file_path}: {pickle_err}")
        raise


def safe_load_with_fallback(
    primary_path: str,
    fallback_paths: Optional[List[str]] = None,
    prefer_joblib: bool = True,
    validate_path: bool = True
) -> Tuple[Any, str]:
    """
    Try loading from primary path, then fallback paths if it fails.

    Args:
        primary_path: First path to try.
        fallback_paths: Alternative paths to try if primary fails.
        prefer_joblib: Whether to try joblib first.
        validate_path: Whether to validate paths.

    Returns:
        Tuple of (loaded_model, path_used).

    Raises:
        Exception: If all paths fail.
    """
    paths_to_try = [primary_path] + (fallback_paths or [])
    last_error = None

    for path in paths_to_try:
        if not path or not os.path.exists(path):
            continue
        try:
            model = safe_load(path, prefer_joblib=prefer_joblib, validate_path=validate_path)
            return model, path
        except UnsafePathError:
            raise  # Don't try fallbacks for security errors
        except Exception as e:
            last_error = e
            logger.debug(f"Failed to load from {path}: {e}")
            continue

    raise last_error or FileNotFoundError(f"No model found at any of: {paths_to_try}")
