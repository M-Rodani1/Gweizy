#!/usr/bin/env python3
"""
Migrate Spike Detector Models to Joblib Format

This script converts spike detector models from pickle format to joblib format.
It handles the backwards compatibility issue where older models were saved with
pickle.dump() but the loading code now uses joblib.load().

Run this script once to migrate existing models, or it will be called
automatically on application startup.
"""

import os
import sys
import pickle
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model file names
SPIKE_DETECTOR_FILES = [
    'spike_detector_1h.pkl',
    'spike_detector_4h.pkl',
    'spike_detector_24h.pkl'
]


def get_model_directories():
    """Get all possible model directories in priority order."""
    dirs = []

    # Priority 1: Persistent storage (Railway)
    if os.path.exists('/data/models'):
        dirs.append('/data/models')

    # Priority 2: Local development paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)

    dirs.extend([
        os.path.join(backend_dir, 'models', 'saved_models'),
        'models/saved_models',
        'backend/models/saved_models'
    ])

    return dirs


def is_joblib_format(filepath):
    """Check if a file is in joblib format by reading its magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            # Joblib files typically start with specific bytes
            # We'll try loading with joblib and see if it works without errors
            pass

        # Try loading with joblib
        try:
            data = joblib.load(filepath)
            # If it loads successfully and returns a dict with expected keys, it's joblib
            if isinstance(data, dict) and 'model' in data:
                return True
            return True  # Loaded successfully
        except (KeyError, ValueError, pickle.UnpicklingError):
            return False
    except Exception:
        return False


def migrate_model(filepath):
    """
    Migrate a single model file from pickle to joblib format.

    Returns:
        tuple: (success: bool, message: str)
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    filename = os.path.basename(filepath)
    backup_path = filepath + '.pickle_backup'

    try:
        # First, try loading with joblib (already correct format)
        try:
            data = joblib.load(filepath)
            if isinstance(data, dict) and 'model' in data:
                return True, f"{filename}: Already in joblib format"
        except (KeyError, ValueError) as e:
            logger.debug(f"joblib.load failed for {filename}, trying pickle: {e}")

        # Try loading with pickle (old format)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            return False, f"{filename}: Failed to load with pickle: {e}"

        # Validate the loaded data
        if not isinstance(data, dict):
            return False, f"{filename}: Loaded data is not a dict: {type(data)}"

        if 'model' not in data:
            return False, f"{filename}: Missing 'model' key in data"

        # Create backup of original file
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Re-save with joblib
        joblib.dump(data, filepath)

        # Verify the new file loads correctly
        verify_data = joblib.load(filepath)
        if 'model' not in verify_data:
            raise ValueError("Verification failed: model key not found after migration")

        return True, f"{filename}: Successfully migrated to joblib format"

    except Exception as e:
        # Restore from backup if migration failed
        if os.path.exists(backup_path):
            import shutil
            shutil.copy2(backup_path, filepath)
            logger.warning(f"Restored {filename} from backup after failed migration")

        return False, f"{filename}: Migration failed: {e}"


def migrate_all_models():
    """
    Migrate all spike detector models to joblib format.

    Returns:
        dict: Summary of migration results
    """
    results = {
        'migrated': [],
        'already_correct': [],
        'failed': [],
        'not_found': []
    }

    model_dirs = get_model_directories()
    processed_files = set()

    logger.info("=" * 60)
    logger.info("Migrating Spike Detector Models to Joblib Format")
    logger.info("=" * 60)

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue

        logger.info(f"\nChecking directory: {model_dir}")

        for filename in SPIKE_DETECTOR_FILES:
            filepath = os.path.join(model_dir, filename)

            # Skip if already processed (from higher priority directory)
            if filepath in processed_files:
                continue

            if not os.path.exists(filepath):
                continue

            processed_files.add(filepath)

            success, message = migrate_model(filepath)
            logger.info(f"  {message}")

            if success:
                if "Already in joblib" in message:
                    results['already_correct'].append(filepath)
                else:
                    results['migrated'].append(filepath)
            else:
                results['failed'].append((filepath, message))

    # Check for files not found anywhere
    for filename in SPIKE_DETECTOR_FILES:
        found = any(
            os.path.exists(os.path.join(d, filename))
            for d in model_dirs if os.path.exists(d)
        )
        if not found:
            results['not_found'].append(filename)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"  Migrated:        {len(results['migrated'])}")
    logger.info(f"  Already correct: {len(results['already_correct'])}")
    logger.info(f"  Failed:          {len(results['failed'])}")
    logger.info(f"  Not found:       {len(results['not_found'])}")

    if results['failed']:
        logger.warning("\nFailed migrations:")
        for filepath, message in results['failed']:
            logger.warning(f"  - {message}")

    if results['not_found']:
        logger.info("\nModels not found (will use fallback):")
        for filename in results['not_found']:
            logger.info(f"  - {filename}")

    return results


def run_migration_if_needed():
    """
    Run migration only if needed (called from app startup).
    Returns True if migration was successful or not needed.
    """
    model_dirs = get_model_directories()
    needs_migration = False

    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue

        for filename in SPIKE_DETECTOR_FILES:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                # Check if it needs migration
                try:
                    joblib.load(filepath)
                except (KeyError, ValueError, pickle.UnpicklingError):
                    needs_migration = True
                    break

        if needs_migration:
            break

    if needs_migration:
        logger.info("Detected models in pickle format, running migration...")
        results = migrate_all_models()
        return len(results['failed']) == 0
    else:
        logger.debug("All models already in correct format or not present")
        return True


if __name__ == '__main__':
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = migrate_all_models()

    # Exit with error code if any migrations failed
    if results['failed']:
        sys.exit(1)
    else:
        sys.exit(0)
