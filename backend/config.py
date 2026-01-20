import os
import sys
from dotenv import load_dotenv

load_dotenv()


class ConfigValidationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def validate_environment():
    """
    Validate required environment variables on startup.
    Fails fast with helpful error messages if critical config is missing.
    """
    errors = []
    warnings = []

    # Detect production environment
    is_production = (
        os.getenv('FLASK_ENV') == 'production' or
        os.getenv('RAILWAY_ENVIRONMENT') == 'production' or
        os.getenv('DEBUG', 'True').lower() == 'false'
    )

    # Check RPC URL - critical for operation
    rpc_url = os.getenv('BASE_RPC_URL', '')
    if not rpc_url or rpc_url == 'https://mainnet.base.org':
        if is_production:
            warnings.append("BASE_RPC_URL: Using public RPC endpoint. Consider a dedicated provider for reliability.")

    # Check database URL
    db_url = os.getenv('DATABASE_URL', '')
    if is_production and (not db_url or 'sqlite' in db_url.lower()):
        warnings.append("DATABASE_URL: Using SQLite in production. Consider PostgreSQL for better concurrency.")

    # Check Sentry DSN for error tracking
    sentry_dsn = os.getenv('SENTRY_DSN', '')
    if is_production and not sentry_dsn:
        warnings.append("SENTRY_DSN: Not configured. Error tracking will be disabled.")

    # Check models directory exists and is writable
    models_dir = os.getenv('MODELS_DIR', '')
    if models_dir:
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"MODELS_DIR: Cannot create directory '{models_dir}': {e}")
        elif not os.access(models_dir, os.W_OK):
            errors.append(f"MODELS_DIR: Directory '{models_dir}' is not writable.")

    # Check data directory for Railway persistence
    if is_production:
        data_dir = '/data'
        if os.path.exists(data_dir) and not os.access(data_dir, os.W_OK):
            errors.append(f"Data directory '{data_dir}' is not writable.")

    # Validate PORT is a valid number
    try:
        port = int(os.getenv('PORT', 5001))
        if port < 1 or port > 65535:
            errors.append(f"PORT: Invalid port number {port}. Must be between 1 and 65535.")
    except ValueError:
        errors.append(f"PORT: '{os.getenv('PORT')}' is not a valid port number.")

    # Log validation results
    if errors or warnings:
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION")
        print("=" * 60)

        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   • {warning}")

        if errors:
            print("\n❌ ERRORS (will prevent startup):")
            for error in errors:
                print(f"   • {error}")
            print("\n" + "=" * 60)
            raise ConfigValidationError(f"Configuration validation failed with {len(errors)} error(s)")

        print("\n" + "=" * 60 + "\n")

    return True


# Run validation on import (fail fast)
try:
    validate_environment()
except ConfigValidationError as e:
    print(f"\n❌ FATAL: {e}")
    print("Please fix the configuration errors and restart.\n")
    sys.exit(1)


class Config:
    # Flask
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    PORT = int(os.getenv('PORT', 5001))
    
    # Base Network
    BASE_RPC_URL = os.getenv('BASE_RPC_URL', 'https://mainnet.base.org')
    BASE_CHAIN_ID = 8453
    BASESCAN_API_KEY = os.getenv('BASESCAN_API_KEY', '')
    
    # APIs
    OWLRACLE_API_KEY = os.getenv('OWLRACLE_API_KEY', '')
    
    # Database
    # Use /data for persistent storage on Railway, fallback to local for development
    DATABASE_URL = os.getenv('DATABASE_URL',
                            'sqlite:////data/gas_data.db' if os.path.exists('/data')
                            else 'sqlite:///gas_data.db')
    
    # Model Storage
    # Use /data/models for persistent storage on Railway, fallback to local for development
    MODELS_DIR = os.getenv('MODELS_DIR',
                          '/data/models' if os.path.exists('/data')
                          else 'backend/models/saved_models')
    
    # Data Collection
    COLLECTION_INTERVAL = 5  # 5 seconds
    # Rationale: High-frequency collection for optimal data granularity and spike detection
    # - 720 records/hour (17,280/day, 518,400/month)
    # - Excellent spike detection and pattern recognition
    # - Fast model convergence with maximum training data
    # - Requires dedicated RPC provider (Alchemy/Infura) for reliable rate limits
    # - Database: PostgreSQL recommended for production (SQLite OK for dev)
    # Expected impact: Maximum data quality for production-ready predictions
    
    # Model
    MODEL_PATH = 'models/gas_predictor.pkl'
    RETRAIN_INTERVAL = 86400  # 24 hours
    
    # Notifications
    SMTP_SERVER = os.getenv('SMTP_SERVER', '')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USER = os.getenv('SMTP_USER', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'noreply@gweizy.com')
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')

