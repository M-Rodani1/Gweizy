import os
from dotenv import load_dotenv

load_dotenv()


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
    COLLECTION_INTERVAL = 5  # 5 seconds (3x faster data collection)
    # Rationale: Base gas prices can spike rapidly. 5-second sampling provides:
    # - 3x more training data compared to 15s intervals
    # - Excellent spike detection and pattern recognition
    # - Fast model convergence while staying well within API rate limits (17% daily usage)
    # Expected impact: 14 days of data in just 5 days, production-ready models faster
    
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

