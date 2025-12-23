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
    
    # Data Collection
    COLLECTION_INTERVAL = 60  # 1 minute (changed from 300s for better spike capture)
    # Rationale: Gas spikes occur in seconds, not minutes. 1-min sampling captures
    # MEV events and congestion spikes that 5-min intervals miss entirely.
    # Expected impact: +0.08 R², +5% directional accuracy
    
    # Model
    MODEL_PATH = 'models/gas_predictor.pkl'
    RETRAIN_INTERVAL = 86400  # 24 hours

