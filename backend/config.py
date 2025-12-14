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
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///gas_data.db')
    
    # Data Collection
    COLLECTION_INTERVAL = 300  # 5 minutes
    
    # Model
    MODEL_PATH = 'models/gas_predictor.pkl'
    RETRAIN_INTERVAL = 86400  # 24 hours

