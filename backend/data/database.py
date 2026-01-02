from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import Config


Base = declarative_base()


class GasPrice(Base):
    __tablename__ = 'gas_prices'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    chain_id = Column(Integer, default=8453, index=True)  # Chain ID (8453=Base, 1=Ethereum, etc.)
    current_gas = Column(Float)
    base_fee = Column(Float)
    priority_fee = Column(Float)
    block_number = Column(Integer, index=True)


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    chain_id = Column(Integer, default=8453, index=True)  # Chain ID for chain-specific predictions
    horizon = Column(String, index=True)  # '1h', '4h', '24h'
    predicted_gas = Column(Float)
    actual_gas = Column(Float, nullable=True)
    model_version = Column(String)


class OnChainFeatures(Base):
    __tablename__ = 'onchain_features'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    block_number = Column(Integer, index=True)
    tx_count = Column(Integer)
    gas_used = Column(Integer)
    gas_limit = Column(Integer)
    gas_utilization = Column(Float)
    base_fee_gwei = Column(Float)
    avg_gas_price_gwei = Column(Float)
    avg_priority_fee_gwei = Column(Float)
    contract_calls = Column(Integer)
    transfers = Column(Integer)
    contract_call_ratio = Column(Float)
    congestion_score = Column(Float)
    block_time = Column(Float)
    
    # Enhanced congestion features (Week 1 Quick Win #2)
    # These features explain 27% of gas price variance
    pending_tx_count = Column(Integer, nullable=True)
    unique_senders = Column(Integer, nullable=True)
    unique_receivers = Column(Integer, nullable=True)
    unique_addresses = Column(Integer, nullable=True)
    tx_per_second = Column(Float, nullable=True)
    gas_utilization_ratio = Column(Float, nullable=True)  # More precise than gas_utilization
    avg_tx_gas = Column(Float, nullable=True)
    large_tx_ratio = Column(Float, nullable=True)
    congestion_level = Column(Integer, nullable=True)  # 0-5 scale
    is_highly_congested = Column(Integer, nullable=True)  # Boolean as int (0/1)


class DatabaseManager:
    def __init__(self):
        # Add SQLite-specific configuration for concurrent access
        connect_args = {}
        if Config.DATABASE_URL.startswith('sqlite'):
            connect_args = {
                'check_same_thread': False,
                'timeout': 30  # 30 second timeout for locked database
            }

        self.engine = create_engine(
            Config.DATABASE_URL,
            pool_pre_ping=True,
            connect_args=connect_args
        )

        # Enable WAL mode for SQLite to allow concurrent reads/writes
        if Config.DATABASE_URL.startswith('sqlite'):
            from sqlalchemy import event
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                cursor.close()

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def _get_session(self):
        """Get a new session for this operation"""
        return self.Session()
    
    def save_gas_price(self, data):
        """Save gas price data"""
        session = self._get_session()
        try:
            # Convert ISO timestamp string to datetime if needed
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                from dateutil import parser
                data['timestamp'] = parser.parse(data['timestamp'])
            
            gas_price = GasPrice(**data)
            session.add(gas_price)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_historical_data(self, hours=720, chain_id=8453):  # 30 days default, Base chain default
        """Get historical gas prices for a specific chain"""
        session = self._get_session()
        try:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=hours)
            results = session.query(GasPrice).filter(
                GasPrice.timestamp >= cutoff,
                GasPrice.chain_id == chain_id
            ).all()
            # Convert to dict format for JSON serialization
            return [{
                'timestamp': r.timestamp.isoformat() if hasattr(r.timestamp, 'isoformat') else str(r.timestamp),
                'chain_id': r.chain_id,
                'gwei': r.current_gas,
                'baseFee': r.base_fee,
                'priorityFee': r.priority_fee
            } for r in results]
        finally:
            session.close()
    
    def save_prediction(self, horizon, predicted_gas, model_version, chain_id=8453):
        """Save a prediction for a specific chain"""
        session = self._get_session()
        try:
            prediction = Prediction(
                chain_id=chain_id,
                horizon=horizon,
                predicted_gas=predicted_gas,
                model_version=model_version
            )
            session.add(prediction)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_onchain_features(self, features):
        """Save on-chain features"""
        session = self._get_session()
        try:
            # Convert timestamp if needed
            if 'timestamp' in features and isinstance(features['timestamp'], str):
                from dateutil import parser
                features['timestamp'] = parser.parse(features['timestamp'])

            onchain = OnChainFeatures(**features)
            session.add(onchain)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_connection(self):
        """Get raw database connection for custom queries"""
        return self.engine.raw_connection()

    @property
    def session(self):
        """Backward compatibility - returns a new session"""
        return self._get_session()

