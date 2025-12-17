from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import Config


Base = declarative_base()


class GasPrice(Base):
    __tablename__ = 'gas_prices'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    current_gas = Column(Float)
    base_fee = Column(Float)
    priority_fee = Column(Float)
    block_number = Column(Integer)


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    horizon = Column(String)  # '1h', '4h', '24h'
    predicted_gas = Column(Float)
    actual_gas = Column(Float, nullable=True)
    model_version = Column(String)


class OnChainFeatures(Base):
    __tablename__ = 'onchain_features'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    block_number = Column(Integer)
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


class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
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
    
    def get_historical_data(self, hours=720):  # 30 days default
        """Get historical gas prices"""
        session = self._get_session()
        try:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=hours)
            results = session.query(GasPrice).filter(
                GasPrice.timestamp >= cutoff
            ).all()
            # Convert to dict format for JSON serialization
            return [{
                'timestamp': r.timestamp.isoformat() if hasattr(r.timestamp, 'isoformat') else str(r.timestamp),
                'gwei': r.current_gas,
                'baseFee': r.base_fee,
                'priorityFee': r.priority_fee
            } for r in results]
        finally:
            session.close()
    
    def save_prediction(self, horizon, predicted_gas, model_version):
        """Save a prediction"""
        session = self._get_session()
        try:
            prediction = Prediction(
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

