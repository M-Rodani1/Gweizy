from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
from datetime import datetime
from typing import Dict, List, Any, Optional
from config import Config
from utils.logger import logger
import threading
import time
import os


# Connection pool configuration for production reliability
POOL_CONFIG = {
    # Number of connections to keep open
    'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
    # Max additional connections when pool is exhausted
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '10')),
    # Seconds to wait for a connection from pool before timeout
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
    # Seconds before recycling a connection (prevents stale connections)
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '1800')),  # 30 minutes
}

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


class UserTransaction(Base):
    """User transaction history for personalization"""
    __tablename__ = 'user_transactions'

    id = Column(Integer, primary_key=True)
    user_address = Column(String, index=True)  # Wallet address
    chain_id = Column(Integer, default=8453, index=True)
    tx_hash = Column(String, unique=True, index=True)
    block_number = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    gas_price_gwei = Column(Float)
    gas_used = Column(Integer)
    gas_cost_eth = Column(Float)  # Total cost in ETH
    tx_type = Column(String)  # 'swap', 'transfer', 'contract_call', etc.
    status = Column(String)  # 'success', 'failed'
    saved_by_waiting = Column(Float, nullable=True)  # Potential savings if waited
    optimal_time = Column(DateTime, nullable=True)  # When would have been optimal


class MempoolSnapshotRecord(Base):
    """Mempool snapshot data for leading indicator analysis."""
    __tablename__ = 'mempool_snapshots'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    block_number = Column(Integer, index=True)
    pending_count = Column(Integer)
    total_gas = Column(Integer)
    large_tx_count = Column(Integer)
    avg_gas_price = Column(Float)
    median_gas_price = Column(Float)
    p90_gas_price = Column(Float)
    tx_arrival_rate = Column(Float)
    is_congested = Column(Integer)  # Boolean as int
    count_momentum = Column(Float, nullable=True)
    gas_price_momentum = Column(Float, nullable=True)


class DatabaseManager:
    def __init__(self):
        # Ensure database directory exists for SQLite
        if Config.DATABASE_URL.startswith('sqlite'):
            db_path = Config.DATABASE_URL.replace('sqlite:///', '')
            if db_path.startswith('/'):
                # Absolute path - ensure directory exists
                import os
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                        logger.info(f"Created database directory: {db_dir}")
                    except Exception as e:
                        logger.error(f"Could not create database directory {db_dir}: {e}")
                        # Fallback to local database
                        db_path = 'gas_data.db'
                        Config.DATABASE_URL = f'sqlite:///{db_path}'
                        logger.warning(f"Falling back to local database: {db_path}")
                elif db_dir and os.path.exists(db_dir):
                    # Check if directory is writable
                    if not os.access(db_dir, os.W_OK):
                        logger.error(f"Database directory not writable: {db_dir}")
                        # Fallback to local database
                        db_path = 'gas_data.db'
                        Config.DATABASE_URL = f'sqlite:///{db_path}'
                        logger.warning(f"Falling back to local database: {db_path}")
        
        # Add SQLite-specific configuration for concurrent access
        connect_args = {}
        engine_kwargs = {
            'pool_pre_ping': True,  # Verify connections before use
        }

        if Config.DATABASE_URL.startswith('sqlite'):
            connect_args = {
                'check_same_thread': False,
                'timeout': 30  # 30 second timeout for locked database
            }
            # SQLite uses NullPool or StaticPool, not QueuePool
            engine_kwargs['poolclass'] = NullPool
        else:
            # Production database (PostgreSQL, MySQL, etc.) - use connection pooling
            engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': POOL_CONFIG['pool_size'],
                'max_overflow': POOL_CONFIG['max_overflow'],
                'pool_timeout': POOL_CONFIG['pool_timeout'],
                'pool_recycle': POOL_CONFIG['pool_recycle'],
            })
            logger.info(f"Database pool configured: size={POOL_CONFIG['pool_size']}, "
                       f"max_overflow={POOL_CONFIG['max_overflow']}, "
                       f"timeout={POOL_CONFIG['pool_timeout']}s, "
                       f"recycle={POOL_CONFIG['pool_recycle']}s")

        self.engine = create_engine(
            Config.DATABASE_URL,
            connect_args=connect_args,
            **engine_kwargs
        )

        # Enable WAL mode for SQLite to allow concurrent reads/writes
        if Config.DATABASE_URL.startswith('sqlite'):
            from sqlalchemy import event
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                try:
                    # Try to enable WAL mode (may fail on some filesystems/volumes)
                    cursor.execute("PRAGMA journal_mode=WAL")
                    result = cursor.fetchone()
                    if result and result[0] != 'wal':
                        # WAL mode not supported, fallback to DELETE mode
                        logger.warning(f"WAL mode not supported, using {result[0]} mode")
                except Exception as e:
                    # If WAL fails, continue with default journal mode
                    logger.warning(f"Could not set WAL mode: {e}, using default journal mode")
                
                try:
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
                except Exception as e:
                    logger.warning(f"Could not set SQLite pragmas: {e}")
                finally:
                    cursor.close()

        # Try to create tables with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                Base.metadata.create_all(self.engine)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    logger.warning(f"Database initialization failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to initialize database after {max_retries} attempts: {e}")
                    raise
        
        # Run migration to add chain_id column if it doesn't exist
        try:
            self._migrate_add_chain_id()
        except Exception as e:
            logger.warning(f"Migration failed (non-critical): {e}")
        
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
    
    def get_historical_data(self, hours=720, chain_id=8453, limit: Optional[int] = None, order: str = 'asc'):
        """Get historical gas prices for a specific chain."""
        from sqlalchemy import or_
        session = self._get_session()
        try:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=hours)
            query = session.query(
                GasPrice.timestamp,
                GasPrice.chain_id,
                GasPrice.current_gas,
                GasPrice.base_fee,
                GasPrice.priority_fee
            ).filter(
                GasPrice.timestamp >= cutoff,
                # Include records with matching chain_id OR NULL chain_id (legacy Base data)
                or_(GasPrice.chain_id == chain_id, GasPrice.chain_id.is_(None)) if chain_id == 8453 else GasPrice.chain_id == chain_id
            )
            if order == 'desc':
                query = query.order_by(GasPrice.timestamp.desc())
            else:
                query = query.order_by(GasPrice.timestamp.asc())
            if limit:
                query = query.limit(limit)

            results = query.all()
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

    def save_user_transaction(self, transaction_data: Dict):
        """Save user transaction for personalization"""
        session = self._get_session()
        try:
            # Check if transaction already exists
            existing = session.query(UserTransaction).filter(
                UserTransaction.tx_hash == transaction_data.get('tx_hash')
            ).first()
            
            if existing:
                return  # Already exists
            
            transaction = UserTransaction(**transaction_data)
            session.add(transaction)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Failed to save user transaction: {e}")
        finally:
            session.close()
    
    def get_user_transactions(self, user_address: str, chain_id: int = 8453, limit: int = 100, only_success: bool = False) -> List[Dict]:
        """Get user transaction history"""
        session = self._get_session()
        try:
            query = session.query(
                UserTransaction.tx_hash,
                UserTransaction.block_number,
                UserTransaction.timestamp,
                UserTransaction.gas_price_gwei,
                UserTransaction.gas_used,
                UserTransaction.gas_cost_eth,
                UserTransaction.tx_type,
                UserTransaction.status,
                UserTransaction.saved_by_waiting,
                UserTransaction.optimal_time
            ).filter(
                UserTransaction.user_address == user_address.lower(),
                UserTransaction.chain_id == chain_id
            )
            if only_success:
                query = query.filter(UserTransaction.status == 'success')
            transactions = query.order_by(UserTransaction.timestamp.desc()).limit(limit).all()
            
            return [{
                'tx_hash': t.tx_hash,
                'block_number': t.block_number,
                'timestamp': t.timestamp.isoformat() if hasattr(t.timestamp, 'isoformat') else str(t.timestamp),
                'gas_price_gwei': t.gas_price_gwei,
                'gas_used': t.gas_used,
                'gas_cost_eth': t.gas_cost_eth,
                'tx_type': t.tx_type,
                'status': t.status,
                'saved_by_waiting': t.saved_by_waiting,
                'optimal_time': t.optimal_time.isoformat() if t.optimal_time and hasattr(t.optimal_time, 'isoformat') else None
            } for t in transactions]
        finally:
            session.close()
    
    def get_user_savings_stats(self, user_address: str, chain_id: int = 8453) -> Dict:
        """Get user savings statistics"""
        session = self._get_session()
        try:
            totals = session.query(
                func.count(UserTransaction.id),
                func.coalesce(func.sum(UserTransaction.gas_cost_eth), 0),
                func.coalesce(func.sum(UserTransaction.saved_by_waiting), 0),
                func.coalesce(func.avg(UserTransaction.gas_price_gwei), 0)
            ).filter(
                UserTransaction.user_address == user_address.lower(),
                UserTransaction.chain_id == chain_id,
                UserTransaction.status == 'success'
            ).one()

            total_transactions = int(totals[0] or 0)
            total_gas_paid = float(totals[1] or 0)
            potential_savings = float(totals[2] or 0)
            avg_gas_price = float(totals[3] or 0)

            if total_transactions == 0:
                return {
                    'total_transactions': 0,
                    'total_gas_paid': 0,
                    'potential_savings': 0,
                    'savings_percentage': 0,
                    'avg_gas_price': 0
                }

            return {
                'total_transactions': total_transactions,
                'total_gas_paid': total_gas_paid,
                'potential_savings': potential_savings,
                'savings_percentage': (potential_savings / total_gas_paid * 100) if total_gas_paid > 0 else 0,
                'avg_gas_price': avg_gas_price
            }
        finally:
            session.close()
    
    def _migrate_add_chain_id(self):
        """Migrate database to add chain_id column if it doesn't exist."""
        from sqlalchemy import inspect, text
        
        inspector = inspect(self.engine)
        
        with self.engine.connect() as conn:
            # Migrate gas_prices table
            if 'gas_prices' in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns('gas_prices')]
                
                if 'chain_id' not in columns:
                    try:
                        conn.execute(text("ALTER TABLE gas_prices ADD COLUMN chain_id INTEGER DEFAULT 8453"))
                        conn.commit()
                        logger.info("✓ Added chain_id column to gas_prices table")
                        
                        # Create index
                        try:
                            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gas_prices_chain_id ON gas_prices(chain_id)"))
                            conn.commit()
                        except:
                            pass  # Index may already exist
                    except Exception as e:
                        logger.warning(f"Could not add chain_id to gas_prices (may already exist): {e}")
                        conn.rollback()
                
                # Fix NULL chain_id values - set them to Base (8453)
                try:
                    result = conn.execute(text("UPDATE gas_prices SET chain_id = 8453 WHERE chain_id IS NULL"))
                    conn.commit()
                    if result.rowcount > 0:
                        logger.info(f"✓ Updated {result.rowcount} gas_prices records with NULL chain_id to 8453 (Base)")
                except Exception as e:
                    logger.warning(f"Could not update NULL chain_id values: {e}")
                    try:
                        conn.rollback()
                    except:
                        pass
            
            # Migrate predictions table
            if 'predictions' in inspector.get_table_names():
                columns = [col['name'] for col in inspector.get_columns('predictions')]
                
                if 'chain_id' not in columns:
                    try:
                        conn.execute(text("ALTER TABLE predictions ADD COLUMN chain_id INTEGER DEFAULT 8453"))
                        conn.commit()
                        logger.info("✓ Added chain_id column to predictions table")
                        
                        # Create index
                        try:
                            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_chain_id ON predictions(chain_id)"))
                            conn.commit()
                        except:
                            pass  # Index may already exist
                    except Exception as e:
                        logger.warning(f"Could not add chain_id to predictions (may already exist): {e}")
                        conn.rollback()
                
                # Fix NULL chain_id values in predictions
                try:
                    result = conn.execute(text("UPDATE predictions SET chain_id = 8453 WHERE chain_id IS NULL"))
                    conn.commit()
                    if result.rowcount > 0:
                        logger.info(f"✓ Updated {result.rowcount} predictions records with NULL chain_id to 8453 (Base)")
                except Exception as e:
                    logger.warning(f"Could not update NULL chain_id values in predictions: {e}")
                    try:
                        conn.rollback()
                    except:
                        pass

    def get_connection(self):
        """Get raw database connection for custom queries"""
        return self.engine.raw_connection()

    @property
    def session(self):
        """Backward compatibility - returns a new session"""
        return self._get_session()

    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status for monitoring."""
        pool = self.engine.pool

        # Basic pool info
        status = {
            'pool_class': pool.__class__.__name__,
            'database_url_type': 'sqlite' if Config.DATABASE_URL.startswith('sqlite') else 'other',
            'timestamp': datetime.now().isoformat()
        }

        # QueuePool-specific metrics (not applicable to SQLite's NullPool/StaticPool)
        if hasattr(pool, 'size'):
            status['pool_size'] = pool.size()
        if hasattr(pool, 'checkedin'):
            status['checked_in'] = pool.checkedin()
        if hasattr(pool, 'checkedout'):
            status['checked_out'] = pool.checkedout()
        if hasattr(pool, 'overflow'):
            status['overflow'] = pool.overflow()

        # Calculate utilization
        if 'pool_size' in status and status['pool_size'] > 0:
            checked_out = status.get('checked_out', 0)
            overflow = status.get('overflow', 0)
            total_capacity = status['pool_size'] + 10  # Default max_overflow
            status['utilization_percent'] = round((checked_out + overflow) / total_capacity * 100, 2)
            status['is_saturated'] = status['utilization_percent'] > 80
        else:
            status['utilization_percent'] = 0
            status['is_saturated'] = False

        return status

    def get_health_check(self) -> Dict[str, Any]:
        """Perform database health check with timing."""
        start_time = time.time()
        health = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Try to execute a simple query
            session = self._get_session()
            try:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
                session.commit()
                health['status'] = 'healthy'
            finally:
                session.close()

            # Get recent record counts
            session = self._get_session()
            try:
                from datetime import timedelta
                one_hour_ago = datetime.now() - timedelta(hours=1)

                gas_count = session.query(GasPrice).filter(
                    GasPrice.timestamp >= one_hour_ago
                ).count()

                onchain_count = session.query(OnChainFeatures).filter(
                    OnChainFeatures.timestamp >= one_hour_ago
                ).count()

                health['recent_gas_records'] = gas_count
                health['recent_onchain_records'] = onchain_count
                health['data_collection_active'] = gas_count > 0 or onchain_count > 0
            finally:
                session.close()

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Database health check failed: {e}")

        health['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
        health['pool_status'] = self.get_pool_status()

        return health
