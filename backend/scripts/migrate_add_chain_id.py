"""
Database migration script to add chain_id column to gas_prices and predictions tables.
This script safely adds the chain_id column with a default value of 8453 (Base chain).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text, inspect
from config import Config
from utils.logger import logger


def migrate_database():
    """Add chain_id column to gas_prices and predictions tables if they don't exist."""
    
    # Create engine
    connect_args = {}
    if Config.DATABASE_URL.startswith('sqlite'):
        connect_args = {
            'check_same_thread': False,
            'timeout': 30
        }
    
    engine = create_engine(
        Config.DATABASE_URL,
        pool_pre_ping=True,
        connect_args=connect_args
    )
    
    # Enable WAL mode for SQLite
    if Config.DATABASE_URL.startswith('sqlite'):
        from sqlalchemy import event
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()
    
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        # Check and migrate gas_prices table
        if 'gas_prices' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('gas_prices')]
            
            if 'chain_id' not in columns:
                logger.info("Adding chain_id column to gas_prices table...")
                try:
                    # For SQLite, we need to use ALTER TABLE ADD COLUMN
                    conn.execute(text("ALTER TABLE gas_prices ADD COLUMN chain_id INTEGER DEFAULT 8453"))
                    conn.commit()
                    logger.info("✓ Successfully added chain_id column to gas_prices table")
                    
                    # Create index for better query performance
                    try:
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gas_prices_chain_id ON gas_prices(chain_id)"))
                        conn.commit()
                        logger.info("✓ Created index on gas_prices.chain_id")
                    except Exception as e:
                        logger.warning(f"Could not create index (may already exist): {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to add chain_id to gas_prices: {e}")
                    conn.rollback()
                    raise
            else:
                logger.info("chain_id column already exists in gas_prices table")
        else:
            logger.warning("gas_prices table does not exist - will be created on next model initialization")
        
        # Check and migrate predictions table
        if 'predictions' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('predictions')]
            
            if 'chain_id' not in columns:
                logger.info("Adding chain_id column to predictions table...")
                try:
                    conn.execute(text("ALTER TABLE predictions ADD COLUMN chain_id INTEGER DEFAULT 8453"))
                    conn.commit()
                    logger.info("✓ Successfully added chain_id column to predictions table")
                    
                    # Create index
                    try:
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_predictions_chain_id ON predictions(chain_id)"))
                        conn.commit()
                        logger.info("✓ Created index on predictions.chain_id")
                    except Exception as e:
                        logger.warning(f"Could not create index (may already exist): {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to add chain_id to predictions: {e}")
                    conn.rollback()
                    raise
            else:
                logger.info("chain_id column already exists in predictions table")
        else:
            logger.warning("predictions table does not exist - will be created on next model initialization")
    
    logger.info("=" * 60)
    logger.info("Database migration completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        migrate_database()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

