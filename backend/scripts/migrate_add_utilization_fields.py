#!/usr/bin/env python3
"""
Migration Script: Add gas_used, gas_limit, and utilization columns to gas_prices table

This migration adds supply-side features to the gas_prices table for Base L2 block tracking.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.logger import logger

def migrate_database():
    """Add gas_used, gas_limit, and utilization columns to gas_prices table"""
    
    # Get database path
    db_url = Config.DATABASE_URL
    if db_url.startswith('sqlite:///'):
        db_path = db_url.replace('sqlite:///', '')
        if not db_path.startswith('/'):
            # Relative path - use backend directory
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db_path)
    else:
        logger.error(f"Unsupported database URL: {db_url}. This migration is for SQLite only.")
        return False
    
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path}, skipping migration (tables will be created on first run)")
        return True
    
    logger.info(f"Migrating database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(gas_prices)")
        columns = [col[1] for col in cursor.fetchall()]
        
        migrations_applied = []
        
        # Add gas_used column
        if 'gas_used' not in columns:
            logger.info("Adding gas_used column...")
            cursor.execute("ALTER TABLE gas_prices ADD COLUMN gas_used INTEGER")
            migrations_applied.append("gas_used")
        else:
            logger.info("Column gas_used already exists")
        
        # Add gas_limit column
        if 'gas_limit' not in columns:
            logger.info("Adding gas_limit column...")
            cursor.execute("ALTER TABLE gas_prices ADD COLUMN gas_limit INTEGER")
            migrations_applied.append("gas_limit")
        else:
            logger.info("Column gas_limit already exists")
        
        # Add utilization column
        if 'utilization' not in columns:
            logger.info("Adding utilization column...")
            cursor.execute("ALTER TABLE gas_prices ADD COLUMN utilization REAL")
            migrations_applied.append("utilization")
        else:
            logger.info("Column utilization already exists")
        
        conn.commit()
        conn.close()
        
        if migrations_applied:
            logger.info(f"✅ Migration completed! Added columns: {', '.join(migrations_applied)}")
        else:
            logger.info("✅ Migration already applied - all columns exist")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Database Migration: Add Utilization Fields")
    logger.info("="*60)
    
    success = migrate_database()
    
    if success:
        logger.info("Migration script completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration script failed")
        sys.exit(1)
