#!/usr/bin/env python3
"""
Database Migration Script: Add Enhanced Congestion Features

This script adds the new enhanced congestion feature columns to existing
onchain_features tables. Safe to run multiple times (idempotent).

Week 1 Quick Win #2: Enhanced Congestion Features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from data.database import DatabaseManager
from config import Config


def migrate_database():
    """Add enhanced congestion feature columns to onchain_features table"""
    db = DatabaseManager()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    print("="*60)
    print("Database Migration: Enhanced Congestion Features")
    print("="*60)
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(onchain_features)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    new_columns = {
        'pending_tx_count': 'INTEGER',
        'unique_senders': 'INTEGER',
        'unique_receivers': 'INTEGER',
        'unique_addresses': 'INTEGER',
        'tx_per_second': 'REAL',
        'gas_utilization_ratio': 'REAL',
        'avg_tx_gas': 'REAL',
        'large_tx_ratio': 'REAL',
        'congestion_level': 'INTEGER',
        'is_highly_congested': 'INTEGER',
    }
    
    added_count = 0
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            try:
                # SQLite ALTER TABLE ADD COLUMN
                alter_sql = f"ALTER TABLE onchain_features ADD COLUMN {col_name} {col_type}"
                cursor.execute(alter_sql)
                conn.commit()
                print(f"✅ Added column: {col_name}")
                added_count += 1
            except Exception as e:
                print(f"⚠️  Could not add {col_name}: {e}")
        else:
            print(f"⏭️  Column {col_name} already exists, skipping")
    
    conn.close()
    
    print("="*60)
    if added_count > 0:
        print(f"✅ Migration complete! Added {added_count} new columns.")
    else:
        print("✅ Migration complete! All columns already exist.")
    print("="*60)
    
    return added_count


if __name__ == '__main__':
    try:
        migrate_database()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
