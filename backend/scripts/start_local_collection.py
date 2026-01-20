#!/usr/bin/env python3
"""
Start Local Data Collection

This script starts collecting gas price data and saves it to the local database file.
The database will be created at backend/gas_data.db (or configured DATABASE_URL).

Usage:
    python scripts/start_local_collection.py [--interval SECONDS]

Options:
    --interval SECONDS    Collection interval in seconds (default: 300 = 5 minutes)
                          Common values:
                            - 60 seconds (1 minute) - Fast collection
                            - 300 seconds (5 minutes) - Default, balanced
                            - 600 seconds (10 minutes) - Slower, less data

Examples:
    # Collect every 1 minute
    python scripts/start_local_collection.py --interval 60
    
    # Collect every 30 seconds (very fast)
    python scripts/start_local_collection.py --interval 30
    
    # Collect every 5 minutes (default)
    python scripts/start_local_collection.py

The script will:
1. Create/connect to local database
2. Start collecting gas prices at specified interval
3. Save data to backend/gas_data.db
4. Run until you press Ctrl+C
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.gas_collector_service import GasCollectorService
from data.database import DatabaseManager
from config import Config

# Global service reference for signal handling
collector_service = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down data collection...")
    if collector_service:
        collector_service.stop()
    print("✅ Collection stopped. Data saved to database.")
    sys.exit(0)

def main():
    global collector_service
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Start local gas price data collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect every 1 minute
  python scripts/start_local_collection.py --interval 60
  
  # Collect every 30 seconds (very fast)
  python scripts/start_local_collection.py --interval 30
  
  # Collect every 5 minutes (default)
  python scripts/start_local_collection.py
        """
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Collection interval in seconds (default: 300 = 5 minutes)'
    )
    args = parser.parse_args()
    
    # Validate interval
    if args.interval < 5:
        print("⚠️  WARNING: Interval too short (<5 seconds) may cause rate limiting!")
        print("   Recommended minimum: 10-30 seconds")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return
    
    print("="*70)
    print("🚀 Starting Local Gas Price Data Collection")
    print("="*70)
    print()
    
    # Ensure we're using local database (not Railway /data path)
    if os.path.exists('/data'):
        print("⚠️  WARNING: /data directory exists - this might be Railway!")
        print("   Setting DATABASE_URL to local path...")
        os.environ['DATABASE_URL'] = 'sqlite:///gas_data.db'
        # Reload config
        from importlib import reload
        import config
        reload(config)
        from config import Config
    
    # Check database configuration
    print(f"📊 Database Configuration:")
    print(f"   DATABASE_URL: {Config.DATABASE_URL}")
    
    # Check if database exists
    db = DatabaseManager()
    if Config.DATABASE_URL.startswith('sqlite'):
        db_path = Config.DATABASE_URL.replace('sqlite:///', '')
        if db_path.startswith('/'):
            db_file = db_path
        else:
            # Relative path - use backend directory
            backend_dir = os.path.dirname(os.path.dirname(__file__))
            db_file = os.path.join(backend_dir, db_path)
        
        if os.path.exists(db_file):
            file_size_mb = os.path.getsize(db_file) / (1024 * 1024)
            print(f"   Database file: {db_file}")
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Check record count
            try:
                conn = db.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM gas_prices")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"   Current records: {count:,}")
            except:
                print(f"   Current records: 0 (new database)")
        else:
            print(f"   Database file: {db_file} (will be created)")
            print(f"   Current records: 0 (new database)")
    
    print()
    print(f"⏱️  Collection Settings:")
    interval_seconds = args.interval if 'args' in locals() else 300
    interval_minutes = interval_seconds / 60
    print(f"   Interval: {interval_minutes:.1f} minutes ({interval_seconds} seconds)")
    print(f"   Data will be saved to: {Config.DATABASE_URL}")
    print()
    
    # Show data collection rate estimate
    records_per_hour = 3600 / interval_seconds
    records_per_day = records_per_hour * 24
    print(f"📈 Collection Rate:")
    print(f"   ~{records_per_hour:.1f} records/hour")
    print(f"   ~{records_per_day:.0f} records/day")
    print()
    print("💡 Press Ctrl+C to stop collection")
    print("="*70)
    print()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start collection service
    collector_service = GasCollectorService(interval_seconds=interval_seconds, register_signals=False)
    
    try:
        collector_service.start()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()
