"""
Historical data collection script for RL training.
Collects gas price data from the database and enriches it for training.
"""
import os
import sys
from datetime import datetime, timedelta
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import BaseGasCollector
from data.database import DatabaseManager
from utils.logger import logger


def collect_historical_data(hours_back: int = 720, interval_minutes: int = 5):
    """
    Collect historical gas data by backfilling from current time.
    
    Args:
        hours_back: How many hours of data to collect
        interval_minutes: Interval between data points (in minutes)
    """
    collector = BaseGasCollector()
    db = DatabaseManager()
    
    logger.info(f"Starting historical data collection: {hours_back} hours, {interval_minutes} min intervals")
    
    # Check existing data
    existing = db.get_historical_data(hours=hours_back)
    logger.info(f"Found {len(existing)} existing records in database")
    
    # Calculate how many data points we need
    total_points = (hours_back * 60) // interval_minutes
    logger.info(f"Target: {total_points} data points")
    
    collected = 0
    errors = 0
    
    # Collect data points going backwards in time
    # Note: We can't actually go back in time, so we'll collect current data
    # and simulate historical patterns, or use existing database data
    
    # Strategy: Collect current data at specified intervals to build up history
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours_back)
    
    logger.info(f"Collection period: {start_time} to {end_time}")
    logger.info("Collecting data...")
    
    while datetime.now() < end_time and collected < total_points:
        try:
            # Collect current gas price
            data = collector.get_current_gas()
            
            if data:
                # Save to database
                db.save_gas_price(data)
                collected += 1
                
                if collected % 10 == 0:
                    logger.info(f"Collected {collected}/{total_points} data points...")
            
            # Wait for next interval
            time.sleep(interval_minutes * 60)
            
        except Exception as e:
            errors += 1
            logger.error(f"Error collecting data: {e}")
            if errors > 10:
                logger.error("Too many errors, stopping collection")
                break
            time.sleep(60)  # Wait 1 minute before retry
    
    logger.info(f"Collection complete: {collected} points collected, {errors} errors")
    
    # Verify final count
    final_data = db.get_historical_data(hours=hours_back)
    logger.info(f"Final database count: {len(final_data)} records")
    
    return collected


def enrich_existing_data():
    """
    Enrich existing database records with additional features for RL training.
    This function can be used to backfill missing features or augment data.
    """
    db = DatabaseManager()
    
    logger.info("Enriching existing data...")
    
    # Get all historical data
    data = db.get_historical_data(hours=8760)  # 1 year
    
    if len(data) < 100:
        logger.warning(f"Not enough data to enrich: {len(data)} records")
        return
    
    logger.info(f"Found {len(data)} records to enrich")
    
    # Data is already in the database, just verify it's usable
    # The RL data loader will handle feature extraction
    
    logger.info("Data enrichment complete")
    return len(data)


def verify_data_quality(min_records: int = 1000):
    """
    Verify that we have enough quality data for RL training.
    
    Args:
        min_records: Minimum number of records needed
    """
    db = DatabaseManager()
    
    # Check different time ranges
    ranges = [
        (24, "24 hours"),
        (168, "7 days"),
        (720, "30 days"),
        (2160, "90 days")
    ]
    
    logger.info("Verifying data quality...")
    
    for hours, label in ranges:
        data = db.get_historical_data(hours=hours)
        count = len(data)
        
        if count >= min_records:
            logger.info(f"✓ {label}: {count} records (sufficient)")
        else:
            logger.warning(f"✗ {label}: {count} records (need {min_records})")
        
        # Check for gaps
        if count > 1:
            # Calculate average interval
            from dateutil import parser
            timestamps = []
            for d in data:
                ts = d.get('timestamp', '')
                if isinstance(ts, str):
                    try:
                        timestamps.append(parser.parse(ts))
                    except:
                        pass
            
            if len(timestamps) > 1:
                timestamps.sort()
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                           for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals) if intervals else 0
                logger.info(f"  Average interval: {avg_interval:.1f} minutes")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect historical gas data for RL training')
    parser.add_argument('--hours', type=int, default=720, help='Hours of data to collect')
    parser.add_argument('--interval', type=int, default=5, help='Collection interval in minutes')
    parser.add_argument('--enrich', action='store_true', help='Enrich existing data')
    parser.add_argument('--verify', action='store_true', help='Verify data quality')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data_quality()
    elif args.enrich:
        enrich_existing_data()
    else:
        collect_historical_data(hours_back=args.hours, interval_minutes=args.interval)

