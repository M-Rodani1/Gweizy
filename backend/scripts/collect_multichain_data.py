"""
Multi-chain historical data collection script.
Collects gas price data from all supported chains for ML training.
"""
import os
import sys
import time
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.multichain_collector import MultiChainGasCollector, CHAINS
from data.database import DatabaseManager
from utils.logger import logger


def collect_all_chains_data(hours_back: int = 720, interval_minutes: int = 5, chains: list = None):
    """
    Collect historical gas data for all supported chains.
    
    Args:
        hours_back: How many hours of data to collect
        interval_minutes: Interval between data points (in minutes)
        chains: List of chain IDs to collect (None = all chains)
    """
    collector = MultiChainGasCollector()
    db = DatabaseManager()
    
    if chains is None:
        chains = list(CHAINS.keys())
    
    logger.info(f"Starting multi-chain data collection")
    logger.info(f"Chains: {[CHAINS[c]['name'] for c in chains]}")
    logger.info(f"Hours: {hours_back}, Interval: {interval_minutes} minutes")
    
    collected_by_chain = {}
    errors_by_chain = {}
    
    for chain_id in chains:
        chain_name = CHAINS[chain_id]['name']
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting data for {chain_name} (Chain ID: {chain_id})")
        logger.info(f"{'='*60}")
        
        collected = 0
        errors = 0
        
        # Check existing data
        existing = db.get_historical_data(hours=hours_back, chain_id=chain_id)
        logger.info(f"Found {len(existing)} existing records for {chain_name}")
        
        # Calculate how many data points we need
        total_points = (hours_back * 60) // interval_minutes
        logger.info(f"Target: {total_points} data points")
        
        # Collect data points
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours_back)
        
        logger.info(f"Collection period: {start_time} to {end_time}")
        logger.info("Collecting data...")
        
        while datetime.now() < end_time and collected < total_points:
            try:
                # Collect current gas price
                data = collector.get_current_gas(chain_id)
                
                if data:
                    # Ensure chain_id is set
                    data['chain_id'] = chain_id
                    
                    # Save to database
                    db.save_gas_price(data)
                    collected += 1
                    
                    if collected % 10 == 0:
                        logger.info(f"  {chain_name}: Collected {collected}/{total_points} data points...")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                errors += 1
                logger.error(f"  Error collecting data for {chain_name}: {e}")
                if errors > 10:
                    logger.error(f"  Too many errors for {chain_name}, stopping collection")
                    break
                time.sleep(60)  # Wait 1 minute before retry
        
        collected_by_chain[chain_id] = collected
        errors_by_chain[chain_id] = errors
        
        # Verify final count
        final_data = db.get_historical_data(hours=hours_back, chain_id=chain_id)
        logger.info(f"  {chain_name}: Final database count: {len(final_data)} records")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    for chain_id in chains:
        chain_name = CHAINS[chain_id]['name']
        logger.info(f"{chain_name}: {collected_by_chain[chain_id]} collected, {errors_by_chain[chain_id]} errors")
    
    return collected_by_chain


def verify_all_chains_data(min_records: int = 500):
    """
    Verify data quality for all chains.
    
    Args:
        min_records: Minimum number of records needed per chain
    """
    db = DatabaseManager()
    
    logger.info("Verifying data quality for all chains...")
    
    ranges = [
        (24, "24 hours"),
        (168, "7 days"),
        (720, "30 days"),
        (2160, "90 days")
    ]
    
    for chain_id, chain_info in CHAINS.items():
        chain_name = chain_info['name']
        logger.info(f"\n{chain_name} (Chain ID: {chain_id}):")
        
        for hours, label in ranges:
            data = db.get_historical_data(hours=hours, chain_id=chain_id)
            count = len(data)
            
            if count >= min_records:
                logger.info(f"  ✓ {label}: {count} records (sufficient)")
            else:
                logger.warning(f"  ✗ {label}: {count} records (need {min_records})")
            
            # Check for gaps
            if count > 1:
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
                    logger.info(f"    Average interval: {avg_interval:.1f} minutes")


def collect_chain_data_backfill(chain_id: int, hours_back: int = 720):
    """
    Backfill historical data for a specific chain using existing database records.
    This is useful when you have some data but need more.
    
    Args:
        chain_id: Chain ID to backfill
        hours_back: Hours of data to ensure
    """
    collector = MultiChainGasCollector()
    db = DatabaseManager()
    
    chain_name = CHAINS[chain_id]['name']
    logger.info(f"Backfilling data for {chain_name} (Chain ID: {chain_id})")
    
    # Check existing data
    existing = db.get_historical_data(hours=hours_back, chain_id=chain_id)
    logger.info(f"Found {len(existing)} existing records")
    
    if len(existing) >= hours_back * 2:  # At least 2 data points per hour
        logger.info(f"Sufficient data already exists ({len(existing)} records)")
        return
    
    # Collect more data
    needed = (hours_back * 2) - len(existing)
    logger.info(f"Need approximately {needed} more data points")
    
    collected = 0
    while collected < needed:
        try:
            data = collector.get_current_gas(chain_id)
            if data:
                data['chain_id'] = chain_id
                db.save_gas_price(data)
                collected += 1
                
                if collected % 10 == 0:
                    logger.info(f"  Collected {collected}/{needed} additional points...")
            
            time.sleep(300)  # 5 minute intervals
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            time.sleep(60)
    
    logger.info(f"Backfill complete: {collected} additional points collected")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect historical gas data for all chains')
    parser.add_argument('--hours', type=int, default=720, help='Hours of data to collect')
    parser.add_argument('--interval', type=int, default=5, help='Collection interval in minutes')
    parser.add_argument('--chains', type=str, help='Comma-separated chain IDs (e.g., 8453,1,42161)')
    parser.add_argument('--verify', action='store_true', help='Verify data quality for all chains')
    parser.add_argument('--backfill', type=int, help='Backfill data for specific chain ID')
    parser.add_argument('--backfill-hours', type=int, default=720, help='Hours to backfill')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_all_chains_data()
    elif args.backfill:
        collect_chain_data_backfill(args.backfill, args.backfill_hours)
    else:
        chains = None
        if args.chains:
            chains = [int(c.strip()) for c in args.chains.split(',')]
        
        collect_all_chains_data(
            hours_back=args.hours,
            interval_minutes=args.interval,
            chains=chains
        )

