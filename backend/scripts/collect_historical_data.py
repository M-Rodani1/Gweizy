"""
Historical Base Gas Data Collection Script

Fetches historical gas price data from Base mainnet by scanning blocks backwards
from the current block. Collects base fee and priority fee data for ML training.

Usage:
    python scripts/collect_historical_data.py --months 6
    python scripts/collect_historical_data.py --days 30
    python scripts/collect_historical_data.py --start-block 10000000 --end-block 11000000
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from datetime import datetime, timedelta
from web3 import Web3
from tqdm import tqdm
from data.database import DatabaseManager, GasPrice
from config import Config


class HistoricalDataCollector:
    """Collects historical gas price data from Base blockchain"""

    # Base mainnet launched July 2023, estimate ~2 seconds per block
    BLOCKS_PER_HOUR = 1800  # ~2 second block time
    BLOCKS_PER_DAY = 43200
    BLOCKS_PER_MONTH = 1296000  # 30 days

    def __init__(self, rpc_urls=None):
        """Initialize with multiple RPC endpoints for redundancy"""
        self.rpc_urls = rpc_urls or [
            Config.BASE_RPC_URL,
            "https://mainnet.base.org",
            "https://base.llamarpc.com",
            "https://base-rpc.publicnode.com"
        ]
        self.current_rpc_index = 0
        self.w3 = self._get_web3()
        self.db = DatabaseManager()

    def _get_web3(self):
        """Get Web3 instance with current RPC URL"""
        url = self.rpc_urls[self.current_rpc_index]
        return Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 60}))

    def _rotate_rpc(self):
        """Rotate to next RPC endpoint on failure"""
        self.current_rpc_index = (self.current_rpc_index + 1) % len(self.rpc_urls)
        self.w3 = self._get_web3()
        print(f"Rotated to RPC: {self.rpc_urls[self.current_rpc_index]}")

    def get_block_with_retry(self, block_number, max_retries=3):
        """Fetch block with retry logic and RPC rotation"""
        for attempt in range(max_retries):
            try:
                block = self.w3.eth.get_block(block_number, full_transactions=True)
                return block
            except Exception as e:
                if attempt < max_retries - 1:
                    self._rotate_rpc()
                    time.sleep(1)
                else:
                    raise e

    def extract_gas_data(self, block):
        """Extract gas price data from a block"""
        try:
            # Get base fee (EIP-1559)
            base_fee = block.get('baseFeePerGas', 0)

            # Calculate average priority fee from transactions
            priority_fees = []
            for tx in block.transactions:
                if hasattr(tx, 'maxPriorityFeePerGas') and tx.maxPriorityFeePerGas:
                    priority_fees.append(tx.maxPriorityFeePerGas)

            avg_priority_fee = sum(priority_fees) / len(priority_fees) if priority_fees else 0
            total_gas_wei = base_fee + avg_priority_fee

            # Convert block timestamp to datetime
            timestamp = datetime.fromtimestamp(block.timestamp)

            return {
                'timestamp': timestamp,
                'current_gas': round(total_gas_wei / 1e9, 6),  # Convert to Gwei
                'base_fee': round(base_fee / 1e9, 6),
                'priority_fee': round(avg_priority_fee / 1e9, 6),
                'block_number': block.number
            }
        except Exception as e:
            print(f"Error extracting data from block {block.number}: {e}")
            return None

    def collect_by_time_range(self, months=None, days=None, hours=None, sample_rate=1):
        """Collect data for a specific time range going backwards from now"""
        # Calculate target time range
        now = datetime.now()
        if months:
            target_datetime = now - timedelta(days=months * 30)
        elif days:
            target_datetime = now - timedelta(days=days)
        elif hours:
            target_datetime = now - timedelta(hours=hours)
        else:
            raise ValueError("Must specify months, days, or hours")

        print(f"Collecting data from {target_datetime} to {now}")

        # Get current block number
        current_block = self.w3.eth.block_number
        print(f"Current block: {current_block}")

        # Estimate starting block (going backwards)
        if months:
            estimated_blocks_back = months * self.BLOCKS_PER_MONTH
        elif days:
            estimated_blocks_back = days * self.BLOCKS_PER_DAY
        else:
            estimated_blocks_back = hours * self.BLOCKS_PER_HOUR

        start_block = current_block - estimated_blocks_back

        # Fine-tune start block to match exact target datetime
        start_block = self._find_block_by_timestamp(start_block, target_datetime)

        print(f"Collecting from block {start_block} to {current_block}")
        print(f"Estimated blocks to process: {current_block - start_block:,}")

        return self.collect_by_block_range(start_block, current_block, sample_rate=sample_rate)

    def _find_block_by_timestamp(self, approximate_block, target_datetime):
        """Binary search to find block closest to target datetime"""
        print(f"Fine-tuning start block to match {target_datetime}...")

        low = max(0, approximate_block - 100000)
        high = approximate_block + 100000
        target_timestamp = target_datetime.timestamp()

        for _ in range(20):  # Max 20 iterations
            mid = (low + high) // 2
            block = self.get_block_with_retry(mid)
            block_timestamp = block.timestamp

            if abs(block_timestamp - target_timestamp) < 60:  # Within 1 minute
                return mid

            if block_timestamp > target_timestamp:
                high = mid - 1
            else:
                low = mid + 1

        return (low + high) // 2

    def collect_by_block_range(self, start_block, end_block, sample_rate=1):
        """
        Collect data for a specific block range

        Args:
            start_block: Starting block number
            end_block: Ending block number
            sample_rate: Collect every Nth block (1 = every block, 10 = every 10th block)
        """
        total_blocks = (end_block - start_block) // sample_rate
        collected = 0
        skipped = 0
        errors = 0

        print(f"\nCollecting gas data from blocks {start_block:,} to {end_block:,}")
        print(f"Sample rate: Every {sample_rate} block(s)")
        print(f"Total blocks to process: {total_blocks:,}\n")

        # Create progress bar
        with tqdm(total=total_blocks, desc="Collecting blocks", unit="blocks") as pbar:
            for block_num in range(start_block, end_block + 1, sample_rate):
                try:
                    # Check if block already exists in database
                    session = self.db._get_session()
                    existing = session.query(GasPrice).filter_by(block_number=block_num).first()
                    session.close()

                    if existing:
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Fetch and process block
                    block = self.get_block_with_retry(block_num)
                    gas_data = self.extract_gas_data(block)

                    if gas_data:
                        self.db.save_gas_price(gas_data)
                        collected += 1
                    else:
                        errors += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'collected': collected,
                        'skipped': skipped,
                        'errors': errors
                    })

                    # Rate limiting - don't hammer the RPC
                    if collected % 100 == 0:
                        time.sleep(0.5)

                except KeyboardInterrupt:
                    print("\n\nCollection interrupted by user")
                    break
                except Exception as e:
                    errors += 1
                    print(f"\nError processing block {block_num}: {e}")
                    time.sleep(1)
                    continue

        print(f"\n✓ Collection complete!")
        print(f"  Collected: {collected:,} blocks")
        print(f"  Skipped (already in DB): {skipped:,} blocks")
        print(f"  Errors: {errors:,} blocks")

        return {
            'collected': collected,
            'skipped': skipped,
            'errors': errors
        }

    def get_collection_stats(self):
        """Get statistics about collected data"""
        session = self.db._get_session()
        try:
            total_records = session.query(GasPrice).count()

            if total_records == 0:
                return {
                    'total_records': 0,
                    'date_range': None,
                    'avg_gas': None
                }

            oldest = session.query(GasPrice).order_by(GasPrice.timestamp.asc()).first()
            newest = session.query(GasPrice).order_by(GasPrice.timestamp.desc()).first()

            # Calculate average gas
            from sqlalchemy import func
            avg_gas = session.query(func.avg(GasPrice.current_gas)).scalar()

            return {
                'total_records': total_records,
                'oldest_timestamp': oldest.timestamp,
                'newest_timestamp': newest.timestamp,
                'date_range_days': (newest.timestamp - oldest.timestamp).days,
                'avg_gas_gwei': round(avg_gas, 6)
            }
        finally:
            session.close()


def main():
    parser = argparse.ArgumentParser(description='Collect historical Base gas data')

    # Time-based collection
    parser.add_argument('--months', type=int, help='Collect data for last N months')
    parser.add_argument('--days', type=int, help='Collect data for last N days')
    parser.add_argument('--hours', type=int, help='Collect data for last N hours')

    # Block-based collection
    parser.add_argument('--start-block', type=int, help='Starting block number')
    parser.add_argument('--end-block', type=int, help='Ending block number')

    # Sampling
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Collect every Nth block (default: 1)')

    # Stats
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics and exit')

    args = parser.parse_args()

    collector = HistoricalDataCollector()

    # Show stats if requested
    if args.stats:
        print("\n=== Collection Statistics ===")
        stats = collector.get_collection_stats()
        print(f"Total records: {stats['total_records']:,}")
        if stats['total_records'] > 0:
            print(f"Date range: {stats['oldest_timestamp']} to {stats['newest_timestamp']}")
            print(f"Days of data: {stats['date_range_days']}")
            print(f"Average gas: {stats['avg_gas_gwei']} Gwei")
        return

    # Collect by time range
    if any([args.months, args.days, args.hours]):
        collector.collect_by_time_range(
            months=args.months,
            days=args.days,
            hours=args.hours,
            sample_rate=args.sample_rate
        )

    # Collect by block range
    elif args.start_block and args.end_block:
        collector.collect_by_block_range(
            start_block=args.start_block,
            end_block=args.end_block,
            sample_rate=args.sample_rate
        )

    else:
        print("Error: Must specify either time range (--months/--days/--hours) or block range (--start-block and --end-block)")
        parser.print_help()
        return

    # Show final stats
    print("\n=== Final Statistics ===")
    stats = collector.get_collection_stats()
    print(f"Total records in database: {stats['total_records']:,}")
    if stats['total_records'] > 0:
        print(f"Date range: {stats['oldest_timestamp']} to {stats['newest_timestamp']}")
        print(f"Days of data: {stats['date_range_days']}")
        print(f"Average gas price: {stats['avg_gas_gwei']} Gwei")


if __name__ == "__main__":
    main()
