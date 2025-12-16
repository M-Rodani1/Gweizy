"""
On-Chain Feature Collection Script

Collects blockchain-specific features alongside gas prices for improved ML predictions.
These features capture network state that directly influences gas prices:
- Transaction volume and types
- Block utilization
- Network congestion
- Gas price volatility

Usage:
    python scripts/collect_onchain_features.py --hours 24
    python scripts/collect_onchain_features.py --backfill --days 30
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime, timedelta
from web3 import Web3
from tqdm import tqdm
from sqlalchemy import Column, Integer, Float, DateTime, Boolean
from data.database import DatabaseManager, Base
from utils.onchain_features import OnChainFeatureExtractor
from config import Config


class OnChainFeatures(Base):
    """Table for storing on-chain features"""
    __tablename__ = 'onchain_features'

    id = Column(Integer, primary_key=True)
    block_number = Column(Integer, unique=True, index=True)
    timestamp = Column(DateTime, index=True)

    # Block metrics
    gas_used = Column(Float)
    gas_limit = Column(Float)
    block_utilization = Column(Float)
    tx_count = Column(Integer)

    # Transaction types
    simple_transfers = Column(Integer)
    contract_calls = Column(Integer)
    contract_creations = Column(Integer)
    contract_call_ratio = Column(Float)

    # Transaction metrics
    avg_gas_limit = Column(Float)
    total_value_eth = Column(Float)
    high_value_txs = Column(Integer)

    # Gas statistics
    base_fee = Column(Float)
    avg_max_fee = Column(Float)
    avg_priority_fee = Column(Float)
    max_priority_fee = Column(Float)
    min_priority_fee = Column(Float)
    priority_fee_std = Column(Float)
    priority_fee_range = Column(Float)


class OnChainFeatureCollector:
    """Collects on-chain features from Base blockchain"""

    BLOCKS_PER_HOUR = 1800  # ~2 second block time

    def __init__(self):
        self.db = DatabaseManager()
        self.w3 = Web3(Web3.HTTPProvider(Config.BASE_RPC_URL))
        self.extractor = OnChainFeatureExtractor(self.w3)
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """Create on-chain features table if it doesn't exist"""
        Base.metadata.create_all(self.db.engine)

    def collect_block_features(self, block_number: int) -> bool:
        """
        Collect and store features for a single block

        Args:
            block_number: Block number to process

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already exists
            session = self.db._get_session()
            existing = session.query(OnChainFeatures).filter_by(
                block_number=block_number
            ).first()
            session.close()

            if existing:
                return True  # Already collected

            # Extract features
            features = self.extractor.extract_block_features(block_number)

            if not features:
                return False

            # Save to database
            session = self.db._get_session()
            try:
                onchain_features = OnChainFeatures(
                    block_number=features['block_number'],
                    timestamp=features['timestamp'],
                    gas_used=features['gas_used'],
                    gas_limit=features['gas_limit'],
                    block_utilization=features['block_utilization'],
                    tx_count=features['tx_count'],
                    simple_transfers=features['simple_transfers'],
                    contract_calls=features['contract_calls'],
                    contract_creations=features['contract_creations'],
                    contract_call_ratio=features['contract_call_ratio'],
                    avg_gas_limit=features['avg_gas_limit'],
                    total_value_eth=features['total_value_eth'],
                    high_value_txs=features['high_value_txs'],
                    base_fee=features['base_fee'],
                    avg_max_fee=features['avg_max_fee'],
                    avg_priority_fee=features['avg_priority_fee'],
                    max_priority_fee=features['max_priority_fee'],
                    min_priority_fee=features['min_priority_fee'],
                    priority_fee_std=features['priority_fee_std'],
                    priority_fee_range=features['priority_fee_range']
                )
                session.add(onchain_features)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                print(f"Error saving features for block {block_number}: {e}")
                return False
            finally:
                session.close()

        except Exception as e:
            print(f"Error collecting features for block {block_number}: {e}")
            return False

    def collect_recent_features(self, hours: int = 24):
        """
        Collect features for recent blocks

        Args:
            hours: Number of hours to look back
        """
        current_block = self.w3.eth.block_number
        blocks_to_collect = hours * self.BLOCKS_PER_HOUR
        start_block = current_block - blocks_to_collect

        print(f"Collecting on-chain features for last {hours} hours")
        print(f"Blocks: {start_block:,} to {current_block:,}")
        print(f"Total blocks: {blocks_to_collect:,}\n")

        # Sample every 100th block for efficiency
        sample_rate = 100
        blocks_to_process = list(range(start_block, current_block, sample_rate))

        collected = 0
        skipped = 0
        errors = 0

        with tqdm(total=len(blocks_to_process), desc="Collecting features", unit="blocks") as pbar:
            for block_num in blocks_to_process:
                success = self.collect_block_features(block_num)

                if success:
                    collected += 1
                else:
                    errors += 1

                pbar.update(1)
                pbar.set_postfix({
                    'collected': collected,
                    'errors': errors
                })

                # Rate limiting
                if collected % 10 == 0:
                    import time
                    time.sleep(0.1)

        print(f"\n✓ Collection complete!")
        print(f"  Collected: {collected:,} blocks")
        print(f"  Errors: {errors:,} blocks")

    def backfill_features_for_gas_data(self):
        """
        Backfill on-chain features for existing gas price data

        This matches on-chain features to existing gas price records by block number
        """
        print("Backfilling on-chain features for existing gas price data...")

        from data.database import GasPrice

        session = self.db._get_session()
        try:
            # Get all gas prices that don't have matching on-chain features
            gas_prices = session.query(GasPrice).filter(
                GasPrice.block_number.isnot(None)
            ).all()

            print(f"Found {len(gas_prices):,} gas price records")

            # Get existing feature block numbers
            existing_features = session.query(OnChainFeatures.block_number).all()
            existing_blocks = {f[0] for f in existing_features}

            # Find missing blocks
            missing_blocks = [
                gp.block_number for gp in gas_prices
                if gp.block_number and gp.block_number not in existing_blocks
            ]

            print(f"Need to collect features for {len(missing_blocks):,} blocks")

            session.close()

            if not missing_blocks:
                print("All blocks already have on-chain features!")
                return

            # Collect features for missing blocks
            collected = 0
            errors = 0

            with tqdm(total=len(missing_blocks), desc="Backfilling", unit="blocks") as pbar:
                for block_num in missing_blocks:
                    success = self.collect_block_features(block_num)

                    if success:
                        collected += 1
                    else:
                        errors += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'collected': collected,
                        'errors': errors
                    })

                    # Rate limiting
                    if collected % 10 == 0:
                        import time
                        time.sleep(0.1)

            print(f"\n✓ Backfill complete!")
            print(f"  Collected: {collected:,} blocks")
            print(f"  Errors: {errors:,} blocks")

        except Exception as e:
            print(f"Error during backfill: {e}")
        finally:
            if session:
                session.close()

    def get_collection_stats(self):
        """Get statistics about collected on-chain features"""
        session = self.db._get_session()
        try:
            total = session.query(OnChainFeatures).count()

            if total == 0:
                return {
                    'total_blocks': 0,
                    'date_range': None
                }

            oldest = session.query(OnChainFeatures).order_by(
                OnChainFeatures.timestamp.asc()
            ).first()

            newest = session.query(OnChainFeatures).order_by(
                OnChainFeatures.timestamp.desc()
            ).first()

            from sqlalchemy import func
            avg_utilization = session.query(
                func.avg(OnChainFeatures.block_utilization)
            ).scalar()

            avg_tx_count = session.query(
                func.avg(OnChainFeatures.tx_count)
            ).scalar()

            return {
                'total_blocks': total,
                'oldest_timestamp': oldest.timestamp,
                'newest_timestamp': newest.timestamp,
                'date_range_days': (newest.timestamp - oldest.timestamp).days,
                'avg_block_utilization': round(avg_utilization, 4) if avg_utilization else 0,
                'avg_tx_per_block': round(avg_tx_count, 2) if avg_tx_count else 0
            }

        finally:
            session.close()


def main():
    parser = argparse.ArgumentParser(description='Collect on-chain features')

    parser.add_argument('--hours', type=int, help='Collect features for last N hours')
    parser.add_argument('--backfill', action='store_true',
                       help='Backfill features for existing gas price data')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics')

    args = parser.parse_args()

    collector = OnChainFeatureCollector()

    # Show stats
    if args.stats:
        print("\n=== On-Chain Feature Collection Statistics ===")
        stats = collector.get_collection_stats()
        print(f"Total blocks: {stats['total_blocks']:,}")
        if stats['total_blocks'] > 0:
            print(f"Date range: {stats['oldest_timestamp']} to {stats['newest_timestamp']}")
            print(f"Days of data: {stats['date_range_days']}")
            print(f"Avg block utilization: {stats['avg_block_utilization']:.2%}")
            print(f"Avg transactions/block: {stats['avg_tx_per_block']:.1f}")
        return

    # Backfill existing data
    if args.backfill:
        collector.backfill_features_for_gas_data()
        return

    # Collect recent data
    if args.hours:
        collector.collect_recent_features(hours=args.hours)
    else:
        print("Error: Must specify --hours, --backfill, or --stats")
        parser.print_help()
        return

    # Show final stats
    print("\n=== Final Statistics ===")
    stats = collector.get_collection_stats()
    print(f"Total blocks: {stats['total_blocks']:,}")
    if stats['total_blocks'] > 0:
        print(f"Avg block utilization: {stats['avg_block_utilization']:.2%}")
        print(f"Avg transactions/block: {stats['avg_tx_per_block']:.1f}")


if __name__ == "__main__":
    main()
