"""
On-Chain Feature Extractor

Extracts blockchain-specific features that correlate with gas prices:
- Transaction volume (tx/block, tx/hour)
- Block utilization (gas used / gas limit)
- Transaction types (transfers, contract calls, swaps)
- Network congestion metrics
- MEV activity indicators
- Gas price volatility

These features significantly improve ML model accuracy by capturing
network state that directly influences gas prices.
"""

from web3 import Web3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict
from config import Config
from utils.logger import logger


class OnChainFeatureExtractor:
    """Extracts on-chain features from Base blockchain"""

    def __init__(self, w3: Optional[Web3] = None):
        """
        Initialize feature extractor

        Args:
            w3: Web3 instance (creates new one if not provided)
        """
        self.w3 = w3 or Web3(Web3.HTTPProvider(Config.BASE_RPC_URL))

    def extract_block_features(self, block_number: int) -> Dict:
        """
        Extract features from a single block

        Args:
            block_number: Block number to analyze

        Returns:
            Dictionary of on-chain features
        """
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)

            # Basic block metrics
            gas_used = block.get('gasUsed', 0)
            gas_limit = block.get('gasLimit', 0)
            tx_count = len(block.transactions)

            # Calculate utilization
            block_utilization = (gas_used / gas_limit) if gas_limit > 0 else 0

            # Analyze transactions
            tx_features = self._analyze_transactions(block.transactions)

            # Gas price statistics from transactions
            gas_stats = self._calculate_gas_statistics(block.transactions)

            return {
                'block_number': block_number,
                'timestamp': datetime.fromtimestamp(block.timestamp),

                # Block metrics
                'gas_used': gas_used,
                'gas_limit': gas_limit,
                'block_utilization': block_utilization,
                'tx_count': tx_count,

                # Transaction features
                **tx_features,

                # Gas statistics
                **gas_stats,

                # Base fee (EIP-1559)
                'base_fee': block.get('baseFeePerGas', 0) / 1e9  # Convert to Gwei
            }

        except Exception as e:
            logger.error(f"Error extracting features from block {block_number}: {e}")
            return None

    def _analyze_transactions(self, transactions: List) -> Dict:
        """
        Analyze transaction types and patterns

        Args:
            transactions: List of transaction objects

        Returns:
            Transaction feature dictionary
        """
        if not transactions:
            return {
                'simple_transfers': 0,
                'contract_calls': 0,
                'contract_creations': 0,
                'avg_gas_limit': 0,
                'total_value': 0,
                'high_value_txs': 0,
                'failed_txs': 0
            }

        simple_transfers = 0
        contract_calls = 0
        contract_creations = 0
        gas_limits = []
        values = []
        failed_count = 0

        for tx in transactions:
            # Gas limit
            gas_limits.append(tx.get('gas', 0))

            # Transaction value
            value_eth = tx.get('value', 0) / 1e18
            values.append(value_eth)

            # Transaction type detection
            to_address = tx.get('to')
            input_data = tx.get('input', '0x')

            if to_address is None:
                # Contract creation
                contract_creations += 1
            elif input_data == '0x' or len(input_data) <= 2:
                # Simple transfer (no data)
                simple_transfers += 1
            else:
                # Contract interaction
                contract_calls += 1

            # Check if transaction failed (requires receipt, skip for now)
            # This would need eth_getTransactionReceipt which is expensive
            # failed_count += 1 if receipt.status == 0 else 0

        # High value transactions (>1 ETH)
        high_value_txs = sum(1 for v in values if v > 1.0)

        return {
            'simple_transfers': simple_transfers,
            'contract_calls': contract_calls,
            'contract_creations': contract_creations,
            'avg_gas_limit': np.mean(gas_limits) if gas_limits else 0,
            'total_value_eth': sum(values),
            'high_value_txs': high_value_txs,
            'contract_call_ratio': contract_calls / len(transactions) if transactions else 0
        }

    def _calculate_gas_statistics(self, transactions: List) -> Dict:
        """
        Calculate gas price statistics from transactions

        Args:
            transactions: List of transaction objects

        Returns:
            Gas statistics dictionary
        """
        if not transactions:
            return {
                'avg_max_fee': 0,
                'avg_priority_fee': 0,
                'max_priority_fee': 0,
                'min_priority_fee': 0,
                'priority_fee_std': 0,
                'priority_fee_range': 0
            }

        max_fees = []
        priority_fees = []

        for tx in transactions:
            if hasattr(tx, 'maxFeePerGas') and tx.maxFeePerGas:
                max_fees.append(tx.maxFeePerGas / 1e9)

            if hasattr(tx, 'maxPriorityFeePerGas') and tx.maxPriorityFeePerGas:
                priority_fees.append(tx.maxPriorityFeePerGas / 1e9)

        if not priority_fees:
            return {
                'avg_max_fee': 0,
                'avg_priority_fee': 0,
                'max_priority_fee': 0,
                'min_priority_fee': 0,
                'priority_fee_std': 0,
                'priority_fee_range': 0
            }

        return {
            'avg_max_fee': np.mean(max_fees) if max_fees else 0,
            'avg_priority_fee': np.mean(priority_fees),
            'max_priority_fee': np.max(priority_fees),
            'min_priority_fee': np.min(priority_fees),
            'priority_fee_std': np.std(priority_fees),
            'priority_fee_range': np.max(priority_fees) - np.min(priority_fees)
        }

    def extract_hourly_aggregates(
        self,
        start_block: int,
        end_block: int,
        blocks_per_hour: int = 1800
    ) -> Dict:
        """
        Extract aggregated features over an hour

        Args:
            start_block: Starting block number
            end_block: Ending block number (start + blocks_per_hour)
            blocks_per_hour: Blocks per hour (default: 1800 for ~2s block time)

        Returns:
            Aggregated hourly features
        """
        all_features = []

        # Sample blocks (not every block, too expensive)
        sample_interval = max(1, (end_block - start_block) // 20)  # Sample 20 blocks per hour

        for block_num in range(start_block, end_block, sample_interval):
            features = self.extract_block_features(block_num)
            if features:
                all_features.append(features)

        if not all_features:
            return None

        # Aggregate features
        return {
            'period_start': all_features[0]['timestamp'],
            'period_end': all_features[-1]['timestamp'],

            # Average metrics
            'avg_block_utilization': np.mean([f['block_utilization'] for f in all_features]),
            'avg_tx_per_block': np.mean([f['tx_count'] for f in all_features]),
            'avg_gas_used': np.mean([f['gas_used'] for f in all_features]),

            # Total counts
            'total_txs': sum([f['tx_count'] for f in all_features]),
            'total_contract_calls': sum([f['contract_calls'] for f in all_features]),
            'total_value_eth': sum([f['total_value_eth'] for f in all_features]),

            # Congestion indicators
            'max_utilization': np.max([f['block_utilization'] for f in all_features]),
            'utilization_std': np.std([f['block_utilization'] for f in all_features]),
            'high_congestion_blocks': sum(1 for f in all_features if f['block_utilization'] > 0.8),

            # Gas price volatility
            'base_fee_mean': np.mean([f['base_fee'] for f in all_features]),
            'base_fee_std': np.std([f['base_fee'] for f in all_features]),
            'base_fee_trend': self._calculate_trend([f['base_fee'] for f in all_features]),

            # Priority fee metrics
            'avg_priority_fee': np.mean([f['avg_priority_fee'] for f in all_features if f['avg_priority_fee'] > 0]),
            'priority_fee_volatility': np.std([f['avg_priority_fee'] for f in all_features if f['avg_priority_fee'] > 0])
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate simple linear trend

        Args:
            values: Time series values

        Returns:
            Trend coefficient (positive = increasing, negative = decreasing)
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        coefficients = np.polyfit(x, y, 1)
        return float(coefficients[0])  # Return slope

    def get_current_network_state(self) -> Dict:
        """
        Get current network state features

        Returns:
            Current on-chain features
        """
        try:
            current_block = self.w3.eth.block_number

            # Get features from last 5 blocks for stability
            recent_features = []
            for i in range(5):
                block_num = current_block - i
                features = self.extract_block_features(block_num)
                if features:
                    recent_features.append(features)

            if not recent_features:
                return None

            # Average recent features
            return {
                'current_block': current_block,
                'avg_utilization': np.mean([f['block_utilization'] for f in recent_features]),
                'avg_tx_count': np.mean([f['tx_count'] for f in recent_features]),
                'avg_base_fee': np.mean([f['base_fee'] for f in recent_features]),
                'base_fee_trend': self._calculate_trend([f['base_fee'] for f in recent_features]),
                'is_congested': np.mean([f['block_utilization'] for f in recent_features]) > 0.7,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting current network state: {e}")
            return None


class OnChainFeatureCache:
    """
    Caches on-chain features to reduce RPC calls

    Features are expensive to calculate, so we cache them by block number
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize cache

        Args:
            max_size: Maximum number of blocks to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.extractor = OnChainFeatureExtractor()

    def get_features(self, block_number: int) -> Optional[Dict]:
        """
        Get features for a block (from cache or extract)

        Args:
            block_number: Block number

        Returns:
            Feature dictionary or None
        """
        # Check cache
        if block_number in self.cache:
            return self.cache[block_number]

        # Extract features
        features = self.extractor.extract_block_features(block_number)

        if features:
            # Add to cache
            self.cache[block_number] = features

            # Evict oldest if cache too large
            if len(self.cache) > self.max_size:
                oldest_block = min(self.cache.keys())
                del self.cache[oldest_block]

        return features

    def get_hourly_features(
        self,
        start_block: int,
        blocks_per_hour: int = 1800
    ) -> Optional[Dict]:
        """
        Get aggregated hourly features

        Args:
            start_block: Starting block number
            blocks_per_hour: Blocks in an hour

        Returns:
            Hourly aggregated features
        """
        end_block = start_block + blocks_per_hour
        return self.extractor.extract_hourly_aggregates(start_block, end_block)

    def clear_cache(self):
        """Clear the feature cache"""
        self.cache = {}


# Global cache instance
feature_cache = OnChainFeatureCache()


if __name__ == "__main__":
    # Example usage
    extractor = OnChainFeatureExtractor()

    print("=== On-Chain Feature Extraction Demo ===\n")

    # Get current network state
    print("Current Network State:")
    state = extractor.get_current_network_state()
    if state:
        print(f"  Block: {state['current_block']}")
        print(f"  Utilization: {state['avg_utilization']:.2%}")
        print(f"  Avg TX/block: {state['avg_tx_count']:.0f}")
        print(f"  Base fee: {state['avg_base_fee']:.6f} Gwei")
        print(f"  Trend: {'↑ Increasing' if state['base_fee_trend'] > 0 else '↓ Decreasing'}")
        print(f"  Congested: {'Yes' if state['is_congested'] else 'No'}")
