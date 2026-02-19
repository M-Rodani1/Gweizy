"""
Multi-chain gas price collector.
Collects gas prices from multiple EVM chains for ML training and predictions.
"""
import requests
import time
from datetime import datetime
from web3 import Web3
from typing import Dict, Optional, List
from config import Config


# Chain configurations matching multichain_routes.py
CHAINS = {
    8453: {
        'name': 'Base',
        'shortName': 'BASE',
        'rpcs': [
            'https://mainnet.base.org',
            'https://base.llamarpc.com',
            'https://base.drpc.org'
        ],
        'isL2': True,
        'owlracle_endpoint': 'base'
    },
    1: {
        'name': 'Ethereum',
        'shortName': 'ETH',
        'rpcs': [
            'https://eth.llamarpc.com',
            'https://ethereum.publicnode.com',
            'https://rpc.ankr.com/eth'
        ],
        'isL2': False,
        'owlracle_endpoint': 'eth'
    },
    42161: {
        'name': 'Arbitrum',
        'shortName': 'ARB',
        'rpcs': [
            'https://arb1.arbitrum.io/rpc',
            'https://arbitrum.llamarpc.com',
            'https://arbitrum-one.publicnode.com'
        ],
        'isL2': True,
        'owlracle_endpoint': 'arbitrum'
    },
    10: {
        'name': 'Optimism',
        'shortName': 'OP',
        'rpcs': [
            'https://mainnet.optimism.io',
            'https://optimism.llamarpc.com',
            'https://optimism.publicnode.com'
        ],
        'isL2': True,
        'owlracle_endpoint': 'optimism'
    },
    137: {
        'name': 'Polygon',
        'shortName': 'MATIC',
        'rpcs': [
            'https://polygon-bor-rpc.publicnode.com',
            'https://rpc.ankr.com/polygon',
            'https://1rpc.io/matic'
        ],
        'isL2': True,
        'owlracle_endpoint': 'polygon'
    }
}


class MultiChainGasCollector:
    """Collects gas prices from multiple EVM chains."""
    
    def __init__(self):
        self.chains = CHAINS
        self._web3_instances = {}
        self._session = requests.Session()
    
    def _get_web3(self, chain_id: int) -> Optional[Web3]:
        """Get or create Web3 instance for a chain."""
        if chain_id not in self._web3_instances:
            chain = self.chains.get(chain_id)
            if not chain:
                return None
            
            # Try first RPC
            try:
                w3 = Web3(Web3.HTTPProvider(chain['rpcs'][0]))
                if w3.is_connected():
                    self._web3_instances[chain_id] = w3
                    return w3
            except Exception:
                pass  # RPC connection failed
        
        return self._web3_instances.get(chain_id)
    
    def get_current_gas(self, chain_id: int) -> Optional[Dict]:
        """
        Fetch current gas price for a specific chain.
        
        Args:
            chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
        
        Returns:
            Dict with gas price data or None if failed
        """
        chain = self.chains.get(chain_id)
        if not chain:
            return None
        
        # Try direct RPC first
        w3 = self._get_web3(chain_id)
        if w3:
            try:
                latest_block = w3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', 0)
                
                # Get recent transactions to estimate priority fee
                block = w3.eth.get_block('latest', full_transactions=True)
                transactions = block.transactions[:10] if hasattr(block, 'transactions') else []
                
                priority_fees = []
                for tx in transactions:
                    if hasattr(tx, 'maxPriorityFeePerGas') and tx.maxPriorityFeePerGas:
                        priority_fees.append(tx.maxPriorityFeePerGas)
                
                avg_priority_fee = sum(priority_fees) / len(priority_fees) if priority_fees else 0
                
                # For L1 chains without baseFeePerGas, use gasPrice
                if base_fee == 0:
                    gas_price = w3.eth.gas_price
                    total_gas = gas_price / 1e9
                    base_fee_gwei = total_gas * 0.9
                    priority_fee_gwei = total_gas * 0.1
                else:
                    total_gas = (base_fee + avg_priority_fee) / 1e9
                    base_fee_gwei = base_fee / 1e9
                    priority_fee_gwei = avg_priority_fee / 1e9
                
                return {
                    'chain_id': chain_id,
                    'chain_name': chain['name'],
                    'timestamp': datetime.now().isoformat(),
                    'current_gas': round(total_gas, 6),
                    'base_fee': round(base_fee_gwei, 6),
                    'priority_fee': round(priority_fee_gwei, 6),
                    'block_number': block.number
                }
            except Exception as e:
                print(f"Error fetching gas from RPC for chain {chain_id}: {e}")
        
        # Fallback to Owlracle API
        return self._fetch_from_owlracle(chain_id)
    
    def _fetch_from_owlracle(self, chain_id: int) -> Optional[Dict]:
        """Fallback: Fetch from Owlracle API."""
        chain = self.chains.get(chain_id)
        if not chain or 'owlracle_endpoint' not in chain:
            return None
        
        try:
            url = f"https://api.owlracle.info/v4/{chain['owlracle_endpoint']}/gas"
            headers = {}
            if hasattr(Config, 'OWLRACLE_API_KEY') and Config.OWLRACLE_API_KEY:
                headers['Authorization'] = Config.OWLRACLE_API_KEY
            
            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Owlracle returns speeds array
            standard_gas = data['speeds'][1]['gasPrice'] if len(data.get('speeds', [])) > 1 else 0
            
            return {
                'chain_id': chain_id,
                'chain_name': chain['name'],
                'timestamp': datetime.now().isoformat(),
                'current_gas': round(standard_gas, 6),
                'base_fee': round(standard_gas * 0.9, 6),  # Estimate
                'priority_fee': round(standard_gas * 0.1, 6),  # Estimate
                'block_number': data.get('timestamp', 0)
            }
        except Exception as e:
            print(f"Error fetching from Owlracle for chain {chain_id}: {e}")
            return None
    
    def get_all_chains_gas(self) -> Dict[int, Dict]:
        """
        Fetch gas prices for all supported chains.
        
        Returns:
            Dict mapping chain_id to gas data
        """
        results = {}
        for chain_id in self.chains.keys():
            gas_data = self.get_current_gas(chain_id)
            if gas_data:
                results[chain_id] = gas_data
            else:
                results[chain_id] = {
                    'chain_id': chain_id,
                    'chain_name': self.chains[chain_id]['name'],
                    'error': 'Failed to fetch gas price'
                }
        
        return results
    
    def get_supported_chains(self) -> List[int]:
        """Get list of supported chain IDs."""
        return list(self.chains.keys())


# Convenience function for backward compatibility
def get_collector_for_chain(chain_id: int = 8453):
    """Get a collector instance for a specific chain (backward compatibility)."""
    collector = MultiChainGasCollector()
    return collector.get_current_gas(chain_id)


if __name__ == "__main__":
    collector = MultiChainGasCollector()
    
    # Test single chain
    print("Testing Base chain (8453):")
    base_gas = collector.get_current_gas(8453)
    print(base_gas)
    
    print("\nTesting all chains:")
    all_gas = collector.get_all_chains_gas()
    for chain_id, data in all_gas.items():
        if 'error' not in data:
            print(f"{data['chain_name']} ({chain_id}): {data['current_gas']} gwei")
        else:
            print(f"{data['chain_name']} ({chain_id}): {data.get('error', 'Unknown error')}")
