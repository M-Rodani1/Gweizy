import requests
import time
from datetime import datetime
from web3 import Web3
from config import Config
from utils.circuit_breaker import rpc_circuit, owlracle_circuit, CircuitOpenError
from utils.logger import logger, capture_exception
from utils.rpc_manager import get_rpc_manager


class BaseGasCollector:
    def __init__(self):
        self.rpc_manager = get_rpc_manager()
        # Initialize with current RPC (will be rotated as needed)
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_manager.get_current_rpc()))
        self._session = requests.Session()

    def get_current_gas(self):
        """Fetch current Base gas price with circuit breaker protection and RPC rotation."""
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Get current RPC endpoint
                current_rpc = self.rpc_manager.get_current_rpc()
                # Update Web3 provider if RPC changed
                if self.w3.provider.endpoint_uri != current_rpc:
                    self.w3 = Web3(Web3.HTTPProvider(current_rpc))
                    logger.debug(f"Updated Web3 provider to: {current_rpc}")
                
                result = self._fetch_from_rpc(current_rpc)
                # Record success
                self.rpc_manager.record_success(current_rpc)
                return result
                
            except CircuitOpenError:
                logger.warning(f"RPC circuit open for {current_rpc}, trying Owlracle fallback")
                return self._fetch_from_owlracle()
                
            except requests.exceptions.HTTPError as e:
                # Check for rate limit (429)
                if e.response and e.response.status_code == 429:
                    logger.warning(f"Rate limit detected (429) for {current_rpc}")
                    self.rpc_manager.record_failure(current_rpc, is_rate_limit=True)
                    last_exception = e
                    # Try next RPC if available
                    continue
                else:
                    logger.error(f"HTTP error fetching gas from RPC: {e}")
                    self.rpc_manager.record_failure(current_rpc)
                    last_exception = e
                    # Try next RPC if available
                    continue
                    
            except Exception as e:
                logger.error(f"Error fetching gas from RPC: {e}")
                capture_exception(e, {'source': 'rpc_fetch', 'attempt': attempt + 1})
                self.rpc_manager.record_failure(current_rpc)
                last_exception = e
                # Try next RPC if available
                continue
        
        # All RPC attempts failed, try Owlracle fallback
        logger.warning("All RPC attempts failed, trying Owlracle fallback")
        return self._fetch_from_owlracle()

    def _fetch_from_rpc(self, rpc_url: str):
        """Fetch gas data from RPC with circuit breaker protection."""
        def rpc_call():
            # Use requests timeout for better control
            latest_block = self.w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', 0)

            # Get recent transactions to estimate priority fee
            # Reduce transaction sampling for faster requests
            block = self.w3.eth.get_block('latest', full_transactions=True)
            transactions = block.transactions[:10]  # Sample

            priority_fees = []
            for tx in transactions:
                if hasattr(tx, 'maxPriorityFeePerGas'):
                    priority_fees.append(tx.maxPriorityFeePerGas)

            avg_priority_fee = sum(priority_fees) / len(priority_fees) if priority_fees else 0

            total_gas = (base_fee + avg_priority_fee) / 1e9  # Convert to Gwei

            return {
                'timestamp': datetime.now().isoformat(),
                'current_gas': round(total_gas, 6),
                'base_fee': round(base_fee / 1e9, 6),
                'priority_fee': round(avg_priority_fee / 1e9, 6),
                'block_number': block.number,
                'rpc_source': rpc_url  # Track which RPC was used
            }

        return rpc_circuit.call(rpc_call)
    
    def _fetch_from_owlracle(self):
        """Fallback: Fetch from Owlracle API with circuit breaker protection."""
        def owlracle_call():
            url = "https://api.owlracle.info/v4/base/gas"
            headers = {}
            if Config.OWLRACLE_API_KEY:
                headers['Authorization'] = Config.OWLRACLE_API_KEY

            response = self._session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Owlracle returns speeds array
            standard_gas = data['speeds'][1]['gasPrice'] if len(data['speeds']) > 1 else 0

            return {
                'timestamp': datetime.now().isoformat(),
                'current_gas': round(standard_gas, 6),
                'base_fee': round(standard_gas * 0.9, 6),  # Estimate
                'priority_fee': round(standard_gas * 0.1, 6),  # Estimate
                'block_number': data.get('timestamp', 0)
            }

        try:
            return owlracle_circuit.call(owlracle_call)
        except CircuitOpenError:
            logger.error("Both RPC and Owlracle circuits are open - no data available")
            return None
        except Exception as e:
            logger.error(f"Error fetching from Owlracle: {e}")
            capture_exception(e, {'source': 'owlracle_fetch'})
            return None


# Usage
if __name__ == "__main__":
    collector = BaseGasCollector()
    data = collector.get_current_gas()
    print(data)
