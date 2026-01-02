from web3 import Web3
from config import Config
from utils.logger import logger
from datetime import datetime
import requests


class BaseScanner:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(Config.BASE_RPC_URL))
        
    def get_recent_transactions(self, limit=10):
        """
        Fetch recent transactions from Base network
        Returns list of formatted transaction data
        """
        try:
            latest_block = self.w3.eth.get_block('latest', full_transactions=True)
            
            transactions = []
            for tx in latest_block.transactions[:limit]:
                # Decode method name (first 4 bytes of input data)
                method = self._decode_method(tx.input)
                
                # Calculate age
                block_timestamp = latest_block.timestamp
                age = self._format_age(datetime.now().timestamp() - block_timestamp)
                
                # Calculate gas price
                gas_price_gwei = tx.get('gasPrice', 0) / 1e9 if tx.get('gasPrice') else 0
                
                # If EIP-1559 transaction
                if hasattr(tx, 'maxFeePerGas'):
                    gas_price_gwei = tx.maxFeePerGas / 1e9
                
                transactions.append({
                    'txHash': tx.hash.hex()[:10] + '...' + tx.hash.hex()[-4:],
                    'method': method,
                    'age': age,
                    'gasUsed': tx.get('gas', 0),
                    'gasPrice': round(gas_price_gwei, 4),
                    'timestamp': int(block_timestamp)
                })
            
            logger.info(f"Fetched {len(transactions)} recent transactions")
            return transactions
            
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return self._get_fallback_transactions()
    
    def _decode_method(self, input_data):
        """Decode transaction method from input data"""
        if not input_data or input_data == '0x':
            return 'Transfer'
        
        # Common method signatures
        methods = {
            '0xa9059cbb': 'Transfer',
            '0x095ea7b3': 'Approve',
            '0x23b872dd': 'TransferFrom',
            '0x40c10f19': 'Mint',
            '0x42842e0e': 'SafeTransferFrom',
            '0x7ff36ab5': 'Swap',
            '0x38ed1739': 'SwapExactTokensForTokens',
        }
        
        signature = input_data[:10]
        return methods.get(signature, 'Contract Call')
    
    def _format_age(self, seconds):
        """Format timestamp as human-readable age"""
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        else:
            return f"{int(seconds/3600)}h ago"
    
    def _get_fallback_transactions(self):
        """Return sample transactions if RPC fails"""
        logger.warning("Using fallback transaction data")
        return [
            {'txHash': '0x1a2b...c3d4', 'method': 'Swap', 'age': '12s ago', 
             'gasUsed': 154321, 'gasPrice': 0.0025, 'timestamp': int(datetime.now().timestamp())},
            {'txHash': '0x5e6f...g7h8', 'method': 'Transfer', 'age': '25s ago', 
             'gasUsed': 21000, 'gasPrice': 0.0023, 'timestamp': int(datetime.now().timestamp())},
        ]


# Test
if __name__ == "__main__":
    scanner = BaseScanner()
    txs = scanner.get_recent_transactions(5)
    for tx in txs:
        print(tx)

