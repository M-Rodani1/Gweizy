"""
Multi-chain gas price API routes
Provides gas prices across multiple EVM chains
"""

from flask import Blueprint, jsonify, request
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
import time

multichain_bp = Blueprint('multichain', __name__)

# Chain configurations
CHAINS = {
    8453: {
        'name': 'Base',
        'shortName': 'BASE',
        'rpcs': [
            'https://mainnet.base.org',
            'https://base.llamarpc.com',
            'https://base.drpc.org'
        ],
        'isL2': True
    },
    1: {
        'name': 'Ethereum',
        'shortName': 'ETH',
        'rpcs': [
            'https://eth.llamarpc.com',
            'https://ethereum.publicnode.com',
            'https://rpc.ankr.com/eth'
        ],
        'isL2': False
    },
    42161: {
        'name': 'Arbitrum',
        'shortName': 'ARB',
        'rpcs': [
            'https://arb1.arbitrum.io/rpc',
            'https://arbitrum.llamarpc.com',
            'https://arbitrum-one.publicnode.com'
        ],
        'isL2': True
    },
    10: {
        'name': 'Optimism',
        'shortName': 'OP',
        'rpcs': [
            'https://mainnet.optimism.io',
            'https://optimism.llamarpc.com',
            'https://optimism.publicnode.com'
        ],
        'isL2': True
    },
    137: {
        'name': 'Polygon',
        'shortName': 'MATIC',
        'rpcs': [
            'https://polygon-bor-rpc.publicnode.com',
            'https://rpc.ankr.com/polygon',
            'https://1rpc.io/matic'
        ],
        'isL2': True
    }
}

# Cache for gas prices (chain_id -> {price, timestamp})
_gas_cache: Dict[int, Dict[str, Any]] = {}
CACHE_TTL = 10  # seconds


async def fetch_gas_price(session: aiohttp.ClientSession, chain_id: int, rpc_url: str) -> Optional[float]:
    """Fetch gas price from a single RPC endpoint"""
    try:
        payload = {
            'jsonrpc': '2.0',
            'method': 'eth_gasPrice',
            'params': [],
            'id': 1
        }

        async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 200:
                data = await response.json()
                if 'result' in data:
                    gas_wei = int(data['result'], 16)
                    gas_gwei = gas_wei / 1e9
                    return gas_gwei
    except Exception as e:
        pass
    return None


async def fetch_chain_gas(session: aiohttp.ClientSession, chain_id: int) -> Dict[str, Any]:
    """Fetch gas price for a chain, trying multiple RPCs"""
    chain = CHAINS.get(chain_id)
    if not chain:
        return {
            'chainId': chain_id,
            'error': 'Unknown chain',
            'gasPrice': None
        }

    # Check cache first
    cached = _gas_cache.get(chain_id)
    if cached and (time.time() - cached['timestamp']) < CACHE_TTL:
        return {
            'chainId': chain_id,
            'name': chain['name'],
            'shortName': chain['shortName'],
            'gasPrice': cached['price'],
            'timestamp': cached['timestamp'],
            'cached': True,
            'isL2': chain['isL2']
        }

    # Try each RPC
    for rpc_url in chain['rpcs']:
        gas_price = await fetch_gas_price(session, chain_id, rpc_url)
        if gas_price is not None:
            # Update cache
            _gas_cache[chain_id] = {
                'price': gas_price,
                'timestamp': time.time()
            }

            return {
                'chainId': chain_id,
                'name': chain['name'],
                'shortName': chain['shortName'],
                'gasPrice': gas_price,
                'timestamp': time.time(),
                'cached': False,
                'isL2': chain['isL2']
            }

    return {
        'chainId': chain_id,
        'name': chain['name'],
        'shortName': chain['shortName'],
        'error': 'All RPCs failed',
        'gasPrice': None,
        'isL2': chain['isL2']
    }


async def fetch_all_chains() -> Dict[int, Dict[str, Any]]:
    """Fetch gas prices for all chains in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_chain_gas(session, chain_id) for chain_id in CHAINS.keys()]
        results = await asyncio.gather(*tasks)

        return {result['chainId']: result for result in results}


def run_async(coro):
    """Run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


@multichain_bp.route('/multichain/gas', methods=['GET'])
def get_all_chain_gas():
    """
    Get gas prices for all supported chains

    Returns:
        {
            "success": true,
            "timestamp": "2025-01-01T00:00:00Z",
            "chains": {
                "8453": { "chainId": 8453, "name": "Base", "gasPrice": 0.001, ... },
                "1": { "chainId": 1, "name": "Ethereum", "gasPrice": 15.5, ... },
                ...
            },
            "cheapest": { "chainId": 8453, "name": "Base", "gasPrice": 0.001 },
            "mostExpensive": { "chainId": 1, "name": "Ethereum", "gasPrice": 15.5 }
        }
    """
    try:
        results = run_async(fetch_all_chains())

        # Find cheapest and most expensive
        valid_results = [r for r in results.values() if r.get('gasPrice') is not None]

        cheapest = None
        most_expensive = None

        if valid_results:
            cheapest = min(valid_results, key=lambda x: x['gasPrice'])
            most_expensive = max(valid_results, key=lambda x: x['gasPrice'])

        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'chains': results,
            'cheapest': cheapest,
            'mostExpensive': most_expensive
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multichain_bp.route('/multichain/gas/<int:chain_id>', methods=['GET'])
def get_chain_gas(chain_id: int):
    """
    Get gas price for a specific chain

    Args:
        chain_id: The chain ID (e.g., 8453 for Base)

    Returns:
        Gas price info for the specified chain
    """
    if chain_id not in CHAINS:
        return jsonify({
            'success': False,
            'error': f'Chain {chain_id} not supported',
            'supportedChains': list(CHAINS.keys())
        }), 400

    try:
        async def fetch_single():
            async with aiohttp.ClientSession() as session:
                return await fetch_chain_gas(session, chain_id)

        result = run_async(fetch_single())

        return jsonify({
            'success': result.get('gasPrice') is not None,
            **result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multichain_bp.route('/multichain/compare', methods=['GET'])
def compare_chains():
    """
    Compare gas prices across chains and recommend the best one

    Query params:
        - gas_units: Number of gas units for the transaction (default: 21000)
        - tx_type: Transaction type (transfer, swap, mint, etc.)

    Returns:
        Comparison data with recommendations
    """
    gas_units = request.args.get('gas_units', 21000, type=int)
    tx_type = request.args.get('tx_type', 'transfer')

    # Gas estimates by transaction type
    TX_GAS_ESTIMATES = {
        'transfer': 21000,
        'erc20_transfer': 65000,
        'swap': 150000,
        'nft_mint': 100000,
        'nft_transfer': 80000,
        'contract_deploy': 500000,
        'bridge': 200000,
        'approve': 46000
    }

    if tx_type in TX_GAS_ESTIMATES:
        gas_units = TX_GAS_ESTIMATES[tx_type]

    try:
        results = run_async(fetch_all_chains())

        # Calculate costs for each chain
        comparisons = []
        for chain_id, data in results.items():
            if data.get('gasPrice') is not None:
                # Cost in ETH
                cost_eth = (data['gasPrice'] * gas_units) / 1e9

                comparisons.append({
                    'chainId': chain_id,
                    'name': data['name'],
                    'shortName': data['shortName'],
                    'gasPrice': data['gasPrice'],
                    'gasUnits': gas_units,
                    'costEth': cost_eth,
                    'isL2': data['isL2']
                })

        # Sort by cost
        comparisons.sort(key=lambda x: x['costEth'])

        # Calculate savings vs most expensive
        if len(comparisons) >= 2:
            cheapest_cost = comparisons[0]['costEth']
            most_expensive_cost = comparisons[-1]['costEth']

            for comp in comparisons:
                comp['savingsVsCheapest'] = comp['costEth'] - cheapest_cost
                comp['savingsVsMostExpensive'] = most_expensive_cost - comp['costEth']
                comp['savingsPercent'] = ((most_expensive_cost - comp['costEth']) / most_expensive_cost * 100) if most_expensive_cost > 0 else 0

        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'txType': tx_type,
            'gasUnits': gas_units,
            'comparisons': comparisons,
            'recommendation': comparisons[0] if comparisons else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multichain_bp.route('/multichain/supported', methods=['GET'])
def get_supported_chains():
    """Get list of supported chains"""
    return jsonify({
        'success': True,
        'chains': [
            {
                'chainId': chain_id,
                'name': chain['name'],
                'shortName': chain['shortName'],
                'isL2': chain['isL2']
            }
            for chain_id, chain in CHAINS.items()
        ]
    })
